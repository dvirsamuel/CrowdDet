import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
from torch.nn import CrossEntropyLoss

from model.rcnn_fpn_deepsets.config import config
from lib.backbone.resnet50 import ResNet50
from lib.backbone.fpn import FPN
from lib.module.rpn import RPN
from lib.layers.pooler import roi_pooler
from lib.det_oprs.bbox_opr import bbox_transform_inv_opr
from lib.det_oprs.fpn_roi_target import fpn_roi_target
from lib.det_oprs.loss_opr import softmax_loss, smooth_l1_loss
from lib.det_oprs.utils import get_padded_tensor


def bbox_overlaps(bboxes1, bboxes2, mode='iou', is_aligned=False):
    """Calculate overlap between two set of bboxes.

    If ``is_aligned`` is ``False``, then calculate the ious between each bbox
    of bboxes1 and bboxes2, otherwise the ious between each aligned pair of
    bboxes1 and bboxes2.

    Args:
        bboxes1 (Tensor): shape (m, 4)
        bboxes2 (Tensor): shape (n, 4), if is_aligned is ``True``, then m and n
            must be equal.
        mode (str): "iou" (intersection over union) or iof (intersection over
            foreground).

    Returns:
        ious(Tensor): shape (m, n) if is_aligned == False else shape (m, 1)
    """

    assert mode in ['iou', 'iof']

    rows = bboxes1.size(0)
    cols = bboxes2.size(0)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        return bboxes1.new(rows, 1) if is_aligned else bboxes1.new(rows, cols)

    if is_aligned:
        lt = torch.max(bboxes1[:, :2], bboxes2[:, :2])  # [rows, 2]
        rb = torch.min(bboxes1[:, 2:], bboxes2[:, 2:])  # [rows, 2]

        wh = (rb - lt + 1).clamp(min=0)  # [rows, 2]
        overlap = wh[:, 0] * wh[:, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
            bboxes1[:, 3] - bboxes1[:, 1] + 1)

        if mode == 'iou':
            area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
                bboxes2[:, 3] - bboxes2[:, 1] + 1)
            ious = overlap / (area1 + area2 - overlap)
        else:
            ious = overlap / area1
    else:
        lt = torch.max(bboxes1[:, None, :2], bboxes2[:, :2])  # [rows, cols, 2]
        rb = torch.min(bboxes1[:, None, 2:], bboxes2[:, 2:])  # [rows, cols, 2]

        wh = (rb - lt + 1).clamp(min=0)  # [rows, cols, 2]
        overlap = wh[:, :, 0] * wh[:, :, 1]
        area1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (
            bboxes1[:, 3] - bboxes1[:, 1] + 1)

        if mode == 'iou':
            area2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (
                bboxes2[:, 3] - bboxes2[:, 1] + 1)
            ious = overlap / (area1[:, None] + area2 - overlap)
        else:
            ious = overlap / (area1[:, None])

    return ious


class PermEqui2_mean(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.Gamma = nn.Linear(in_dim, out_dim)
        self.Lambda = nn.Linear(in_dim, out_dim, bias=False)
        self.weight = self.Gamma.weight
        self.bias = self.Gamma.bias


    def forward(self, x):
        xm = x.mean(0, keepdim=True)
        xm = self.Lambda(xm)
        x = self.Gamma(x)
        x = x - xm
        return x


class DeepsetsHead(nn.Module):
    def __init__(self,
                 indim=1033,
                 ds1=1000,
                 ds2=600,
                 ds3=300,
                 ds4=150):
        super(DeepsetsHead, self).__init__()
        self.loss_ce = CrossEntropyLoss()
        self.ds1 = PermEqui2_mean(indim, ds1)
        self.ds2 = PermEqui2_mean(ds1, ds2)
        self.ds3 = PermEqui2_mean(ds2, ds3)
        self.ds4 = PermEqui2_mean(ds3, ds4)
        self.ds5 = PermEqui2_mean(ds4, 1)
        self.reg = 5 # todo: config


    def set_forward(self, x):
        x = F.elu(self.ds1(x))
        x = F.elu(self.ds2(x))
        x = F.elu(self.ds3(x))
        x = F.elu(self.ds4(x))
        pred = F.elu(self.ds5(x))
        return pred


    def forward(self, multi_bboxes, cls_score, last_layer_feats,
                mode='train', ds_cfg=None, img_shape=None, gt_labels=None):
        preds = []
        inputs = []
        set_bboxes = []
        input_labels = []
        top_c = ds_cfg['top_c']
        max_num = ds_cfg['max_num']
        iou_threshold = ds_cfg['iou_thresh']
        # if mode == 'train':
        #     classes = torch.unique(gt_labels)
        #     classes = [0]
        # else:
        #     classes = torch.unique(torch.topk(cls_score[:, 1:], top_c, dim=1)[1]) + 1  # top_classes_on_set
        classes = [1]
        for c in classes:
            sets = []
            #bboxes = multi_bboxes[:, c * 4:(c + 1) * 4]
            bboxes = multi_bboxes
            scores = cls_score[:, c]
            scores, inds = scores.sort(descending=True)
            #inds = inds[:max_num]
            bboxes = bboxes[inds]
            feats = last_layer_feats[inds]
            ious = bbox_overlaps(bboxes, bboxes)
            is_clustered = torch.ones(ious.shape[0]).cuda()
            for j, row in enumerate(ious):
                if is_clustered[j] == 1:
                    selected_indices = torch.nonzero((row > iou_threshold) * is_clustered).squeeze()
                    is_clustered *= torch.where(row > iou_threshold, torch.zeros(1).cuda(), torch.ones(1).cuda())
                    sets.append(selected_indices)

            for s, _set in enumerate(sets):
                if _set.ndim == 0:  # were set includes only one object
                    continue
                else:
                    x1 = bboxes[sets[s], 0].unsqueeze(1) / img_shape[1]
                    x2 = bboxes[sets[s], 2].unsqueeze(1) / img_shape[1]
                    y1 = bboxes[sets[s], 1].unsqueeze(1) / img_shape[0]
                    y2 = bboxes[sets[s], 3].unsqueeze(1) / img_shape[0]
                    width = (x2 - x1) / img_shape[1]
                    height = (y2 - y1) / img_shape[0]
                    area = width * height
                    aspect_ratio = torch.div(width, height+sys.float_info.epsilon)
                    input = torch.cat([bboxes[sets[s]], feats[sets[s]], width, height, aspect_ratio, area,
                                       scores[sets[s]].unsqueeze(1)], dim=1)
                    _set_bboxes = bboxes[sets[s]]
                    _, top_scores_idx = input[:, -1].sort(descending=True)
                    #top_scores_idx = top_scores_idx[:self.set_size]
                    input = input[top_scores_idx]
                    _set_bboxes = _set_bboxes[top_scores_idx]
                    randperm = torch.randperm(len(input))
                    input = input[randperm]
                    set_bboxes.append(_set_bboxes[randperm])
                    inputs.append(input)
                    input_labels.append(c)
                    pred = self.set_forward(input)
                    preds.append(pred)
        return inputs, preds, input_labels, set_bboxes


    def get_target(self,
                   sets,
                   set_labels,
                   set_bboxes,
                   gt_bboxes,
                   gt_labels,
                   preds):
        """
        returns deepsets labels.
        a. for each class on each set, find closest gt,
        b. calculate iou between set and gt box,
        c. normalize with max iou.
        """
        one_hot_targets = []
        soft_targets = []
        valid_preds = []

        non_padded_gt_boxes = []
        for g in gt_bboxes:
            non_padded_gt_boxes += [
                g[:torch.where(torch.all(torch.isclose(g, torch.tensor([0], dtype=torch.float).cuda()), axis=1))[0][0]]]

        valid_ious = []
        for j, set in enumerate(sets):
            c = set_labels[j]
            if len(set) > 0 and c in gt_labels:
                #gt_class_inds = torch.where(gt_labels == c)[0].squeeze()
                # takes boxes of relevant gt
                #class_boxes = torch.index_select(torch.tensor(non_padded_gt_boxes[0]), 0, gt_class_inds)
                gt_iou = bbox_overlaps(set_bboxes[j], non_padded_gt_boxes[0])
                vals, row_idx = gt_iou.max(0)  # row idx: indices of set elements with max ious with each GT (1, num_gts)
                col_idx = vals.argmax(0)  # col idx: index of GT with highest iou with any set element
                if torch.max(gt_iou[:, col_idx]) < 0.5:
                    continue
                trg = torch.zeros(len(set), dtype=torch.int64).cuda()
                trg[row_idx[col_idx]] = 1
                one_hot_targets.append(trg)
                valid_preds.append(preds[j])
                valid_ious.append(gt_iou[:, col_idx])
        return valid_preds, one_hot_targets, valid_ious


    def loss(self, preds, one_hot_targets, valid_ious):
        losses = dict()
        c = sys.float_info.epsilon
        if preds is not None:
            _mse_loss = torch.zeros(1).cuda()
            _ce_loss = torch.zeros(1).cuda()
            _soft_ce = torch.zeros(1).cuda()
            _error = torch.zeros(1).cuda()
            _ds_acc = torch.zeros(1).cuda()
            _ds_pred_on_max = torch.zeros(1).cuda()
            iou_error = torch.zeros(1).cuda()
            for i, pred in enumerate(preds):
                _ce_loss += self.loss_ce(
                    pred.T,
                    torch.argmax(one_hot_targets[i]).unsqueeze(0))
                _ds_acc += torch.argmax(pred) == torch.argmax(one_hot_targets[i])
                iou_error += torch.max(valid_ious[i]) - valid_ious[i][torch.argmax(pred)]
                c += 1
            _ce_loss /= c
            _ds_acc /= c
            iou_error /= c
            losses['loss_deepsets_ce'] = _ce_loss + self.reg*iou_error
            losses['ds_acc'] = _ds_acc
            losses['iou_error'] = iou_error
        else:
            print('set with no valid predictions')
            dl = torch.zeros(1).cuda()
            dl.requires_grad = True
            losses['loss_deepsets_ce'] = dl
            losses['ds_acc'] = torch.zeros(1).cuda()
            losses['iou_error'] = torch.zeros(1).cuda()
        return losses


class Network(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet50 = ResNet50(config.backbone_freeze_at, False)
        self.FPN = FPN(self.resnet50, 2, 6)
        self.RPN = RPN(config.rpn_channel)
        self.RCNN = RCNN()

    def forward(self, image, im_info, gt_boxes=None):
        image = (image - torch.tensor(config.image_mean[None, :, None, None]).type_as(image)) / (
                torch.tensor(config.image_std[None, :, None, None]).type_as(image))
        image = get_padded_tensor(image, 64)
        if self.training:
            return self._forward_train(image, im_info, gt_boxes)
        else:
            return self._forward_test(image, im_info)

    def _forward_train(self, image, im_info, gt_boxes):
        loss_dict = {}
        fpn_fms = self.FPN(image)
        # fpn_fms stride: 64,32,16,8,4, p6->p2
        rpn_rois, loss_dict_rpn = self.RPN(fpn_fms, im_info, gt_boxes)
        rcnn_rois, rcnn_labels, rcnn_bbox_targets = fpn_roi_target(
                rpn_rois, im_info, gt_boxes, top_k=1)
        loss_dict_rcnn = self.RCNN(fpn_fms, rcnn_rois,
                rcnn_labels, rcnn_bbox_targets, image.shape[2:], gt_boxes)
        loss_dict.update(loss_dict_rpn)
        loss_dict.update(loss_dict_rcnn)
        return loss_dict

    def _forward_test(self, image, im_info):
        fpn_fms = self.FPN(image)
        rpn_rois = self.RPN(fpn_fms, im_info)
        pred_bbox = self.RCNN(fpn_fms, rpn_rois, image_shape=image.shape[2:])
        return pred_bbox.cpu().detach()

class RCNN(nn.Module):
    def __init__(self):
        super().__init__()
        # roi head
        self.fc1 = nn.Linear(256*7*7, 1024)
        self.fc2 = nn.Linear(1024, 1024)
        self.deepsets_head = DeepsetsHead()

        for l in [self.fc1, self.fc2]:
            nn.init.kaiming_uniform_(l.weight, a=1)
            nn.init.constant_(l.bias, 0)
        # box predictor
        self.pred_cls = nn.Linear(1024, config.num_classes)
        self.pred_delta = nn.Linear(1024, config.num_classes * 4)
        for l in [self.pred_cls]:
            nn.init.normal_(l.weight, std=0.01)
            nn.init.constant_(l.bias, 0)
        for l in [self.pred_delta]:
            nn.init.normal_(l.weight, std=0.001)
            nn.init.constant_(l.bias, 0)

    def forward(self, fpn_fms, rcnn_rois, labels=None, bbox_targets=None, image_shape=None, gt_boxes=None):
        ds_cfg=dict(
                top_c=3,
                max_num=500,
                iou_thresh=0.5)
        # input p2-p5
        fpn_fms = fpn_fms[1:][::-1]
        stride = [4, 8, 16, 32]
        pool_features = roi_pooler(fpn_fms, rcnn_rois, stride, (7, 7), "ROIAlignV2")
        flatten_feature = torch.flatten(pool_features, start_dim=1)
        flatten_feature = F.relu_(self.fc1(flatten_feature))
        flatten_feature = F.relu_(self.fc2(flatten_feature))
        pred_cls = self.pred_cls(flatten_feature)
        pred_delta = self.pred_delta(flatten_feature)
        if self.training:
            # loss for regression
            labels = labels.long().flatten()
            fg_masks = labels > 0
            valid_masks = labels >= 0
            # multi class
            class_num = pred_cls.shape[-1] - 1
            pred_delta2 = pred_delta[:, 4:].reshape(-1, 4)
            base_rois = rcnn_rois[:, 1:5].repeat(1, class_num).reshape(-1, 4)
            pred_bbox = restore_bbox(base_rois, pred_delta2, True)

            pred_delta = pred_delta.reshape(-1, config.num_classes, 4)
            fg_gt_classes = labels[fg_masks]
            pred_delta = pred_delta[fg_masks, fg_gt_classes, :]
            localization_loss = smooth_l1_loss(
                pred_delta,
                bbox_targets[fg_masks],
                config.rcnn_smooth_l1_beta)
            # loss for classification
            objectness_loss = softmax_loss(pred_cls, labels)
            objectness_loss = objectness_loss * valid_masks
            normalizer = 1.0 / valid_masks.sum().item()
            loss_rcnn_loc = localization_loss.sum() * normalizer
            loss_rcnn_cls = objectness_loss.sum() * normalizer
            loss_dict = {}
            loss_dict['loss_rcnn_loc'] = loss_rcnn_loc
            loss_dict['loss_rcnn_cls'] = loss_rcnn_cls
            ####### deepsets ########
            pred_scores = F.softmax(pred_cls, dim=-1)
            sets, preds, set_labels, set_bboxes = self.deepsets_head.forward(pred_bbox, pred_scores, flatten_feature,
                                                img_shape=image_shape, gt_labels=labels, ds_cfg=ds_cfg)
            valid_preds, one_hot_targets, valid_ious = self.deepsets_head.get_target(sets, set_labels, set_bboxes,
                                                                         gt_boxes[:,:,:-1], labels, preds)
            loss_deepsets = self.deepsets_head.loss(valid_preds, one_hot_targets, valid_ious)
            loss_dict['loss_deepsets'] = loss_deepsets["loss_deepsets_ce"][0]
            ##########################
            return loss_dict
        else:
            class_num = pred_cls.shape[-1] - 1
            tag = torch.arange(class_num).type_as(pred_cls)+1
            tag = tag.repeat(pred_cls.shape[0], 1).reshape(-1, 1)
            pred_scores = F.softmax(pred_cls, dim=-1)[:, 1:].reshape(-1, 1)
            pred_delta = pred_delta[:, 4:].reshape(-1, 4)
            base_rois = rcnn_rois[:, 1:5].repeat(1, class_num).reshape(-1, 4)
            pred_bbox = restore_bbox(base_rois, pred_delta, True)
            pred_bbox_new = torch.cat([pred_bbox, pred_scores, tag], axis=1) #TOdo
            # ####### deepsets ########
            sets, preds, set_labels, set_bboxes = self.deepsets_head.forward(pred_bbox, F.softmax(pred_cls, dim=-1), flatten_feature,
                                                                             img_shape=image_shape, mode='test',
                                                                             ds_cfg=ds_cfg)
            det_bboxes = []
            #det_labels = []
            for i, _set in enumerate(sets):
                correct_indices = torch.argmax(preds[i][:len(set_bboxes[i])])  # soft ds
                bbox = set_bboxes[i][correct_indices].unsqueeze(0)
                score = torch.mean(_set[:, -1][_set[:, -1] != 0.0]) * torch.ones(1, 1).cuda()
                #bbox = torch.cat([bbox, score], dim=1)
                det_bboxes.append(bbox)
                label = set_labels[i]
                #det_labels.append(label - 1)
            det_bboxes = torch.stack(det_bboxes, dim=1).squeeze()
            #det_labels = torch.stack(det_labels).squeeze()
            k = 100 #rcnn_test_cfg.max_per_img
            _, inds = det_bboxes[:, -1].sort(descending=True)
            inds = inds[:k]
            det_bboxes = det_bboxes[inds]
            det_scores = pred_scores[inds]
            tag = tag[inds]
            #det_labels = det_labels[inds]
            if det_bboxes.shape[0] < k:
                det_bboxes = torch.cat([det_bboxes, torch.zeros((k - det_bboxes.shape[0]), det_bboxes.shape[1]).cuda()],
                                       0)
                det_scores = torch.cat([det_scores, torch.zeros((k - det_scores.shape[0]), det_scores.shape[1]).cuda()],
                                       0)
                #det_labels = torch.cat([det_labels, torch.zeros((k - det_labels.shape[0]), dtype=int).cuda()], 0)

            # return det_bboxes, det_labels
            # #########################
            tag = torch.arange(class_num).type_as(det_scores) + 1
            tag = tag.repeat(det_scores.shape[0], 1).reshape(-1, 1)
            pred_all = torch.cat([det_bboxes, det_scores, tag], axis=1)
            return pred_all

def restore_bbox(rois, deltas, unnormalize=True):
    if unnormalize:
        std_opr = torch.tensor(config.bbox_normalize_stds[None, :]).type_as(deltas)
        mean_opr = torch.tensor(config.bbox_normalize_means[None, :]).type_as(deltas)
        deltas = deltas * std_opr
        deltas = deltas + mean_opr
    pred_bbox = bbox_transform_inv_opr(rois, deltas)
    return pred_bbox

import sys