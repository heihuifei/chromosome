# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import kaiming_init, trunc_normal_init
from timm.models.layers import trunc_normal_, DropPath
from mmcv.runner import BaseModule, auto_fp16, force_fp32
from torch.nn import init

from mmcv_custom import load_checkpoint
from mmdet.utils import get_root_logger
from mmdet.models.builder import HEADS, build_loss
from mmdet.models.losses import accuracy


@HEADS.register_module()
class CountHead(BaseModule):

    def __init__(self,
                 loss_count=dict(type='CrossEntropyLoss', loss_weight=1.0),
                 in_channels=256,
                 num_levels=5,
                 end_level_h=13,
                 end_level_w=15,
                 strides=None,
                 init_cfg=None):
        super(CountHead, self).__init__(init_cfg)
        assert isinstance(loss_count, dict)
        assert num_levels == len(strides), "num_levels must equal to len(strides)"
        self.in_channels = in_channels
        self.strides = strides
        self.end_level_h = end_level_h
        self.end_level_w = end_level_w
        self.count_convs = self.build_count_convs(strides)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(self.in_channels * self.end_level_h * self.end_level_w, 1)
        self.loss_count = build_loss(loss_count)
        # self.apply(self._init_weights)

    def build_count_convs(self, strides):
        count_convs = nn.ModuleList(
            [nn.Conv2d(self.in_channels, self.in_channels, kernel_size=s, stride=s) for s in strides])
        return count_convs

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                kaiming_init(m)
            elif isinstance(m, nn.Linear):
                    trunc_normal_init(m, std=.02, bias=0.)

    @auto_fp16()
    def forward(self, x):
        count_pred = []
        for i in range(len(x)):
            print("this is x[i] in forward in count_head: ", type(x[i]), x[i].shape, x[i])
            feat = F.relu(x[i], inplace=True)
            print("this is feat in forward in count_head: ", type(feat), feat.shape, feat)
            # 针对每层x特征图进行卷积到相同尺寸并扁平化处理为一维Tensor
            fc_input = self.count_convs[i](feat)
            print("this is fc_input.shape[1] in forward in count_head: ", type(fc_input), fc_input.shape, fc_input)
            assert fc_input.shape[1] == self.in_channels * self.end_level_h * self.end_level_w, "fc_input must equal to fc in_channels"
            fc_output = self.fc(fc_input)
            count_pred.append(fc_output)
        return count_pred

    @force_fp32(apply_to=('count_pred'))
    def loss(self,
             count_pred,
             count_gt,
             reduction_override=None):
        losses = dict()
        if count_pred is not None:
            print("this is in count_loss, loss_count's input count_pred and count_gt: ", type(count_pred), count_pred, type(count_gt), count_gt)
            loss = self.loss_count(count_pred, count_gt)
        losses['count'] = loss
        return losses

    def onnx_export(self,
                    rois,
                    cls_score,
                    bbox_pred,
                    img_shape,
                    cfg=None,
                    **kwargs):
        """Transform network output for a batch into bbox predictions.

        Args:
            rois (Tensor): Boxes to be transformed.
                Has shape (B, num_boxes, 5)
            cls_score (Tensor): Box scores. has shape
                (B, num_boxes, num_classes + 1), 1 represent the background.
            bbox_pred (Tensor, optional): Box energies / deltas for,
                has shape (B, num_boxes, num_classes * 4) when.
            img_shape (torch.Tensor): Shape of image.
            cfg (obj:`ConfigDict`): `test_cfg` of Bbox Head. Default: None

        Returns:
            tuple[Tensor, Tensor]: dets of shape [N, num_det, 5]
                and class labels of shape [N, num_det].
        """

        assert rois.ndim == 3, 'Only support export two stage ' \
                               'model to ONNX ' \
                               'with batch dimension. '
        if self.custom_cls_channels:
            scores = self.loss_cls.get_activation(cls_score)
        else:
            scores = F.softmax(
                cls_score, dim=-1) if cls_score is not None else None

        if bbox_pred is not None:
            bboxes = self.bbox_coder.decode(
                rois[..., 1:], bbox_pred, max_shape=img_shape)
        else:
            bboxes = rois[..., 1:].clone()
            if img_shape is not None:
                max_shape = bboxes.new_tensor(img_shape)[..., :2]
                min_xy = bboxes.new_tensor(0)
                max_xy = torch.cat(
                    [max_shape] * 2, dim=-1).flip(-1).unsqueeze(-2)
                bboxes = torch.where(bboxes < min_xy, min_xy, bboxes)
                bboxes = torch.where(bboxes > max_xy, max_xy, bboxes)

        # Replace multiclass_nms with ONNX::NonMaxSuppression in deployment
        from mmdet.core.export import add_dummy_nms_for_onnx
        max_output_boxes_per_class = cfg.nms.get('max_output_boxes_per_class',
                                                 cfg.max_per_img)
        iou_threshold = cfg.nms.get('iou_threshold', 0.5)
        score_threshold = cfg.score_thr
        nms_pre = cfg.get('deploy_nms_pre', -1)

        scores = scores[..., :self.num_classes]
        if self.reg_class_agnostic:
            return add_dummy_nms_for_onnx(
                bboxes,
                scores,
                max_output_boxes_per_class,
                iou_threshold,
                score_threshold,
                pre_top_k=nms_pre,
                after_top_k=cfg.max_per_img)
        else:
            batch_size = scores.shape[0]
            labels = torch.arange(
                self.num_classes, dtype=torch.long).to(scores.device)
            labels = labels.view(1, 1, -1).expand_as(scores)
            labels = labels.reshape(batch_size, -1)
            scores = scores.reshape(batch_size, -1)
            bboxes = bboxes.reshape(batch_size, -1, 4)

            max_size = torch.max(img_shape)
            # Offset bboxes of each class so that bboxes of different labels
            #  do not overlap.
            offsets = (labels * max_size + 1).unsqueeze(2)
            bboxes_for_nms = bboxes + offsets

            batch_dets, labels = add_dummy_nms_for_onnx(
                bboxes_for_nms,
                scores.unsqueeze(2),
                max_output_boxes_per_class,
                iou_threshold,
                score_threshold,
                pre_top_k=nms_pre,
                after_top_k=cfg.max_per_img,
                labels=labels)
            # Offset the bboxes back after dummy nms.
            offsets = (labels * max_size + 1).unsqueeze(2)
            # Indexing + inplace operation fails with dynamic shape in ONNX
            # original style: batch_dets[..., :4] -= offsets
            bboxes, scores = batch_dets[..., 0:4], batch_dets[..., 4:5]
            bboxes -= offsets
            batch_dets = torch.cat([bboxes, scores], dim=2)
            return batch_dets, labels
