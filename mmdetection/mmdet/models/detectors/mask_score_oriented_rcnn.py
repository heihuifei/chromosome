# Copyright (c) OpenMMLab. All rights reserved.
from ..builder import DETECTORS
from .oriented_two_stage import OrientedTwoStageDetector


@DETECTORS.register_module()
class MaskScoringOrientedRCNN(OrientedTwoStageDetector):
    """Mask Scoring RCNN.

    https://arxiv.org/abs/1903.00241
    """

    def __init__(self,
                 backbone,
                 rpn_head,
                 rrpn_head,
                 roi_head,
                 rroi_head,
                 train_cfg,
                 test_cfg,
                 neck=None,
                 pretrained=None,
                 init_cfg=None):
        super(MaskScoringOrientedRCNN, self).__init__(
            backbone=backbone,
            neck=neck,
            rpn_head=rpn_head,
            rrpn_head=rrpn_head,
            roi_head=roi_head,
            rroi_head=rroi_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
