# Copyright (c) OpenMMLab. All rights reserved.
import warnings

import torch
import mmcv
import numpy as np

from ..builder import DETECTORS, build_backbone, build_head, build_neck
from .base import BaseDetector
from mmdet.core.visualization import imshow_det_bboxes, imshow_det_bboxes_rbboxes


@DETECTORS.register_module()
class OrientedTwoStageDetector(BaseDetector):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone,
                 neck=None,
                 rpn_head=None,
                 rrpn_head=None,
                 roi_head=None,
                 rroi_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None):
        super(OrientedTwoStageDetector, self).__init__(init_cfg)
        if pretrained:
            warnings.warn('DeprecationWarning: pretrained is deprecated, '
                          'please use "init_cfg" instead')
            backbone.pretrained = pretrained
        self.backbone = build_backbone(backbone)

        if neck is not None:
            self.neck = build_neck(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            self.rpn_head = build_head(rpn_head_)
        
        if rrpn_head is not None:
            rrpn_train_cfg = train_cfg.rrpn if train_cfg is not None else None
            rrpn_head_ = rrpn_head.copy()
            rrpn_head_.update(train_cfg=rrpn_train_cfg, test_cfg=test_cfg.rrpn)
            self.rrpn_head = build_head(rrpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            roi_head.pretrained = pretrained
            self.roi_head = build_head(roi_head)
        
        if rroi_head is not None:
            # TODO: refactor assigner & sampler
            r_rcnn_train_cfg = train_cfg.r_rcnn if train_cfg is not None else None
            rroi_head.update(train_cfg=r_rcnn_train_cfg)
            rroi_head.update(test_cfg=test_cfg.r_rcnn)
            rroi_head.pretrained = pretrained
            self.rroi_head = build_head(rroi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    @property
    def with_rpn(self):
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_rrpn(self):
        """bool: whether the detector has RRPN"""
        return hasattr(self, 'rrpn_head') and self.rrpn_head is not None

    @property
    def with_roi_head(self):
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    @property
    def with_rroi_head(self):
        """bool: whether the detector has a RRoI head"""
        return hasattr(self, 'rroi_head') and self.rroi_head is not None

    def extract_feat(self, img):
        """Directly extract features from the backbone+neck."""
        x = self.backbone(img)
        if self.with_neck:
            x = self.neck(x)
        return x

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/analysis_tools/get_flops.py`
        """
        outs = ()
        # backbone
        x = self.extract_feat(img)
        # rpn
        if self.with_rpn:
            rpn_outs = self.rpn_head(x)
            outs = outs + (rpn_outs, )
        proposals = torch.randn(1000, 4).to(img.device)
        # roi_head
        roi_outs = self.roi_head.forward_dummy(x, proposals)
        outs = outs + (roi_outs, )
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_rbboxes,
                      gt_bboxes_ignore=None,
                      gt_rbboxes_ignore=None,
                      gt_masks=None,
                      proposals=None,
                      rproposals=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmdet/datasets/pipelines/formatting.py:Collect`.

            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.

            gt_labels (list[Tensor]): class indices corresponding to each box

            gt_bboxes_ignore (None | list[Tensor]): specify which bounding
                boxes can be ignored when computing the loss.

            gt_masks (None | Tensor) : true segmentation masks for each box
                used if the architecture supports a segmentation task.

            proposals : override rpn proposals with custom proposals. Use when
                `with_rpn` is False.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """
        x = self.extract_feat(img)

        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal',
                                              self.test_cfg.rpn)
            rpn_losses, proposal_list = self.rpn_head.forward_train(
                x,
                img_metas,
                gt_bboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_bboxes_ignore,
                proposal_cfg=proposal_cfg,
                **kwargs)
            losses.update(rpn_losses)
        else:
            proposal_list = proposals

        # RRPN forward and loss
        if self.with_rrpn:
            rproposal_cfg = self.train_cfg.get('rrpn_proposal',
                                              self.test_cfg.rrpn)
            rrpn_losses, rproposal_list = self.rrpn_head.forward_train(
                x,
                img_metas,
                gt_rbboxes,
                gt_labels=None,
                gt_bboxes_ignore=gt_rbboxes_ignore,
                proposal_cfg=rproposal_cfg,
                **kwargs)
            losses.update(rrpn_losses)
        else:
            rproposal_list = rproposals

        roi_losses = self.roi_head.forward_train(x, img_metas, proposal_list,
                                                 gt_bboxes, gt_labels,
                                                 gt_bboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(roi_losses)
        
        rroi_losses = self.rroi_head.forward_train(x, img_metas, rproposal_list,
                                                 gt_rbboxes, gt_labels,
                                                 gt_rbboxes_ignore, gt_masks,
                                                 **kwargs)
        losses.update(rroi_losses)

        return losses

    async def async_simple_test(self,
                                img,
                                img_meta,
                                proposals=None,
                                rescale=False):
        """Async test without augmentation."""
        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)

        if proposals is None:
            proposal_list = await self.rpn_head.async_simple_test_rpn(
                x, img_meta)
        else:
            proposal_list = proposals

        return await self.roi_head.async_simple_test(
            x, proposal_list, img_meta, rescale=rescale)

    # TODO: add rroi_head.simple_test
    def simple_test(self, img, img_metas, proposals=None, rproposals=None, rescale=False):
        """Test without augmentation."""

        assert self.with_bbox, 'Bbox head must be implemented.'
        x = self.extract_feat(img)
        if proposals is None:
            proposal_list = self.rpn_head.simple_test_rpn(x, img_metas)
            rproposal_list = self.rrpn_head.simple_test_rpn(x, img_metas)
        else:
            proposal_list = proposals
            rproposal_list = rproposals
        
        # roi_head_results为[([array(n, 4)], [[array(w, h), array(w, h)...]])]
        # rroi_head_result为[[array(n, 4)], ...]，list元素是以图像为单位，因此可以对其进行zip为tuple
        roi_head_results = self.roi_head.simple_test(
            x, proposal_list, img_metas, rescale=rescale)
        rroi_head_results = self.rroi_head.simple_test(
            x, rproposal_list, img_metas, rescale=rescale)
        bbox_results, segm_results = zip(*roi_head_results)
        rbbox_results = rroi_head_results
        return list(zip(list(bbox_results), list(segm_results), rbbox_results))

    def aug_test(self, imgs, img_metas, rescale=False):
        """Test with augmentations.

        If rescale is False, then returned bboxes and masks will fit the scale
        of imgs[0].
        """
        x = self.extract_feats(imgs)
        proposal_list = self.rpn_head.aug_test_rpn(x, img_metas)
        return self.roi_head.aug_test(
            x, proposal_list, img_metas, rescale=rescale)

    def onnx_export(self, img, img_metas):

        img_shape = torch._shape_as_tensor(img)[2:]
        img_metas[0]['img_shape_for_onnx'] = img_shape
        x = self.extract_feat(img)
        proposals = self.rpn_head.onnx_export(x, img_metas)
        if hasattr(self.roi_head, 'onnx_export'):
            return self.roi_head.onnx_export(x, proposals, img_metas)
        else:
            raise NotImplementedError(
                f'{self.__class__.__name__} can not '
                f'be exported to ONNX. Please refer to the '
                f'list of supported models,'
                f'https://mmdetection.readthedocs.io/en/latest/tutorials/pytorch2onnx.html#list-of-supported-models-exportable-to-onnx'  # noqa E501
            )

    def show_result(self,
                    img,
                    result,
                    score_thr=0.3,
                    bbox_color=(72, 101, 241),
                    text_color=(72, 101, 241),
                    mask_color=None,
                    thickness=2,
                    font_size=13,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (Tensor or tuple): The results to draw over `img`
                bbox_result or (bbox_result, segm_result).
            score_thr (float, optional): Minimum score of bboxes to be shown.
                Default: 0.3.
            bbox_color (str or tuple(int) or :obj:`Color`):Color of bbox lines.
               The tuple of color should be in BGR order. Default: 'green'
            text_color (str or tuple(int) or :obj:`Color`):Color of texts.
               The tuple of color should be in BGR order. Default: 'green'
            mask_color (None or str or tuple(int) or :obj:`Color`):
               Color of masks. The tuple of color should be in BGR order.
               Default: None
            thickness (int): Thickness of lines. Default: 2
            font_size (int): Font size of texts. Default: 13
            win_name (str): The window name. Default: ''
            wait_time (float): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """
        img = mmcv.imread(img)
        img = img.copy()
        if isinstance(result, tuple):
            if len(result) == 2:
                bbox_result, segm_result = result
            elif len(result) == 3:
                bbox_result, segm_result, rbbox_result = result
            if isinstance(segm_result, tuple):
                segm_result = segm_result[0]  # ms rcnn
        else:
            bbox_result, segm_result, rbbox_result = result, None, None
        bboxes = np.vstack(bbox_result)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_result)
        ]
        labels = np.concatenate(labels)
        rbboxes = np.vstack(rbbox_result)
        rlabels = [
            np.full(rbbox.shape[0], i, dtype=np.int32)
            for i, rbbox in enumerate(rbbox_result)
        ]
        rlabels = np.concatenate(rlabels)
        # draw segmentation masks
        segms = None
        if segm_result is not None and len(labels) > 0:  # non empty
            segms = mmcv.concat_list(segm_result)
            if isinstance(segms[0], torch.Tensor):
                segms = torch.stack(segms, dim=0).detach().cpu().numpy()
            else:
                segms = np.stack(segms, axis=0)
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False
        # draw bounding boxes
        # img = imshow_det_bboxes(
        #     img,
        #     bboxes,
        #     labels,
        #     segms,
        #     class_names=self.CLASSES,
        #     score_thr=score_thr,
        #     bbox_color=bbox_color,
        #     text_color=text_color,
        #     mask_color=mask_color,
        #     thickness=thickness,
        #     font_size=font_size,
        #     win_name=win_name,
        #     show=show,
        #     wait_time=wait_time,
        #     out_file=out_file)
        img = imshow_det_bboxes_rbboxes(
            img,
            bboxes,
            labels,
            rbboxes,
            rlabels,
            segms,
            class_names=self.CLASSES,
            score_thr=score_thr,
            bbox_color=bbox_color,
            text_color=text_color,
            mask_color=mask_color,
            thickness=thickness,
            font_size=font_size,
            win_name=win_name,
            show=show,
            wait_time=wait_time,
            out_file=out_file)

        if not (show or out_file):
            return img
