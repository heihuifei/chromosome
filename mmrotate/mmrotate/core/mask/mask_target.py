# Copyright (c) OpenMMLab. All rights reserved.
import numpy as np
import torch
from torch.nn.modules.utils import _pair

from mmrotate.core import obb2xyxy

# 主要用于针对旋转proposal进行计算其偏导数targets
def mask_target(pos_proposals_list, pos_assigned_gt_inds_list, gt_masks_list,
                cfg):
    """Compute mask target for positive proposals in multiple images.

    Args:
        pos_proposals_list (list[Tensor]): Positive proposals in multiple
            images.
        pos_assigned_gt_inds_list (list[Tensor]): Assigned GT indices for each
            positive proposals.
        gt_masks_list (list[:obj:`BaseInstanceMasks`]): Ground truth masks of
            each image.
        cfg (dict): Config dict that specifies the mask size.

    Returns:
        list[Tensor]: Mask target of each image.

    Example:
        >>> import mmcv
        >>> import mmdet
        >>> from mmdet.core.mask import BitmapMasks
        >>> from mmdet.core.mask.mask_target import *
        >>> H, W = 17, 18
        >>> cfg = mmcv.Config({'mask_size': (13, 14)})
        >>> rng = np.random.RandomState(0)
        >>> # Positive proposals (tl_x, tl_y, br_x, br_y) for each image
        >>> pos_proposals_list = [
        >>>     torch.Tensor([
        >>>         [ 7.2425,  5.5929, 13.9414, 14.9541],
        >>>         [ 7.3241,  3.6170, 16.3850, 15.3102],
        >>>     ]),
        >>>     torch.Tensor([
        >>>         [ 4.8448, 6.4010, 7.0314, 9.7681],
        >>>         [ 5.9790, 2.6989, 7.4416, 4.8580],
        >>>         [ 0.0000, 0.0000, 0.1398, 9.8232],
        >>>     ]),
        >>> ]
        >>> # Corresponding class index for each proposal for each image
        >>> pos_assigned_gt_inds_list = [
        >>>     torch.LongTensor([7, 0]),
        >>>     torch.LongTensor([5, 4, 1]),
        >>> ]
        >>> # Ground truth mask for each true object for each image
        >>> gt_masks_list = [
        >>>     BitmapMasks(rng.rand(8, H, W), height=H, width=W),
        >>>     BitmapMasks(rng.rand(6, H, W), height=H, width=W),
        >>> ]
        >>> mask_targets = mask_target(
        >>>     pos_proposals_list, pos_assigned_gt_inds_list,
        >>>     gt_masks_list, cfg)
        >>> assert mask_targets.shape == (5,) + cfg['mask_size']
    """
    cfg_list = [cfg for _ in range(len(pos_proposals_list))]
    mask_targets = map(mask_target_single, pos_proposals_list,
                       pos_assigned_gt_inds_list, gt_masks_list, cfg_list)
    mask_targets = list(mask_targets)
    if len(mask_targets) > 0:
        mask_targets = torch.cat(mask_targets)
    return mask_targets


def mask_target_single(pos_proposals, pos_assigned_gt_inds, gt_masks, cfg):
    """Compute mask target for each positive proposal in the image.

    Args:
        pos_proposals (Tensor): Positive proposals.
        pos_assigned_gt_inds (Tensor): Assigned GT inds of positive proposals.
        gt_masks (:obj:`BaseInstanceMasks`): GT masks in the format of Bitmap
            or Polygon.
        cfg (dict): Config dict that indicate the mask size.

    Returns:
        Tensor: Mask target of each positive proposals in the image.

    Example:
        >>> import mmcv
        >>> import mmdet
        >>> from mmdet.core.mask import BitmapMasks
        >>> from mmdet.core.mask.mask_target import *  # NOQA
        >>> H, W = 32, 32
        >>> cfg = mmcv.Config({'mask_size': (7, 11)})
        >>> rng = np.random.RandomState(0)
        >>> # Masks for each ground truth box (relative to the image)
        >>> gt_masks_data = rng.rand(3, H, W)
        >>> gt_masks = BitmapMasks(gt_masks_data, height=H, width=W)
        >>> # Predicted positive boxes in one image
        >>> pos_proposals = torch.FloatTensor([
        >>>     [ 16.2,   5.5, 19.9, 20.9],
        >>>     [ 17.3,  13.6, 19.3, 19.3],
        >>>     [ 14.8,  16.4, 17.0, 23.7],
        >>>     [  0.0,   0.0, 16.0, 16.0],
        >>>     [  4.0,   0.0, 20.0, 16.0],
        >>> ])
        >>> # For each predicted proposal, its assignment to a gt mask
        >>> pos_assigned_gt_inds = torch.LongTensor([0, 1, 2, 1, 1])
        >>> mask_targets = mask_target_single(
        >>>     pos_proposals, pos_assigned_gt_inds, gt_masks, cfg)
        >>> assert mask_targets.shape == (5,) + cfg['mask_size']
    """
    # print("this is pos_proposals in mask_target: ", len(pos_proposals), pos_proposals)
    device = pos_proposals.device
    mask_size = _pair(cfg.mask_size)
    binarize = not cfg.get('soft_mask_target', False)
    num_pos = pos_proposals.size(0)
    if num_pos > 0:
        # 可以先将所有的rbbox格式转为hbbox格式
        pos_hproposals = obb2xyxy(pos_proposals, 'oc')
        proposals_np = pos_hproposals.cpu().numpy()
        maxh, maxw = gt_masks.height, gt_masks.width
        # 使用clip将proposal_np的值限制在0, maxw之间
        # 由于是rotated proposal, 因此proposals_np并非[batchid,x1,y1,x2,y2]而是[batchid,cx,cy,w,h,a]
        proposals_np[:, [0, 2]] = np.clip(proposals_np[:, [0, 2]], 0, maxw)
        proposals_np[:, [1, 3]] = np.clip(proposals_np[:, [1, 3]], 0, maxh)
        pos_assigned_gt_inds = pos_assigned_gt_inds.cpu().numpy()

        # gt_masks是Bitmap类,用于存储一张图像上的所有instance,其中crop_and_resize用于
        # Crop each bitmask by the given box, and resize results to (mask_size, mask_size).
        mask_targets = gt_masks.crop_and_resize(
            proposals_np,
            mask_size,
            device=device,
            inds=pos_assigned_gt_inds,
            binarize=binarize).to_ndarray()

        mask_targets = torch.from_numpy(mask_targets).float().to(device)
    else:
        mask_targets = pos_proposals.new_zeros((0, ) + mask_size)

    return mask_targets