# Copyright (c) OpenMMLab. All rights reserved.
import torch
from mmcv.utils import to_2tuple
from mmdet.core.anchor import AnchorGenerator

from .builder import ROTATED_ANCHOR_GENERATORS


@ROTATED_ANCHOR_GENERATORS.register_module()
class RotatedAnchorGenerator(AnchorGenerator):
    """Standard rotate anchor generator for 2D anchor-based detectors."""

    # 在AnchorGenerator基础上，修改其内部的单层特征图anchors传播生成的函数
    def single_level_grid_priors(self,
                                 featmap_size,
                                 level_idx,
                                 dtype=torch.float32,
                                 device='cuda'):
        """Generate grid anchors of a single level.

        Note:
            This function is usually called by method ``self.grid_priors``.

        Args:
            featmap_size (tuple[int]): Size of the feature maps.
            level_idx (int): The index of corresponding feature map level.
            dtype (obj:`torch.dtype`): Date type of points.Defaults to
            ``torch.float32``.
            device (str, optional): The device the tensor will be put on.
            Defaults to 'cuda'.

        Returns:
            torch.Tensor: Anchors in the overall feature maps.
        """
        # 先根据父类的该函数获取到anchors, anchor格式: [xmin, ymin, xmax,ymax]
        anchors = super(RotatedAnchorGenerator, self).single_level_grid_priors(
            featmap_size, level_idx, dtype=dtype, device=device)

        num_anchors = anchors.size(0)
        # xy表示(xmin, ymin)+(xmax, ymax)/2, 即anchor的中心点坐标
        xy = (anchors[:, 2:] + anchors[:, :2]) / 2
        # wh表示(xmax, ymax)-(xmin, ymin), 即anchor的宽高
        wh = anchors[:, 2:] - anchors[:, :2]
        # 创建num_anchors*1维的数值维0的数组
        theta = xy.new_zeros((num_anchors, 1))
        # cat将两个tensor拼接在一起, 按照列(axis=1)拼接, 即生成单层的多个anchors
        # 但是这些anchors是在轴对准[xmin, ymin, xmax, ymax]基础上变为[x, y, w, h, theta]含角度
        anchors = torch.cat([xy, wh, theta], axis=1)

        return anchors


@ROTATED_ANCHOR_GENERATORS.register_module()
class PseudoAnchorGenerator(AnchorGenerator):
    """Non-Standard pseudo anchor generator that is used to generate valid
    flags only!"""

    def __init__(self, strides):
        self.strides = [to_2tuple(stride) for stride in strides]

    @property
    def num_base_anchors(self):
        """list[int]: total number of base anchors in a feature grid"""
        return [1 for _ in self.strides]

    def single_level_grid_anchors(self, featmap_sizes, device='cuda'):
        """Calling its grid_anchors() method will raise NotImplementedError!"""
        raise NotImplementedError

    def __repr__(self):
        indent_str = '    '
        repr_str = self.__class__.__name__ + '(\n'
        repr_str += f'{indent_str}strides={self.strides})'
        return repr_str
