# Copyright (c) OpenMMLab. All rights reserved.
from .inference import inference_detector_by_patches
from .test import multi_gpu_test, single_gpu_test
from .train import train_detector

__all__ = ['inference_detector_by_patches', 'train_detector',
           'multi_gpu_test', 'single_gpu_test']
