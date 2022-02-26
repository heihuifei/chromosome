# The new config inherits a base config to highlight the necessary modification
_base_ = [
    'mask_rcnn_r50_fpn_model.py',
    'mask_rcnn_r50_fpn_dataset.py',
    'mask_rcnn_r50_fpn_schedule.py', '../_base_/default_runtime.py'
]

# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = '/home/guest01/projects/mmdetection/checkpoints/mask_rcnn_r50_fpn_1x_coco_20200205-d4b0c5d6.pth'