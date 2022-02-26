# The new config inherits a base config to highlight the necessary modification
_base_ = '/home/guest01/projects/mmdetection/configs/cascade_rcnn/cascade_mask_rcnn_r50_fpn_20e_coco.py'

# 1. dataset settings
dataset_type = 'CocoDataset'
classes = ('chromos',)
data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file='/home/guest01/projects/mmdetection/data/chromos/annotations/instances_train2017.json',
        img_prefix='/home/guest01/projects/mmdetection/data/chromos/train2017'),
    val=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file='/home/guest01/projects/mmdetection/data/chromos/annotations/instances_val2017.json',
        img_prefix='/home/guest01/projects/mmdetection/data/chromos/val2017'),
    test=dict(
        type=dataset_type,
        # explicitly add your class names to the field `classes`
        classes=classes,
        ann_file='/home/guest01/projects/mmdetection/data/chromos/annotations/instances_val2017.json',
        img_prefix='/home/guest01/projects/mmdetection/data/chromos/val2017'))

# 2. model settings
# explicitly over-write all the `num_classes` field from default 80 to 5.
model = dict(
    roi_head=dict(
        bbox_head=[
            dict(
                type='Shared2FCBBoxHead',
                # explicitly over-write all the `num_classes` field from default 80 to 5.
                num_classes=1),
            dict(
                type='Shared2FCBBoxHead',
                # explicitly over-write all the `num_classes` field from default 80 to 5.
                num_classes=1),
            dict(
                type='Shared2FCBBoxHead',
                # explicitly over-write all the `num_classes` field from default 80 to 5.
                num_classes=1)],
    # explicitly over-write all the `num_classes` field from default 80 to 5.
    mask_head=dict(num_classes=1)))

# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = 'checkpoints/cascade_mask_rcnn_r50_fpn_20e_coco_bbox_mAP-0.419__segm_mAP-0.365_20200504_174711-4af8e66e.pth'
