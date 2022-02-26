# the new config inherits the base configs to highlight the necessary modification
_base_ = '/home/guest01/projects/mmdetection/configs/ms_rcnn/ms_rcnn_r50_fpn_1x_coco.py'

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