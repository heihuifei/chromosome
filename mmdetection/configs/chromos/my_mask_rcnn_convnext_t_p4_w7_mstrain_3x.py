# The new config inherits a base config to highlight the necessary modification
_base_ = '../convnext/mask_rcnn_convnext_tiny_patch4_window7_mstrain_480-800_adamw_3x_coco_in1k.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(num_classes=1)))

# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('chromos',)
data = dict(
    train=dict(
        img_prefix='/home/guest01/projects/chromos/dataset/segmentation_dataset/chromosome_coco_format/chromos/train_origin_77and187',
        classes=classes,
        ann_file='/home/guest01/projects/chromos/dataset/segmentation_dataset/chromosome_coco_format/chromos/annotations/instances_train_origin_77and187images.json'),
    val=dict(
        img_prefix='/home/guest01/projects/chromos/dataset/segmentation_dataset/chromosome_coco_format/chromos/val_origin_23and37',
        classes=classes,
        ann_file='/home/guest01/projects/chromos/dataset/segmentation_dataset/chromosome_coco_format/chromos/annotations/instances_val_origin_23and37images.json'),
    test=dict(
        img_prefix='/home/guest01/projects/chromos/dataset/segmentation_dataset/chromosome_coco_format/chromos/val_origin_23and37',
        classes=classes,
        ann_file='/home/guest01/projects/chromos/dataset/segmentation_dataset/chromosome_coco_format/chromos/annotations/instances_val_origin_23and37images.json'))

# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = '/home/guest01/projects/chromos/checkpoint/mask_rcnn_convnext_tiny_1k_3x.pth'
