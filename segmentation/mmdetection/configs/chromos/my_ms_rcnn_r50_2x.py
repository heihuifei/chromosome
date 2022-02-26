# The new config inherits a base config to highlight the necessary modification
_base_ = '/home/guest01/projects/mmdetection/configs/ms_rcnn/ms_rcnn_r50_caffe_fpn_2x_coco.py'

# We also need to change the num_classes in head to match the dataset's annotation
model = dict(
    roi_head=dict(
        bbox_head=dict(num_classes=1),
        mask_head=dict(num_classes=1),
        mask_iou_head=dict(num_classes=1)))

# Modify dataset related settings
dataset_type = 'COCODataset'
classes = ('chromos',)
data = dict(
    train=dict(
        img_prefix='/home/guest01/projects/mmdetection/data/chromos/train2017',
        classes=classes,
        ann_file='/home/guest01/projects/mmdetection/data/chromos/annotations/instances_train2017.json'),
    val=dict(
        img_prefix='/home/guest01/projects/mmdetection/data/chromos/val2017',
        classes=classes,
        ann_file='/home/guest01/projects/mmdetection/data/chromos/annotations/instances_val2017.json'),
    test=dict(
        img_prefix='/home/guest01/projects/mmdetection/data/chromos/val2017',
        classes=classes,
        ann_file='/home/guest01/projects/mmdetection/data/chromos/annotations/instances_val2017.json'))

# We can use the pre-trained Mask RCNN model to obtain higher performance
load_from = 'checkpoints/ms_rcnn_r50_caffe_fpn_2x_coco_bbox_mAP-0.388__segm_mAP-0.363_20200506_004738-ee87b137.pth'
