# The new config inherits a base config to highlight the necessary modification
from random import sample


_base_ = './rotated_faster_rcnn_r50_fpn_1x_dota_le90.py'

# Modify dataset related settings
dataset_type = 'DOTADataset'
# classes = ('chromos',)
classes = ('plane', 'baseball-diamond', 'bridge', 'ground-track-field',
               'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
               'basketball-court', 'storage-tank', 'soccer-ball-field',
               'roundabout', 'harbor', 'swimming-pool', 'helicopter')
data_root = "/home/guest01/projects/chromos/dataset/detection_dataset/split_1024_dota1_0"

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'trainsplit/annfiles/',
        img_prefix=data_root + 'trainsplit/images/'),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'valsplit/annfiles/',
        img_prefix=data_root + 'valsplit/images/'),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file=data_root + 'testsplit/annfiles/',
        img_prefix=data_root + 'testsplit/images/'))

model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=len(classes))))
