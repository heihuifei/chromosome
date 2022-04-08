_base_ = '../oriented_rcnn/oriented_rcnn_r50_fpn_1x_dota_le90.py'

# Modify dataset related settings
dataset_type = 'DOTADataset'
classes = ('chromos',)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=2,
    train=dict(
        type=dataset_type,
        classes=classes,
        ann_file='/home/guest01/projects/chromos/dataset/detection_dataset/chromos/chromos_origin_fake1500/splitTrain/annfiles',
        img_prefix='/home/guest01/projects/chromos/dataset/detection_dataset/chromos/chromos_origin_fake1500/splitTrain/images'),
    val=dict(
        type=dataset_type,
        classes=classes,
        ann_file='/home/guest01/projects/chromos/dataset/detection_dataset/chromos/chromos_origin_fake1500/splitVal/annfiles',
        img_prefix='/home/guest01/projects/chromos/dataset/detection_dataset/chromos/chromos_origin_fake1500/splitVal/images'),
    test=dict(
        type=dataset_type,
        classes=classes,
        ann_file='/home/guest01/projects/chromos/dataset/detection_dataset/chromos/chromos_origin_fake1500/splitTest/annfiles',
        img_prefix='/home/guest01/projects/chromos/dataset/detection_dataset/chromos/chromos_origin_fake1500/splitTest/images'))

print("this is test num_classes: ", len(classes))

model = dict(
    roi_head=dict(
        bbox_head=dict(
            num_classes=len(classes))))
