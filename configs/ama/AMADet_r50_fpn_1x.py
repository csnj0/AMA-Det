_base_ = [
    '../_base_/datasets/coco_detection.py',
    '../_base_/schedules/schedule_1x.py', '../_base_/default_runtime.py'
]
model = dict(
    type='ATSS',
    pretrained='torchvision://resnet50',
    backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=True),
        norm_eval=True,
        style='pytorch'),
    neck=dict(
        type='FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs='on_output',
        num_outs=5),
    bbox_head=dict(
        type='AMAHead',
        num_classes=80,
        in_channels=256,
        stacked_convs=3,
        feat_channels=256,
        norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
        point_strides=[8, 16, 32, 64, 128],
        point_base_scale=8,
        num_points=9,
        loc_type='bbox',
        GFLV2_cfg=dict(
            bins_full=5,
            bins_select=5,
            is_sorted=False,
            channels=64),
        loss_cls=dict(
            type='QualityFocalLoss',
            use_sigmoid=False,
            beta=2.0,
            loss_weight=1.0),
        loss_bbox=dict(type='GIoULoss', loss_weight=2.0)
    ))
# training and testing settings
train_cfg = dict(
    type='ATSS',
    anchor_guiding=False,
    assigner=dict(type='ATSSAssigner', topk=9),
    allowed_border=-1,
    pos_weight=-1,
    debug=False)

# train_cfg = dict(
#     type='RepPoints',
#     anchor_guiding=True,
#     init=dict(
#         assigner=dict(type='PointAssigner', scale=4, pos_num=1),
#         allowed_border=-1,
#         pos_weight=-1,
#         debug=False),
#     refine=dict(
#         assigner=dict(
#             type='MaxIoUAssigner',
#             pos_iou_thr=0.5,
#             neg_iou_thr=0.4,
#             min_pos_iou=0,
#             ignore_iof_thr=-1),
#         allowed_border=-1,
#         pos_weight=-1,
#         debug=False))

# train_cfg = dict(
#     type='RepPoints-ATSS',
#     anchor_guiding=True,
#     init=dict(
#         assigner=dict(type='ATSSAssigner', topk=9),
#         allowed_border=-1,
#         pos_weight=-1,
#         debug=False),
#     refine=dict(
#         assigner=dict(type='ATSSAssigner', topk=9),
#         allowed_border=-1,
#         pos_weight=-1,
#         debug=False))

test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    score_thr=0.05,
    nms=dict(type='nms', iou_threshold=0.6),
    max_per_img=100)
# optimizer
optimizer = dict(type='SGD', lr=0.01, momentum=0.9, weight_decay=0.0001)

checkpoint_config = dict(max_keep_ckpts=5)

data_root = '/share/Datasets/coco/'
data = dict(
    test=dict(
        ann_file=data_root + 'annotations/image_info_test-dev2017.json',
        img_prefix=data_root + 'test2017/'))
