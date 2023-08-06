_base_ = [
    "../_base_/datasets/coco_detection.py",
    "../_base_/schedules/schedule_1x.py",
    "../_base_/default_runtime.py",
]

# model settings
norm_cfg = dict(type="GN", num_groups=32, requires_grad=True)

model = dict(
    type="RepPointsDetector",
    pretrained="modelzoo://resnet50",
    backbone=dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        style="pytorch",
    ),
    neck=dict(
        type="FPN",
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs=True,
        num_outs=5,
        norm_cfg=norm_cfg,
    ),
    bbox_head=dict(
        type="RepPointsHead",
        num_classes=80,
        in_channels=256,
        feat_channels=256,
        point_feat_channels=256,
        stacked_convs=3,
        num_points=9,
        gradient_mul=0.1,
        point_strides=[8, 16, 32, 64, 128],
        point_base_scale=4,
        norm_cfg=norm_cfg,
        loss_cls=dict(
            type="FocalLoss", use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0
        ),
        loss_bbox_init=dict(type="SmoothL1Loss", beta=0.11, loss_weight=0.5),
        loss_bbox_refine=dict(type="SmoothL1Loss", beta=0.11, loss_weight=1.0),
        transform_method="minmax",
    ),
)
# training and testing settings
train_cfg = dict(
    init=dict(
        assigner=dict(type="PointAssigner", scale=4, pos_num=1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False,
    ),
    refine=dict(
        assigner=dict(
            type="MaxIoUAssigner",
            pos_iou_thr=0.5,
            neg_iou_thr=0.4,
            min_pos_iou=0,
            ignore_iof_thr=-1,
        ),
        allowed_border=-1,
        pos_weight=-1,
        debug=False,
    ),
)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    # score_thr=0.05,
    # nms=dict(type="nms", iou_thr=0.5),
    # max_per_img=100,
    score_thr=0.0,
    nms=dict(type="nms", iou_thr=1.0),
    max_per_img=1000,
)


# optimizer
optimizer = dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0001)
# fp16 settings
fp16 = dict(loss_scale=512.0)


dataset_type = "CocoDataset"
data_root = "/share/Datasets/coco/"
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
show_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="LoadAnnotations", with_bbox=True),
    dict(
        type="MultiScaleFlipAug",
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="DefaultFormatBundle"),
            dict(type="Collect", keys=["img", "gt_bboxes", "gt_labels"]),
        ],
    ),
]
data = dict(
    show=dict(
        type=dataset_type,
        ann_file=data_root + "annotations/instances_val2017.json",
        img_prefix=data_root + "val2017/",
        pipeline=show_pipeline,
    )
)
