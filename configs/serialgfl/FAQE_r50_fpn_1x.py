_base_ = [
    "../_base_/datasets/coco_detection.py",
    "../_base_/schedules/schedule_1x.py",
    "../_base_/default_runtime.py",
]
model = dict(
    type="GFL",
    pretrained="torchvision://resnet50",
    backbone=dict(
        type="ResNet",
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type="BN", requires_grad=True),
        norm_eval=True,
        style="pytorch",
    ),
    neck=dict(
        type="FPN",
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        start_level=1,
        add_extra_convs="on_output",
        num_outs=5,
    ),
    bbox_head=dict(
        type="SERIALGFLHead",
        num_classes=80,
        in_channels=256,
        stacked_convs=4,
        feat_channels=256,
        norm_cfg=dict(type="GN", num_groups=32, requires_grad=True),
        anchor_generator=dict(
            type="AnchorGenerator",
            ratios=[1.0],
            octave_base_scale=8,
            scales_per_octave=1,
            strides=[8, 16, 32, 64, 128],
        ),
        loss_cls=dict(
            type="QualityFocalLoss", use_sigmoid=False, beta=2.0, loss_weight=1.0
        ),
        bins=5,
        points=9,
        DGQP_cfg=dict(_delete_=True, channels=64),
        loss_bbox=dict(type="GIoULoss", loss_weight=2.0),
    ),
)
# training and testing settings
train_cfg = dict(
    assigner=dict(type="ATSSAssigner", topk=9),
    allowed_border=-1,
    pos_weight=-1,
    debug=False,
)
test_cfg = dict(
    nms_pre=1000,
    min_bbox_size=0,
    # score_thr=0.05,
    # nms=dict(type="nms", iou_threshold=0.6),
    # max_per_img=100,
    score_thr=0.0,
    nms=dict(type="nms", iou_threshold=1.0),
    max_per_img=1000,
)
# optimizer
optimizer = dict(type="SGD", lr=0.01, momentum=0.9, weight_decay=0.0001)

checkpoint_config = dict(max_keep_ckpts=5)

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
    samples_per_gpu=2,
    workers_per_gpu=2,
    show=dict(
        type=dataset_type,
        ann_file=data_root + "annotations/instances_val2017.json",
        img_prefix=data_root + "val2017/",
        pipeline=show_pipeline,
    ),
    test=dict(
        ann_file=data_root + "annotations/image_info_test-dev2017.json",
        img_prefix=data_root + "test2017/",
    ),
)
