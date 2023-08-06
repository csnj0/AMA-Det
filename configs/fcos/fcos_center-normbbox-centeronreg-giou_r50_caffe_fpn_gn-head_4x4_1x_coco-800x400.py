_base_ = 'fcos_r50_caffe_fpn_gn-head_4x4_1x_coco.py'

model = dict(
    pretrained='open-mmlab://detectron2/resnet50_caffe',
    bbox_head=dict(
        norm_on_bbox=True,
        centerness_on_reg=True,
        dcn_on_last_conv=False,
        center_sampling=True,
        conv_bias=True,
        loss_bbox=dict(type='GIoULoss', loss_weight=1.0)))
# training and testing settings
test_cfg = dict(nms=dict(type='nms', iou_threshold=0.6))

# dataset settings
img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)
train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', img_scale=(1333, 800), keep_ratio=True),
    dict(type='RandomFlip', flip_ratio=0.5),
    dict(type='Normalize', **img_norm_cfg),
    dict(type='Pad', size_divisor=32),
    dict(type='DefaultFormatBundle'),
    dict(type='Collect', keys=['img', 'gt_bboxes', 'gt_labels']),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(
        type='MultiScaleFlipAug',
        img_scale=(1333, 800),
        flip=False,
        transforms=[
            dict(type='Resize', keep_ratio=True),
            dict(type='RandomFlip'),
            dict(type='Normalize', **img_norm_cfg),
            dict(type='Pad', size_divisor=32),
            dict(type='ImageToTensor', keys=['img']),
            dict(type='Collect', keys=['img']),
        ])
]
data = dict(
    samples_per_gpu=16,
    workers_per_gpu=4,
    train=dict(pipeline=train_pipeline),
    val=dict(pipeline=test_pipeline),
    test=dict(pipeline=test_pipeline))

# total_batchsize=16, lr=0.01
# total_batchsize=64, lr=0.04
optimizer = dict(type='SGD', lr=0.04, momentum=0.9, weight_decay=0.0001)
optimizer_config = dict(_delete_=True, grad_clip=None)

# optimizer_config = dict(
#     _delete_=True, grad_clip=dict(max_norm=35, norm_type=2))

# learning policy
lr_config = dict(
    policy='step',
    warmup='linear',
    # warmup_iters=500,
    warmup_iters=1000,
    warmup_ratio=0.001,
    step=[8, 11])

# fp16 settings
fp16 = dict(loss_scale=512.)

