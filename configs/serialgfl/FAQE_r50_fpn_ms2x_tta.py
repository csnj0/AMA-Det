_base_ = "./FAQE_r50_fpn_1x.py"

# learning policy
lr_config = dict(step=[16, 22])
total_epochs = 24

# # 5 scales and flip
# tta_flip = True
# tta_scale = [(667, 400), (1000, 600), (1333, 800), (1667, 1000), (2000, 1200)]
# scale_ranges = [(96, 10000), (64, 10000), (0, 10000), (0, 10000), (0, 256)]

# 13 scales and flip (slow)
tta_flip = True
tta_scale = [
    (667, 400),
    (833, 500),
    (1000, 600),
    (1067, 640),
    (1167, 700),
    (1333, 800),
    (1500, 900),
    (1667, 1000),
    (1833, 1100),
    (2000, 1200),
    (2167, 1300),
    (2333, 1400),
    (3000, 1800),
]
scale_ranges = [
    (96, 10000),
    (96, 10000),
    (64, 10000),
    (64, 10000),
    (64, 10000),
    (0, 10000),
    (0, 10000),
    (0, 10000),
    (0, 256),
    (0, 256),
    (0, 192),
    (0, 192),
    (0, 96),
]

test_cfg = dict(fusion_cfg=dict(type="soft_vote", scale_ranges=scale_ranges))

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(
        type="MultiScaleFlipAug",
        img_scale=tta_scale,
        flip=tta_flip,
        # # VFNet
        # img_scale=[(1200, 800), (1300, 900), (1500, 1000), (1600, 1200)],
        # # SCNet
        # img_scale=[(600, 900), (800, 1200), (1000, 1500), (1200, 1800), (1400, 2100)],
        # #UniverseNet 5 Scales
        # img_scale=[
        #     (400, 667),
        #     (600, 1000),
        #     (800, 1334),
        #     (1000, 1667),
        #     (1200, 2000),
        # ],
        # # UniverseNet 13 Scales
        # img_scale=[
        #     (400, 667),
        #     (500, 834),
        #     (600, 1000),
        #     (640, 1067),
        #     (700, 1167),
        #     (800, 1334),
        #     (900, 1500),
        #     (1000, 1667),
        #     (1100, 1834),
        #     (1200, 2000),
        #     (1300, 2167),
        #     (1400, 2334),
        #     (1800, 3000),
        # ],
        # flip=True,
        transforms=[
            dict(type="Resize", keep_ratio=True),
            dict(type="RandomFlip"),
            dict(type="Normalize", **img_norm_cfg),
            dict(type="Pad", size_divisor=32),
            dict(type="ImageToTensor", keys=["img"]),
            dict(type="Collect", keys=["img"]),
        ],
    ),
]
data = dict(test=dict(pipeline=test_pipeline))
