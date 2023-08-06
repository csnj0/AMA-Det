_base_ = "./FAQE_r50_fpn_ms2x_tta.py"
model = dict(
    pretrained="open-mmlab://resnext101_64x4d",
    backbone=dict(
        type="ResNeXt",
        depth=101,
        groups=64,
        base_width=4,
        dcn=dict(type="DCNv2", deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True),
    ),
    bbox_head=dict(dcn_on_last_conv=True),
)
