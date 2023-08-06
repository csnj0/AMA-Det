_base_ = './FAQE_r50_fpn_ms2x.py'
model = dict(
    backbone=dict(
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)
    ),
    bbox_head=dict(dcn_on_last_conv=True)
)
