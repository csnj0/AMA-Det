_base_ = './FAQE_r50_fpn_ms2x.py'
model = dict(
    pretrained='open-mmlab://res2net101_v1d_26w_4s',
    backbone=dict(
        type='Res2Net',
        depth=101,
        base_width=26,
        scales=4,
        dcn=dict(type='DCNv2', deform_groups=1, fallback_on_stride=False),
        stage_with_dcn=(False, True, True, True)
    ),
)
