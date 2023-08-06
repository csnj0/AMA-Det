_base_ = ['../gflv3/gflv3_r50_fpn_1x_coco.py']
model = dict(
    bbox_head=dict(
        RGQP_cfg=dict(_delete_=True),
    )
)
