_base_ = ['../gflv3/gflv3_r50_fpn_1x_coco.py']
model = dict(
    bbox_head=dict(
        DGQP_cfg=dict(_delete_=True, topk=4, sorted=False, channels=64, add_mean=False),
    )
)

data = dict(
    samples_per_gpu=2,
    workers_per_gpu=4
)

