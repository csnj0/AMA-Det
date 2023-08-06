_base_ = './reppoints_moment_r50_fpn_1x_coco.py'
norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
model = dict(neck=dict(norm_cfg=norm_cfg), bbox_head=dict(norm_cfg=norm_cfg))
optimizer = dict(lr=0.01)

train_cfg = dict(
    init=dict(
        _delete_=True,
        assigner=dict(type='PointAssigner', scale=4, pos_num=1),
        allowed_border=-1,
        pos_weight=-1,
        debug=False),
    refine=dict(
        _delete_=True,
        assigner=dict(type='ATSSAssigner', topk=9),
        allowed_border=-1,
        pos_weight=-1,
        debug=False)
)


model = dict(bbox_head=dict(point_base_scale=8))
