_base_ = ['../serialgfl/serialgfl_r50_fpn_1x_coco.py']
model = dict(
    # neck=dict(
    #     norm_cfg=dict(type='GN', num_groups=32, requires_grad=True)
    #     ),
    bbox_head=dict(
        bins=5,
        points=9,
        DGQP_cfg=dict(_delete_=True, channels=64),
        # loss_cls=dict(_delete_=True, type='FocalLoss', use_sigmoid=True, gamma=2.0, alpha=0.25, loss_weight=1.0),
        # loss_bbox_init=dict(_delete_=True, type='SmoothL1Loss', beta=1.0/9.0, loss_weight=0.5),
        # loss_bbox_refine=dict(_delete_=True, type='SmoothL1Loss', beta=1.0/9.0, loss_weight=1.0),
    )
)

# # mix-precision training
# fp16 = dict(loss_scale=512.)

data = dict(
    # samples_per_gpu=8,
    # samples_per_gpu=4,
    samples_per_gpu=2,
    workers_per_gpu=2
)

######### 备注 #####################
# Range(0,2): cls:x reg:x
# Range(3,4): cls:x+g(x) reg:x+g(x)
# Range(5,6): cls:x reg:x+g(x)
###################################

# gpu_ids = range(0, 2)
# gpu_ids = range(2, 4)
# gpu_ids = range(4, 6)

# gpu_ids = range(0, 4)
# gpu_ids = range(4, 8)

gpu_ids = range(0, 8)


checkpoint_config = dict(
    out_dir='./work_dirs/serialgfl_r50_fpn_1x_coco_[DGQP-c64]/result' + str(gpu_ids)[5:].replace(' ', ''),
    max_keep_ckpts=1)


# train_cfg = dict(
#     _delete_=True,

#     # init=dict(
#     #     _delete_=True,
#     #     assigner=dict(type='PointAssigner', scale=4, pos_num=1),
#     #     allowed_border=-1,
#     #     pos_weight=-1,
#     #     debug=False),

#     init=dict(
#         _delete_=True,
#         assigner=dict(type='ATSSAssigner', topk=1),
#         allowed_border=-1,
#         pos_weight=-1,
#         debug=False),

#     # refine=dict(
#     #     assigner=dict(
#     #         type='MaxIoUAssigner',
#     #         pos_iou_thr=0.5,
#     #         neg_iou_thr=0.4,
#     #         min_pos_iou=0,
#     #         ignore_iof_thr=-1),
#     #     allowed_border=-1,
#     #     pos_weight=-1,
#     #     debug=False)
#     refine=dict(
#         _delete_=True,
#         assigner=dict(type='ATSSAssigner', topk=9),
#         allowed_border=-1,
#         pos_weight=-1,
#         debug=False)
# )

# test_cfg = dict(
#     nms_pre=1000,
#     min_bbox_size=0,
#     score_thr=0.05,
#     nms=dict(type='nms', iou_threshold=0.5),
#     max_per_img=100)


# norm_cfg = dict(type='GN', num_groups=32, requires_grad=True)
# model = dict(neck=dict(norm_cfg=norm_cfg), bbox_head=dict(norm_cfg=norm_cfg))


### bins=5 ###
### points ###
# 9: 0.424 ###
# 7: 0.425 ###
# 5: 0.422 ###
# 3: 0.420 ###

### bins=4 ###
### points ###
# 9: 0. ###
# 7: 0.426 ###
# 5: 0. ###
# 3: 0. ###

### bins=3 ###
### points ###
# 9: 0. ###
# 7: 0.424 ###
# 5: 0. ###
# 3: 0. ###

### bins=2 ###
### points ###
# 9: 0. ###
# 7: 0.417 ###
# 5: 0. ###
# 3: 0. ###
