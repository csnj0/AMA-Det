_base_ = './FAQE_r50_fpn_1x.py'

model = dict(
    pretrained='torchvision://resnet101',
    backbone=dict(depth=101)
)
