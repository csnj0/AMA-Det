_base_ = './FAQE_r50_fpn_ms2x.py'

model = dict(
    pretrained='torchvision://resnet101',
    backbone=dict(depth=101)
)
