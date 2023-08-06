from cProfile import label
from collections import OrderedDict
from heapq import merge
from operator import index
from numpy.lib.polynomial import polyint
from torch.functional import norm
from torch.nn.modules import padding
from torch.nn.modules.activation import ReLU, Sigmoid
from torch.nn.modules.fold import Unfold
from mmdet.core.bbox.transforms import bbox2distance
from mmcv.cnn.utils.weight_init import constant_init, kaiming_init
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import (
    ConvModule,
    Scale,
    bias_init_with_prob,
    normal_init,
    build_norm_layer,
)
from mmcv.runner import force_fp32
from mmcv.ops import DeformConv2d, ModulatedDeformConv2d
from mmdet.core import (
    anchor_inside_flags,
    bbox,
    bbox_overlaps,
    build_assigner,
    build_sampler,
    distance2bbox,
    images_to_levels,
    multi_apply,
    multiclass_nms,
    post_processing,
    reduce_mean,
    unmap,
)
from ..builder import HEADS, build_loss
from .anchor_head import AnchorHead

from mmcv.ops.nms import batched_nms

import numpy as np
import math

# from torchvision.ops import DeformConv2d
# from torch.nn.modules.utils import _pair, _single


class Integral(nn.Module):
    def __init__(self, start=0.0, stop=4.0, bins=5, with_softmax=False):
        super(Integral, self).__init__()
        self.bins = bins
        self.with_softmax = with_softmax
        self.register_buffer("project", torch.linspace(start, stop, self.bins))

    def forward(self, x):
        x = x.reshape(-1, self.bins)
        if self.with_softmax:
            x = F.softmax(x, dim=1)
        x = F.linear(x, self.project.type_as(x)).reshape(-1, 4)
        return x


class Integral_v2(nn.Module):
    def __init__(self, start=-2.0, stop=2.0, bins=5, with_softmax=True):
        super(Integral_v2, self).__init__()
        self.bins = bins
        self.register_buffer("project", torch.linspace(start, stop, self.bins))
        self.with_softmax = with_softmax

    def forward(self, x):
        N, _, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(-1, self.bins)  # NHW*4, bins
        if self.with_softmax:
            x = F.softmax(x, dim=1)
        x = F.linear(x, self.project.type_as(x))
        x = x.reshape(N, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x


class DeformConv2dModule(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        groups=1,
        deform_groups=1,
        norm_cfg=None,
        with_act=True,
    ):
        super(DeformConv2dModule, self).__init__()
        self.conv = DeformConv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            deform_groups=deform_groups,
        )
        # self.conv = DeformConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
        #                             groups=groups, bias=False)

        self.norm_cfg = norm_cfg
        if self.norm_cfg is not None:
            self.norm_name, norm = build_norm_layer(norm_cfg, out_channels)
            self.add_module(self.norm_name, norm)
            constant_init(self.norm, 1, bias=0)
        else:
            self.bias = nn.Parameter(torch.Tensor(1, out_channels, 1, 1))
            nn.init.constant_(self.bias, 0.0)

        self.with_act = with_act
        if with_act:
            self.act = nn.ReLU(inplace=True)

    @property
    def norm(self):
        return getattr(self, self.norm_name)

    def forward(self, x, offset):
        x = self.conv(x, offset)
        if self.norm_cfg is not None:
            x = self.norm(x)
        else:
            x = x + self.bias

        if self.with_act:
            x = self.act(x)
        return x


class MaskedDeformConv2dModule(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size=3,
        stride=1,
        padding=1,
        groups=1,
        deform_groups=1,
        norm_cfg=None,
    ):
        super(MaskedDeformConv2dModule, self).__init__()
        self.conv = ModulatedDeformConv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            groups=groups,
            deform_groups=deform_groups,
        )

        self.norm_cfg = norm_cfg
        if self.norm_cfg is not None:
            self.norm_name, norm = build_norm_layer(norm_cfg, out_channels)
            self.add_module(self.norm_name, norm)
            constant_init(self.norm, 1, bias=0)
        else:
            self.bias = nn.Parameter(torch.Tensor(1, out_channels, 1, 1))
            nn.init.constant_(self.bias, 0.0)

        self.act = nn.ReLU(inplace=True)

    @property
    def norm(self):
        return getattr(self, self.norm_name)

    def forward(self, x, offset, mask):
        x = self.conv(x, offset, mask)
        if self.norm_cfg is not None:
            x = self.norm(x)
        else:
            x = x + self.bias
        x = self.act(x)
        return x


class DualDeformModule(nn.Module):
    def __init__(self, channels, points=9, norm_cfg=None):
        super(DualDeformModule, self).__init__()
        self.points = points

        self.base_offset = self.get_base_offset(self.points)
        self.channels = channels

        # self.cls_conv = DeformConv2dModule(channels, channels, [1, self.points], stride=1,
        #                 padding=[0, self.points//2], norm_cfg=norm_cfg)
        self.reg_conv = DeformConv2dModule(
            channels,
            channels,
            [1, self.points],
            stride=1,
            padding=[0, self.points // 2],
            norm_cfg=norm_cfg,
        )

        self.offset_conv = ConvModule(
            channels, channels, 3, padding=1, norm_cfg=norm_cfg
        )
        normal_init(self.offset_conv.conv, std=0.01)
        self.offset_out = nn.Conv2d(channels, self.points * 2, 3, padding=1)
        normal_init(self.offset_out, std=0.01)

        # self.bbox_conv = ConvModule(channels, channels, 3, padding=1, norm_cfg=norm_cfg)
        # normal_init(self.bbox_conv.conv, std=0.01)
        # self.bbox_out = nn.Conv2d(channels, 4, 3, padding=1)
        # normal_init(self.bbox_out, std=0.01)

        # self.cls_offset_conv = ConvModule(channels, channels, 3, padding=1, norm_cfg=norm_cfg)
        # self.cls_offset_out = nn.Conv2d(channels, self.points*2, 3, padding=1)
        # normal_init(self.cls_offset_conv.conv, std=0.01)
        # normal_init(self.cls_offset_out, std=0.01)

        # self.integral = Integral(-15, 15, bins=31)
        # self.integral = Integral(0, 16, bins=17)
        # self.integral = Integral(0, 16, bins=2)

    def get_base_offset(self, points):
        dcn_base_x = np.arange(-((points - 1) // 2), points // 2 + 1).astype(np.float64)
        dcn_base_y = dcn_base_x * 0.0
        dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=1).reshape((-1))
        dcn_base_offset = torch.tensor(dcn_base_offset).view(1, -1, 1, 1)
        return dcn_base_offset

    def points2distance(self, pts):
        if pts.dim() == 4:
            pts_reshape = pts.view(pts.shape[0], -1, 2, *pts.shape[2:])
            pts_y = pts_reshape[:, :, 0, ...]
            pts_x = pts_reshape[:, :, 1, ...]
        elif pts.dim() == 2:
            pts_reshape = pts.view(pts.shape[0], -1, 2)
            pts_y = pts_reshape[:, :, 0]
            pts_x = pts_reshape[:, :, 1]

        bbox_left, _ = pts_x.min(dim=1, keepdim=True)
        bbox_right, _ = pts_x.max(dim=1, keepdim=True)
        bbox_up, _ = pts_y.min(dim=1, keepdim=True)
        bbox_bottom, _ = pts_y.max(dim=1, keepdim=True)
        bbox = torch.cat([-bbox_left, -bbox_up, bbox_right, bbox_bottom], dim=1)

        return bbox

    # def forward(self, cls_feat, reg_feat, scale_pts, scale_bbox, scale_cls_pts):
    # def forward(self, cls_feat, reg_feat, scale_pts, scale_bbox):
    def forward(self, cls_feat, reg_feat, scale_pts):
        # def forward(self, cls_feat, reg_feat, scale):
        N, _, H, W = cls_feat.shape
        base_offset = self.base_offset.type_as(reg_feat)

        # dist_pts = scale_pts(self.offset_out(self.offset_conv(reg_feat)))
        # pts = self.integral(dist_pts)

        pts = scale_pts(self.offset_out(self.offset_conv(reg_feat)))
        # bbox = self.integral(scale_bbox(self.bbox_out(reg_feat)))

        # bbox = scale_bbox(self.bbox_out(self.bbox_conv(reg_feat))).exp()

        # reg_feat = self.bbox_conv(reg_feat)
        # dist_bbox = scale_bbox(self.bbox_out(reg_feat))

        # dist_bbox = scale_bbox(self.bbox_out(self.bbox_conv(reg_feat)))
        # bbox = scale_bbox(self.bbox_out(self.bbox_conv(reg_feat))).exp()
        # bbox = self.integral(dist_bbox)
        # l, u, r, b = bbox.chunk(4, dim=1)

        # pts_y, pts_x = scale_pts(self.offset_out(self.offset_conv(reg_feat))).exp().split([self.points, self.points], dim=1)
        # pts_y, pts_x = scale_pts(self.offset_out(reg_feat)).exp().split([self.points, self.points], dim=1)

        # pts1_y, pts2_y, pts3_y, pts4_x, pts5_x, pts6_y, pts7_y, pts8_x, pts9_x = scale_pts(self.offset_out(self.offset_conv(reg_feat))).sigmoid().chunk(self.points, dim=1)
        # pts1 = torch.cat([pts1_y*(u+b)-u, -l], dim=1)
        # pts2 = torch.cat([pts2_y*(u+b)-u, -l], dim=1)
        # pts3 = torch.cat([pts3_y*(u+b)-u, -l], dim=1)
        # pts4 = torch.cat([-u, pts4_x*(l+r)-l], dim=1)
        # pts5 = torch.cat([-u, pts5_x*(l+r)-l], dim=1)
        # pts6 = torch.cat([pts6_y*(u+b)-u, r], dim=1)
        # pts7 = torch.cat([pts7_y*(u+b)-u, r], dim=1)
        # pts8 = torch.cat([b, pts8_x*(l+r)-l], dim=1)
        # pts9 = torch.cat([b, pts9_x*(l+r)-l], dim=1)
        # pts = torch.cat([pts1, pts2, pts3, pts4, pts5, pts6, pts7, pts8, pts9], dim=1)

        # pts = scale_pts(self.offset_out(self.offset_conv(reg_feat)))
        # pts_y, pts_x = pts.reshape(N, -1, 2, H, W).chunk(2, dim=2)
        # pts_y = pts_y.reshape(N, -1, H, W)
        # pts_x = pts_x.reshape(N, -1, H, W)
        # bbox, left_indx, up_indx, right_indx, bottom_indx = self.points2distance(pts)
        bbox = self.points2distance(pts)

        # pts = torch.cat([
        #     torch.gather(pts_y, 1, on left_indx),
        #     torch.gather(pts_x, 1, left_indx),
        #     torch.gather(pts_y, 1, left_indx),
        #     torch.gather(pts_x, 1, left_indx),
        #     torch.gather(pts_y, 1, up_indx),
        #     torch.gather(pts_x, 1, up_indx),
        #     torch.gather(pts_y, 1, right_indx),
        #     torch.gather(pts_x, 1, right_indx),
        #     torch.gather(pts_y, 1, bottom_indx),
        #     torch.gather(pts_x, 1, bottom_indx)
        # ], dim=1)

        # pts_y = (pts_y - pts_y.min(dim=1, keepdim=True)[0]) / (pts_y.max(dim=1, keepdim=True)[0] - pts_y.min(dim=1, keepdim=True)[0])
        # pts_y = pts_y * (u+b) - u
        # pts_x = (pts_x - pts_x.min(dim=1, keepdim=True)[0]) / (pts_x.max(dim=1, keepdim=True)[0] - pts_x.min(dim=1, keepdim=True)[0])
        # pts_x = pts_x * (l+r) - l
        # pts = torch.stack([pts_y, pts_x], dim=2).reshape(N, -1, H, W)

        # bbox = self.points2distance(pts)

        # cls_pts_y, cls_pts_x = scale_cls_pts(self.cls_offset_out(self.cls_offset_conv(reg_feat))).sigmoid().split([self.points, self.points], dim=1)
        # cls_pts_y = cls_pts_y * (u+b) - u
        # cls_pts_x = cls_pts_x * (l+r) - l
        # cls_pts = torch.stack([cls_pts_y, cls_pts_x], dim=2).reshape(N, -1, H, W)

        # cls_feat = self.cls_conv(cls_feat, cls_pts - base_offset)
        reg_feat = self.reg_conv(reg_feat, pts - base_offset)

        # return pts, cls_feat, reg_feat, dist_bbox, bbox
        return pts, cls_feat, reg_feat, bbox


# class SampleFeat(nn.Module):
#     def __init__(self):
#         super(SampleFeat, self).__init__()

#     def sample_offset(self, x, flow, padding_mode):
#         N, _, H, W = flow.size()
#         x_ = torch.arange(W).view(1, -1).expand(H, -1)
#         y_ = torch.arange(H).view(-1, 1).expand(-1, W)
#         grid = torch.stack([x_, y_], dim=0).type_as(x)

#         grid = grid.unsqueeze(0).expand(N, -1, -1, -1)
#         grid = grid + flow
#         gx = 2 * grid[:, 0, :, :] / (W - 1) - 1
#         gy = 2 * grid[:, 1, :, :] / (H - 1) - 1
#         grid = torch.stack([gx, gy], dim=1)
#         grid = grid.permute(0, 2, 3, 1)
#         return F.grid_sample(x, grid, padding_mode=padding_mode, align_corners=True)

#     def forward(self, x, offset):
#         B, C, H, W = x.shape
#         offset_reshape = offset.reshape(-1, 2, H, W)
#         points = offset_reshape.shape[0] // B

#         offset_reshape = torch.stack([offset_reshape[:, 1, ...], offset_reshape[:, 0, ...]], dim=1)

#         x = x.unsqueeze(1).repeat(1, points, 1, 1, 1).reshape(B*points, C, H, W)
#         sampled_feat = self.sample_offset(x, offset_reshape, padding_mode='zeros').reshape(B, points*C, H, W)
#         return sampled_feat

# class Merge(nn.Module):
#     def __init__(self, initial=0.0):
#         super(Merge, self).__init__()
#         self.scale_a = nn.Parameter(torch.tensor(initial, dtype=torch.float))
#         self.scale_b = nn.Parameter(torch.tensor(initial, dtype=torch.float))

#     def forward(self, a, b):
#         scale_a = self.scale_a.exp()
#         scale_b = self.scale_b.exp()
#         out = (scale_a * a + scale_b * b) / (scale_a + scale_b)
#         return out

# class Scale2(nn.Module):
#     def __init__(self, scale=0.0):
#         super(Scale2, self).__init__()
#         self.scale_a = nn.Parameter(torch.tensor(scale, dtype=torch.float))
#         self.scale_b = nn.Parameter(torch.tensor(scale, dtype=torch.float))

#     def forward(self, a, b):
#         # return (a * self.scale_a.exp() + b * self.scale_b.exp()) / (self.scale_a.exp() + self.scale_b.exp())
#         return a * self.scale_a.sigmoid() + b * self.scale_b.sigmoid()


@HEADS.register_module()
class SERIALGFLHead(AnchorHead):
    def __init__(
        self,
        num_classes,
        in_channels,
        stacked_convs=4,
        conv_cfg=None,
        norm_cfg=dict(type="GN", num_groups=32, requires_grad=True),
        bins=5,
        points=9,
        DGQP_cfg=dict(channels=64),
        dcn_on_last_conv=False,
        loss_dfl=dict(type="DistributionFocalLoss", loss_weight=0.25),
        **kwargs
    ):
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.bins = bins
        # self.bins = 17
        # self.bins = 5
        # self.bins = 17

        # self.points = points
        # self.num_points = self.points

        self.points_cls = points
        self.points_reg = points
        self.num_points = self.points_reg

        self.dcn_on_last_conv = dcn_on_last_conv
        self.DGQP_cfg = DGQP_cfg

        self.with_DGQP = DGQP_cfg is not None
        # self.with_DGQP = None

        super(SERIALGFLHead, self).__init__(num_classes, in_channels, **kwargs)

        self.sampling = False
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            # SSD sampling=False so use PseudoSampler
            sampler_cfg = dict(type="PseudoSampler")
            self.sampler = build_sampler(sampler_cfg, context=self)

        self.loss_dfl = build_loss(loss_dfl)

        # self.integral_initial = Integral_v2(0, 15, bins=16)
        # self.integral_refine = Integral(
        #     0, self.bins - 1, bins=self.bins, with_softmax=True
        # )
        self.integral_refine = Integral(0, 4, bins=self.bins, with_softmax=True)
        # self.integral_refine = Integral(-1, 5, bins=self.bins)

        # self.integral_refine = Integral(4, 0, bins=5)

        # self.integral_refine = Integral(0.0, 2.0, bins=3)

        # self.integral_initial = Integral(0, 16, bins=17)

        # self.integral = Integral(self.bins-1, 0, bins=self.bins, with_softmax=False)
        # self.integral = Integral(0, 17+4-1, bins=17+4, with_softmax=False)
        # self.integral = Integral(0, 16, bins=17, with_softmax=True)

        # self.integral = Integral(0, 18, bins=19, with_softmax=False)

        # self.unfold = nn.Unfold([1, 5], padding=[0, 4])
        # self.unfold = nn.Unfold([1, 17], padding=[0, 16])

        # self.scales_pts_cls = nn.ModuleList(
        #     [Scale(1.0) for _ in self.anchor_generator.strides])
        # self.scales_pts_reg = nn.ModuleList(
        #     [Scale(1.0) for _ in self.anchor_generator.strides])
        # self.scales_initial = nn.ModuleList(
        #     [Scale(1.0) for _ in self.anchor_generator.strides])

        self.scales_pts = nn.ModuleList(
            [Scale(1.0) for _ in self.anchor_generator.strides]
        )
        self.scales_refine = nn.ModuleList(
            [Scale(1.0) for _ in self.anchor_generator.strides]
        )

        # self.scales_add_0 = nn.ModuleList(
        #     [Scale2(0.0) for _ in self.anchor_generator.strides])
        # self.scales_add_1 = nn.ModuleList(
        #     [Scale2(0.0) for _ in self.anchor_generator.strides])

        # self.scales_cls_pts = nn.ModuleList(
        #     [Scale(1.0) for _ in self.anchor_generator.strides])
        # self.scales_mask = nn.ModuleList(
        #     [Scale(1.0) for _ in self.anchor_generator.strides])
        # self.scales_bbox = nn.ModuleList(
        #     [Scale(1.0) for _ in self.anchor_generator.strides])

        # self.scale_cls = nn.Parameter(torch.tensor(0.0, dtype=torch.float))
        # self.scale_reg = nn.Parameter(torch.tensor(0.0, dtype=torch.float))

        # self.merges = nn.ModuleList([
        #     Merge(initial=0.0) for _ in self.anchor_generator.strides])
        # self.merge = Merge(initial=0.0)

        # self.cls_dconv = DeformConv2dModule(self.feat_channels, self.feat_channels, [1, self.points], stride=1,
        #                 padding=[0, self.points//2], norm_cfg=norm_cfg)
        # self.reg_dconv = DeformConv2dModule(self.feat_channels, self.feat_channels, [1, self.points], stride=1,
        #                 padding=[0, self.points//2], norm_cfg=norm_cfg)

        self.cls_dconv = DeformConv2dModule(
            self.feat_channels,
            self.feat_channels,
            [1, self.points_cls],
            stride=1,
            padding=[0, self.points_cls // 2],
            norm_cfg=norm_cfg,
        )
        self.reg_dconv = DeformConv2dModule(
            self.feat_channels,
            self.feat_channels,
            [1, self.points_reg],
            stride=1,
            padding=[0, self.points_reg // 2],
            norm_cfg=norm_cfg,
        )
        normal_init(self.cls_dconv.conv, std=0.01)
        normal_init(self.reg_dconv.conv, std=0.01)

        # self.cls_dconv = DeformConv2dModule(self.feat_channels, self.feat_channels, [1, self.points], stride=1,
        #                 padding=[0, self.points//2], norm_cfg=None)
        # self.reg_dconv = DeformConv2dModule(self.feat_channels, self.feat_channels, [1, self.points], stride=1,
        #                 padding=[0, self.points//2], norm_cfg=None)

        # self.transition = nn.Conv2d(256, 256, 1, bias=False)
        # normal_init(self.transition, std=0.01)

        # self.cls_dconv = ConvModule(self.feat_channels, self.feat_channels, 3, stride=1,
        #                 padding=1, norm_cfg=norm_cfg)
        # self.reg_dconv = ConvModule(self.feat_channels, self.feat_channels, 3, stride=1,
        #                 padding=1, norm_cfg=norm_cfg)

        self.offset_conv = ConvModule(
            self.feat_channels, self.feat_channels, 3, padding=1, norm_cfg=norm_cfg
        )
        normal_init(self.offset_conv.conv, std=0.01)
        # self.offset_out = nn.Conv2d(
        #     self.feat_channels, self.points_reg*2, 3, padding=1)
        # self.offset_out = nn.Conv2d(self.feat_channels, self.points_reg * 2, 1)
        self.offset_out = nn.Conv2d(self.feat_channels, self.points_reg * 2, 1)
        normal_init(self.offset_out, std=0.01)

        # self.mask = nn.Conv2d(self.feat_channels, 1, 3, padding=1)
        # normal_init(self.mask, std=0.01)

        # self.bbox_out = nn.Conv2d(self.feat_channels, 4 + 2, 1)
        # normal_init(self.bbox_out, std=0.01)

        # self.offset_conv = ConvModule(self.feat_channels, self.feat_channels, 3, padding=1, norm_cfg=norm_cfg)
        # # DeformConv2dModule(self.feat_channels, self.feat_channels, [1, self.points], stride=1,
        # #                 padding=[0, self.points//2], norm_cfg=norm_cfg)
        # normal_init(self.offset_conv.conv, std=0.01)
        # # self.offset_conv = DeformConv2dModule(self.feat_channels, self.feat_channels, 1, stride=1, padding=0, norm_cfg=norm_cfg)
        # self.offset_out = nn.Conv2d(self.feat_channels, 4 + self.points*2 + self.points*2, 3, padding=1)
        # normal_init(self.offset_out, std=0.01)

        # self.offset_conv = ConvModule(self.feat_channels, self.feat_channels, 3, padding=1, norm_cfg=norm_cfg)
        # self.offset_conv = ConvModule(self.feat_channels, self.feat_channels, 3, padding=1, norm_cfg=None)
        # self.offset_out = nn.Conv2d(self.feat_channels, self.points*2, 3, padding=1)
        # normal_init(self.offset_out, std=0.01)
        # self.offset_out = nn.Conv2d(self.feat_channels, 4*2 * 33 + (self.points*2- 4*2), 1)
        # self.offset_out = nn.Conv2d(self.feat_channels, 4 + self.points*2, 1)

        # self.offset_conv_1 = ConvModule(
        #     self.feat_channels, self.feat_channels, 3, padding=1, norm_cfg=norm_cfg
        # )
        # self.offset_out_1 = nn.Conv2d(self.feat_channels, self.points_reg*2, 1)
        # normal_init(self.offset_conv_1.conv, std=0.01)
        # normal_init(self.offset_out_1, std=0.01)

        # self.offset_conv_2 = DeformConv2dModule(
        #     self.feat_channels, self.feat_channels, 3, padding=1, norm_cfg=norm_cfg
        # )
        # self.offset_out_2 = nn.Conv2d(self.feat_channels, self.points_reg*2, 1)
        # normal_init(self.offset_conv_2.conv, std=0.01)
        # normal_init(self.offset_out_2, std=0.01)

        # self.offset_reg_conv = DeformConv2dModule(self.feat_channels, self.feat_channels, 3, padding=1, norm_cfg=norm_cfg)
        # self.offset_reg_out = nn.Conv2d(self.feat_channels, self.points*2, 3, padding=1)
        # normal_init(self.offset_reg_out, std=0.01)

        # # self.bbox_conv = ConvModule(self.feat_channels, self.feat_channels, 3, padding=1, norm_cfg=norm_cfg)
        # self.bbox_conv = DeformConv2dModule(self.feat_channels, self.feat_channels, 1, stride=1, padding=0, norm_cfg=norm_cfg)
        # normal_init(self.bbox_conv.conv, std=0.01)
        # self.bbox_out = nn.Conv2d(self.feat_channels, self.points*2, 3, padding=1)
        # # nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        # normal_init(self.bbox_out, std=0.01)

        # self.cls_offset_conv = ConvModule(self.feat_channels, self.feat_channels, 3, padding=1, norm_cfg=norm_cfg)
        # normal_init(self.cls_offset_conv.conv, std=0.01)
        # self.cls_offset_out = nn.Conv2d(self.feat_channels, self.points*2, 3, padding=1)
        # normal_init(self.cls_offset_out, std=0.01)

        # self.base_offset = self.get_base_offset(self.points)
        self.base_offset_cls = self.get_base_offset(self.points_cls)
        self.base_offset_reg = self.get_base_offset(self.points_reg)

        # self.merge_cls = nn.Conv2d(4, 2, kernel_size=7, stride=1, padding=3)
        # self.merge_reg = nn.Conv2d(4, 2, kernel_size=7, stride=1, padding=3)
        # normal_init(self.merge_cls, std=0.01)
        # normal_init(self.merge_reg, std=0.01)

        # self.shrink = nn.Conv2d(self.feat_channels*2, self.feat_channels, 1)
        # normal_init(self.shrink, std=0.01)

        # self.shrink_0 = nn.Conv2d(self.feat_channels, self.feat_channels//2, 1)
        # normal_init(self.shrink_0, std=0.01)

        # self.shrink_1 = nn.Conv2d(self.feat_channels, self.feat_channels//2, 1)
        # normal_init(self.shrink_1, std=0.01)

        # self.sample_feat = SampleFeat()
        # self.edge_conv = ConvModule(4*self.feat_channels, self.feat_channels, 3, padding=1, norm_cfg=norm_cfg)
        # self.edge_out = nn.Conv2d(self.feat_channels, self.cls_out_channels, 3, padding=1)
        # normal_init(self.edge_conv.conv, std=0.01)
        # normal_init(self.edge_out, std=0.01, bias=bias_init_with_prob(0.01))

        # self.loss_edge = build_loss(
        #     dict(
        #     type='FocalLoss',
        #     use_sigmoid=True,
        #     gamma=2.0,
        #     alpha=0.25,
        #     loss_weight=1.0))

    def get_base_offset(self, points):
        dcn_base_x = np.arange(-((points - 1) // 2), points // 2 + 1).astype(np.float64)
        dcn_base_y = dcn_base_x * 0.0
        dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=1).reshape((-1))
        dcn_base_offset = torch.tensor(dcn_base_offset).view(1, -1, 1, 1)
        return dcn_base_offset

    def _init_layers(self):
        """Initialize layers of the head."""
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()

        for i in range(self.stacked_convs - 1):
            chn = self.in_channels if i == 0 else self.feat_channels
            if self.dcn_on_last_conv and (i == self.stacked_convs - 2):
                conv_cfg = dict(type="DCNv2")
            else:
                conv_cfg = self.conv_cfg
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                )
            )
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=self.norm_cfg,
                )
            )

        # self.gfl_dual = DualDeformModule(self.feat_channels, points=self.points, norm_cfg=self.norm_cfg)

        assert self.num_anchors == 1, "anchor free version"

        # self.gfl_cls = nn.Conv2d(
        #     self.feat_channels, self.cls_out_channels, 3, padding=1)
        # self.gfl_reg = nn.Conv2d(
        #     self.feat_channels, 4 * self.bins, 3, padding=1)

        self.gfl_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 1)
        self.gfl_reg = nn.Conv2d(self.feat_channels, 4 * self.bins, 1)

        if self.with_DGQP:
            self.reg_conf = nn.Sequential(
                nn.Conv2d(4 * self.bins, self.DGQP_cfg["channels"], 1),
                nn.GroupNorm(8, self.DGQP_cfg["channels"]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.DGQP_cfg["channels"], 1, 1),
                nn.Sigmoid(),
            )

        # self.cls_conv = DeformConv2dModule(256, 256, [1, self.points], stride=1,
        #                 padding=[0, self.points//2], norm_cfg=self.norm_cfg)

        # self.cls_offset_conv = ConvModule(256, 256, 3, padding=1, norm_cfg=self.norm_cfg)
        # self.cls_offset_out = nn.Conv2d(256, self.points*2, 3, padding=1)
        # normal_init(self.cls_offset_conv.conv, std=0.01)
        # normal_init(self.cls_offset_out, std=0.01)

    def init_weights(self):
        """Initialize weights of the head."""
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)

        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.gfl_cls, std=0.01, bias=bias_cls)
        normal_init(self.gfl_reg, std=0.01)

        if self.with_DGQP:
            for m in self.reg_conf:
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)

    def points2distance(self, pts, return_indx=False):
        if pts.dim() == 4:
            pts_reshape = pts.view(pts.shape[0], -1, 2, *pts.shape[2:])
            pts_y = pts_reshape[:, :, 0, ...]
            pts_x = pts_reshape[:, :, 1, ...]
        elif pts.dim() == 2:
            pts_reshape = pts.view(pts.shape[0], -1, 2)
            pts_y = pts_reshape[:, :, 0]
            pts_x = pts_reshape[:, :, 1]

        bbox_left, left_indx = pts_x.min(dim=1, keepdim=True)
        bbox_right, right_indx = pts_x.max(dim=1, keepdim=True)
        bbox_up, up_indx = pts_y.min(dim=1, keepdim=True)
        bbox_bottom, bottom_indx = pts_y.max(dim=1, keepdim=True)
        bbox = torch.cat([-bbox_left, -bbox_up, bbox_right, bbox_bottom], dim=1)

        if return_indx:
            return bbox, left_indx, up_indx, right_indx, bottom_indx
        else:
            return bbox

    def forward(self, feats):
        return multi_apply(
            self.forward_single,
            feats,
            # self.scales_reg_pts, self.scales_initial, self.scales_refine, self.scales_cls_pts)
            # self.scales_pts, self.scales_refine, self.scales_cls_pts)
            # self.scales_initial, self.scales_refine)
            # self.scales_pts, self.scales_initial, self.scales_refine)
            self.scales_pts,
            self.scales_refine,
        )
        # self.scales_pts, self.scales_refine, self.scales_cls_pts)
        # self.scales_reg_pts, self.scales_refine, self.scales_bbox)

    # def forward_single(self, x, scale_initial, scale_refine):
    # def forward_single(self, x, scale_reg_pts, scale_initial, scale_refine, scale_cls_pts):
    # def forward_single(self, x, scale_pts, scale_refine, scale_cls_pts):
    # def forward_single(self, x, scale_pts, scale_initial, scale_refine):
    # def forward_single(self, x, scale_pts, scale_refine, scale_bbox):
    def forward_single(self, x, scale_pts, scale_refine):
        # def forward_single(self, x, scale_pts, scale_refine, scales_cls_pts):
        # base_offset = self.base_offset.type_as(x)
        base_offset_cls = self.base_offset_cls.type_as(x)
        base_offset_reg = self.base_offset_reg.type_as(x)

        N, _, H, W = x.shape

        cls_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
            # cls_feat = cls_feat + x

        # merge_feat = self.transition(cls_feat) + x
        # merge_feat = torch.cat([cls_feat, x], dim=1)
        # merge_feat = torch.where(x.gt(cls_feat), x, cls_feat)
        # reg_feat = merge_feat
        # reg_feat = cls_feat * 0.0 + x
        # cls_feat = merge_feat
        # cls_feat = x

        merge_feat = cls_feat + x
        reg_feat = merge_feat
        cls_feat = merge_feat

        # merge_cls = self.merge_cls(
        #     torch.cat([x.max(dim=1, keepdim=True)[0], x.mean(dim=1, keepdim=True),
        #     cls_feat.max(dim=1, keepdim=True)[0], cls_feat.mean(dim=1, keepdim=True)], dim=1)
        # ).sigmoid()
        # merge_reg = self.merge_reg(
        #     torch.cat([x.max(dim=1, keepdim=True)[0], x.mean(dim=1, keepdim=True),
        #     cls_feat.max(dim=1, keepdim=True)[0], cls_feat.mean(dim=1, keepdim=True)], dim=1)
        # ).sigmoid()

        # cls_feat = x * merge_cls[:, 0:1, ...] + cls_feat * merge_cls[:, 1:2, ...]
        # reg_feat = x * merge_reg[:, 0:1, ...] + cls_feat * merge_reg[:, 1:2, ...]

        # cls_feat = merge_feat
        # reg_feat = merge_feat
        # reg_feat = x

        # cls_feat = cls_feat + x
        # reg_feat = cls_feat

        # reg_feat = x

        # reg_feat = self.transition(x) + cls_feat
        # reg_feat = torch.cat([cls_feat, x], dim=1)

        # cls_feat = self.relu(cls_feat) + x
        # cls_feat = torch.cat([self.shrink_0(cls_feat), self.shrink_1(x)], dim=1)
        # cls_feat = torch.cat([cls_feat, x], dim=1)
        # reg_feat = self.shrink_0(cls_feat)
        # cls_feat = self.shrink(cls_feat)
        # cls_feat = x + cls_feat
        # cls_feat = (cls_feat + x) / math.sqrt(2.0)

        # cls_feat = torch.cat([x, cls_feat], dim=1)
        # cls_feat = x + cls_feat

        # reg_feat = x + cls_feat
        # cls_feat = scale_add(cls_feat, x)
        # reg_feat = cls_feat
        # reg_feat = self.transfer(x) + cls_feat

        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)

        # reg_feat = reg_feat + merge_feat

        # reg_feat = reg_feat + cls_feat
        # cls_feat = reg_feat

        # mask = self.mask(reg_feat).sigmoid()

        # reg_feat = reg_feat * mask

        # reg_feat = torch.cat([reg_feat, x], dim=1)

        # reg_feat_backup = reg_feat
        # cls_feat_backup = cls_feat

        # pts_pred_initial, cls_feat, reg_feat = self.gfl_dual(cls_feat, reg_feat, scale_initial)
        # pts_pred_initial, cls_feat, reg_feat, prob_initial, bbox_pred_initial = self.gfl_dual(cls_feat, reg_feat, scale_pts, scale_initial, scale_cls_pts)
        # pts_pred_initial, cls_feat, reg_feat, bbox_pred_initial = self.gfl_dual(cls_feat, reg_feat, scale_pts, scale_initial, scale_cls_pts)
        # pts_pred_initial, cls_feat, reg_feat, bbox_pred_initial = self.gfl_dual(cls_feat, reg_feat, scale_pts)
        # pts_pred_initial, cls_feat, reg_feat, bbox_pred_initial = self.gfl_dual(cls_feat, reg_feat, scale_pts, scale_initial)
        # pts_pred_initial, cls_feat, reg_feat, bbox_pred_initial = self.gfl_dual(cls_feat, reg_feat, scale_pts)

        # pts_reg = scale_pts(self.offset_out(self.offset_conv(reg_feat)))

        # pts_reg = scale_pts(self.offset_out(self.reg_dconv(reg_feat, -base_offset.repeat(N, 1, H, W))))
        # reg_feat = self.reg_dconv(reg_feat, pts_reg - base_offset)

        # pts_reg = scale_pts(self.offset_out_2(self.reg_dconv(reg_feat, pts_reg - base_offset))) + pts_reg

        # bbox = scale_initial(self.bbox_out(self.bbox_conv(reg_feat)))
        # .split([4, 2], dim=1)
        # # bbox = self.points2distance(pts_bbox)
        # l, u, r, b = bbox.chunk(4, dim=1)

        # offset = torch.cat([(r-l)/2, (b-u)/2], dim=1)

        # pts_reg= scale_pts(self.offset_out(self.offset_conv(reg_feat, offset.contiguous())))
        # pts_reg= scale_pts(self.offset_out(self.offset_conv(reg_feat)))

        # bbox = self.points2distance(pts)

        # pts_reg, pts_cls = scale_pts(self.offset_out(self.offset_conv(reg_feat, pts_bbox - base_offset))).split([self.points, self.points*2], dim=1)
        # pts_reg, pts_cls = scale_pts(self.offset_out(self.offset_conv(reg_feat))).split([self.points*2, self.points*2], dim=1)

        # pts_y, pts_x = pts_reg.exp().chunk(2, dim=1)
        # pts_y = (pts_y - pts_y.min(dim=1, keepdim=True)[0]) / (pts_y.max(dim=1, keepdim=True)[0] - pts_y.min(dim=1, keepdim=True)[0])
        # pts_x = (pts_x - pts_x.min(dim=1, keepdim=True)[0]) / (pts_x.max(dim=1, keepdim=True)[0] - pts_x.min(dim=1, keepdim=True)[0])
        # pts_y = pts_y * (u+b) - u
        # pts_x = pts_x * (l+r) - l
        # pts_reg = torch.stack([pts_y, pts_x], dim=2).reshape(N, -1, H, W)

        # pts_y, pts_x = pts_cls.exp().chunk(2, dim=1)
        # pts_y = (pts_y - pts_y.min(dim=1, keepdim=True)[0]) / (pts_y.max(dim=1, keepdim=True)[0] - pts_y.min(dim=1, keepdim=True)[0])
        # pts_x = (pts_x - pts_x.min(dim=1, keepdim=True)[0]) / (pts_x.max(dim=1, keepdim=True)[0] - pts_x.min(dim=1, keepdim=True)[0])
        # pts_y = pts_y * (u+b) - u
        # pts_x = pts_x * (l+r) - l
        # pts_cls = torch.stack([pts_y, pts_x], dim=2).reshape(N, -1, H, W)

        # pts = pts_reg.sigmoid()
        # pts_l = torch.stack([
        #     pts[:, 0:3, ...] * (u+b) - u,
        #     -l.repeat(1, 3, 1, 1)
        # ], dim=2)
        # pts_u = torch.stack([
        #     -u.repeat(1, 2, 1, 1),
        #     pts[:, 3:5, ...] * (l+r) - l
        # ], dim=2)
        # pts_r = torch.stack([
        #     pts[:, 5:7, ...] * (u+b) - u,
        #     r.repeat(1, 2, 1, 1)
        # ], dim=2)
        # pts_b = torch.stack([
        #     b.repeat(1, 2, 1, 1),
        #     pts[:, 7:9, ...] * (l+r) - l
        # ], dim=2)
        # pts_reg = torch.cat([pts_l, pts_u, pts_r, pts_b], dim=1).reshape(N, -1, H, W)

        # pts_y, pts_x = pts_cls.sigmoid().chunk(2, dim=1)
        # pts_y = pts_y * (u+b) - u
        # pts_x = pts_x * (l+r) - l
        # pts_cls = torch.stack([pts_y, pts_x], dim=2).reshape(N, -1, H, W)

        # pts_reg = torch.stack([pts_reg_y, pts_reg_x], dim=2).reshape(N, -1, H, W)
        # mask_reg = mask_reg.sigmoid()
        # mask_cls = mask_cls.sigmoid()

        # bbox_initial = self.integral_initial(scale_initial(bbox_initial))
        # l, u, r, b = bbox_initial.chunk(4, dim=1)

        # pts_reg_l_y, pts_reg_u_x, pts_reg_r_y, pts_reg_b_x = scale_reg_pts(pts_reg).sigmoid().split([3, 2, 2, 2], dim=1)
        # pts_reg = torch.cat([
        #     torch.stack([pts_reg_l_y*(u+b)-u, -l.repeat(1, 3, 1, 1)], dim=2),
        #     torch.stack([-u.repeat(1, 2, 1, 1), pts_reg_u_x*(l+r)-l], dim=2),
        #     torch.stack([pts_reg_r_y*(u+b)-u, r.repeat(1, 2, 1, 1)], dim=2),
        #     torch.stack([b.repeat(1, 2, 1, 1), pts_reg_b_x*(l+r)-l], dim=2)
        # ], dim=1).reshape(N, -1, H, W)

        # pts_cls_y, pts_cls_x = scale_cls_pts(pts_cls).sigmoid().chunk(2, dim=1)
        # pts_cls = torch.stack([
        #     pts_cls_y*(u+b)-u, pts_cls_x*(l+r)-l
        # ], dim=2).reshape(N, -1, H, W)

        # pts_cls = self.cls_offset_out(self.cls_offset_conv(reg_feat))
        # pts_cls = scales_cls_pts(pts_cls).sigmoid()

        # offset = scale_initial(self.offset_out(reg_feat))
        # pts_reg = scale_pts(self.bbox_out(self.bbox_conv(reg_feat, offset))) + offset.repeat(1, 9, 1, 1)
        # bbox = self.points2distance(pts_reg)

        # bbox, pts_reg, pts_cls = scale_pts(self.offset_out(self.offset_conv(reg_feat))).split(
        #     [4, self.points*2, self.points*2], dim=1)
        # l, u, r, b = bbox.chunk(4, dim=1)
        # # pts_y = (pts_y.reshape(N, -1, 2, H, W).softmax(dim=2) * torch.stack([-u, b], dim=2)).sum(dim=2)
        # # pts_x = (pts_x.reshape(N, -1, 2, H, W).softmax(dim=2) * torch.stack([-l, r], dim=2)).sum(dim=2)
        # # pts_reg = torch.stack([pts_y, pts_x], dim=2).reshape(N, -1, H, W)

        # pts_y, pts_x = pts_reg.chunk(2, dim=1)
        # pts_y = torch.where(pts_y.lt(b.repeat(1, 9, 1, 1)), pts_y, b.repeat(1, 9, 1, 1))
        # pts_y = torch.where(pts_y.gt(-u.repeat(1, 9, 1, 1)), pts_y, -u.repeat(1, 9, 1, 1))
        # pts_x = torch.where(pts_x.lt(r.repeat(1, 9, 1, 1)), pts_x, r.repeat(1, 9, 1, 1))
        # pts_x = torch.where(pts_x.gt(-l.repeat(1, 9, 1, 1)), pts_x, -l.repeat(1, 9, 1, 1))
        # pts_reg = torch.stack([pts_y, pts_x], dim=2).reshape(N, -1, H, W)

        # reg_feat = self.reg_dconv(reg_feat, pts - base_offset)

        # prob_initial = prob_initial.reshape(
        #     N, 4, -1, H, W).softmax(dim=2)

        # pts_cls, pts_reg = scale_pts(self.offset_out(
        #     self.offset_conv(reg_feat))).split([self.points_cls*2, self.points_reg*2], dim=1)

        # pts_reg_1 = scale_pts(self.offset_out_1(self.offset_conv_1(reg_feat)))
        # pts_reg_2 = scale_pts(self.offset_out_2(
        #     self.offset_conv_2(reg_feat, pts_reg_1 - base_offset_reg)))
        # pts_reg = pts_reg_1 + pts_reg_2
        # pts_cls = pts_reg

        # pts_reg = scale_pts(self.offset_out(self.offset_conv(reg_feat)))
        # pts_cls = pts_reg

        # bbox, pts = self.offset_out(self.offset_conv(reg_feat)).split([4, self.points*2], dim=1)
        # bbox_pred_initial_cls, pts_cls, bbox_pred_initial_reg, pts_reg = self.offset_out(
        #     self.offset_conv(reg_feat)).split([4, self.points_cls*2, 4, self.points_reg*2], dim=1)

        # bbox_pred_initial, pts_reg = self.offset_out(self.offset_conv(reg_feat)).split(
        #     [4, self.points_reg * 2], dim=1
        # )
        # bbox_pred_initial = self.integral_initial(scale_pts(dist_pred_initial))
        # bbox_pred_initial, pts_reg = self.offset_out(
        #     self.offset_conv(reg_feat)).split([4, self.points_reg*2], dim=1)
        # bbox_pred_initial = scale_pts(bbox_pred_initial).exp()
        # l, u, r, b = bbox_pred_initial.chunk(4, dim=1)

        # lr, ub = scale_pts(lrub).chunk(2, dim=1)

        # bbox_dist = scale_pts(bbox_dist)
        # bbox = scale_pts(bbox).exp()
        # bbox = scale_pts(bbox)
        # lr, ub = bbox.chunk(2, dim=1)
        # l = lr.min(dim=1, keepdim=True)[0]
        # r = lr.max(dim=1, keepdim=True)[0]
        # u = ub.min(dim=1, keepdim=True)[0]
        # b = ub.max(dim=1, keepdim=True)[0]
        # bbox = torch.cat([-l, -u, r, b], dim=1)

        # l, u, r, b = bbox.chunk(4, dim=1)
        # l, u, r, b = self.integral_initial(bbox_dist).chunk(4, dim=1)

        # # bbox = self.integral_initial(bbox_dist)

        # pts_y, pts_x = pts_reg.chunk(2, dim=1)
        # pts_y = pts_y.exp()
        # pts_x = pts_x.exp()
        # pts_y = (pts_y - pts_y.min(dim=1, keepdim=True)[0]) / (
        #     pts_y.max(dim=1, keepdim=True)[0] -
        #     pts_y.min(dim=1, keepdim=True)[0]
        # )
        # pts_x = (pts_x - pts_x.min(dim=1, keepdim=True)[0]) / (
        #     pts_x.max(dim=1, keepdim=True)[0] -
        #     pts_x.min(dim=1, keepdim=True)[0]
        # )
        # pts_y = pts_y * (b + u) - u
        # pts_x = pts_x * (r + l) - l
        # pts_reg = torch.stack([pts_y, pts_x], dim=2).reshape(N, -1, H, W)

        # pts_cls = pts_reg

        # bbox_pred_initial_cls = scale_pts(bbox_pred_initial_cls)
        # l, u, r, b = bbox_pred_initial_cls.chunk(4, dim=1)
        # pts_y, pts_x = pts_cls.chunk(2, dim=1)
        # pts_y = pts_y.exp()
        # pts_x = pts_x.exp()
        # pts_y = (pts_y - pts_y.min(dim=1, keepdim=True)[0]) / (
        #     pts_y.max(dim=1, keepdim=True)[0] - pts_y.min(dim=1, keepdim=True)[0])
        # pts_x = (pts_x - pts_x.min(dim=1, keepdim=True)[0]) / (
        #     pts_x.max(dim=1, keepdim=True)[0] - pts_x.min(dim=1, keepdim=True)[0])
        # pts_y = pts_y * (b+u) - u
        # pts_x = pts_x * (r+l) - l
        # pts_cls = torch.stack([pts_y, pts_x], dim=2).reshape(N, -1, H, W)

        # bbox_pred_initial_reg = scale_pts(bbox_pred_initial_reg)
        # l, u, r, b = bbox_pred_initial_reg.chunk(4, dim=1)
        # pts_y, pts_x = pts_reg.chunk(2, dim=1)
        # pts_y = pts_y.exp()
        # pts_x = pts_x.exp()
        # pts_y = (pts_y - pts_y.min(dim=1, keepdim=True)[0]) / (
        #     pts_y.max(dim=1, keepdim=True)[0] - pts_y.min(dim=1, keepdim=True)[0])
        # pts_x = (pts_x - pts_x.min(dim=1, keepdim=True)[0]) / (
        #     pts_x.max(dim=1, keepdim=True)[0] - pts_x.min(dim=1, keepdim=True)[0])
        # pts_y = pts_y * (b+u) - u
        # pts_x = pts_x * (r+l) - l
        # pts_reg = torch.stack([pts_y, pts_x], dim=2).reshape(N, -1, H, W)

        # pts = pts * 0.0
        # pts = scale_pts(self.offset_out(reg_feat))

        # pts_y, pts_x, bbox = scale_pts(self.offset_out(self.offset_conv(reg_feat))).split([self.points, self.points, 4], dim=1)
        # pts_y, pts_x, bbox = self.offset_out(self.offset_conv(reg_feat)).split([self.points, self.points, 4], dim=1)

        # l, u, r, b = scale_bbox(bbox).chunk(4, dim=1)
        # pts_y = scale_pts(pts_y).exp()
        # pts_x = scale_pts(pts_x).exp()
        # pts_y = (pts_y - pts_y.min(dim=1, keepdim=True)[0]) / (pts_y.max(dim=1, keepdim=True)[0] - pts_y.min(dim=1, keepdim=True)[0])
        # pts_x = (pts_x - pts_x.min(dim=1, keepdim=True)[0]) / (pts_x.max(dim=1, keepdim=True)[0] - pts_x.min(dim=1, keepdim=True)[0])

        # pts_y = pts_y * (u+b) - u
        # pts_x = pts_x * (l+r) - l
        # pts = torch.stack([pts_y, pts_x], dim=2).reshape(N, -1, H, W)

        # bbox, left_indx, up_indx, right_indx, bottom_indx = self.points2distance(pts, return_indx=True)
        # bbox = self.points2distance(pts)

        # pts_reg = scale_pts_reg(self.offset_reg_out(self.offset_reg_conv(reg_feat, pts_cls - base_offset))) + pts_cls

        # bbox = self.points2distance(pts_reg)

        # pts_reg = torch.cat([
        #     pts_left, pts_left,
        #     pts_up, pts_up,
        #     pts_right, pts_right,
        #     pts_bottom, pts_bottom, pts_bottom
        # ], dim=1)

        # cls_feat = self.cls_dconv(cls_feat + reg_feat, pts_cls - base_offset)

        # cls_feat = self.cls_dconv(cls_feat, pts_cls - base_offset)
        # reg_feat = self.reg_dconv(reg_feat, pts_reg - base_offset)

        # cls_feat = self.cls_dconv(
        #     cls_feat, pts_cls - base_offset_cls)
        # reg_feat = self.reg_dconv(reg_feat, pts_reg - base_offset_reg)

        # merge_feat = reg_feat
        # merge_feat = cls_feat + reg_feat

        # pts_pred_initial_reg = scale_pts(self.offset_out(self.offset_conv(reg_feat)))
        # pts_pred_initial_cls = pts_pred_initial_reg
        # pts_cls, pts_reg = scale_pts(self.offset_out(
        #     self.offset_conv(reg_feat))).chunk(2, dim=1)

        # pts_pred_initial_reg = scale_pts(self.offset_out(self.offset_conv(reg_feat)))
        pts_feat = self.offset_conv(reg_feat)
        pts_pred_initial_reg = scale_pts(self.offset_out(pts_feat))

        pts_pred_initial_cls = pts_pred_initial_reg

        # l, u, r, b = bbox_pred_initial.chunk(4, dim=1)
        # pts_l = torch.stack(
        #     [offset_pred_initial[:, 0:3, ...], -l.repeat(1, 3, 1, 1)], dim=2
        # ).reshape(N, -1, H, W)
        # pts_u = torch.stack(
        #     [-u.repeat(1, 2, 1, 1), offset_pred_initial[:, 3:5, ...]], dim=2
        # ).reshape(N, -1, H, W)
        # pts_r = torch.stack(
        #     [offset_pred_initial[:, 5:7, ...], r.repeat(1, 2, 1, 1)], dim=2
        # ).reshape(N, -1, H, W)
        # pts_b = torch.stack(
        #     [b.repeat(1, 2, 1, 1), offset_pred_initial[:, 7:9, ...]], dim=2
        # ).reshape(N, -1, H, W)
        # pts_pred_initial_reg = torch.cat([pts_l, pts_u, pts_r, pts_b], dim=1)
        # l, u, r, b = scale_pts(pred_box).exp().chunk(4, dim=1)
        # pts_y, pts_x = pts.exp().chunk(2, dim=1)
        # pts_y = (pts_y - pts_y.min(dim=1, keepdim=True)[0]) / (
        #     pts_y.max(dim=1, keepdim=True)[0] - pts_y.min(dim=1, keepdim=True)[0]
        # )
        # pts_x = (pts_x - pts_x.min(dim=1, keepdim=True)[0]) / (
        #     pts_x.max(dim=1, keepdim=True)[0] - pts_x.min(dim=1, keepdim=True)[0]
        # )
        # pts_y = pts_y * (u + b) - u
        # pts_x = pts_x * (l + r) - l
        # pts_pred_initial_reg = torch.stack([pts_y, pts_x], dim=2).reshape(N, -1, H, W)
        # pts_pred_initial_cls = pts_pred_initial_reg

        cls_feat = self.cls_dconv(cls_feat, pts_pred_initial_cls - base_offset_cls)
        reg_feat = self.reg_dconv(reg_feat, pts_pred_initial_reg - base_offset_reg)

        # cls_feat = self.cls_dconv(cls_feat)
        # reg_feat = self.reg_dconv(reg_feat)

        # prob_refine = scale_refine(self.gfl_reg(reg_feat))
        # prob_refine = scale_refine(self.gfl_reg(reg_feat, pts - base_offset))

        # bbox_pred_initial_reg = self.points2distance(pts_reg)
        # bbox_pred_refine = (
        #     self.integral_refine(prob_refine.reshape(N, -1, H, W))
        #     + bbox_pred_initial_reg
        # )
        # bbox_pred_refine = self.integral_refine(prob_refine.reshape(N, -1, H, W))
        # self.points2distance(pts)
        # bbox_pred_refine = prob_refine + bbox

        # l, u, r, b = bbox_pred_refine.chunk(4, dim=1)
        # pts_y, pts_x = pts_cls.chunk(2, dim=1)
        # pts_y = pts_y.exp()
        # pts_x = pts_x.exp()
        # pts_y = (pts_y - pts_y.min(dim=1, keepdim=True)[0]) / (pts_y.max(dim=1, keepdim=True)[0] - pts_y.min(dim=1, keepdim=True)[0])
        # pts_x = (pts_x - pts_x.min(dim=1, keepdim=True)[0]) / (pts_x.max(dim=1, keepdim=True)[0] - pts_x.min(dim=1, keepdim=True)[0])
        # pts_y = pts_y * (b+u) - u
        # pts_x = pts_x * (r+l) - l
        # pts_cls = torch.stack([pts_y, pts_x], dim=2).reshape(N, -1, H, W)

        # cls_feat = self.cls_dconv(cls_feat, pts_cls - base_offset_cls)

        # l, u, r, b = bbox.chunk(4, dim=1)
        # pts_cls_y, pts_cls_x = pts_cls.chunk(2, dim=1)
        # pts_cls_y = torch.where(pts_cls_y.lt(b), pts_cls_y, b)
        # pts_cls_y = torch.where(pts_cls_y.gt(-u), pts_cls_y, -u)
        # pts_cls_x = torch.where(pts_cls_x.lt(r), pts_cls_x, r)
        # pts_cls_x = torch.where(pts_cls_x.gt(-l), pts_cls_x, -l)
        # pts_cls = torch.stack([pts_cls_y, pts_cls_x], dim=2).reshape(N, -1, H, W)

        # pts_y, pts_x = pts.reshape(N, -1, 2, H, W).chunk(2, dim=2)
        # pts_y = pts_y.reshape(N, -1, H, W)
        # pts_x = pts_x.reshape(N, -1, H, W)

        # pts_y = torch.scatter_add(pts_y, dim=1, index=up_indx, src=-bbox_pred_refine[:, 1:2, ...])
        # pts_y = torch.scatter_add(pts_y, dim=1, index=bottom_indx, src=bbox_pred_refine[:, 3:, ...])
        # pts_x = torch.scatter_add(pts_x, dim=1, index=left_indx, src=-bbox_pred_refine[:, 0:1, ...])
        # pts_x = torch.scatter_add(pts_x, dim=1, index=right_indx, src=bbox_pred_refine[:, 2:3, ...])

        # pts = torch.stack([pts_y, pts_x], dim=2).reshape(N, -1, H, W)

        # cls_feat = self.cls_dconv(cls_feat, pts - base_offset)
        # cls_feat = self.cls_dconv(cls_feat)

        # l, u, r, b = bbox_pred_refine.chunk(4, dim=1)
        # pts_y, pts_x = pts_cls.chunk(2, dim=1)
        # pts_y = torch.where(pts_y.lt(b.repeat(1, 9, 1, 1)), pts_y, b.repeat(1, 9, 1, 1))
        # pts_y = torch.where(pts_y.gt(-u.repeat(1, 9, 1, 1)), pts_y, -u.repeat(1, 9, 1, 1))
        # pts_x = torch.where(pts_x.lt(r.repeat(1, 9, 1, 1)), pts_x, r.repeat(1, 9, 1, 1))
        # pts_x = torch.where(pts_x.gt(-l.repeat(1, 9, 1, 1)), pts_x, -l.repeat(1, 9, 1, 1))
        # pts_cls = torch.stack([pts_y, pts_x], dim=2).reshape(N, -1, H, W)

        # cls_feat = self.cls_dconv(cls_feat, pts_cls - base_offset)

        # l, u, r, b = bbox_pred_refine.chunk(4, dim=1)
        # pts_cls_y = pts_cls_y.sigmoid() * (u+b) - u
        # pts_cls_x = pts_cls_x.sigmoid() * (l+r) - l
        # pts_cls = torch.stack([pts_cls_y, pts_cls_x], dim=2).reshape(N, -1, H, W)

        # cls_feat = self.cls_dconv(cls_feat, pts_cls - base_offset)

        # l, u, r, b = bbox_pred_refine.chunk(4, dim=1)
        # pts_cls_y, pts_cls_x = pts_cls.reshape(N, -1, 2, H, W).chunk(2, dim=2)
        # pts_cls_y = pts_cls_y.squeeze()
        # pts_cls_x = pts_cls_x.squeeze()

        # pts_cls_y = torch.where(pts_cls_y.lt(b), pts_cls_y, b)
        # pts_cls_y = torch.where(pts_cls_y.gt(-u), pts_cls_y, -u)
        # pts_cls_x = torch.where(pts_cls_x.lt(r), pts_cls_x, r)
        # pts_cls_x = torch.where(pts_cls_x.gt(-l), pts_cls_x, -l)

        # pts_cls = torch.stack([pts_cls_y, pts_cls_x], dim=2).reshape(N, -1, H, W)

        # cls_feat = self.cls_dconv(cls_feat, pts_cls - base_offset)

        # l, u, r, b = (self.points2distance(pts_pred_initial) + bbox_pred_refine).chunk(4, dim=1)
        # pts_y, pts_x = pts_cls.chunk(2, dim=1)
        # pts_y = pts_y * (u+b) - u
        # pts_x = pts_x * (l+r) - l
        # pts_cls = torch.stack([pts_y, pts_x], dim=2).reshape(N, -1, H, W)

        # cls_feat = self.cls_dconv(cls_feat, pts_reg - base_offset)

        # bbox_pred_refine = self.integral(prob_refine)

        # prob_DGQP = prob_refine
        # prob_DGQP = torch.cat([prob_initial, prob_refine], dim=2)

        # prob_initial = prob_initial.permute(0, 1, 3, 4, 2).reshape(N, 4*H*W, 1, -1)
        # prob_initial = self.unfold(prob_initial).reshape(N, 4, H, W, 5, -1)

        # prob_refine = prob_refine.permute(0, 1, 3, 4, 2).reshape(N, 4, H, W, 5, 1)

        # prob = (prob_initial * prob_refine).sum(dim=-2).permute(0, 1, 4, 2, 3)

        # bbox_pred_refine = self.integral(dist_pred_refine)
        # bbox_pred_refine = scale_refine(self.gfl_reg(reg_feat))

        # bbox_pred_refine = self.integral(prob_DGQP.reshape(N, -1, H, W))
        # bbox_pred_refine = self.integral(prob.reshape(N, -1, H, W))

        # pts_pred_initial = pts_pred_initial * 0.0

        # l, u, r, b = bbox_pred_refine.chunk(4, dim=1)
        # cls_pts_y, cls_pts_x = scale_cls_pts(self.cls_offset_out(self.cls_offset_conv(reg_feat_backup))).sigmoid().split([self.points, self.points], dim=1)
        # cls_pts_y = cls_pts_y * (u+b) - u
        # cls_pts_x = cls_pts_x * (l+r) - l
        # cls_pts = torch.stack([cls_pts_y, cls_pts_x], dim=2).reshape(N, -1, H, W)
        # cls_feat = self.cls_conv(cls_feat, cls_pts - base_offset)

        # cls_feat = self.cls_conv(cls_feat, pts_pred_initial - base_offset)

        # cls_score = self.gfl_cls(cls_feat, pts - base_offset).sigmoid()

        dist_pred_refine = scale_refine(self.gfl_reg(reg_feat))

        cls_score = self.gfl_cls(cls_feat).sigmoid()
        # cls_score = self.gfl_cls(cls_feat)

        if self.with_DGQP:
            # prob_initial = torch.zeros(N, 4, 17, H, W).type_as(x)
            # prob_initial_data = torch.stack([
            #                     1.0 - bbox_pred_initial + bbox_pred_initial.floor(),
            #                     bbox_pred_initial - bbox_pred_initial.floor()], dim=2)
            # prob_initial_indx = torch.stack([bbox_pred_initial.long(), bbox_pred_initial.long()+1], dim=2)
            # prob_initial = torch.scatter(prob_initial, dim=2, index=prob_initial_indx, src=prob_initial_data)
            # # # prob_initial = bbox_dist.reshape(N, 4, -1, H, W).softmax(dim=2)
            # prob_initial = prob_initial.permute(0, 1, 3, 4, 2).reshape(N, 4*H*W, 1, -1)
            # prob_initial = self.unfold(prob_initial).reshape(N, 4, H, W, 5, -1)
            # prob_refine = prob_refine.reshape(N, 4, -1, H, W).softmax(dim=2)
            # prob_refine = prob_refine.permute(0, 1, 3, 4, 2).reshape(N, 4, H, W, -1, 1)
            # prob = (prob_initial * prob_refine).sum(dim=-2).permute(0, 1, 4, 2, 3)

            # prob_initial_left = 1.0 - bbox_pred_initial + bbox_pred_initial.floor()
            # prob_initial_right = bbox_pred_initial - bbox_pred_initial.floor()
            # prob_refine = prob_refine.reshape(N, 4, -1, H, W).softmax(dim=2)
            # prob_refine_left = torch.cat([
            #     torch.zeros(N, 4, 1, H, W).type_as(prob_refine),
            #     prob_refine
            # ], dim=2)
            # prob_refine_right = torch.cat([
            #     prob_refine,
            #     torch.zeros(N, 4, 1, H, W).type_as(prob_refine)
            # ], dim=2)
            # prob = (prob_initial_left[:, :, None, ...] * prob_refine_left) + (prob_initial_right[:, :, None, ...] * prob_refine_right)

            # prob_refine = prob_refine.reshape(N, 4, -1, H, W).softmax(dim=2)
            # prob = prob_refine * (bbox_pred_initial_reg[:, :, None, ...] + torch.arange(
            #     self.bins).type_as(x)[None, None, :, None, None])
            # prob = prob_refine * (bbox[:, :, None, ...] + torch.arange(self.bins).type_as(x)[None, None, :, None, None])
            # prob = torch.cat([prob_refine * (bbox_pred_initial - bbox_pred_initial.long())[:, :, None, ...],
            #                   prob_refine * (1.0 - bbox_pred_initial + bbox_pred_initial.long())[:, :, None, ...]], dim=2)
            # prob = topk_unsorted(prob_refine, k=5, dim=2)[0]

            # print((prob-prob_refine).abs().sum())

            # prob = prob_refine

            # prob_refine = prob_refine.reshape(N, 4, -1, H, W).softmax(dim=2)

            # prob_refine = torch.cat([prob_refine, bbox_dist], dim=2)

            # bbox_dist = torch.stack([bbox - bbox.long(), 1.0 - bbox + bbox.long()], dim=2) # N, 4, 2, H, W
            # bbox_dist = torch.stack([1.0 - bbox + bbox.long(), bbox - bbox.long()], dim=2) # N, 4, 2, H, W

            # prob_refine_0 = torch.cat([
            #     prob_refine.new_zeros([N, 4, 1, H, W]),
            #     prob_refine
            # ], dim=2)
            # prob_refine_1 = torch.cat([
            #     prob_refine,
            #     prob_refine.new_zeros([N, 4, 1, H, W])
            # ], dim=2)
            # prob_refine = torch.stack([prob_refine_0, prob_refine_1], dim=3) # N, 4, 6, 2, H, W
            # prob_refine = (prob_refine * bbox_dist[:, :, None, ...]).sum(dim=3)

            # prob_refine = torch.cat([prob_refine, bbox_dist], dim=2)
            # prob_refine = (
            #     dist_pred_refine.reshape(N, 4, -1, H, W)
            #     .softmax(dim=2)
            #     .topk(self.bins, dim=2)[0]
            # )

            prob_refine = dist_pred_refine.reshape(N, 4, -1, H, W).softmax(dim=2)
            conf = self.reg_conf(prob_refine.reshape(N, -1, H, W))

            # cls_score = cls_score * self.reg_conf(prob_refine.reshape(N, -1, H, W))

            # cls_score = cls_score * self.reg_conf(prob.reshape(N, -1, H, W))
            # cls_score = cls_score * self.reg_conf(prob_refine.reshape(N, -1, H, W))

        # select_inds = cls_score.permute(0, 2, 3, 1).max(dim=-1)[0].reshape(-1).gt(0.6).nonzero().reshape(-1)
        # bbox_pred_refine_topk = bbox_pred_refine.permute(0, 2, 3, 1).reshape(-1, 4)[select_inds]

        # hist = torch.histc(bbox_pred_refine.reshape(-1).float(), bins=15, min=-3.0, max=12.0)
        # file = open("./faqe_output.txt","a")
        # file.write(str([*hist.cpu().numpy()]))
        # file.write('\n')
        # file.close()

        return cls_score, conf, pts_pred_initial_reg, dist_pred_refine
        # return cls_score, pts_pred_initial* 0.0, bbox_pred_initial + bbox_pred_refine

    def anchor_center(self, anchors):
        anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
        anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2
        return torch.stack([anchors_cx, anchors_cy], dim=-1)

    def loss_single(
        self,
        anchors,
        cls_score,
        conf,
        pts_pred_initial,
        dist_pred_refine,
        labels,
        label_weights,
        bbox_targets,
        stride,
        num_total_samples,
    ):
        assert stride[0] == stride[1], "h stride is not equal to w stride!"
        anchors = anchors.reshape(-1, 4)
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        conf = conf.permute(0, 2, 3, 1).reshape(-1, 1)
        cls_score = cls_score * conf

        bbox_pred_initial = self.points2distance(pts_pred_initial)
        bbox_pred_initial = bbox_pred_initial.permute(0, 2, 3, 1).reshape(-1, 4)
        bbox_pred_refine = self.integral_refine(
            dist_pred_refine.permute(0, 2, 3, 1).reshape(-1, 4 * self.bins)
        )
        bbox_pred = bbox_pred_initial + bbox_pred_refine

        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = (
            ((labels >= 0) & (labels < bg_class_ind))
            .nonzero(as_tuple=False)
            .squeeze(-1)
        )
        score = label_weights.new_zeros(labels.shape)

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred = bbox_pred[pos_inds]

            pos_anchors = anchors[pos_inds]
            pos_anchor_centers = self.anchor_center(pos_anchors) / stride[0]
            pos_decode_bbox_pred = distance2bbox(pos_anchor_centers, pos_bbox_pred)
            pos_decode_bbox_targets = pos_bbox_targets / stride[0]

            score[pos_inds] = bbox_overlaps(
                pos_decode_bbox_pred.detach(), pos_decode_bbox_targets, is_aligned=True
            ).clamp(min=0.0)

            weight_targets = cls_score.detach().max(dim=1)[0][pos_inds]
            # weight_targets = cls_score.detach().max(dim=1)[0][pos_inds].sigmoid()

            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred,
                pos_decode_bbox_targets,
                weight=weight_targets,
                avg_factor=1.0,
            )

            # print("{} {} {}".format(pos_anchor_centers.size(), pos_decode_bbox_targets.size(), pos_bbox_pred.size()))
            # exit(0)

            # loss_dfl = self.loss_dfl(
            #     dist_pred_refine.permute(0, 2, 3, 1)
            #     .reshape(-1, 4 * self.bins)[pos_inds]
            #     .reshape(-1, self.bins),
            #     (
            #         bbox2distance(pos_anchor_centers, pos_decode_bbox_targets).reshape(
            #             -1
            #         )
            #         - pos_bbox_pred.reshape(-1).detach()
            #     ).clamp(min=0.0, max=3.999),
            #     weight=weight_targets[:, None].expand(-1, 4).reshape(-1),
            #     avg_factor=4.0,
            # )
            loss_dfl = bbox_pred.mul(0.0).sum()

        else:
            # loss_bbox = bbox_pred_initial.sum() * 0.0
            loss_bbox = bbox_pred.mul(0.0).sum()
            loss_dfl = bbox_pred.mul(0.0).sum()
            weight_targets = torch.tensor(0.0).cuda()

        loss_cls = self.loss_cls(
            cls_score,
            (labels, score),
            weight=label_weights,
            avg_factor=num_total_samples,
        )
        # loss_cls = self.loss_cls(
        #     cls_score, labels,
        #     weight=label_weights,
        #     avg_factor=num_total_samples)

        return loss_cls, loss_bbox, loss_dfl, weight_targets.sum()

    # def loss_single(self, anchors, cls_score, pts_pred_initial, bbox_pred_refine, labels, label_weights,
    #                 bbox_targets, stride, num_total_samples):
    #     assert stride[0] == stride[1], 'h stride is not equal to w stride!'
    #     anchors = anchors.reshape(-1, 4)
    #     cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)

    #     # bbox_pred_initial = self.points2distance(pts_pred_initial)

    #     # bbox_pred_initial = bbox_pred_initial.permute(0, 2, 3, 1).reshape(-1, 4)
    #     pts_pred_initial = pts_pred_initial.permute(0, 2, 3, 1).reshape(-1, self.points, 2)
    #     pts_y = pts_pred_initial[:, :, 0]
    #     pts_x = pts_pred_initial[:, :, 1]
    #     bbox_l = -torch.topk(pts_x, k=2, dim=1, largest=False)[0]
    #     bbox_u = -torch.topk(pts_y, k=2, dim=1, largest=False)[0]
    #     bbox_r = torch.topk(pts_x, k=2, dim=1, largest=True)[0]
    #     bbox_b = torch.topk(pts_y, k=2, dim=1, largest=True)[0]
    #     bbox_l = bbox_l[:, None, None, None, :].repeat(1, 2, 2, 2, 1)
    #     bbox_u = bbox_u[:, None, None, :, None].repeat(1, 2, 2, 1, 2)
    #     bbox_r = bbox_r[:, None, :, None, None].repeat(1, 2, 1, 2, 2)
    #     bbox_b = bbox_b[:, :, None, None, None].repeat(1, 1, 2, 2, 2)
    #     bbox_pred_initial = torch.stack([bbox_l, bbox_u, bbox_r, bbox_b], dim=-1).reshape(-1, 16, 4)

    #     bbox_pred_refine = bbox_pred_refine.permute(0, 2, 3, 1).reshape(-1, 4)

    #     bbox_targets = bbox_targets.reshape(-1, 4)
    #     labels = labels.reshape(-1)
    #     label_weights = label_weights.reshape(-1)

    #     # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
    #     bg_class_ind = self.num_classes
    #     pos_inds = ((labels >= 0)
    #                 & (labels < bg_class_ind)).nonzero(as_tuple=False).squeeze(-1)
    #     score = label_weights.new_zeros(labels.shape)

    #     if len(pos_inds) > 0:
    #         pos_bbox_targets = bbox_targets[pos_inds]
    #         pos_bbox_pred_initial = bbox_pred_initial[pos_inds]
    #         pos_bbox_pred_refine = bbox_pred_refine[pos_inds]

    #         pos_anchors = anchors[pos_inds]
    #         pos_anchor_centers = self.anchor_center(pos_anchors) / stride[0]

    #         weight_targets = cls_score.detach()
    #         weight_targets = weight_targets.max(dim=1)[0][pos_inds]

    #         pos_decode_bbox_pred = distance2bbox(pos_anchor_centers[:, None, :].repeat(1, 16, 1).reshape(-1, 2),
    #                                 (pos_bbox_pred_initial + pos_bbox_pred_refine[:, None, :]).reshape(-1, 4))

    #         pos_decode_bbox_targets = pos_bbox_targets / stride[0]

    #         # score[pos_inds] = bbox_overlaps(
    #         #     pos_decode_bbox_pred.detach(),
    #         #     pos_decode_bbox_targets,
    #         #     is_aligned=True).clamp(min=0.0)

    #         loss_bbox, loss_bbox_inds = self.loss_bbox(
    #                 pos_decode_bbox_pred,
    #                 pos_decode_bbox_targets[:, None, :].repeat(1, 16, 1).reshape(-1, 4),
    #                 weight=weight_targets[:, None].repeat(1, 16).reshape(-1),
    #                 reduction_override='none').reshape(-1, 16).max(dim=-1)
    #         loss_bbox = loss_bbox.sum()

    #         iou = bbox_overlaps(
    #             pos_decode_bbox_pred.detach(),
    #             pos_decode_bbox_targets[:, None, :].repeat(1, 16, 1).reshape(-1, 4),
    #             is_aligned=True).clamp(min=0.0).reshape(-1, 16)

    #         score[pos_inds] = torch.gather(iou, dim=1, index=loss_bbox_inds[:, None]).reshape(-1)

    #     else:
    #         loss_bbox = bbox_pred_refine.sum() * 0.0
    #         weight_targets = torch.tensor(0).cuda()

    #     loss_cls = self.loss_cls(
    #         cls_score, (labels, score),
    #         weight=label_weights,
    #         avg_factor=num_total_samples)

    #     return loss_cls, loss_bbox, weight_targets.sum()

    @force_fp32(
        apply_to=("cls_scores", "confs", "pts_preds_initial", "dist_preds_refine")
    )
    def loss(
        self,
        cls_scores,
        confs,
        pts_preds_initial,
        dist_preds_refine,
        gt_bboxes,
        gt_labels,
        img_metas,
        gt_bboxes_ignore=None,
    ):
        """Compute losses of the head.

        Args:
            cls_scores (list[Tensor]): Cls and quality scores for each scale
                level has shape (N, num_classes, H, W).
            bbox_preds (list[Tensor]): Box distribution logits for each scale
                level with shape (N, 4*(n+1), H, W), n is max value of integral
                set.
            gt_bboxes (list[Tensor]): Ground truth bboxes for each image with
                shape (num_gts, 4) in [tl_x, tl_y, br_x, br_y] format.
            gt_labels (list[Tensor]): class indices corresponding to each box
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            gt_bboxes_ignore (list[Tensor] | None): specify which bounding
                boxes can be ignored when computing the loss.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == self.anchor_generator.num_levels

        device = cls_scores[0].device
        anchor_list, valid_flag_list = self.get_anchors(
            featmap_sizes, img_metas, device=device
        )
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1

        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
        )
        if cls_reg_targets is None:
            return None

        (
            anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_pos,
            num_total_neg,
        ) = cls_reg_targets

        num_total_samples = reduce_mean(
            torch.tensor(num_total_pos, dtype=torch.float, device=device)
        ).item()
        num_total_samples = max(num_total_samples, 1.0)

        losses_cls, losses_bbox, losses_dfl, avg_factor = multi_apply(
            self.loss_single,
            anchor_list,
            cls_scores,
            confs,
            pts_preds_initial,
            dist_preds_refine,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            self.anchor_generator.strides,
            num_total_samples=num_total_samples,
        )

        avg_factor = sum(avg_factor)
        avg_factor = reduce_mean(avg_factor).item()
        losses_bbox = list(map(lambda x: x / avg_factor, losses_bbox))
        losses_dfl = list(map(lambda x: x / avg_factor, losses_dfl))

        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox, loss_dfl=losses_dfl)

    @force_fp32(
        apply_to=("cls_scores", "confs", "pts_preds_initial", "dist_preds_refine")
    )
    def get_bboxes(
        self,
        cls_scores,
        confs,
        pts_preds_initial,
        dist_preds_refine,
        img_metas,
        cfg=None,
        rescale=False,
        with_nms=True,
    ):
        assert len(cls_scores) == len(pts_preds_initial) == len(dist_preds_refine)
        num_levels = len(cls_scores)

        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_anchors(featmap_sizes, device=device)

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [cls_scores[i][img_id].detach() for i in range(num_levels)]
            confs_list = [confs[i][img_id].detach() for i in range(num_levels)]
            pts_pred_initial_list = [
                pts_preds_initial[i][img_id].detach() for i in range(num_levels)
            ]
            dist_pred_refine_list = [
                dist_preds_refine[i][img_id].detach() for i in range(num_levels)
            ]

            img_shape = img_metas[img_id]["img_shape"]
            scale_factor = img_metas[img_id]["scale_factor"]
            if with_nms:
                # some heads don't support with_nms argument
                proposals = self._get_bboxes_single(
                    cls_score_list,
                    confs_list,
                    pts_pred_initial_list,
                    dist_pred_refine_list,
                    mlvl_anchors,
                    img_shape,
                    scale_factor,
                    cfg,
                    rescale,
                )
            else:
                proposals = self._get_bboxes_single(
                    cls_score_list,
                    confs_list,
                    pts_pred_initial_list,
                    dist_pred_refine_list,
                    mlvl_anchors,
                    img_shape,
                    scale_factor,
                    cfg,
                    rescale,
                    with_nms,
                )
            result_list.append(proposals)
        return result_list

    def _get_bboxes_single(
        self,
        cls_scores,
        confs,
        pts_preds_initial,
        dist_preds_refine,
        mlvl_anchors,
        img_shape,
        scale_factor,
        cfg,
        rescale=False,
        with_nms=True,
    ):
        """Transform outputs for a single batch item into labeled boxes.

        Args:
            cls_scores (list[Tensor]): Box scores for a single scale level
                has shape (num_classes, H, W).
            bbox_preds (list[Tensor]): Box distribution logits for a single
                scale level with shape (4*(n+1), H, W), n is max value of
                integral set.
            mlvl_anchors (list[Tensor]): Box reference for a single scale level
                with shape (num_total_anchors, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            tuple(Tensor):
                det_bboxes (Tensor): Bbox predictions in shape (N, 5), where
                    the first 4 columns are bounding box positions
                    (tl_x, tl_y, br_x, br_y) and the 5-th column is a score
                    between 0 and 1.
                det_labels (Tensor): A (N,) tensor where each item is the
                    predicted class label of the corresponding box.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert (
            len(cls_scores)
            == len(pts_preds_initial)
            == len(dist_preds_refine)
            == len(mlvl_anchors)
        )
        mlvl_bboxes = []
        mlvl_centers = []
        mlvl_pts_init = []
        mlvl_dist_refine = []
        mlvl_scores = []
        for cls_score, conf, pts_pred_initial, dist_pred_refine, stride, anchors in zip(
            cls_scores,
            confs,
            pts_preds_initial,
            dist_preds_refine,
            self.anchor_generator.strides,
            mlvl_anchors,
        ):
            assert (
                cls_score.size()[-2:]
                == pts_pred_initial.size()[-2:]
                == dist_pred_refine.size()[-2:]
            )
            assert stride[0] == stride[1]

            scores = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels)
            confs = conf.permute(1, 2, 0).reshape(-1, 1)
            scores = scores * confs
            # scores = torch.cat([confs, confs.repeat(1, 80) * 0.0], dim=1)

            # scores = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels).sigmoid()

            points = self.anchor_center(anchors)

            pts_pred_initial = pts_pred_initial.permute(1, 2, 0).reshape(
                -1, self.num_points * 2
            )

            dist_pred_refine = dist_pred_refine.permute(1, 2, 0).reshape(
                -1, 4 * self.bins
            )

            bbox_pred_initial = self.points2distance(pts_pred_initial)
            bbox_pred_refine = self.integral_refine(dist_pred_refine)
            bbox_pred = bbox_pred_initial + bbox_pred_refine

            bbox_pred = (
                bbox_pred
                * torch.tensor([-1.0, -1.0, 1.0, 1.0]).type_as(bbox_pred)[None, :]
            )

            nms_pre = cfg.get("nms_pre", -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = scores.max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                # anchors = anchors[topk_inds, :]
                points = points[topk_inds, :]
                pts_pred_initial = pts_pred_initial[topk_inds, :]
                dist_pred_refine = dist_pred_refine[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]

            bbox_pos_center = torch.cat([points[:, :2], points[:, :2]], dim=1)
            bboxes = bbox_pred * stride[0] + bbox_pos_center
            x1 = bboxes[:, 0].clamp(min=0, max=img_shape[1])
            y1 = bboxes[:, 1].clamp(min=0, max=img_shape[0])
            x2 = bboxes[:, 2].clamp(min=0, max=img_shape[1])
            y2 = bboxes[:, 3].clamp(min=0, max=img_shape[0])
            bboxes = torch.stack([x1, y1, x2, y2], dim=-1)

            pts_pred_initial = pts_pred_initial.reshape(-1, self.num_points, 2)
            pts_init_y = pts_pred_initial[:, :, 0] * stride[0] + points[:, 1:2]
            pts_init_y = pts_init_y.clamp(min=0, max=img_shape[0])
            pts_init_x = pts_pred_initial[:, :, 1] * stride[0] + points[:, 0:1]
            pts_init_x = pts_init_x.clamp(min=0, max=img_shape[1])
            pts_init = torch.stack([pts_init_x, pts_init_y], dim=-1).reshape(
                -1, self.num_points * 2
            )

            mlvl_bboxes.append(bboxes)
            mlvl_centers.append(points[:, :2])
            mlvl_pts_init.append(pts_init)
            dist_pred_refine = (
                dist_pred_refine.reshape(-1, 4, self.bins)
                .softmax(dim=-1)
                .reshape(-1, 4 * self.bins)
            )
            mlvl_dist_refine.append(dist_pred_refine)
            mlvl_scores.append(scores)

        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_centers = torch.cat(mlvl_centers)
        mlvl_pts_init = torch.cat(mlvl_pts_init)
        mlvl_dist_refine = torch.cat(mlvl_dist_refine)

        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
            mlvl_centers /= mlvl_centers.new_tensor(scale_factor[0:2])
            mlvl_pts_init /= mlvl_pts_init.new_tensor(
                scale_factor[0:2].repeat(self.num_points)
            )

        # Add a dummy background class to the backend when using sigmoid
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)

        # show_results = True
        show_results = False

        if show_results:
            if with_nms:
                (
                    det_bboxes,
                    det_labels,
                    det_centers,
                    det_pts_init,
                    det_dist_refine,
                ) = multiclass_nms_expanded(
                    mlvl_bboxes,
                    mlvl_scores,
                    mlvl_centers,
                    mlvl_pts_init,
                    mlvl_dist_refine,
                    cfg.score_thr,
                    cfg.nms,
                    cfg.max_per_img,
                )
                return (
                    det_bboxes,
                    det_centers,
                    det_pts_init,
                    det_dist_refine,
                    det_labels,
                )
            else:
                return (
                    mlvl_bboxes,
                    mlvl_centers,
                    mlvl_pts_init,
                    mlvl_dist_refine,
                    mlvl_scores,
                )
        else:
            if with_nms:
                det_bboxes, det_labels = multiclass_nms(
                    mlvl_bboxes,
                    mlvl_scores,
                    cfg.score_thr,
                    cfg.nms,
                    cfg.max_per_img,
                    return_inds=False,
                )
                return det_bboxes, det_labels
            else:
                return mlvl_bboxes, mlvl_scores

    # @force_fp32(apply_to=("cls_scores", "pts_preds_initial", "dist_preds_refine"))
    @force_fp32(
        apply_to=("cls_scores", "confs", "pts_preds_initial", "dist_preds_refine")
    )
    def get_results(
        self,
        cls_scores,
        confs,
        pts_preds_initial,
        dist_preds_refine,
        gt_bboxes,
        gt_labels,
        img_metas,
    ):

        assert len(cls_scores) == len(pts_preds_initial) == len(dist_preds_refine)
        num_levels = len(cls_scores)

        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_anchors(featmap_sizes, device=device)

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id]
                .detach()
                .permute(1, 2, 0)
                .reshape(-1, self.cls_out_channels)
                for i in range(num_levels)
            ]
            conf_list = [
                confs[i][img_id].detach().permute(1, 2, 0).reshape(-1, 1)
                for i in range(num_levels)
            ]
            pts_pred_initial_list = [
                pts_preds_initial[i][img_id]
                .detach()
                .permute(1, 2, 0)
                .reshape(-1, self.num_points * 2)
                for i in range(num_levels)
            ]
            dist_pred_refine_list = [
                dist_preds_refine[i][img_id]
                .detach()
                .permute(1, 2, 0)
                .reshape(-1, 4 * self.bins)
                for i in range(num_levels)
            ]

            cls_score, conf, iou = self._get_result_single(
                cls_score_list,
                conf_list,
                pts_pred_initial_list,
                dist_pred_refine_list,
                mlvl_anchors,
                self.anchor_generator.strides,
                featmap_sizes,
                gt_bboxes[img_id],
                gt_labels[img_id],
                img_metas[img_id]["img_shape"],
            )
            result_list.append([cls_score, conf, iou])

        return result_list

    def _get_result_single(
        self,
        cls_scores,
        confs,
        pts_preds_initial,
        dist_preds_refine,
        anchors,
        strides,
        featmap_sizes,
        gt_bboxes,
        gt_labels,
        img_shape,
    ):

        bbox_list = []
        for cls_score, conf, pts_pred_initial, dist_pred_refine, anchor, stride in zip(
            cls_scores, confs, pts_preds_initial, dist_preds_refine, anchors, strides
        ):
            point = self.anchor_center(anchor)
            bbox_pred = self.points2distance(pts_pred_initial) + self.integral_refine(
                dist_pred_refine
            )
            bbox_pred = (
                bbox_pred
                * torch.tensor([-1.0, -1.0, 1.0, 1.0]).type_as(bbox_pred)[None, :]
            )

            bbox_center = torch.cat([point[:, :2], point[:, :2]], dim=1)
            bbox = bbox_pred * stride[0] + bbox_center
            x1 = bbox[:, 0].clamp(min=0, max=img_shape[1])
            y1 = bbox[:, 1].clamp(min=0, max=img_shape[0])
            x2 = bbox[:, 2].clamp(min=0, max=img_shape[1])
            y2 = bbox[:, 3].clamp(min=0, max=img_shape[0])
            bbox = torch.stack([x1, y1, x2, y2], dim=-1)
            bbox_list.append(bbox)

        cls_score_list = []
        conf_list = []
        iou_list = []
        for gt_bbox, gt_label in zip(gt_bboxes, gt_labels):
            cls_score_lvl_list = []
            conf_lvl_list = []
            iou_lvl_list = []
            for cls_score, conf, bbox, dist_pred_refine, feat_size in zip(
                cls_scores, confs, bbox_list, dist_preds_refine, featmap_sizes
            ):
                cls_score = cls_score[:, gt_label] * conf.reshape(-1)
                # cls_score = (
                #     dist_pred_refine.reshape(-1, 4, self.bins)
                #     .softmax(dim=2)
                #     .topk(1, dim=-1)[0].sum(dim=-1)
                #     .mean(dim=-1)
                # )

                iou = bbox_overlaps(gt_bbox[None, :], bbox, is_aligned=False)

                # select = conf.max(dim=0)[1]
                # print('{}, {}, {}'.format(
                #     conf[select],
                #     iou[0, select],
                #     refine[select]))
                cls_score_lvl_list.append(cls_score.reshape(feat_size))
                conf_lvl_list.append(conf.reshape(feat_size))
                iou_lvl_list.append(iou.reshape(feat_size))
            cls_score_list.append(cls_score_lvl_list)
            conf_list.append(conf_lvl_list)
            iou_list.append(iou_lvl_list)

        # conf_list: list[gt]-list[lvl]-tensor[H, W, 1]
        # iou_list: list[gt]-list[lvl]-tensor[H, W, 1]
        return cls_score_list, conf_list, iou_list

    def get_targets(
        self,
        anchor_list,
        valid_flag_list,
        gt_bboxes_list,
        img_metas,
        gt_bboxes_ignore_list=None,
        gt_labels_list=None,
        label_channels=1,
        unmap_outputs=True,
    ):
        """Get targets for GFL head.

        This method is almost the same as `AnchorHead.get_targets()`. Besides
        returning the targets as the parent method does, it also returns the
        anchors as the first element of the returned tuple.
        """
        num_imgs = len(img_metas)
        assert len(anchor_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        num_level_anchors_list = [num_level_anchors] * num_imgs

        # concat all level anchors and flags to a single tensor
        for i in range(num_imgs):
            assert len(anchor_list[i]) == len(valid_flag_list[i])
            anchor_list[i] = torch.cat(anchor_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        (
            all_anchors,
            all_labels,
            all_label_weights,
            all_bbox_targets,
            all_bbox_weights,
            pos_inds_list,
            neg_inds_list,
        ) = multi_apply(
            self._get_target_single,
            anchor_list,
            valid_flag_list,
            num_level_anchors_list,
            gt_bboxes_list,
            gt_bboxes_ignore_list,
            gt_labels_list,
            img_metas,
            label_channels=label_channels,
            unmap_outputs=unmap_outputs,
        )
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        anchors_list = images_to_levels(all_anchors, num_level_anchors)
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights, num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets, num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights, num_level_anchors)
        return (
            anchors_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_pos,
            num_total_neg,
        )

    def _get_target_single(
        self,
        flat_anchors,
        valid_flags,
        num_level_anchors,
        gt_bboxes,
        gt_bboxes_ignore,
        gt_labels,
        img_meta,
        label_channels=1,
        unmap_outputs=True,
    ):
        """Compute regression, classification targets for anchors in a single
        image.

        Args:
            flat_anchors (Tensor): Multi-level anchors of the image, which are
                concatenated into a single tensor of shape (num_anchors, 4)
            valid_flags (Tensor): Multi level valid flags of the image,
                which are concatenated into a single tensor of
                    shape (num_anchors,).
            num_level_anchors Tensor): Number of anchors of each scale level.
            gt_bboxes (Tensor): Ground truth bboxes of the image,
                shape (num_gts, 4).
            gt_bboxes_ignore (Tensor): Ground truth bboxes to be
                ignored, shape (num_ignored_gts, 4).
            gt_labels (Tensor): Ground truth labels of each box,
                shape (num_gts,).
            img_meta (dict): Meta info of the image.
            label_channels (int): Channel of label.
            unmap_outputs (bool): Whether to map outputs back to the original
                set of anchors.

        Returns:
            tuple: N is the number of total anchors in the image.
                anchors (Tensor): All anchors in the image with shape (N, 4).
                labels (Tensor): Labels of all anchors in the image with shape
                    (N,).
                label_weights (Tensor): Label weights of all anchor in the
                    image with shape (N,).
                bbox_targets (Tensor): BBox targets of all anchors in the
                    image with shape (N, 4).
                bbox_weights (Tensor): BBox weights of all anchors in the
                    image with shape (N, 4).
                pos_inds (Tensor): Indices of postive anchor with shape
                    (num_pos,).
                neg_inds (Tensor): Indices of negative anchor with shape
                    (num_neg,).
        """
        inside_flags = anchor_inside_flags(
            flat_anchors,
            valid_flags,
            img_meta["img_shape"][:2],
            self.train_cfg.allowed_border,
        )
        if not inside_flags.any():
            return (None,) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        num_level_anchors_inside = self.get_num_level_anchors_inside(
            num_level_anchors, inside_flags
        )
        assign_result = self.assigner.assign(
            anchors, num_level_anchors_inside, gt_bboxes, gt_bboxes_ignore, gt_labels
        )

        sampling_result = self.sampler.sample(assign_result, anchors, gt_bboxes)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full(
            (num_valid_anchors,), self.num_classes, dtype=torch.long
        )
        label_weights = anchors.new_zeros(num_valid_anchors, dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        if len(pos_inds) > 0:
            pos_bbox_targets = sampling_result.pos_gt_bboxes
            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0

            if gt_labels is None:
                # Only rpn gives gt_labels as None
                # Foreground is the first class
                labels[pos_inds] = 0
            else:
                labels[pos_inds] = gt_labels[sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_anchors = flat_anchors.size(0)
            anchors = unmap(anchors, num_total_anchors, inside_flags)
            labels = unmap(
                labels, num_total_anchors, inside_flags, fill=self.num_classes
            )
            label_weights = unmap(label_weights, num_total_anchors, inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        return (
            anchors,
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            pos_inds,
            neg_inds,
        )

    def get_num_level_anchors_inside(self, num_level_anchors, inside_flags):
        split_inside_flags = torch.split(inside_flags, num_level_anchors)
        num_level_anchors_inside = [int(flags.sum()) for flags in split_inside_flags]
        return num_level_anchors_inside


def topk_unsorted(x, k=1, dim=0, largest=True):
    val, idx = torch.topk(x, k, dim=dim, largest=largest)
    sorted_idx, new_idx = torch.sort(idx, dim=dim)
    val = torch.gather(val, dim=dim, index=new_idx)

    return val, sorted_idx


def multiclass_nms_expanded(
    multi_bboxes,
    multi_scores,
    mlvl_points,
    mlvl_pts_init,
    mlvl_dist_refine,
    score_thr,
    nms_cfg,
    max_num=-1,
    score_factors=None,
    return_inds=False,
):
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(multi_scores.size(0), num_classes, 4)

    points = mlvl_points[:, None].expand(
        mlvl_points.size(0), num_classes, mlvl_points.size(1)
    )
    pts_init = mlvl_pts_init[:, None].expand(
        mlvl_pts_init.size(0), num_classes, mlvl_pts_init.size(1)
    )
    dist_refine = mlvl_dist_refine[:, None].expand(
        mlvl_dist_refine.size(0), num_classes, mlvl_dist_refine.size(1)
    )

    scores = multi_scores[:, :-1]
    if score_factors is not None:
        scores = scores * score_factors[:, None]

    labels = torch.arange(num_classes, dtype=torch.long)
    labels = labels.view(1, -1).expand_as(scores)

    bboxes = bboxes.reshape(-1, 4)
    scores = scores.reshape(-1)
    labels = labels.reshape(-1)
    points = points.reshape(-1, points.size(-1))
    pts_init = pts_init.reshape(-1, pts_init.size(-1))
    dist_refine = dist_refine.reshape(-1, dist_refine.size(-1))

    # remove low scoring boxes
    valid_mask = scores > score_thr
    inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
    bboxes, scores, labels = bboxes[inds], scores[inds], labels[inds]
    points, pts_init = points[inds], pts_init[inds]
    dist_refine = dist_refine[inds]

    if inds.numel() == 0:
        if torch.onnx.is_in_onnx_export():
            raise RuntimeError(
                "[ONNX Error] Can not record NMS "
                "as it has not been executed this time"
            )
        if return_inds:
            return bboxes, labels, points, pts_init, dist_refine, inds
        else:
            return bboxes, labels, points, pts_init, dist_refine

    # TODO: add size check before feed into batched_nms
    dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)

    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]

    if return_inds:
        return dets, labels[keep], points[keep], pts_init[keep], dist_refine[keep], keep
    else:
        return dets, labels[keep], points[keep], pts_init[keep], dist_refine[keep]
