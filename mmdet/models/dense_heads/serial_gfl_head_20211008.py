from collections import OrderedDict
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
from mmcv.cnn import ConvModule, Scale, bias_init_with_prob, normal_init
from mmcv.runner import force_fp32
from mmcv.ops import DeformConv2d, ModulatedDeformConv2d, DeformConv2dPack, DeformUnfold
from torch.autograd import Function


from mmdet.core import (anchor_inside_flags, bbox, bbox_overlaps,
                        build_assigner, build_sampler, distance2bbox,
                        images_to_levels, multi_apply, multiclass_nms,
                        reduce_mean, unmap)
from ..builder import HEADS, build_loss
from .anchor_head import AnchorHead

import numpy as np
import math

from mmcv.cnn import build_norm_layer


class Integral(nn.Module):
    def __init__(self, start=-2.0, stop=2.0, bins=5):
        super(Integral, self).__init__()
        self.bins = bins
        self.register_buffer('project',
                              torch.linspace(start, stop, self.bins))

    def forward(self, x):
        x = F.softmax(x.reshape(-1, self.bins), dim=1)
        x = F.linear(x, self.project.type_as(x)).reshape(-1, 4)
        return x


# class AbsScale(nn.Module):
#     def __init__(self, scale=0.0):
#         super(AbsScale, self).__init__()
#         self.scale = nn.Parameter(torch.tensor(scale, dtype=torch.float))
#     def forward(self, x):
#         return x * self.scale.exp() * self.scale.exp()

class Integral_v2(nn.Module):
    def __init__(self, start=-2.0, stop=2.0, bins=5):
        super(Integral_v2, self).__init__()
        self.bins = bins

        self.register_buffer('project',
                              torch.linspace(start, stop, self.bins))

    def forward(self, x):
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(-1, self.bins) # BHW*4, bins
        x = F.softmax(x, dim=1)
        x = F.linear(x, self.project.type_as(x))
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return x


class MaskDeformConv2dModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, norm_cfg=None):
        super(MaskDeformConv2dModule, self).__init__()
        self.conv = ModulatedDeformConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.norm_name, norm = build_norm_layer(norm_cfg, out_channels)
        self.add_module(self.norm_name, norm)
        self.act = nn.ReLU(inplace=True)
        constant_init(self.norm, 1, bias=0)
    
    @property
    def norm(self):
        return getattr(self, self.norm_name)

    def forward(self, x, offset, mask):
        x = self.conv(x, offset, mask)
        x = self.norm(x)
        x = self.act(x)
        return x

class DeformConv2dModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, deform_groups=1, groups=1, norm_cfg=None):
        super(DeformConv2dModule, self).__init__()
        self.conv = DeformConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, deform_groups=deform_groups, groups=groups)
        self.norm_name, norm = build_norm_layer(norm_cfg, out_channels)
        self.add_module(self.norm_name, norm)
        self.act = nn.ReLU(inplace=True)
        constant_init(self.norm, 1, bias=0)

    @property
    def norm(self):
        return getattr(self, self.norm_name)

    def forward(self, x, offset):
        x = self.conv(x, offset)
        x = self.norm(x)
        x = self.act(x)
        return x


# class DeformAvgPool(nn.Module):
#     def __init__(self, channels, points):
#         super(DeformAvgPool, self).__init__()
#         # self.conv = DeformConv2d(channels, channels, [1, points], stride=[1,1], padding=[0, points//2], groups=channels)
#         self.conv = DeformConv2d(channels, channels, [1, points], stride=[1,1], padding=[0, points//2])
#         # constant_init(self.conv, 1.0/points)
#         # self.conv.weight.detach_()

#     def forward(self, x, offset):
#         out = self.conv(x, offset)
#         return out



# class MergeBolck(nn.Module):
#     def __init__(self):
#         super(MergeBolck, self).__init__()
#         self.hpool = nn.MaxPool2d([7, 1], stride=1, padding=[1, 0])
#         self.wpool = nn.MaxPool2d([1, 7], stride=1, padding=[0, 1])
#     def forward(self, x):
#         x_hpool = self.hpool(x)
#         x_wpool = self.wpool(x)

#         out = (x_hpool + x_wpool) / 2.0
#         return out


# class LocalAttention(nn.Module):
#     def __init__(self, channels):
#         super(LocalAttention, self).__init__()
#         self.conv_vec = nn.Conv2d(channels, 64, 1, stride=1, padding=0)
#         self.dconv_stdvec = DeformConv2d(channels, 64, [3,3], stride=[1,1], padding=[1,1])
#         self.dconv_iden = DeformConv2d(64, 64, [1,1], stride=[1,1], padding=[0,0], groups=64)

#         nn.init.constant_(self.dconv_iden.weight, 1)
#         self.dconv_stdvec.weight.detach_()

#         normal_init(self.conv_vec, std=0.01)

#     def forward(self, x, offset):
#         x_vec = self.conv_vec(x)
#         x_vec = x_vec - x_vec.mean(dim=1, keepdim=True)
#         x_vec = x_vec / x_vec.pow(2).sum(dim=1, keepdim=True).add(1E-5).sqrt()

#         std_vec = self.dconv_stdvec(x, offset)
#         std_vec = std_vec - std_vec.mean(dim=1, keepdim=True)
#         std_vec = std_vec / std_vec.pow(2).sum(dim=1, keepdim=True).add(1E-5).sqrt()


#         pt0 = (self.dconv_iden(x_vec, offset[:, 0:2, ...].contiguous()) * std_vec).sum(dim=1, keepdim=True)
#         pt1 = (self.dconv_iden(x_vec, offset[:, 2:4, ...].contiguous()) * std_vec).sum(dim=1, keepdim=True)
#         pt2 = (self.dconv_iden(x_vec, offset[:, 4:6, ...].contiguous()) * std_vec).sum(dim=1, keepdim=True)
#         pt3 = (self.dconv_iden(x_vec, offset[:, 6:8, ...].contiguous()) * std_vec).sum(dim=1, keepdim=True)
#         pt4 = (self.dconv_iden(x_vec, offset[:, 8:10, ...].contiguous()) * std_vec).sum(dim=1, keepdim=True)
#         pt5 = (self.dconv_iden(x_vec, offset[:, 10:12, ...].contiguous()) * std_vec).sum(dim=1, keepdim=True)
#         pt6 = (self.dconv_iden(x_vec, offset[:, 12:14, ...].contiguous()) * std_vec).sum(dim=1, keepdim=True)
#         pt7 = (self.dconv_iden(x_vec, offset[:, 14:16, ...].contiguous()) * std_vec).sum(dim=1, keepdim=True)
#         pt8 = (self.dconv_iden(x_vec, offset[:, 16:, ...].contiguous()) * std_vec).sum(dim=1, keepdim=True)

#         output = torch.cat([pt0, pt1, pt2, pt3, pt4, pt5, pt6, pt7, pt8], dim=1)

#         return output



# class PointsAttention(nn.Module):
#     def __init__(self, channels):
#         super(PointsAttention, self).__init__()
#         # self.conv_obj = DeformConv2d(channels, 64, [3,3], stride=[1,1], padding=[1,1])
#         # self.conv_pts_0 = nn.Conv2d(channels, 256, 1, stride=1, padding=0)
#         # self.conv_pts_1 = DeformConv2d(256, 9 * 64, 1, stride=1, padding=0)

#         # normal_init(self.conv_pts_0, std=0.01)

#     def forward(self, x, offset):
#         N, C, H, W = x.size()
#         x_obj = self.conv_obj(x, offset).reshape(N, 1, 64, H, W)
#         x_pts = self.conv_pts_1(self.conv_pts_0(x), offset).reshape(N, 9, 64, H, W)
#         out = (x_obj * x_pts).sum(dim=2)

#         return out


# class PointsAttention(nn.Module):
#     def __init__(self, channels):
#         super(PointsAttention, self).__init__()
#         self.conv0 = nn.Conv2d(channels, 16, 1, stride=1, padding=0)
#         self.conv1 = DeformConv2d(16, 16, [1,9], stride=[1,1], padding=[0,4])
#         self.conv2 = nn.Conv2d(16, 10 * 16, 1, stride=1, padding=0)
        
#         normal_init(self.conv0, std=0.01)
#         normal_init(self.conv2, std=0.01)

#     def forward(self, x, offset):
#         x = self.conv2(self.conv1(self.conv0(x), offset))
#         B, _, H, W = x.size()
#         x = x.reshape(B, 10, 16, H, W)
#         x = x - x.mean(dim=2, keepdim=True)
#         x = x / (x.pow(2).sum(dim=2, keepdim=True).sqrt() + 1E-5)
#         x = x.chunk(10, dim=1)

#         out = (torch.cat(x[0:-1], dim=1) * x[-1]).sum(dim=2)
#         out = (out + 1.0) / 2.0
#         return out



'''
class PointsAttention(nn.Module):
    def __init__(self, channels):
        super(PointsAttention, self).__init__()
        self.conv1 = DeformConv2dModule(channels, channels, [1,9], stride=[1,1], padding=[0,4])
        self.conv2 = nn.Conv2d(channels, 9, 3, stride=1, padding=1)
        # self.conv2 = nn.Conv2d(channels, 10 * 64, 1, stride=1, padding=0)

        # self.scale = Scale()
        normal_init(self.conv2, std=0.01)

    # def forward(self, x, offset):
    #     x = self.conv2(self.conv1(x, offset))
    #     B, _, H, W = x.size()
    #     x = x.reshape(B, 10, 64, H, W)
    #     x = x - x.mean(dim=2, keepdim=True)
    #     x = x / (x.pow(2).sum(dim=2, keepdim=True).sqrt() + 1E-5)
    #     x = x.chunk(10, dim=1)

    #     out = (torch.cat(x[0:-1], dim=1) * x[-1]).sum(dim=2)
    #     out = (out + 1.0) / 2.0

    #     return out

    def forward(self, x, offset):
        out = self.conv2(self.conv1(x, offset)).sigmoid()
        return out
'''

'''
class PointsAttention(nn.Module):
    def __init__(self, channels):
        super(PointsAttention, self).__init__()
        self.conv0 = nn.Conv2d(channels, 64, 1, stride=1, padding=0)
        # self.conv1 = DeformConv2dModule(64, 64, [3,3], stride=[1,1], padding=[1,1])


        self.conv1 = DeformConv2d(64, 64, [1,1], stride=[1,1], padding=[0,0], groups=64)
        nn.init.constant_(self.conv1.weight, 1)
        self.conv1.weight.detach_()

        self.conv2 = nn.Conv2d(64, 64, 1, stride=1, padding=0)
        self.conv3 = nn.Conv2d(64, 64, 1, stride=1, padding=0)
        self.conv4 = nn.Conv2d(64, 64, 1, stride=1, padding=0)
        self.conv_out = nn.Conv2d(64, 1, 1, stride=1, padding=0)


        normal_init(self.conv0, std=0.01)
        normal_init(self.conv2, std=0.01)
        normal_init(self.conv3, std=0.01)
        normal_init(self.conv4, std=0.01)
        normal_init(self.conv_out, std=0.01)

    def forward(self, x, offset):
        feat = self.conv0(x)

        gather_feats_a = []
        gather_feats_b = []
        gather_outs = []
        for offset_i in offset.chunk(9, dim=1):
            temp = self.conv1(feat, offset_i.contiguous())
            gather_feats_a.append(self.conv2(temp))
            gather_feats_b.append(self.conv3(temp))
            gather_outs.append(self.conv4(temp))

        x_a = torch.stack(gather_feats_a, dim=1)
        x_b = torch.stack(gather_feats_b, dim=1)
        y = torch.stack(gather_outs, dim=1)
        B, _, _, H, W = x_a.size()
        x_a = x_a.permute(0, 3, 4, 1, 2)
        x_b = x_b.permute(0, 3, 4, 2, 1)
        x_relation = torch.matmul(x_a, x_b).softmax(-1)
        x_out = torch.matmul(
                    x_relation, 
                    y.reshape(B, 9, 64, H, W).permute(0, 3, 4, 1, 2))
        x_out = x_out.permute(0, 3, 4, 1, 2).chunk(9, dim=1)

        out = []
        for x_out_i in x_out:
            out.append(
                self.conv_out(x_out_i.squeeze(dim=1)))
        
        out = torch.cat(out, dim=1).sigmoid()

        return out
'''

# class DeformUnfold(nn.Module):
#     def __init__(self, channels, points):
#         super(DeformUnfold, self).__init__()
#         self.channels = channels
#         self.points = points

#         self.identity = DeformConv2d(
#                             channels, channels, [1,1], 
#                             stride=[1,1], padding=[0,0], groups=channels)
#         constant_init(self.identity, 1,0)
#         self.identity.weight.detach_()
    
#     def forward(self, x, offsets):
#         offsets = offsets.chunk(self.points, dim=1)
#         x_unfolded = []
#         for offset in offsets:
#             offset = offset.contiguous()
#             x_unfolded.append(self.identity(x, offset))
#         x_unfolded = torch.stack(x_unfolded, dim=1) # B, P, C, H, W

#         return x_unfolded


class KernelAttention(nn.Module):
    def __init__(self, channels, points=9, inter_channels=128):
        super(KernelAttention, self).__init__()
        self.points = points
        self.channels = channels
        self.inter_channels = inter_channels
        self.conv_query = nn.Conv2d(channels, inter_channels, 1, stride=1, padding=0)
        self.conv_key = nn.Conv2d(channels, inter_channels, 1, stride=1, padding=0)
        self.conv_value = nn.Conv2d(channels, inter_channels, 1, stride=1, padding=0)
        self.unfold = DeformUnfold(inter_channels, 3, 1, 1) # CKK, BHW

        self.conv_out = nn.Conv2d(inter_channels, 1, 1)

        normal_init(self.conv_query, std=0.01)
        normal_init(self.conv_key, std=0.01)
        normal_init(self.conv_out, std=0.01)
        normal_init(self.conv_value, std=0.01)

    def forward(self, x, offset):
        B, _, H, W = x.size()

        query = self.unfold(self.conv_query(x), offset).permute(0,1).reshape(B, H, W, self.inter_channels, self.points).permute(0,1,2,4,3).contiguous() # B, H, W, P, C
        key = self.unfold(self.conv_key(x), offset).permute(0,1).reshape(B, H, W, self.inter_channels, self.points).contiguous() # B, H, W, C, P
        value = self.unfold(self.conv_value(x), offset).permute(0,1).reshape(B, H, W, self.inter_channels, self.points).permute(0,1,2,4,3).contiguous() # B, H, W, P, C

        matrix = torch.matmul(query, key).softmax(-1) # B, H, W, P, P

        value = torch.matmul(matrix, value) # B, H, W, P, C
        value = value.permute(0, 4, 1, 2, 3).reshape(B, self.inter_channels, H, W * self.points)

        out = self.conv_out(value).reshape(B, H, W, self.points).permute(0,3,1,2).contiguous()
        out = out.sigmoid()
        return out

'''
class SeparableInvolution(nn.Module):
    def __init__(self, channels, points=9):
        super(SeparableInvolution, self).__init__()
        self.channels = channels
        self.inter_channels = channels // 2
        self.points = points

        self.conv_se = nn.Conv2d(channels, self.inter_channels, 1)

        self.conv_mask = nn.Conv2d(self.inter_channels, self.inter_channels*self.points, 1)

        self.unfold = DeformUnfold(self.inter_channels, 3, 1, 1) # CKK, BHW
        self.conv_pw = ConvModule(16, channels, 1)

        normal_init(self.conv_se, std=0.01)
        normal_init(self.conv_mask, std=0.01)
        normal_init(self.conv_pw.conv, std=0.01)

    def forward(self, x, offset):
        B, C, H, W = x.size()
        assert C == self.channels

        x = self.conv_se(x)
        mask = self.conv_mask(x).permute(0, 2, 3, 1).reshape(B, H, W, 16, self.points)
        x_unfold = self.unfold(x, offset).permute(0, 1).reshape(B, H, W, 16, self.points)
        x = (x_unfold * mask).sum(-1).permute(0, 3, 1, 2).contiguous()
        x = self.conv_pw(x)
        return x
'''




'''
class SeparableDeform(nn.Module):
    def __init__(self, channels, points):
        super(SeparableDeform, self).__init__()
        self.unfold = DeformUnfold(channels, [3,3], stride=[1,1], padding=[1,1]) # CKK, BHW
        self.points = points

        self.conv_query = nn.Conv2d(channels, channels//4, 1)
        self.conv_key = nn.Conv2d(channels, channels//4, 1)
        # self.conv_value = nn.Conv2d(channels, channels//4, 1)

        self.conv_out = ModulatedDeformConv2d(channels, channels, [3, 3], stride=[1,1], padding=[1,1])
        self.norm = nn.BatchNorm2d(channels)
        self.act = nn.ReLU()

        normal_init(self.conv_query, std=0.01)
        normal_init(self.conv_key, std=0.01)
        # normal_init(self.conv_value, std=0.01)

    def forward(self, x, offset):
        B, C, H, W = x.shape
        x_query = self.conv_query(x)
        x_key = self.conv_key(x)
        x_select = self.unfold(x_query, offset).permute(1, 0).reshape(B, H, W, C//4, self.points)
        x_matrix = x_key.permute(0, 2, 3, 1).reshape(B, H, W, C//4).unsqueeze(-2)

        x_matrix = torch.matmul(x_matrix, x_select).sigmoid().squeeze(dim=-2).permute(0, 3, 1, 2).contiguous()
        x_value = self.conv_out(x, offset, x_matrix)
        out = self.act(self.norm(x_value))
        return out
'''



class DeformSpaceAttention(nn.Module):
    def __init__(self, channels):
        super(DeformSpaceAttention, self).__init__()
        self.unfold = DeformUnfold(channels, 3, 1, 1) # CKK, BHW
        self.conv0 = nn.Conv2d(channels, 1, 1)
        normal_init(self.conv0, std=0.01)

    def forward(self, x, offset):
        B, C, H, W = x.shape
        x = self.unfold(x, offset).permute(0, 1).reshape(B, H, W, C, -1).max(dim=-1).values
        x = x.permute(0, 3, 1, 2).contiguous()
        out = self.conv0(x).sigmoid()
        return out


class DeformSpaceAttentionv2(nn.Module):
    def __init__(self, channels, groups=4):
        super(DeformSpaceAttentionv2, self).__init__()
        self.groups = groups
        self.channels = channels
        self.unfold = DeformUnfold(channels, 3, 1, 1) # CKK, BHW
        self.conv0 = nn.Conv2d(channels, channels //4, 1)
        self.norm = nn.BatchNorm2d(channels//4)
        self.conv1 = nn.Conv2d(channels//4, groups, 1)

        normal_init(self.conv0, std=0.01)
        normal_init(self.conv1, std=0.01)

    def forward(self, x, offset):
        B, C, H, W = x.shape
        x = self.unfold(x, offset).permute(0, 1).reshape(B, H, W, C, -1).max(dim=-1).values

        x = x.permute(0, 3, 1, 2).contiguous()
        out = self.conv1(self.norm(self.conv0(x)))
        out = out.sigmoid().repeat(1, self.channels//self.groups, 1, 1)
        return out

class DeformSpaceAttentionv3(nn.Module):
    def __init__(self, channels):
        super(DeformSpaceAttentionv3, self).__init__()
        self.channels = channels
        self.conv = DeformConv2d(channels, 1, [3,3], stride=[1,1], padding=[1,1])

    def forward(self, x, offset):
        x = self.conv(x, offset).sigmoid()
        return x


'''
class DeformSpaceAttentionv4(nn.Module):
    def __init__(self, channels):
        super(DeformSpaceAttentionv4, self).__init__()
        self.unfold = DeformUnfold(channels, 3, 1, 1) # CKK, BHW
        self.conv = nn.Conv2d(channels, 1, 1)
        normal_init(self.conv, std=0.01)

    def forward(self, x, offset):
        B, C, H, W = x.shape
        x = self.unfold(x, offset).permute(0, 1).reshape(B, H, W, C, -1).max(dim=-1).values
        x = x.permute(0, 3, 1, 2).contiguous()
        out = self.conv(x).sigmoid()
        return out
'''


class DeformSpaceAttentionv4(nn.Module):
    def __init__(self, channels, topk=4):
        super(DeformSpaceAttentionv4, self).__init__()
        self.topk = topk
        self.conv = DeformConv2d(topk, 1, [3,3], stride=[1,1], padding=[1,1])

    def forward(self, x, offset):
        x_topk = torch.topk(x, self.topk, dim=1).values
        out = self.conv(x_topk, offset).sigmoid()
        return out


class DeformSpaceAttentionv5(nn.Module):
    def __init__(self, channels):
        super(DeformSpaceAttentionv5, self).__init__()
        self.unfold = DeformUnfold(channels, 3, 1, 1) # CKK, BHW
        self.conv0 = nn.Conv2d(channels, channels, 1)

        self.conv1 = nn.Conv2d(channels, channels, 1)
        normal_init(self.conv0, std=0.01)
        normal_init(self.conv1, std=0.01)

    def forward(self, x, offset):
        B, C, H, W = x.shape
        query = self.unfold(x, offset).permute(0, 1).reshape(B, H, W, C, -1).max(dim=-1).values
        query = query.permute(0, 3, 1, 2).contiguous()
        query = self.conv0(query)

        key = self.conv1(x)

        query = query - query.mean(dim=1, keepdim=True)
        query = query / (query.pow(2.0).sum(dim=1, keepdim=True) + 1E-5).sqrt()

        key = key - key.mean(dim=1, keepdim=True)
        key = key / (key.pow(2.0).sum(dim=1, keepdim=True) + 1E-5).sqrt()

        out = (query * key).sum(dim=1, keepdim=True)
        return out


class SEAttention(nn.Module):
    def __init__(self, channels):
        super(SEAttention, self).__init__()
        self.conv0 = nn.Conv2d(channels, channels//4, 1)
        self.act = nn.ReLU()
        self.conv1 = nn.Conv2d(channels//4, channels, 1)

        normal_init(self.conv0, std=0.01)
        normal_init(self.conv1, std=0.01)
    
    def forward(self, x):
        x = x * self.conv1(self.act(self.conv0(x))).sigmoid()
        return x


'''
class MaskGenerator(nn.Module):
    def __init__(self, channels):
        super(MaskGenerator, self).__init__()
        self.channels = channels
        self.conv_query = nn.Conv2d(channels, channels//8, 1)
        self.conv_key = DeformConv2d(channels, channels//8, 3, padding=1)
        self.unfold = DeformUnfold(channels, 3, 1, 1) # CKK, BHW

        normal_init(self.conv_query)

    def forward(self, x, offset):
        B, C, H, W = x.shape
        x_q = self.conv_query(x)
        x_unfold = self.unfold(x_q, offset).permute(1, 0).reshape(B, H, W, C//8, 9)
        x_k = self.conv_key(x, offset)
        x_k = x_k.permute(0, 2, 3, 1).unsqueeze(-1)

        out = (x_unfold * x_k).sum(dim=-2).permute(0, 3, 1, 2).contiguous()
        return out
'''



class ChannelAttentionDeformConv(nn.Module):
    def __init__(self, channels, kernel, stride, padding):
        super(ChannelAttentionDeformConv, self).__init__()
        self.channels = channels
        self.kernel = kernel
        self.unfold = DeformUnfold(channels, kernel, stride, padding) #CKK, BHW

        self.conv_att = nn.Conv2d(channels, 1, 1)
        self.weight = nn.Parameter(
            torch.Tensor(channels, channels, kernel, kernel))
        self.norm = nn.GroupNorm(32, channels)
        self.act = nn.ReLU()

        nn.init.normal_(self.weight, std=0.01)
        normal_init(self.conv_att, std=0.01)

    def forward(self, x, offset):
        B, C, H, W = x.shape
        x_unfold = self.unfold(x, offset).reshape(C, -1, B, H, W) # C, KK, B, H, W

        x_unfold_max = x_unfold.max(dim=1).values.permute(1, 0, 2, 3).contiguous() # B, C, H, W

        att = self.conv_att(x_unfold_max).sigmoid().permute(1, 0, 2, 3) # KK, B, H, W

        x_unfold = (att.unsqueeze(0) * x_unfold).reshape(-1, B*H*W) # CKK, BHW
        weight = self.weight.reshape(self.channels, -1) # Co, CKK

        out = torch.matmul(weight, x_unfold).reshape(C, B, H, W).permute(1, 0, 2, 3)
        out = self.act(self.norm(out))
        return out




'''
class DualDeformModule(nn.Module):
    def __init__(self, channels, points=9):
        super(DualDeformModule, self).__init__()
        self.points = points
        self.base_offset = self.get_base_offset(points)
        
        self.cls_conv = DeformConv2dModule(channels, channels, 3, stride=1, padding=1)
        self.reg_conv = DeformConv2dModule(channels, channels, 3, stride=1, padding=1)

        self.offset_conv = nn.Conv2d(channels, points*2, 3, padding=1)
        normal_init(self.offset_conv, std=0.01)


    def get_base_offset(self, points):
        dcn_kernel = int(np.sqrt(points))
        dcn_pad = int((dcn_kernel - 1) / 2)
        dcn_base = np.arange(-dcn_pad, dcn_pad + 1).astype(np.float64)
        dcn_base_y = np.repeat(dcn_base, dcn_kernel)
        dcn_base_x = np.tile(dcn_base, dcn_kernel)
        dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=1).reshape((-1))
        dcn_base_offset = torch.tensor(dcn_base_offset).view(1, -1, 1, 1)
        return dcn_base_offset

    def points2distance(self, pts, y_first=True):
        pts_reshape = pts.view(pts.shape[0], -1, 2, *pts.shape[2:])
        pts_y = pts_reshape[:, :, 0, ...] if y_first else pts_reshape[:, :, 1,
                                                                      ...]
        pts_x = pts_reshape[:, :, 1, ...] if y_first else pts_reshape[:, :, 0,
                                                                      ...]
        bbox_left = pts_x.min(dim=1, keepdim=True)[0]
        bbox_right = pts_x.max(dim=1, keepdim=True)[0]
        bbox_up = pts_y.min(dim=1, keepdim=True)[0]
        bbox_bottom = pts_y.max(dim=1, keepdim=True)[0]
        bbox = torch.cat([-bbox_left, -bbox_up, bbox_right, bbox_bottom],
                            dim=1)
        return bbox

    def forward(self, cls_feat, reg_feat):
        self.base_offset = self.base_offset.type_as(reg_feat)

        offset = self.offset_conv(reg_feat)
        distance = self.points2distance(offset)

        cls_feat = self.cls_conv(cls_feat, offset - self.base_offset)
        reg_feat = self.reg_conv(reg_feat, offset - self.base_offset)

        return distance, cls_feat, reg_feat
'''

'''
class RotateDeformConv(nn.Module):
    def __init__(self, channels, kernel, stride, padding):
        super(RotateDeformConv, self).__init__()
        self.channels = channels
        self.unfold = DeformUnfold(channels, kernel, stride, padding) #CKK, BHW

        self.weight = nn.Parameter(
            torch.Tensor(channels, channels, kernel*kernel))
        nn.init.normal_(self.weight, std=0.01)

    def forward(self, x, offset):
        B, C, H, W = x.shape
        x_unfold = self.unfold(x, offset).reshape(C, -1, B, H, W) # C, KK, B, H, W

        x_unfold = x_unfold.reshape(-1, B*H*W) # CKK, BHW
        weight = self.weight.reshape(self.channels, self.channels, -1) # Co, CKK


        weight_list = weight.split([2, 2, 2, 3], dim=-1)

        weight_1 = weight.reshape(self.channels, -1)
        weight_2 = torch.cat([
            weight_list[1], weight_list[2], weight_list[3], weight_list[0]
        ], dim=-1).reshape(self.channels, -1)
        weight_3 = torch.cat([
            weight_list[2], weight_list[3], weight_list[0], weight_list[1]
        ], dim=-1).reshape(self.channels, -1)
        weight_4 = torch.cat([
            weight_list[3], weight_list[0], weight_list[1], weight_list[2]
        ], dim=-1).reshape(self.channels, -1)

        out_1 = torch.matmul(weight_1, x_unfold).reshape(-1, B, H, W).permute(1, 0, 2, 3)
        out_2 = torch.matmul(weight_2, x_unfold).reshape(-1, B, H, W).permute(1, 0, 2, 3)
        out_3 = torch.matmul(weight_3, x_unfold).reshape(-1, B, H, W).permute(1, 0, 2, 3)
        out_4 = torch.matmul(weight_4, x_unfold).reshape(-1, B, H, W).permute(1, 0, 2, 3)


        out = torch.cat([out_1, out_2, out_3, out_4], dim=1)

        return out
'''



'''
class RotateDeformConv(nn.Module):
    def __init__(self, channels, kernel, stride, padding):
        super(RotateDeformConv, self).__init__()
        self.channels = channels
        self.unfold = DeformUnfold(channels, kernel, stride, padding) #CKK, BHW

        self.weight_1 = nn.Parameter(
            torch.Tensor(channels//4, channels, kernel*kernel))
        nn.init.normal_(self.weight_1, std=0.01)

        self.weight_2 = nn.Parameter(
            torch.Tensor(channels//4, channels, kernel*kernel))
        nn.init.normal_(self.weight_2, std=0.01)

        self.weight_3 = nn.Parameter(
            torch.Tensor(channels//4, channels, kernel*kernel))
        nn.init.normal_(self.weight_3, std=0.01)

        self.weight_4 = nn.Parameter(
            torch.Tensor(channels//4, channels, kernel*kernel))
        nn.init.normal_(self.weight_4, std=0.01)

    def pts2angle(self, pts):
        pts_reshape = pts.reshape(pts.shape[0], -1, 2, *pts.shape[2:])
        offset_y = pts_reshape[:, :, 0, ...]
        offset_x = pts_reshape[:, :, 1, ...]
        pts_sin = torch.atan(offset_y/(offset_x + 1E-8)) + offset_x.lt(0.0) * math.pi
        return pts_sin

    def rotate_pts(self, pts, sorted_index, step=2):
        rotate_index = torch.cat([sorted_index[:, step:, ...], sorted_index[:, 0:step, ...]], dim=1)
        rotate_index = rotate_index.unsqueeze(2) * 2 + torch.arange(2).reshape(1, 1, 2, 1, 1).type_as(rotate_index)
        rotate_index = rotate_index.reshape(pts.shape[0], -1, *pts.shape[2:])
        pts_sorted = torch.gather(pts, dim=1, index=rotate_index)

        sorted_index = sorted_index.unsqueeze(2) * 2 + torch.arange(2).reshape(1, 1, 2, 1, 1).type_as(sorted_index)
        sorted_index = sorted_index.reshape(pts.shape[0], -1, *pts.shape[2:])

        pts_out = torch.scatter(torch.zeros_like(pts_sorted), dim=1, index=sorted_index, src=pts_sorted)
        return pts_out
        
    
    def forward(self, x, offset, base_offset):
        B, _, H, W = x.shape
        angles = self.pts2angle(offset)
        _, sorted_indx = torch.sort(angles, dim=1)

        pts_1 = offset
        pts_2 = self.rotate_pts(offset, sorted_indx, step=2)
        pts_3 = self.rotate_pts(offset, sorted_indx, step=4)
        pts_4 = self.rotate_pts(offset, sorted_indx, step=6)

        unfold_1 = self.unfold(x, pts_1 - base_offset) # CKK, BHW
        unfold_2 = self.unfold(x, pts_2 - base_offset) # CKK, BHW
        unfold_3 = self.unfold(x, pts_3 - base_offset) # CKK, BHW
        unfold_4 = self.unfold(x, pts_4 - base_offset) # CKK, BHW

        weight_1 = self.weight_1.reshape(self.channels // 4, -1)
        weight_2 = self.weight_2.reshape(self.channels // 4, -1)
        weight_3 = self.weight_3.reshape(self.channels // 4, -1)
        weight_4 = self.weight_4.reshape(self.channels // 4, -1)
        
        out_1 = torch.matmul(weight_1, unfold_1).reshape(-1, B, H, W).permute(1, 0, 2, 3)
        out_2 = torch.matmul(weight_2, unfold_2).reshape(-1, B, H, W).permute(1, 0, 2, 3)
        out_3 = torch.matmul(weight_3, unfold_3).reshape(-1, B, H, W).permute(1, 0, 2, 3)
        out_4 = torch.matmul(weight_4, unfold_4).reshape(-1, B, H, W).permute(1, 0, 2, 3)
        out = torch.cat([out_1, out_2, out_3, out_4], dim=1)

        return out
'''

class SampleFeat(nn.Module):
    def __init__(self):
        super(SampleFeat, self).__init__()

    def sample_offset(self, x, flow, padding_mode):
        N, _, H, W = flow.size()
        x_ = torch.arange(W).view(1, -1).expand(H, -1)
        y_ = torch.arange(H).view(-1, 1).expand(-1, W)
        grid = torch.stack([x_, y_], dim=0).type_as(x)

        grid = grid.unsqueeze(0).expand(N, -1, -1, -1)
        grid = grid + flow
        gx = 2 * grid[:, 0, :, :] / (W - 1) - 1
        gy = 2 * grid[:, 1, :, :] / (H - 1) - 1
        grid = torch.stack([gx, gy], dim=1)
        grid = grid.permute(0, 2, 3, 1)
        return F.grid_sample(x, grid, padding_mode=padding_mode, align_corners=True)

    def forward(self, x, offset):
        B, C, H, W = x.shape
        offset_reshape = offset.reshape(-1, 2, H, W)
        points = offset_reshape.shape[0] // B

        offset_reshape = torch.stack([offset_reshape[:, 1, ...], offset_reshape[:, 0, ...]], dim=1)
        
        x = x.unsqueeze(1).repeat(1, points, 1, 1, 1).reshape(B*points, C, H, W)
        sampled_feat = self.sample_offset(x, offset_reshape, padding_mode='zeros').reshape(B, points*C, H, W)
        return sampled_feat


class FeatTransform(nn.Module):
    def __init__(self, channels, points=9, section=4, inner_channels=64, norm_cfg=None):
        super(FeatTransform, self).__init__()
        self.channels = channels
        self.section = section
        self.inner_channels = inner_channels
        self.points = points

        self.sample_feat = SampleFeat()

        self.conv_lr = ConvModule(points*inner_channels, channels//2, 1, norm_cfg=norm_cfg)
        normal_init(self.conv_lr.conv, std=0.01)
        self.conv_ub = ConvModule(points*inner_channels, channels//2, 1, norm_cfg=norm_cfg)
        normal_init(self.conv_ub.conv, std=0.01)

    def forward(self, x, offset, prob_y, prob_x):
        B, _, H, W = x.shape
        x = self.sample_feat(x, offset).reshape(B, self.points, self.section, self.inner_channels, H, W)
        x_lr, x_ub = x.chunk(2, dim=2)

        # prob_lr, prob_ub = prob.reshape(B, self.points, self.section, 1, H, W).chunk(2, dim=2)
        # x = torch.cat([
        #     (x_lr * prob_lr).sum(dim=2),
        #     (x_ub * prob_ub).sum(dim=2),
        # ], dim=2).reshape(B, -1, H, W)
        # x = self.conv_1(x)

        prob_lr, _ = prob_x.reshape(B, self.points, 3, 1, H, W).softmax(dim=2).split([2, 1], dim=2)
        prob_ub, _ = prob_y.reshape(B, self.points, 3, 1, H, W).softmax(dim=2).split([2, 1], dim=2)

        x_lr = self.conv_lr((x_lr * prob_lr).sum(dim=2).reshape(B, -1, H, W))
        x_ub = self.conv_ub((x_ub * prob_ub).sum(dim=2).reshape(B, -1, H, W))

        x = torch.cat([x_lr, x_ub], dim=1)

        return x



# class ClassGuidedConvModule(nn.Module):
#     def __init__(self, channels, section, inner_channels, norm_cfg=None):
#         super(ClassGuidedConvModule, self).__init__()
#         self.channels = channels
#         self.section = section
#         self.inner_channels = inner_channels

#         self.conv = nn.Conv2d(channels, section * inner_channels, 3, padding=1)
#         self.conv_out = nn.Conv2d(inner_channels, channels, 1)
#         self.norm_name, norm = build_norm_layer(norm_cfg, channels)
#         self.add_module(self.norm_name, norm)
#         self.act = nn.ReLU(inplace=True)

#         normal_init(self.conv, std=0.01)
#         normal_init(self.conv_out, std=0.01)
#         constant_init(self.norm, 1, bias=0)

#         self.CGM = nn.Sequential(OrderedDict([
#             ('conv0', nn.Conv2d(channels, 256, 3, padding=1)),
#             ('act0', nn.ReLU(inplace=True)),
#             ('conv1', nn.Conv2d(256, section, 3, padding=1)),
#             ('act1', nn.Softmax(dim=1)),
#         ]))
#         normal_init(self.CGM.conv0, std=0.01)
#         normal_init(self.CGM.conv1, std=0.01)
    
#     @property
#     def norm(self):
#         return getattr(self, self.norm_name)

#     def forward(self, x, cls_feat):
#         B, _, H, W = x.shape
#         mask = self.CGM(x)
#         x = self.conv(x).view(B, self.section, self.inner_channels, H, W)
#         # mask = self.CGM(cls_feat)
#         x = (x * mask.unsqueeze(dim=2)).sum(dim=1)
#         x = self.act(self.norm(self.conv_out(x)))
#         return x


class DualDeformModule(nn.Module):
    def __init__(self, channels, points=9, norm_cfg=None):
        super(DualDeformModule, self).__init__()
        self.points = points
        self.base_offset = self.get_base_offset(points)

        self.channels = channels

        self.cls_conv = DeformConv2dModule(channels, channels, [1, self.points], stride=1, padding=[0, self.points//2], norm_cfg=norm_cfg)
        self.reg_conv = DeformConv2dModule(channels, channels, [1, self.points], stride=1, padding=[0, self.points//2], norm_cfg=norm_cfg)

        self.offset_conv = ConvModule(channels, channels, 3, padding=1, norm_cfg=norm_cfg)
        self.offset_out = nn.Conv2d(channels, self.points*2, 3, padding=1)

        normal_init(self.offset_conv.conv, std=0.01)
        normal_init(self.offset_out, std=0.01)

    @property
    def norm(self):
        return getattr(self, self.norm_name)

    # def points2ratio(self, pts):
    #     pts_reshape = pts.view(pts.shape[0], -1, 2, *pts.shape[2:])
    #     pts_y = pts_reshape[:, :, 0, ...]
    #     pts_x = pts_reshape[:, :, 1, ...]

    #     left = pts_x.min(dim=1, keepdim=True)[0]
    #     right = pts_x.max(dim=1, keepdim=True)[0]
    #     up = pts_y.min(dim=1, keepdim=True)[0]
    #     bottom = pts_y.max(dim=1, keepdim=True)[0]

    #     ratio_x = (pts_x - left) / (right - left) - 0.5
    #     ratio_y = (pts_y - up) / (bottom - up) - 0.5

    #     ratio = torch.cat([ratio_y, ratio_x], dim=1)
    #     return ratio

    def points2ratio(self, pts):
        pts_reshape = pts.view(pts.shape[0], -1, 2, *pts.shape[2:])
        pts_y = pts_reshape[:, :, 0, ...]
        pts_x = pts_reshape[:, :, 1, ...]

        left = pts_x.min(dim=1, keepdim=True)[0]
        right = pts_x.max(dim=1, keepdim=True)[0]
        up = pts_y.min(dim=1, keepdim=True)[0]
        bottom = pts_y.max(dim=1, keepdim=True)[0]

        ratio_y = (pts_y - up) / (bottom - up)
        ratio_x = (pts_x - left) / (right - left)
        ratio = torch.cat([ratio_y, ratio_x], dim=1)

        return ratio

    # def points2mask(self, pts, th=0.2):
    #     pts_reshape = pts.view(pts.shape[0], -1, 2, *pts.shape[2:])
    #     pts_y = pts_reshape[:, :, 0, ...]
    #     pts_x = pts_reshape[:, :, 1, ...]

    #     left = pts_x.min(dim=1, keepdim=True)[0]
    #     up = pts_y.min(dim=1, keepdim=True)[0]
    #     right = pts_x.max(dim=1, keepdim=True)[0]
    #     bottom = pts_y.max(dim=1, keepdim=True)[0]

    #     ratio_x = (pts_x - left) / (right - left)
    #     ratio_y = (pts_y - up) / (bottom - up)

    #     is_left = ratio_x.lt(th)
    #     is_up = ratio_y.lt(th)
    #     is_right = ratio_x.gt(1.0-th)
    #     is_bottom = ratio_y.gt(1.0-th)

    #     mask = torch.stack([is_left, is_right, is_up, is_bottom], dim=2).type_as(pts) # B, P, 4, H, W

    #     return mask

    def points2mask(self, pts, th=0.2):
        pts_reshape = pts.view(pts.shape[0], -1, 2, *pts.shape[2:])
        pts_y = pts_reshape[:, :, 0, ...]
        pts_x = pts_reshape[:, :, 1, ...]

        left = pts_x.min(dim=1, keepdim=True)[0]
        up = pts_y.min(dim=1, keepdim=True)[0]
        right = pts_x.max(dim=1, keepdim=True)[0]
        bottom = pts_y.max(dim=1, keepdim=True)[0]

        ratio_x = (pts_x - left) / (right - left)
        ratio_y = (pts_y - up) / (bottom - up)

        is_left = ratio_x.lt(th)
        is_up = ratio_y.lt(th)
        is_right = ratio_x.gt(1.0-th)
        is_bottom = ratio_y.gt(1.0-th)

        mask = (is_left + is_up + is_right + is_bottom).type_as(pts)

        return mask

    def get_base_offset(self, points):
        dcn_base_x = np.arange(-((points-1)//2), points//2+1).astype(np.float64)
        dcn_base_y = dcn_base_x * 0.0
        dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=1).reshape((-1))
        dcn_base_offset = torch.tensor(dcn_base_offset).view(1, -1, 1, 1)
        return dcn_base_offset

    def points2distance(self, pts, y_first=True):
        pts_reshape = pts.view(pts.shape[0], -1, 2, *pts.shape[2:])
        pts_y = pts_reshape[:, :, 0, ...] if y_first else pts_reshape[:, :, 1,
                                                                      ...]
        pts_x = pts_reshape[:, :, 1, ...] if y_first else pts_reshape[:, :, 0,
                                                                      ...]
        bbox_left = pts_x.min(dim=1, keepdim=True)[0]
        bbox_right = pts_x.max(dim=1, keepdim=True)[0]
        bbox_up = pts_y.min(dim=1, keepdim=True)[0]
        bbox_bottom = pts_y.max(dim=1, keepdim=True)[0]

        bbox = torch.cat([-bbox_left, -bbox_up, bbox_right, bbox_bottom],
                            dim=1)
        return bbox

    def forward(self, cls_feat, reg_feat):
        B, _, H, W = cls_feat.shape
        base_offset = self.base_offset.type_as(reg_feat)

        pts = self.offset_out(self.offset_conv(reg_feat))
        cls_feat = self.cls_conv(cls_feat, pts - base_offset)
        reg_feat = self.reg_conv(reg_feat, pts - base_offset)

        return pts, cls_feat, reg_feat



    # def forward(self, cls_feat, reg_feat, off_feat):
    #     B, _, H, W = cls_feat.shape
    #     # base_offset = self.base_offset.type_as(off_feat)
    #     pts = self.offset_conv(off_feat)
    #     pts_reshape = pts.view(B, -1, 2, H, W)
    #     pts_x = pts_reshape[:, :, 0, ...]
    #     pts_y = pts_reshape[:, :, 1, ...]
    #     # bbox_left = pts_x.min(dim=1, keepdim=True)[0]
    #     # bbox_right = pts_x.max(dim=1, keepdim=True)[0]
    #     # bbox_up = pts_y.min(dim=1, keepdim=True)[0]
    #     # bbox_bottom = pts_y.max(dim=1, keepdim=True)[0]
        
    #     # cls_feat = self.cls_conv(cls_feat, pts - base_offset)
    #     # reg_feat = self.reg_conv(reg_feat, pts - base_offset)

    #     cls_feat = self.sample_feat(cls_feat, pts)
    #     cls_feat = self.cls_conv(cls_feat)

    #     reg_feat = self.reg_conv_trans(reg_feat)
    #     reg_feat = self.sample_feat(reg_feat, pts).reshape(B, self.points, -1, H, W) # B, P, 4C, H, W
    #     reg_indx_0 = pts_x.gt(0.0) * 0
    #     reg_indx_1 = pts_x.le(0.0) * 1
    #     reg_indx_2 = pts_y.gt(0.0) * 2
    #     reg_indx_3 = pts_y.le(0.0) * 3
    #     reg_indx_x = (reg_indx_0 + reg_indx_1).long()
    #     reg_indx_y = (reg_indx_2 + reg_indx_3).long()

    #     reg_indx_x = reg_indx_x.unsqueeze(dim=2) * self.channels + torch.arange(self.channels).view(1, 1, -1, 1, 1).type_as(reg_indx_x)
    #     reg_indx_y = reg_indx_y.unsqueeze(dim=2) * self.channels + torch.arange(self.channels).view(1, 1, -1, 1, 1).type_as(reg_indx_y)
    #     reg_feat_x = torch.gather(reg_feat, dim=2, index=reg_indx_x).view(B, -1, H, W)
    #     reg_feat_y = torch.gather(reg_feat, dim=2, index=reg_indx_y).view(B, -1, H, W)
    #     # reg_feat = torch.cat([reg_feat_x, reg_feat_y], dim=1)
    #     reg_feat_x = self.reg_conv_x(reg_feat_x)
    #     reg_feat_y = self.reg_conv_y(reg_feat_y)

    #     # reg_feat = torch.cat([reg_feat_x, reg_feat_y], dim=1)

    #     # reg_feat = self.sample_feat(reg_feat, pts)
    #     # reg_feat = self.reg_conv(reg_feat)

    #     return pts, cls_feat, reg_feat_x, reg_feat_y


    # def points2distance(self, pts, y_first=True):
    #     pts_reshape = pts.view(pts.shape[0], -1, 2, *pts.shape[2:])
    #     pts_y = pts_reshape[:, :, 0, ...] if y_first else pts_reshape[:, :, 1,
    #                                                                   ...]
    #     pts_x = pts_reshape[:, :, 1, ...] if y_first else pts_reshape[:, :, 0,
    #                                                                   ...]
        
    #     points = pts_x.shape[1]
    #     size_section = [points//4] * 3 + [points - 3 * points//4] 
    #     pts_y_list = torch.split(pts_y, size_section, dim=1)
    #     pts_x_list = torch.split(pts_x, size_section, dim=1)

    #     #0l, 1u, 2r, 3b
    #     bbox_left = pts_x_list[0].min(dim=1, keepdim=True)[0]
    #     bbox_right = pts_x_list[2].max(dim=1, keepdim=True)[0]
    #     bbox_up = pts_y_list[1].min(dim=1, keepdim=True)[0]
    #     bbox_bottom = pts_y_list[3].max(dim=1, keepdim=True)[0]
        
    #     bbox = torch.cat([-bbox_left, -bbox_up, bbox_right, bbox_bottom],
    #                         dim=1)
    #     return bbox


    # def forward(self, cls_feat, reg_feat, off_feat):
    #     B, _, H, W = cls_feat.shape
    #     pts = self.offset_conv(off_feat)

    #     cls_feat = self.sample_feat(cls_feat, pts)
    #     cls_feat = self.cls_conv(cls_feat)

    #     pts_reshape = pts.view(pts.shape[0], -1, 2, *pts.shape[2:])
    #     pts_x = pts_reshape[:, :, 0, ...]
    #     pts_y = pts_reshape[:, :, 1, ...]

    #     pts_x_sorted, pts_x_indx = torch.sort(pts_x, dim=1, descending=False)
    #     pts_y_sorted, pts_y_indx = torch.sort(pts_y, dim=1, descending=False)
    #     distance = torch.stack([
    #                     -pts_x_sorted[:, 0, ...], -pts_y_sorted[:, 0, ...], 
    #                     pts_x_sorted[:, -1, ...], pts_y_sorted[:, -1, ...]], dim=1)
        
    #     pts_x_cast = torch.gather(pts, dim=1, 
    #         index= (pts_x_indx.unsqueeze(2) * 2 + torch.arange(2).view(1, 1, 2, 1, 1).type_as(pts_x_indx)).reshape(B, -1, H, W)
    #     )
    #     pts_y_cast = torch.gather(pts, dim=1, 
    #         index= (pts_y_indx.unsqueeze(2) * 2 + torch.arange(2).view(1, 1, 2, 1, 1).type_as(pts_y_indx)).reshape(B, -1, H, W)
    #     )

    #     # reg_feat = torch.cat([
    #     #     self.sample_feat(reg_feat, pts_x_cast),
    #     #     self.sample_feat(reg_feat, pts_y_cast)
    #     # ], dim=1)

    #     # reg_feat = self.reg_conv(reg_feat)
    #     reg_feat_x = self.reg_conv_x(self.sample_feat(reg_feat, pts_x_cast))
    #     reg_feat_y = self.reg_conv_y(self.sample_feat(reg_feat, pts_y_cast))

    #     reg_feat = torch.cat([reg_feat_x, reg_feat_y], dim=1)

    #     mask = torch.zeros_like(cls_feat)

    #     if self.training:
    #         return distance, pts, cls_feat, reg_feat
    #     else:
    #         return distance, pts, cls_feat, reg_feat, pts, mask


    # def boundpts(self, pts, bord):
    #     pts_reshape = pts.view(pts.shape[0], -1, 2, *pts.shape[2:])
    #     pts_x = pts_reshape[:, :, 0, ...]
    #     pts_y = pts_reshape[:, :, 1, ...]
    #     l = bord[:, 0:1, ...]
    #     u = bord[:, 1:2, ...]
    #     r = bord[:, 2:3, ...]
    #     b = bord[:, 3:, ...]
    #     bound_pts_x = torch.where(pts_x.gt(l), pts_x, l)
    #     bound_pts_x = torch.where(bound_pts_x.gt(r), r, bound_pts_x)
    #     bound_pts_y = torch.where(pts_y.gt(u), pts_y, u)
    #     bound_pts_y = torch.where(bound_pts_y.gt(b), b, bound_pts_y)

    #     bound_pts = torch.cat([bound_pts_x, bound_pts_y], dim=1)
    #     return bound_pts


    # def forward(self, cls_feat, reg_feat, off_feat):

    #     offset_1, offset_2 = self.offset_conv(off_feat).split([16, 6], dim=1)
        
    #     # l_x, u_y, r_x, b_y, l_y0, l_y1, l_y2, u_x0, u_x1, u_x2, r_y0, r_y1, r_y2, b_x0, b_x1, b_x2 = self.offset_conv(off_feat).chunk(16, dim=1) # B, 4 + 4*3, H, W
    #     l_x, u_y, r_x, b_y, l_y0, l_y1, l_y2, u_x0, u_x1, u_x2, r_y0, r_y1, r_y2, b_x0, b_x1, b_x2 = offset_1.chunk(16, dim=1) # B, 4 + 4*3, H, W

    #     # l_y0 = (l_y0 - b_y).clamp(min=0.0) + b_y

    #     pts = torch.cat([
    #         l_x, l_y0, 
    #         l_x, l_y1, 
    #         l_x, l_y2,
    #         u_x0, u_y, 
    #         u_x1, u_y, 
    #         u_x2, u_y, 
    #         r_x, r_y0, 
    #         r_x, r_y1, 
    #         r_x, r_y2, 
    #         b_x0, b_y,
    #         b_x1, b_y,
    #         b_x2, b_y,
    #     ], dim=1)

    #     # pts = self.boundpts(pts, torch.cat([l_x, u_y, r_x, b_y], dim=1))

    #     distance = torch.cat([-l_x, -u_y, r_x, b_y], dim=1)
       
    #     # cls_feat = self.sample_feat(cls_feat, pts)
    #     cls_feat = self.sample_feat(cls_feat, torch.cat([pts, offset_2], dim=1))
    #     cls_feat = self.cls_conv(cls_feat)
    #     reg_feat = self.sample_feat(reg_feat, pts)
    #     reg_feat = self.reg_conv(reg_feat)

    #     mask = torch.zeros_like(cls_feat)

    #     if self.training:
    #         return distance, pts, cls_feat, reg_feat
    #     else:
    #         return distance, pts, cls_feat, reg_feat, pts, mask

    
    # def forward(self, cls_feat, reg_feat, off_feat):
        
    #     pts = self.offset_conv(off_feat) # B, 5*G*2, H, W
    #     # cls_feat = self.sample_feat(cls_feat, pts)
    #     # cls_feat = self.cls_conv(cls_feat)

    #     # distance = self.points2distance(pts, y_first=False)

    #     # reg_feat = self.sample_feat(reg_feat, pts)
    #     # reg_feat = self.reg_conv(reg_feat)

    #     pts_reshape = pts.view(pts.shape[0], self.points, 2, *pts.shape[2:])

    #     cls_feat = self.sample_feat(cls_feat, pts_reshape)
    #     cls_feat = self.cls_conv(cls_feat)

        
    #     pts_x = pts_reshape[:, :, 0, ...]
    #     pts_y = pts_reshape[:, :, 1, ...]

    #     left, left_inds = pts_x.min(dim=1, keepdim=True)
    #     right, right_inds = pts_x.max(dim=1, keepdim=True)
    #     up, up_inds = pts_y.min(dim=1, keepdim=True)
    #     bottom, bottom_inds = pts_y.max(dim=1, keepdim=True)

    #     distance = torch.cat([-left, -up, right, bottom], dim=1)

    #     left_pts = torch.gather(pts, dim=1,
    #         index=(left_inds*2 + torch.arange(2).view([1, 2, 1, 1]).type_as(left_inds)).reshape(pts.shape[0], -1, *pts.shape[2:])
    #     )
    #     right_pts = torch.gather(pts, dim=1,
    #         index=(right_inds*2 + torch.arange(2).view([1, 2, 1, 1]).type_as(right_inds)).reshape(pts.shape[0], -1, *pts.shape[2:])
    #     )
    #     up_pts = torch.gather(pts, dim=1,
    #         index=(up_inds*2 + torch.arange(2).view([1, 2, 1, 1]).type_as(up_inds)).reshape(pts.shape[0], -1, *pts.shape[2:])
    #     )
    #     bottom_pts = torch.gather(pts, dim=1,
    #         index=(bottom_inds*2 + torch.arange(2).view([1, 2, 1, 1]).type_as(bottom_inds)).reshape(pts.shape[0], -1, *pts.shape[2:])
    #     )

    #     reg_feat_list = torch.chunk(reg_feat, 4, dim=1)

    #     reg_feat_l = self.sample_feat(reg_feat_list[0], left_pts)
    #     reg_feat_r = self.sample_feat(reg_feat_list[1], right_pts)
    #     reg_feat_u = self.sample_feat(reg_feat_list[2], up_pts)
    #     reg_feat_b = self.sample_feat(reg_feat_list[3], bottom_pts)


    #     reg_feat = torch.cat([reg_feat_l, reg_feat_r, reg_feat_u, reg_feat_b], dim=1)
    #     reg_feat = self.reg_conv(reg_feat)

    #     mask = torch.zeros_like(cls_feat)

    #     if self.training:
    #         return distance, pts, cls_feat, reg_feat
    #     else:
    #         return distance, pts, cls_feat, reg_feat, pts, mask
    


# class SeparableDeformConv(nn.Module):
#     def __init__(self, channels, inter_channels, kernel, stride, padding):
#         super(SeparableDeformConv, self).__init__()
#         self.unfold = DeformUnfold(channels, kernel, stride, padding)



# class CA2dModule(nn.Module):
#     def __init__(self, feat_channels, r=8, kernel=5):
#         super(CA2dModule, self).__init__()
#         self.kernel = kernel
#         self.inner_channels = max(8, feat_channels//r)
#         self.unfold_h = nn.Unfold(kernel_size=[kernel,1], stride=1)
#         self.unfold_w = nn.Unfold(kernel_size=[1, kernel], stride=1)
#         self.conv1 = nn.Conv2d(feat_channels, self.inner_channels, kernel_size=1, stride=1, padding=0)
#         self.bn1 = nn.BatchNorm2d(self.inner_channels)
#         self.bn2 = nn.BatchNorm2d(self.inner_channels)
#         self.act = nn.ReLU()

#         self.conv_h = nn.Conv2d(self.inner_channels, feat_channels, kernel_size=1, stride=1, padding=0)
#         self.conv_w = nn.Conv2d(self.inner_channels, feat_channels, kernel_size=1, stride=1, padding=0)

#         normal_init(self.conv1, std=0.01)
#         normal_init(self.conv_h, std=0.01)
#         normal_init(self.conv_w, std=0.01)

#     def forward(self, x):
#         N, C, H, W = x.size()

#         x_unfold_h = self.unfold_h(
#             F.pad(x, [0, 0, self.kernel//2, self.kernel//2], 'constant', -1E3)
#             ).view(N, C, self.kernel, H, W).max(dim=2).values
#         x_unfold_w = self.unfold_w(
#             F.pad(x, [self.kernel//2, self.kernel//2, 0, 0], 'constant', -1E3)
#             ).view(N, C, self.kernel, H, W).max(dim=2).values

#         x_h = self.conv_h(self.act(self.bn1(self.conv1(x_unfold_h)))).sigmoid()
#         x_w = self.conv_w(self.act(self.bn2(self.conv1(x_unfold_w)))).sigmoid()
#         out = x * x_h * x_w
#         return out


# class CA2dModule(nn.Module):
#     def __init__(self, feat_channels, r=8, kernel=5, topk=5):
#         super(CA2dModule, self).__init__()
#         self.kernel = kernel
#         self.topk = topk
#         self.inner_channels = topk * r
#         self.unfold_h = nn.Unfold(kernel_size=[kernel,1], stride=1)
#         self.unfold_w = nn.Unfold(kernel_size=[1, kernel], stride=1)

#         self.conv1 = nn.Conv2d(topk, self.inner_channels, kernel_size=1, stride=1, padding=0)
#         self.bn = nn.BatchNorm2d(self.inner_channels)
#         self.act = nn.ReLU()

#         self.conv_h = nn.Conv2d(self.inner_channels, feat_channels, kernel_size=1, stride=1, padding=0)
#         self.conv_w = nn.Conv2d(self.inner_channels, feat_channels, kernel_size=1, stride=1, padding=0)

#         normal_init(self.conv1, std=0.01)
#         normal_init(self.conv_h, std=0.01)
#         normal_init(self.conv_w, std=0.01)

#     def forward(self, x):
#         N, C, H, W = x.size()

#         x_unfold_h = self.unfold_h(
#             F.pad(x, [0, 0, self.kernel//2, self.kernel//2], 'constant', -1E4)).view(N, C, self.kernel, H, W).max(dim=2).values
#         x_unfold_h, _ = topk_unsorted(x_unfold_h, k=self.topk, dim=1, largest=True)
        
#         x_unfold_w = self.unfold_w(
#             F.pad(x, [self.kernel//2, self.kernel//2, 0, 0], 'constant', -1E4)).view(N, C, self.kernel, H, W).max(dim=2).values
#         x_unfold_w, _ = topk_unsorted(x_unfold_w, k=self.topk, dim=1, largest=True)

#         x_h = self.conv_h(self.act(self.bn(self.conv1(x_unfold_h)))).sigmoid()
#         x_w = self.conv_w(self.act(self.bn(self.conv1(x_unfold_w)))).sigmoid()
#         out = x * x_h * x_w
#         return out




# class CA2dModule(nn.Module):
#     def __init__(self, feat_channels, r=8, kernel=5):
#         super(CA2dModule, self).__init__()
#         self.kernel = kernel
#         self.inner_channels = max(8, feat_channels//r)
#         self.avg_h = nn.AdaptiveAvgPool2d((1, None))
#         self.avg_w = nn.AdaptiveAvgPool2d((None, 1))

#         self.conv1 = nn.Conv2d(feat_channels, self.inner_channels, kernel_size=1, stride=1, padding=0)
#         self.bn1 = nn.BatchNorm2d(self.inner_channels)
#         self.act = nn.ReLU()

#         self.conv_h = nn.Conv2d(self.inner_channels, feat_channels, kernel_size=1, stride=1, padding=0)
#         self.conv_w = nn.Conv2d(self.inner_channels, feat_channels, kernel_size=1, stride=1, padding=0)

#         normal_init(self.conv1, std=0.01)
#         normal_init(self.conv_h, std=0.01)
#         normal_init(self.conv_w, std=0.01)

#     def forward(self, x):
        
#         N, C, H, W = x.size()

#         x_h = self.avg_h(x).view(N, C, 1, W)
#         x_w = self.avg_w(x).view(N, C, H, 1)

#         x_h = self.conv_h(self.act(self.bn1(self.conv1(x_h)))).sigmoid()
#         x_w = self.conv_w(self.act(self.bn1(self.conv1(x_w)))).sigmoid()

#         out = x * x_h * x_w
#         return out




@HEADS.register_module()
class SERIALGFLHead(AnchorHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 bins=5,
                 points=9,
                 DGQP_cfg=dict(channels=64),
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.bins = bins
        self.points = points

        self.DGQP_cfg = DGQP_cfg
        self.with_DGQP = DGQP_cfg is not None

        # dcn_base_x = np.arange(-(self.points//2),
        #                      self.points//2 + 1).astype(np.float64)
        # dcn_base_y = dcn_base_x * 0.0
        # dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=1).reshape((-1))
        # self.dcn_base_offset = torch.tensor(dcn_base_offset).view(1, -1, 1, 1)
        

        # dcn_base_x = np.arange(-(self.points//2),
        #                      self.points//2 + 1).astype(np.float64)
        # dcn_base_y = dcn_base_x * 0.0
        # dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=1).reshape((-1))
        # self.dcn_base_offset = torch.tensor(dcn_base_offset).view(1, -1, 1, 1)


        # self.dcn_kernel = int(np.sqrt(self.points))
        # self.dcn_pad = int((self.dcn_kernel - 1) / 2)
        # assert self.dcn_kernel * self.dcn_kernel == self.points, \
        #     'The points number should be a square number.'
        # assert self.dcn_kernel % 2 == 1, \
        #     'The points number should be an odd square number.'
        # dcn_base = np.arange(-self.dcn_pad,
        #                      self.dcn_pad + 1).astype(np.float64)
        # dcn_base_y = np.repeat(dcn_base, self.dcn_kernel)
        # dcn_base_x = np.tile(dcn_base, self.dcn_kernel)
        # dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=1).reshape((-1))
        # self.dcn_base_offset = torch.tensor(dcn_base_offset).view(1, -1, 1, 1)


        super(SERIALGFLHead, self).__init__(num_classes, in_channels, **kwargs)

        self.sampling = False
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            # SSD sampling=False so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

        # self.integral = Integral(-(self.bins-1)//2, self.bins//2, bins=self.bins)
        # self.integral = Integral_v2(-(self.bins-1)//2, self.bins//2, bins=self.bins)
        self.integral = Integral_v2(0, self.bins-1, bins=self.bins)
        self.loss_bbox = build_loss(dict(type='GIoULoss', loss_weight=2.0))

        self.base_offset = self.get_base_offset(points)
    
    
    def get_base_offset(self, points):
        dcn_base_x = np.arange(-(points//2), points//2 + 1).astype(np.float64)
        dcn_base_y = dcn_base_x * 0.0
        dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=1).reshape((-1))
        dcn_base_offset = torch.tensor(dcn_base_offset).view(1, -1, 1, 1)
        return dcn_base_offset

    
    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        # self.off_convs = nn.ModuleList()

        for i in range(self.stacked_convs - 1):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            # self.off_convs.append(
            #     ConvModule(
            #         chn,
            #         self.feat_channels,
            #         3,
            #         stride=1,
            #         padding=1,
            #         conv_cfg=self.conv_cfg,
            #         norm_cfg=self.norm_cfg))


        self.gfl_dual = DualDeformModule(self.feat_channels, points=self.points, norm_cfg=self.norm_cfg)

        assert self.num_anchors == 1, 'anchor free version'

        self.gfl_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.gfl_reg = nn.Conv2d(self.feat_channels, 4 * self.bins, 3, padding=1)

        if self.with_DGQP:
            self.reg_conf = nn.Sequential(
                nn.Conv2d(4 * self.bins, self.DGQP_cfg['channels'], 1),
                self.relu,
                nn.Conv2d(self.DGQP_cfg['channels'], 1, 1),
                nn.Sigmoid()
            )


    def init_weights(self):
        """Initialize weights of the head."""
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        # for m in self.off_convs:
        #     normal_init(m.conv, std=0.01)

        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.gfl_cls, std=0.01, bias=bias_cls)
        normal_init(self.gfl_reg, std=0.01)

        if self.with_DGQP:
            for m in self.reg_conf:
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)


    def forward(self, feats):
        return multi_apply(self.forward_single, feats)

    # def points2distance(self, pts, y_first=True):
    #     pts_reshape = pts.view(pts.shape[0], -1, 2, *pts.shape[2:])
    #     pts_y = pts_reshape[:, :, 0, ...] if y_first else pts_reshape[:, :, 1,
    #                                                                   ...]
    #     pts_x = pts_reshape[:, :, 1, ...] if y_first else pts_reshape[:, :, 0,
    #                                                                   ...]
    #     bbox_left = pts_x.min(dim=1, keepdim=True)[0]
    #     bbox_right = pts_x.max(dim=1, keepdim=True)[0]
    #     bbox_up = pts_y.min(dim=1, keepdim=True)[0]
    #     bbox_bottom = pts_y.max(dim=1, keepdim=True)[0]

    #     bbox = torch.cat([-bbox_left, -bbox_up, bbox_right, bbox_bottom],
    #                         dim=1)
    #     return bbox

    # def points2distance(self, pts, y_first=True):
    #     pts_reshape = pts.view(pts.shape[0], -1, 2, *pts.shape[2:])
    #     pts_y = pts_reshape[:, :, 0, ...] if y_first else pts_reshape[:, :, 1,
    #                                                                   ...]
    #     pts_x = pts_reshape[:, :, 1, ...] if y_first else pts_reshape[:, :, 0,
    #                                                                   ...]
    #     bbox_left, left_inds = pts_x.min(dim=1, keepdim=True)
    #     bbox_right, right_inds = pts_x.max(dim=1, keepdim=True)
    #     bbox_up, up_inds = pts_y.min(dim=1, keepdim=True)
    #     bbox_bottom, bottom_inds = pts_y.max(dim=1, keepdim=True)

    #     bbox = torch.cat([-bbox_left, -bbox_up, bbox_right, bbox_bottom],
    #                         dim=1)
    #     return bbox, left_inds, right_inds, up_inds, bottom_inds
    

    # def forward_single(self, x):
    #     B, _, H, W = x.shape
    #     cls_feat = x
    #     reg_feat = x
    #     # off_feat = x

    #     for cls_conv in self.cls_convs:
    #         cls_feat = cls_conv(cls_feat)
    #     for reg_conv in self.reg_convs:
    #         reg_feat = reg_conv(reg_feat)
    #     # for off_conv in self.off_convs:
    #     #     off_feat = off_conv(off_feat)

    #     # off_feat = reg_feat

    #     pts, cls_feat, reg_feat = self.gfl_dual(cls_feat, reg_feat)


    #     pts_reshape = pts.reshape(B, -1, 2, H, W)
    #     pts_y = pts_reshape[:, :, 0, ...]
    #     pts_x = pts_reshape[:, :, 1, ...]
    #     bbox_left, _ = pts_x.min(dim=1, keepdim=True)
    #     bbox_up, _ = pts_y.min(dim=1, keepdim=True)
    #     bbox_right, _ = pts_x.max(dim=1, keepdim=True)
    #     bbox_bottom, _ = pts_y.max(dim=1, keepdim=True)
    #     bbox_pred_initial = torch.cat([-bbox_left, -bbox_up, bbox_right, bbox_bottom],
    #                         dim=1)


    #     bbox_dist = self.gfl_reg(reg_feat)
    #     bbox_pred_refine = self.integral(bbox_dist)

    #     cls_score = self.gfl_cls(cls_feat).sigmoid()
    #     if self.with_DGQP:
    #         N, _, H, W = bbox_dist.size()
    #         prob = F.softmax(bbox_dist.reshape(N, 4, self.bins, H, W), dim=2)
    #         cls_score = cls_score * self.reg_conf(prob.reshape(N, -1, H, W))

    #     return cls_score, bbox_pred_initial, bbox_pred_refine



    def forward_single(self, x):
        B, _, H, W = x.shape
        cls_feat = x
        
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)

        reg_feat = cls_feat + x
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        
        # off_feat = cls_feat + reg_feat
        # for off_conv in self.off_convs:
        #     off_feat = off_conv(off_feat)

        pts, cls_feat, reg_feat = self.gfl_dual(cls_feat, reg_feat)

        pts_reshape = pts.reshape(B, self.points, 2, H, W)
        pts_y = pts_reshape[:, :, 0, ...]
        pts_x = pts_reshape[:, :, 1, ...]
        bbox_left, _ = pts_x.min(dim=1, keepdim=True)
        bbox_up, _ = pts_y.min(dim=1, keepdim=True)
        bbox_right, _ = pts_x.max(dim=1, keepdim=True)
        bbox_bottom, _ = pts_y.max(dim=1, keepdim=True)

        bbox_pred_initial = torch.cat([-bbox_left, -bbox_up, bbox_right, bbox_bottom],
                            dim=1)

        bbox_dist = self.gfl_reg(reg_feat)

        bbox_pred_refine = self.integral(bbox_dist)

        cls_score = self.gfl_cls(cls_feat).sigmoid()

        if self.with_DGQP:
            N, _, H, W = bbox_dist.size()
            prob = bbox_dist.reshape(N, 4, self.bins, H, W)
            prob = F.softmax(prob, dim=2)
            cls_score = cls_score * self.reg_conf(prob.reshape(N, -1, H, W))

        return cls_score, bbox_pred_initial, bbox_pred_refine

    
    # def forward_single(self, x):
    #     base_offset = self.base_offset.type_as(x)


    #     B, _, H, W = x.shape
    #     cls_feat = x
    #     reg_feat = x
    #     # off_feat = x

    #     for cls_conv in self.cls_convs:
    #         cls_feat = cls_conv(cls_feat)
    #     for reg_conv in self.reg_convs:
    #         reg_feat = reg_conv(reg_feat)

    #     pts, cls_feat, reg_feat = self.gfl_dual(cls_feat, reg_feat)

    #     # pts_reshape = pts.reshape(B, self.points, 2, H, W)
    #     # pts_y = pts_reshape[:, :, 0, ...]
    #     # pts_x = pts_reshape[:, :, 1, ...]
    #     # bbox_left, left_inds = pts_x.min(dim=1, keepdim=True)
    #     # bbox_up, up_inds = pts_y.min(dim=1, keepdim=True)
    #     # bbox_right, right_inds = pts_x.max(dim=1, keepdim=True)
    #     # bbox_bottom, bottom_inds = pts_y.max(dim=1, keepdim=True)

    #     # bbox_pred_initial = torch.cat([-bbox_left, -bbox_up, bbox_right, bbox_bottom],
    #     #                     dim=1)

    #     pts_refine_dist = self.gfl_reg(reg_feat) # B, points*4*bin, H, W
    #     pts_refine_dist_reshape = pts_refine_dist.view(B, self.points, 2, self.bins, H, W)
    #     pts_refine_dist_y = pts_refine_dist_reshape[:, :, 0, ...].reshape(B, self.points * self.bins, H, W)
    #     pts_refine_dist_x = pts_refine_dist_reshape[:, :, 1, ...].reshape(B, self.points * self.bins, H, W)

    #     pts_refine = self.integral(pts_refine_dist).view(B, self.points, 2, H, W)

    #     pts_reshape = pts.reshape(B, self.points, 2, H, W)
    #     pts_y = pts_reshape[:, :, 0, ...] + pts_refine[:, :, 0, ...]
    #     pts_x = pts_reshape[:, :, 1, ...] + pts_refine[:, :, 1, ...]
    #     pts = torch.stack([pts_y, pts_x], dim=2).reshape(B, -1, H, W)
    #     bbox_left, left_inds = pts_x.min(dim=1, keepdim=True)
    #     bbox_up, up_inds = pts_y.min(dim=1, keepdim=True)
    #     bbox_right, right_inds = pts_x.max(dim=1, keepdim=True)
    #     bbox_bottom, bottom_inds = pts_y.max(dim=1, keepdim=True)

    #     bbox_pred_initial = torch.cat([-bbox_left, -bbox_up, bbox_right, bbox_bottom], dim=1)
    #     bbox_pred_refine = torch.zeros_like(bbox_pred_initial)

    #     left_dist = torch.gather(pts_refine_dist_x, dim=1,
    #                     index=left_inds * self.bins + torch.arange(self.bins).reshape(1, self.bins, 1, 1).type_as(left_inds))
    #     up_dist = torch.gather(pts_refine_dist_y, dim=1,
    #                     index=up_inds * self.bins + torch.arange(self.bins).reshape(1, self.bins, 1, 1).type_as(up_inds))
    #     right_dist = torch.gather(pts_refine_dist_x, dim=1,
    #                     index=right_inds * self.bins + torch.arange(self.bins).reshape(1, self.bins, 1, 1).type_as(right_inds))
    #     bottom_dist = torch.gather(pts_refine_dist_y, dim=1,
    #                     index=bottom_inds * self.bins + torch.arange(self.bins).reshape(1, self.bins, 1, 1).type_as(bottom_inds))

    #     bbox_dist = torch.cat([left_dist, up_dist, right_dist, bottom_dist], dim=1)

    #     # bbox_pred_refine = self.integral(bbox_dist)

    #     # cls_score = self.gfl_cls(cls_feat).sigmoid()
    #     cls_score = self.gfl_cls(self.gfl_conf_0(cls_feat, pts - base_offset)).sigmoid()

    #     if self.with_DGQP:
    #         N, _, H, W = bbox_dist.size()
    #         prob = F.softmax(bbox_dist.reshape(N, 4, self.bins, H, W), dim=2)
    #         cls_score = cls_score * self.reg_conf(prob.reshape(N, -1, H, W))

    #     return cls_score, bbox_pred_initial, bbox_pred_refine
        


        

        # pts_dist = self.gfl_reg(reg_feat) # B, P2b, H, W
        # pts = self.integral(pts_dist) + offset

        # bbox_pred, left_inds, right_inds, up_inds, bottom_inds = self.points2distance(pts) # B, 1, H, W

        # pts_dist_y = pts_dist.reshape(pts_dist.shape[0], -1, 2, self.bins, *pts_dist.shape[2:])[:, :, 0, ...].reshape(pts_dist.shape[0], -1, *pts_dist.shape[2:]) # B, Pb, H, W
        # pts_dist_x = pts_dist.reshape(pts_dist.shape[0], -1, 2, self.bins, *pts_dist.shape[2:])[:, :, 1, ...].reshape(pts_dist.shape[0], -1, *pts_dist.shape[2:])

        # left_inds = left_inds * self.bins + torch.arange(self.bins).view(1, -1, 1, 1).type_as(left_inds)
        # right_inds = right_inds * self.bins + torch.arange(self.bins).view(1, -1, 1, 1).type_as(left_inds)
        # up_inds = up_inds * self.bins + torch.arange(self.bins).view(1, -1, 1, 1).type_as(left_inds)
        # bottom_inds = bottom_inds * self.bins + torch.arange(self.bins).view(1, -1, 1, 1).type_as(left_inds)

        # left_dist = torch.gather(pts_dist_x, dim=1, index=left_inds)
        # right_dist = torch.gather(pts_dist_x, dim=1, index=right_inds)
        # up_dist = torch.gather(pts_dist_y, dim=1, index=up_inds)
        # bottom_dist = torch.gather(pts_dist_y, dim=1, index=bottom_inds)

        # bbox_dist = torch.cat([left_dist, right_dist, up_dist, bottom_dist], dim=1)
        

        

        # reg_feat_x, reg_feat_y = reg_feat.chunk(2, dim=1)

        # bbox_dist_x = self.gfl_reg_x(reg_feat_x)
        # bbox_dist_l, bbox_dist_r = bbox_dist_x.chunk(2, dim=1)

        # bbox_dist_y = self.gfl_reg_y(reg_feat_y)
        # bbox_dist_u, bbox_dist_b = bbox_dist_y.chunk(2, dim=1)


        # bbox_dist = torch.cat([bbox_dist_l, bbox_dist_u, bbox_dist_r, bbox_dist_b], dim=1)

        # bbox_pred_refine = self.integral(bbox_dist) + bbox_pred_initial

        

        # if self.with_DGQP:
        #     N, _, H, W = bbox_pred_refine.size()
        #     prob = F.softmax(bbox_pred_refine.reshape(N, 4, self.bins, H, W), dim=2)
        #     cls_score = cls_score * self.reg_conf(prob.reshape(N, -1, H, W))

        # if self.with_DGQP:
        #     N, _, H, W = bbox_pred_refine.size()
        #     prob = F.softmax(bbox_pred_refine.reshape(N, 4, self.bins, H, W), dim=2)
        #     cls_score = cls_score * self.reg_conf(prob.reshape(N, -1, H, W))

       
        # bbox_pred = self.points2distance(self.integral(pts) + offset)


    def anchor_center(self, anchors):
        anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
        anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2
        return torch.stack([anchors_cx, anchors_cy], dim=-1)


    def loss_single(self, anchors, cls_score, bbox_pred_initial, bbox_pred_refine, labels, label_weights,
                    bbox_targets, stride, num_total_samples):
    # def loss_single(self, anchors, cls_score, bbox_pred, labels, label_weights,
    #                 bbox_targets, stride, num_total_samples):
        assert stride[0] == stride[1], 'h stride is not equal to w stride!'
        anchors = anchors.reshape(-1, 4)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        bbox_pred_initial = bbox_pred_initial.permute(0, 2, 3, 1).reshape(-1, 4)
        # bbox_pred_refine = bbox_pred_refine.permute(0, 2, 3, 1).reshape(-1, 4 * self.bins)
        bbox_pred_refine = bbox_pred_refine.permute(0, 2, 3, 1).reshape(-1, 4)

        # bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)


        bbox_targets = bbox_targets.reshape(-1, 4)
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)

        # FG cat_id: [0, num_classes -1], BG cat_id: num_classes
        bg_class_ind = self.num_classes
        pos_inds = ((labels >= 0)
                    & (labels < bg_class_ind)).nonzero(as_tuple=False).squeeze(-1)
        score = label_weights.new_zeros(labels.shape)

        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets[pos_inds]
            pos_bbox_pred_initial_corners = bbox_pred_initial[pos_inds]
            pos_bbox_pred_refine = bbox_pred_refine[pos_inds]
            # pos_bbox_pred_corners = bbox_pred[pos_inds]


            pos_anchors = anchors[pos_inds]
            pos_anchor_centers = self.anchor_center(pos_anchors) / stride[0]

            weight_targets = cls_score.detach()
            weight_targets = weight_targets.max(dim=1)[0][pos_inds]

            # pos_bbox_pred_refine_corners = self.integral(pos_bbox_pred_refine)
            pos_bbox_pred_refine_corners = pos_bbox_pred_refine

            pos_decode_bbox_pred = distance2bbox(pos_anchor_centers,
                                                 (pos_bbox_pred_initial_corners + pos_bbox_pred_refine_corners))
            # pos_decode_bbox_pred = distance2bbox(pos_anchor_centers, pos_bbox_pred_corners)

            pos_decode_bbox_targets = pos_bbox_targets / stride[0]

            score[pos_inds] = bbox_overlaps(
                pos_decode_bbox_pred.detach(),
                pos_decode_bbox_targets,
                is_aligned=True).clamp(min=0.0)

            loss_bbox = self.loss_bbox(
                    distance2bbox(pos_anchor_centers,
                                  (pos_bbox_pred_initial_corners + pos_bbox_pred_refine_corners)),
                    pos_decode_bbox_targets,
                    weight=weight_targets,
                    avg_factor=1.0)
            # loss_bbox = self.loss_bbox(
            #         distance2bbox(pos_anchor_centers, pos_bbox_pred_corners),
            #         pos_decode_bbox_targets,
            #         weight=weight_targets,
            #         avg_factor=1.0)

            # loss_bbox_2 = self.loss_bbox(
            #         distance2bbox(pos_anchor_centers,
            #                       pos_bbox_pred_initial_corners),
            #         pos_decode_bbox_targets,
            #         weight=weight_targets,
            #         avg_factor=1.0) * 0.0

        else:
            loss_bbox = bbox_pred_initial.sum() * 0.0
            # loss_bbox_2 = bbox_pred_initial.sum() * 0.0
            # loss_bbox = bbox_pred.sum() * 0.0
            weight_targets = torch.tensor(0).cuda()


        loss_cls = self.loss_cls(
            cls_score, (labels, score),
            weight=label_weights,
            avg_factor=num_total_samples)

        return loss_cls, loss_bbox, weight_targets.sum()


    @force_fp32(apply_to=('cls_scores', 'bbox_preds_initial', 'bbox_preds_refine'))
    # @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def loss(self,
             cls_scores,
             bbox_preds_initial,
             bbox_preds_refine,
            #  bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
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
            featmap_sizes, img_metas, device=device)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1

        cls_reg_targets = self.get_targets(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels)
        if cls_reg_targets is None:
            return None

        (anchor_list, labels_list, label_weights_list, bbox_targets_list,
         bbox_weights_list, num_total_pos, num_total_neg) = cls_reg_targets

        num_total_samples = reduce_mean(
            torch.tensor(num_total_pos, dtype=torch.float,
                         device=device)).item()
        num_total_samples = max(num_total_samples, 1.0)

        losses_cls, losses_bbox, \
            avg_factor = multi_apply(
                self.loss_single,
                anchor_list,
                cls_scores,
                bbox_preds_initial,
                bbox_preds_refine,
                # bbox_preds,
                labels_list,
                label_weights_list,
                bbox_targets_list,
                self.anchor_generator.strides,
                num_total_samples=num_total_samples)

        avg_factor = sum(avg_factor) 
        avg_factor = reduce_mean(avg_factor).item()
        losses_bbox = list(map(lambda x: x / avg_factor, losses_bbox))
        # losses_bbox_2 = list(map(lambda x: x / avg_factor, losses_bbox_2))


        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)


    @force_fp32(apply_to=('cls_scores', 'bbox_preds_initial', 'bbox_preds_refine', 'offset', 'mask'))
    # @force_fp32(apply_to=('cls_scores', 'bbox_preds'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds_initial,
                   bbox_preds_refine,
                #    bbox_preds,
                #    offset,
                #    mask,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        assert len(cls_scores) == len(bbox_preds_initial) == len(bbox_preds_refine)
        num_levels = len(cls_scores)

        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
        mlvl_anchors = self.anchor_generator.grid_anchors(
            featmap_sizes, device=device)

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_initial_list = [
                bbox_preds_initial[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_refine_list = [
                bbox_preds_refine[i][img_id].detach() for i in range(num_levels)
            ]
            # bbox_pred_list = [
            #     bbox_preds[i][img_id].detach() for i in range(num_levels)
            # ]

            # offset_list = [
            #     offset[i][img_id].detach() for i in range(num_levels)
            # ]
            # mask_list = [
            #     mask[i][img_id].detach() for i in range(num_levels)
            # ]
           
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            if with_nms:
                # some heads don't support with_nms argument
                proposals = self._get_bboxes_single(cls_score_list,
                                                    bbox_pred_initial_list,
                                                    bbox_pred_refine_list,
                                                    # bbox_pred_list,
                                                    # offset_list,
                                                    # mask_list,
                                                    mlvl_anchors, img_shape,
                                                    scale_factor, cfg, rescale)
            else:
                proposals = self._get_bboxes_single(cls_score_list,
                                                    bbox_pred_initial_list,
                                                    bbox_pred_refine_list,
                                                    # bbox_pred_list,
                                                    # offset_list,
                                                    # mask_list,
                                                    mlvl_anchors, img_shape,
                                                    scale_factor, cfg, rescale,
                                                    with_nms)
            result_list.append(proposals)
        return result_list

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds_initial,
                           bbox_preds_refine,
                        #    bbox_preds,
                        #    offsets,
                        #    masks,
                           mlvl_anchors,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False,
                           with_nms=True):
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
        assert len(cls_scores) == len(bbox_preds_initial) == len(bbox_preds_refine) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_distances = []
        mlvl_scores = []
        # mlvl_offsets = []
        # mlvl_masks = []
        for cls_score, bbox_pred_initial, bbox_pred_refine, stride, anchors in zip(
                cls_scores, bbox_preds_initial, bbox_preds_refine, self.anchor_generator.strides,
                mlvl_anchors):
        # for cls_score, bbox_pred, offset, mask, stride, anchors in zip(
        #         cls_scores, bbox_preds, offsets, masks, self.anchor_generator.strides,
        #         mlvl_anchors):
        # for cls_score, bbox_pred, stride, anchors in zip(
        #         cls_scores, bbox_preds, self.anchor_generator.strides,
        #         mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred_initial.size()[-2:] == bbox_pred_refine.size()[-2:]
            assert stride[0] == stride[1]

            scores = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels)

            # offset = offset.permute(1, 2, 0).reshape(-1, self.points*2) * stride[0]
            # mask = mask.permute(1, 2, 0).reshape(-1, 4)
            
            bbox_pred = (bbox_pred_initial.permute(1, 2, 0).reshape(-1, 4) + bbox_pred_refine.permute(1, 2, 0).reshape(-1, 4)) * stride[0]
            # bbox_pred = (bbox_pred_initial.permute(1, 2, 0).reshape(-1, 4) + self.integral(bbox_pred_refine.permute(1, 2, 0))) * stride[0]
            # bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4) * stride[0]
            

            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = scores.max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                # offset = offset[topk_inds, :]
                # mask = mask[topk_inds, :]

            bboxes = distance2bbox(
                self.anchor_center(anchors), bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_distances.append(bbox_pred)
            mlvl_scores.append(scores)
            # mlvl_offsets.append(offset)
            # mlvl_masks.append(mask)

        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)

        mlvl_distances = torch.cat(mlvl_distances) 
        mlvl_scores = torch.cat(mlvl_scores)
        # mlvl_offsets = torch.cat(mlvl_offsets)
        # mlvl_masks = torch.cat(mlvl_masks)
        # Add a dummy background class to the backend when using sigmoid
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)

        if with_nms:
            det_bboxes, det_labels, keep = multiclass_nms(mlvl_bboxes, mlvl_scores,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img, return_inds=True)
            # return det_bboxes, det_labels, mlvl_offsets[keep, :], mlvl_masks[keep, :]

            '''
            valid_mask = mlvl_scores > cfg.score_thr
            inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
            print(det_bboxes[0])
            print(det_labels[0])
            print(mlvl_distances[inds][keep][0, :])
            print(mlvl_offsets[inds][keep][0, :])
            print(mlvl_masks[inds][keep][0, :])
            print('----------------')
            '''

            return det_bboxes, det_labels
        else:
            # return mlvl_bboxes, mlvl_scores, mlvl_offsets, mlvl_masks
            return mlvl_bboxes, mlvl_scores


    def get_targets(self,
                    anchor_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    label_channels=1,
                    unmap_outputs=True):
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
        (all_anchors, all_labels, all_label_weights, all_bbox_targets,
         all_bbox_weights, pos_inds_list, neg_inds_list) = multi_apply(
             self._get_target_single,
             anchor_list,
             valid_flag_list,
             num_level_anchors_list,
             gt_bboxes_list,
             gt_bboxes_ignore_list,
             gt_labels_list,
             img_metas,
             label_channels=label_channels,
             unmap_outputs=unmap_outputs)
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        anchors_list = images_to_levels(all_anchors, num_level_anchors)
        labels_list = images_to_levels(all_labels, num_level_anchors)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_anchors)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_anchors)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_anchors)
        return (anchors_list, labels_list, label_weights_list,
                bbox_targets_list, bbox_weights_list, num_total_pos,
                num_total_neg)

    def _get_target_single(self,
                           flat_anchors,
                           valid_flags,
                           num_level_anchors,
                           gt_bboxes,
                           gt_bboxes_ignore,
                           gt_labels,
                           img_meta,
                           label_channels=1,
                           unmap_outputs=True):
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
        inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
                                           img_meta['img_shape'][:2],
                                           self.train_cfg.allowed_border)
        if not inside_flags.any():
            return (None, ) * 7
        # assign gt and sample anchors
        anchors = flat_anchors[inside_flags, :]

        num_level_anchors_inside = self.get_num_level_anchors_inside(
            num_level_anchors, inside_flags)
        assign_result = self.assigner.assign(anchors, num_level_anchors_inside,
                                             gt_bboxes, gt_bboxes_ignore,
                                             gt_labels)

        sampling_result = self.sampler.sample(assign_result, anchors,
                                              gt_bboxes)

        num_valid_anchors = anchors.shape[0]
        bbox_targets = torch.zeros_like(anchors)
        bbox_weights = torch.zeros_like(anchors)
        labels = anchors.new_full((num_valid_anchors, ),
                                  self.num_classes,
                                  dtype=torch.long)
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
                labels[pos_inds] = gt_labels[
                    sampling_result.pos_assigned_gt_inds]
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
                labels, num_total_anchors, inside_flags, fill=self.num_classes)
            label_weights = unmap(label_weights, num_total_anchors,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_anchors, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_anchors, inside_flags)

        return (anchors, labels, label_weights, bbox_targets, bbox_weights,
                pos_inds, neg_inds)

    def get_num_level_anchors_inside(self, num_level_anchors, inside_flags):
        split_inside_flags = torch.split(inside_flags, num_level_anchors)
        num_level_anchors_inside = [
            int(flags.sum()) for flags in split_inside_flags
        ]
        return num_level_anchors_inside


def topk_unsorted(x, k=1, dim=0, largest=True):
    val, idx = torch.topk(x, k, dim=dim, largest=largest)
    sorted_idx, new_idx = torch.sort(idx, dim=dim)
    val = torch.gather(val, dim=dim, index=new_idx)

    return val, sorted_idx
