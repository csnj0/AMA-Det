from collections import OrderedDict
from numpy.lib.polynomial import polyint
from torch.functional import norm 
from torch.nn.modules import padding
from torch.nn.modules.activation import ReLU, Sigmoid
from torch.nn.modules.fold import Unfold
from mmdet.core.bbox.transforms import bbox2distance
from mmcv.cnn.utils.weight_init import constant_init
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, Scale, bias_init_with_prob, normal_init, build_norm_layer
from mmcv.runner import force_fp32
from mmcv.ops import DeformConv2d, ModulatedDeformConv2d
from mmdet.core import (bbox, bbox_overlaps,
                        build_assigner, build_sampler, distance2bbox,
                        images_to_levels, multi_apply, multiclass_nms, post_processing,
                        reduce_mean, unmap, PointGenerator)
from ..builder import HEADS, build_loss
from .anchor_free_head import AnchorFreeHead

import numpy as np

class Integral(nn.Module):
    def __init__(self, start=-2.0, stop=2.0, bins=5):
        super(Integral, self).__init__()
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


class DeformConv2dModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, groups=1, deform_groups=1, norm_cfg=None):
        super(DeformConv2dModule, self).__init__()
        self.conv = DeformConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, 
                                 groups=groups, deform_groups=deform_groups)
        self.norm_cfg = norm_cfg
        if self.norm_cfg is not None:
            self.norm_name, norm = build_norm_layer(norm_cfg, out_channels)
            self.add_module(self.norm_name, norm)
            constant_init(self.norm, 1, bias=0)
        self.act = nn.ReLU(inplace=True)

    @property
    def norm(self):
        return getattr(self, self.norm_name)

    def forward(self, x, offset):
        x = self.conv(x, offset)
        if self.norm_cfg is not None:
            x = self.norm(x)
        x = self.act(x)
        return x


class DualDeformModule(nn.Module):
    def __init__(self, channels, points=9, norm_cfg=None):
        super(DualDeformModule, self).__init__()
        self.points = points
        self.base_offset = self.get_base_offset(points)
        self.channels = channels

        self.cls_conv = DeformConv2dModule(channels, channels, [1, self.points], stride=1, padding=[0, self.points//2], norm_cfg=norm_cfg)
        self.reg_conv = DeformConv2dModule(channels, channels, [1, self.points], stride=1, padding=[0, self.points//2], norm_cfg=norm_cfg)

        # self.cls_conv = DeformConv2dModule(channels, channels, [1, self.points], stride=1, padding=[0, self.points//2], norm_cfg=None)
        # self.reg_conv = DeformConv2dModule(channels, channels, [1, self.points], stride=1, padding=[0, self.points//2], norm_cfg=None)
        # self.cls_conv = DeformConv2dModule(channels, channels, 3, stride=1, padding=1, norm_cfg=None)
        # self.reg_conv = DeformConv2dModule(channels, channels, 3, stride=1, padding=1, norm_cfg=None)

        # self.cls_conv = DeformConv2dModule(channels, channels, 3, stride=1, padding=1, norm_cfg=norm_cfg)
        # self.reg_conv = DeformConv2dModule(channels, channels, 3, stride=1, padding=1, norm_cfg=norm_cfg)

        self.offset_conv = ConvModule(channels, channels, 3, padding=1, norm_cfg=norm_cfg)
        self.offset_out = nn.Conv2d(channels, self.points * 2 * 16, 3, padding=1)

        # self.offset_conv = ConvModule(channels, channels, 3, padding=1, norm_cfg=None)
        # self.offset_out = nn.Conv2d(channels, self.points * 2, 1)

        normal_init(self.offset_conv.conv, std=0.01)
        normal_init(self.offset_out, std=0.01)

        self.integral = Integral(0, 15, bins=16)

        # self.affiliate_conv = ConvModule(channels, channels, 3, padding=1, norm_cfg=None)
        # self.affiliate_out = nn.Conv2d(channels, self.points * 2, 1)
        # normal_init(self.affiliate_conv.conv, std=0.01)
        # normal_init(self.affiliate_out, std=0.01)

    def get_base_offset(self, points):
        dcn_base_x = np.arange(-((points-1)//2), points//2+1).astype(np.float64)
        dcn_base_y = dcn_base_x * 0.0
        dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=1).reshape((-1))
        dcn_base_offset = torch.tensor(dcn_base_offset).view(1, -1, 1, 1)
        return dcn_base_offset

    # def get_base_offset(self, points):
    #     dcn_kernel = int(np.sqrt(points))
    #     dcn_pad = int((dcn_kernel - 1) / 2)
    #     dcn_base = np.arange(-dcn_pad, dcn_pad + 1).astype(np.float64)
    #     dcn_base_y = np.repeat(dcn_base, dcn_kernel)
    #     dcn_base_x = np.tile(dcn_base, dcn_kernel)
    #     dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=1).reshape((-1))
    #     dcn_base_offset = torch.tensor(dcn_base_offset).view(1, -1, 1, 1)
    #     return dcn_base_offset

    def forward(self, cls_feat, reg_feat):
        base_offset = self.base_offset.type_as(reg_feat)
        pts_dist = self.offset_out(self.offset_conv(reg_feat))
        # pts_aff = self.affiliate_out(self.affiliate_conv(reg_feat))

        # pts_mul = 0.1 * pts + 0.9 * pts.detach()
        pts = self.integral(pts_dist)

        cls_feat = self.cls_conv(cls_feat, pts - base_offset)
        reg_feat = self.reg_conv(reg_feat, pts - base_offset)

        return pts, pts_dist, cls_feat, reg_feat


@HEADS.register_module()
class SERIALGFLHead(AnchorFreeHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 bins=5,
                 points=9,
                 DGQP_cfg=dict(channels=64),
                 train_cfg=None,
                 test_cfg=None,
                 loss_cls=None,
                 loss_bbox_init=None,
                 loss_bbox_refine=None,
                 **kwargs):

        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        
        # self.bins = bins
        self.bins = 16


        self.points = points
        self.point_strides = [8, 16, 32, 64, 128]
        self.point_base_scale = 8

        self.DGQP_cfg = DGQP_cfg
        self.with_DGQP = DGQP_cfg is not None

        super(SERIALGFLHead, self).__init__(num_classes, in_channels, 
            conv_cfg=conv_cfg, norm_cfg=norm_cfg, train_cfg=train_cfg, test_cfg=test_cfg, **kwargs)

        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox_init = build_loss(loss_bbox_init)
        self.loss_bbox_refine = build_loss(loss_bbox_refine)


        self.base_offset = self.get_base_offset(points)
        self.point_generators = [PointGenerator() for _ in self.point_strides]

        # self.moment_transfer = nn.Parameter(
        #         data=torch.zeros(2), requires_grad=True)
        # self.moment_mul = 0.01

        self.sampling = False
        if self.train_cfg:
            # self.assigner = build_assigner(self.train_cfg.assigner)
            self.init_assigner = build_assigner(self.train_cfg.init.assigner)
            self.refine_assigner = build_assigner(self.train_cfg.refine.assigner)

            # SSD sampling=False so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)
        

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

        # assert self.num_anchors == 1, 'anchor free version'

        self.gfl_dual = DualDeformModule(self.feat_channels, points=self.points, norm_cfg=self.norm_cfg)

        self.gfl_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 3, padding=1)
        # self.gfl_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 1)

        self.gfl_reg = nn.Conv2d(self.feat_channels, 4 * self.bins, 3, padding=1)
        # self.gfl_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        # self.gfl_reg = nn.Conv2d(self.feat_channels, 4, 1)
        # self.gfl_reg = nn.Conv2d(self.feat_channels, self.points * 2, 1)
        # self.gfl_reg = nn.Conv2d(self.feat_channels, self.points * 2, 3, padding=1)


        self.scales = nn.ModuleList(
            [Scale(1.0) for _ in self.point_strides])
        
        self.integral = Integral(0, self.bins-1, bins=self.bins)
        # self.integral = Integral(-(self.bins//2), self.bins//2, bins=self.bins)


        if self.with_DGQP:
            self.reg_conf = nn.Sequential(
                # nn.Conv2d(4 * self.bins, self.DGQP_cfg['channels'], 1),
                nn.Conv2d(4 * 5, self.DGQP_cfg['channels'], 1),
                # nn.Conv2d(4 * 10, self.DGQP_cfg['channels'], 1),
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

        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.gfl_cls, std=0.01, bias=bias_cls)
        normal_init(self.gfl_reg, std=0.01)

        if self.with_DGQP:
            for m in self.reg_conf:
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)


    def points2distance(self, pts):
        pts_reshape = pts.view(pts.shape[0], -1, 2, *pts.shape[2:])
        pts_y = pts_reshape[:, :, 0, ...]
        pts_x = pts_reshape[:, :, 1, ...]
        bbox_left, left_indx = pts_x.min(dim=1, keepdim=True)
        bbox_right, right_indx = pts_x.max(dim=1, keepdim=True)
        bbox_up, up_indx = pts_y.min(dim=1, keepdim=True)
        bbox_bottom, bottom_indx = pts_y.max(dim=1, keepdim=True)

        bbox = torch.cat([-bbox_left, -bbox_up, bbox_right, bbox_bottom],
                            dim=1)
        return bbox, left_indx, right_indx, up_indx, bottom_indx

    # def points2distance(self, pts):
    #     pts_reshape = pts.view(pts.shape[0], -1, 2, *pts.shape[2:])
    #     pts_y = pts_reshape[:, :, 0, ...]
    #     pts_x = pts_reshape[:, :, 1, ...]
    #     pts_y_mean = pts_y.mean(dim=1, keepdim=True)
    #     pts_x_mean = pts_x.mean(dim=1, keepdim=True)
    #     pts_y_std = torch.std(pts_y - pts_y_mean, dim=1, keepdim=True)
    #     pts_x_std = torch.std(pts_x - pts_x_mean, dim=1, keepdim=True)
    #     moment_transfer = (self.moment_transfer * self.moment_mul) + (
    #         self.moment_transfer.detach() * (1 - self.moment_mul))
    #     moment_width_transfer = moment_transfer[0]
    #     moment_height_transfer = moment_transfer[1]
    #     half_width = pts_x_std * torch.exp(moment_width_transfer)
    #     half_height = pts_y_std * torch.exp(moment_height_transfer)
    #     bbox = torch.cat([
    #         half_width - pts_x_mean, half_height - pts_y_mean,
    #         pts_x_mean + half_width, pts_y_mean + half_height
    #     ], dim=1)
    #     return bbox


    def forward(self, feats):
        return multi_apply(self.forward_single, feats, self.scales)

    def forward_single(self, x, scale):
        B, _, H, W = x.shape
        cls_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)

        reg_feat = x + cls_feat
        # reg_feat = x
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        
        pts, pts_dist, cls_feat, reg_feat = self.gfl_dual(cls_feat, reg_feat)

        # bbox_pred_init = self.points2distance(pts)
        bbox_pred_init, left_indx, right_indx, up_indx, bottom_indx = self.points2distance(pts)

        pts_dist_reshape = pts_dist.view(B, -1, 2, 16, H, W)
        pts_dist_y = pts_dist_reshape[:, :, 0, ...].reshape(B, -1, H, W)
        pts_dist_x = pts_dist_reshape[:, :, 1, ...].reshape(B, -1, H, W)
        left_dist = torch.gather(pts_dist_x, dim=1, 
            index=left_indx * 16 + torch.arange(16).type_as(left_indx)[None, :, None, None])
        right_dist = torch.gather(pts_dist_x, dim=1, 
            index=right_indx * 16 + torch.arange(16).type_as(right_indx)[None, :, None, None])
        up_dist = torch.gather(pts_dist_y, dim=1, 
            index=up_indx * 16 + torch.arange(16).type_as(up_indx)[None, :, None, None])
        bottom_dist = torch.gather(pts_dist_y, dim=1, 
            index=bottom_indx * 16 + torch.arange(16).type_as(bottom_indx)[None, :, None, None])
        initial_dist = torch.cat([left_dist, up_dist, right_dist, bottom_dist], dim=1)

        # bbox_pred_refine = self.points2distance(self.gfl_reg(reg_feat) + pts.detach())
        # bbox_pred_refine = self.points2distance(self.gfl_reg(reg_feat) + pts)


        bbox_pred_dist = self.gfl_reg(reg_feat)
        bbox_pred_refine = self.integral(bbox_pred_dist) + bbox_pred_init


        # bbox_pred_init = self.points2distance(pts).float()
        # bbox_pred_dist = self.gfl_reg(reg_feat)

        # bbox_pred_refine = scale(self.integral(bbox_pred_dist)).float() + bbox_pred_init.detach()
        # bbox_pred_refine = scale(self.integral(bbox_pred_dist)).float() + bbox_pred_init

        # bbox_pred_refine = scale(bbox_pred_dist).float() + bbox_pred_init.detach()
        # bbox_pred_refine = scale(bbox_pred_dist).float() + bbox_pred_init

        # bbox_pred_refine = bbox_pred_dist.float() + bbox_pred_init.detach()

        
        cls_score = self.gfl_cls(cls_feat).sigmoid()
        # cls_score = self.gfl_cls(cls_feat)

        if self.with_DGQP:
            N, _, H, W = bbox_pred_dist.size()

            # prob = torch.cat([bbox_pred_dist.reshape(N, 4, self.bins, H, W).softmax(dim=2).type_as(x),
            #                   initial_dist.reshape(N, 4, self.bins, H, W).softmax(dim=2).type_as(x)], dim=1)

            prob = bbox_pred_dist.reshape(N, 4, self.bins, H, W).softmax(dim=2).type_as(x)

            # prob = bbox_pred_dist.reshape(N, 4, self.bins, H, W)
            # prob = F.softmax(prob, dim=2).type_as(x)
            prob = topk_unsorted(prob, k=5, dim=2)[0]
            cls_score = cls_score * self.reg_conf(prob.reshape(N, -1, H, W))

        return cls_score, bbox_pred_init, bbox_pred_refine


    def get_points(self, featmap_sizes, img_metas, device):
        """Get points according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.

        Returns:
            tuple: points of each image, valid flags of each image
        """
        num_imgs = len(img_metas)
        num_levels = len(featmap_sizes)

        # since feature map sizes of all images are the same, we only compute
        # points center for one time
        multi_level_points = []
        for i in range(num_levels):
            points = self.point_generators[i].grid_points(
                featmap_sizes[i], self.point_strides[i], device)
            multi_level_points.append(points)
        points_list = [[point.clone() for point in multi_level_points]
                       for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level grids
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = []
            for i in range(num_levels):
                point_stride = self.point_strides[i]
                feat_h, feat_w = featmap_sizes[i]
                h, w = img_meta['pad_shape'][:2]
                valid_feat_h = min(int(np.ceil(h / point_stride)), feat_h)
                valid_feat_w = min(int(np.ceil(w / point_stride)), feat_w)
                flags = self.point_generators[i].valid_flags(
                    (feat_h, feat_w), (valid_feat_h, valid_feat_w), device)
                multi_level_flags.append(flags)
            valid_flag_list.append(multi_level_flags)

        return points_list, valid_flag_list


    def loss_single(self, cls_score, center,
                    bbox_pred_init, labels_init, label_weights_init, 
                    bbox_targets_init,
                    bbox_pred_refine, labels_refine, label_weights_refine, 
                    bbox_targets_refine, 
                    num_total_samples_init=1,
                    num_total_samples_refine=1):
        
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        center_xy, center_stride = center.reshape(-1, 3).split([2, 1], dim=1)
        
        bbox_pred_init = bbox_pred_init.permute(0, 2, 3, 1).reshape(-1, 4)
        labels_init = labels_init.reshape(-1)
        label_weights_init = label_weights_init.reshape(-1)
        bbox_targets_init = bbox_targets_init.reshape(-1, 4)

        bbox_pred_refine = bbox_pred_refine.permute(0, 2, 3, 1).reshape(-1, 4)
        labels_refine = labels_refine.reshape(-1)
        label_weights_refine = label_weights_refine.reshape(-1)
        bbox_targets_refine = bbox_targets_refine.reshape(-1, 4)

        

        # init stage:
        bg_class_ind = self.num_classes
        pos_inds = ((labels_init >= 0)
                    & (labels_init < bg_class_ind)).nonzero(as_tuple=False).squeeze(-1)
        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets_init[pos_inds] / center_stride[pos_inds]
            pos_bbox_preds = distance2bbox(center_xy[pos_inds] / center_stride[pos_inds], 
                                                 bbox_pred_init[pos_inds])

            # loss_bbox_init = self.loss_bbox_init(
            #         pos_bbox_preds / 4.0,
            #         pos_bbox_targets / 4.0,
            #         avg_factor=num_total_samples_init)
            # loss_bbox_init = self.loss_bbox_init(
            #         pos_bbox_preds,
            #         pos_bbox_targets,
            #         avg_factor=num_total_samples_init)

            weight_targets = cls_score.detach()
            weight_targets = weight_targets.max(dim=1)[0][pos_inds]

            loss_bbox_init = self.loss_bbox_init(
                    pos_bbox_preds,
                    pos_bbox_targets,
                    weight=weight_targets,
                    avg_factor=1)
        else:
            loss_bbox_init = bbox_pred_init.sum() * 0.0

        # refine stage:
        bg_class_ind = self.num_classes
        pos_inds = ((labels_refine >= 0)
                    & (labels_refine < bg_class_ind)).nonzero(as_tuple=False).squeeze(-1)
        score = label_weights_refine.new_zeros(labels_refine.shape)
        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets_refine[pos_inds] / center_stride[pos_inds]
            pos_bbox_preds = distance2bbox(center_xy[pos_inds]/center_stride[pos_inds], 
                                                 bbox_pred_refine[pos_inds])
            
            weight_targets = cls_score.detach()
            weight_targets = weight_targets.max(dim=1)[0][pos_inds]

            score[pos_inds] = bbox_overlaps(
                pos_bbox_preds.detach(),
                pos_bbox_targets,
                is_aligned=True).clamp(min=0.0)

            loss_bbox_refine = self.loss_bbox_refine(
                    pos_bbox_preds,
                    pos_bbox_targets,
                    weight=weight_targets,
                    avg_factor=1)
            # loss_bbox_refine = self.loss_bbox_refine(
            #         pos_bbox_preds / 4.0,
            #         pos_bbox_targets / 4.0,
            #         avg_factor=num_total_samples_refine)
            # loss_bbox_refine = self.loss_bbox_refine(
            #         pos_bbox_preds,
            #         pos_bbox_targets,
            #         avg_factor=num_total_samples_refine)
        else:
            loss_bbox_refine = bbox_pred_refine.sum() * 0.0
            weight_targets = torch.tensor(0.0).cuda()
        
        loss_cls = self.loss_cls(
            cls_score, (labels_refine, score),
            weight=label_weights_refine,
            avg_factor=num_total_samples_refine)

        # loss_cls = self.loss_cls(
        #     cls_score, labels_refine,
        #     weight=label_weights_refine,
        #     avg_factor=num_total_samples_refine)

        # return loss_cls, loss_bbox_init, loss_bbox_refine, weight_targets.sum()
        return loss_cls, loss_bbox_init * 0.0, loss_bbox_refine, weight_targets.sum()


    
    def centers_to_bboxes(self, point_list):
        """Get bboxes according to center points.

        Only used in :class:`MaxIoUAssigner`.
        """
        bbox_list = []
        for i_img, point in enumerate(point_list):
            bbox = []
            for i_lvl in range(len(self.point_strides)):
                scale = self.point_base_scale * self.point_strides[i_lvl] * 0.5
                bbox_shift = torch.Tensor([-scale, -scale, scale,
                                           scale]).view(1, 4).type_as(point[0])
                bbox_center = torch.cat(
                    [point[i_lvl][:, :2], point[i_lvl][:, :2]], dim=1)
                bbox.append(bbox_center + bbox_shift)
            bbox_list.append(bbox)
        return bbox_list


    @force_fp32(apply_to=('cls_scores', 'bbox_preds_init', 'bbox_preds_refine'))
    def loss(self,
             cls_scores,
             bbox_preds_init,
             bbox_preds_refine,
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
        device = cls_scores[0].device
        # label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        label_channels = 1

        # target for initial stage:
        center_list, valid_flag_list = self.get_points(
            featmap_sizes, img_metas, device=device)

        bbox_list = self.centers_to_bboxes(center_list) # list{batch}-list{level}-tensor(num, 4)

        cls_reg_targets_init = self.get_targets(
            bbox_list,
            # center_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            stage='init',
            label_channels=label_channels)
        (_, labels_init_list, label_weights_init_list,
                bbox_targets_init_list, bbox_weights_init_list, num_total_pos_init,
                num_total_neg_init) = cls_reg_targets_init
        

        num_total_samples_init = reduce_mean(
            torch.tensor(num_total_pos_init, dtype=torch.float,
                         device=device)).item()
        num_total_samples_init = max(num_total_samples_init, 1.0)
 
        # num_total_samples_init = (
        #     num_total_pos_init +
        #     num_total_neg_init if self.sampling else num_total_pos_init)

        # num_total_samples_init = num_total_pos_init

        # target for refine stage:
        center_list, valid_flag_list = self.get_points(
            featmap_sizes, img_metas, device=device)
        '''
        bbox_list = []
        for i_img, center in enumerate(center_list):
            bbox = []
            for i_lvl in range(len(bbox_preds_init)):
                bbox_shift = bbox_preds_init[i_lvl].detach() * self.point_strides[i_lvl] * torch.tensor([-1.0, -1.0, 1.0, 1.0], device=device)[None, :, None, None]
                bbox_center = torch.cat(
                    [center[i_lvl][:, :2], center[i_lvl][:, :2]], dim=1)
                bbox.append(bbox_center +
                            bbox_shift[i_img].permute(1, 2, 0).reshape(-1, 4))
            bbox_list.append(bbox)
        '''

        bbox_list = self.centers_to_bboxes(center_list) # list{batch}-list{level}-tensor(num, 4)

        cls_reg_targets_refine = self.get_targets(
            bbox_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            stage='refine',
            label_channels=label_channels)
        (_, labels_refine_list, label_weights_refine_list,
                bbox_targets_refine_list, bbox_weights_refine_list, num_total_pos_refine,
                num_total_neg_refine) = cls_reg_targets_refine

        num_total_samples_refine = reduce_mean(
            torch.tensor(num_total_pos_refine, dtype=torch.float,
                         device=device)).item()
        num_total_samples_refine = max(num_total_samples_refine, 1.0)

        # num_total_samples_refine = (
        #     num_total_pos_refine +
        #     num_total_neg_refine if self.sampling else num_total_pos_refine)

        # num_total_samples_refine = num_total_pos_refine
    
    
        center_lvl_list = []
        for i_lvl in range(len(center_list[0])):
            center_batch = []
            for i_img in range(len(center_list)):
                center_batch.append(center_list[i_img][i_lvl])
            center_lvl_list.append(torch.cat(center_batch, dim=0))
        

        losses_cls, losses_bbox_init, losses_bbox_refine, avg_factor = multi_apply(
                self.loss_single,
                cls_scores, center_lvl_list,
                bbox_preds_init, labels_init_list, label_weights_init_list, 
                bbox_targets_init_list,
                bbox_preds_refine, labels_refine_list, label_weights_refine_list, 
                bbox_targets_refine_list, 
                num_total_samples_init=num_total_samples_init,
                num_total_samples_refine=num_total_samples_refine)

        avg_factor = sum(avg_factor)
        avg_factor = reduce_mean(avg_factor).item()
        losses_bbox_refine = list(map(lambda x: x / avg_factor, losses_bbox_refine))

        return dict(loss_cls=losses_cls, 
                    loss_bbox_init=losses_bbox_init, 
                    loss_bbox_refine=losses_bbox_refine)


    @force_fp32(apply_to=('cls_scores', 'bbox_preds_init', 'bbox_preds_refine'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds_init,
                   bbox_preds_refine,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        assert len(cls_scores) == len(bbox_preds_refine)
        num_levels = len(cls_scores)

        device = cls_scores[0].device
        featmap_sizes = [cls_scores[i].shape[-2:] for i in range(num_levels)]
      
        mlvl_points = []
        for i in range(num_levels):
            points = self.point_generators[i].grid_points(
                featmap_sizes[i], self.point_strides[i], device)
            mlvl_points.append(points)

        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_refine_list = [
                bbox_preds_refine[i][img_id].detach() for i in range(num_levels)
            ]
           
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            if with_nms:
                # some heads don't support with_nms argument
                proposals = self._get_bboxes_single(cls_score_list,
                                                    bbox_pred_refine_list,
                                                    mlvl_points, 
                                                    img_shape,
                                                    scale_factor, cfg, rescale)
            else:
                proposals = self._get_bboxes_single(cls_score_list,
                                                    bbox_pred_refine_list,
                                                    mlvl_points, 
                                                    img_shape,
                                                    scale_factor, cfg, rescale,
                                                    with_nms)
            result_list.append(proposals)
        return result_list

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds_refine,
                           mlvl_points,
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
        assert len(cls_scores) == len(bbox_preds_refine) == len(mlvl_points)
        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, stride, points in zip(
                cls_scores, bbox_preds_refine, self.point_strides,
                mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            # assert stride[0] == stride[1]

            # scores = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels)
            scores = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels).sigmoid()
      
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4) * stride

            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = scores.max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]

            bboxes = distance2bbox(
                points, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)

        mlvl_bboxes = torch.cat(mlvl_bboxes)

        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)

        mlvl_scores = torch.cat(mlvl_scores)

        # Add a dummy background class to the backend when using sigmoid
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)

        if with_nms:
            
            det_bboxes, det_labels = multiclass_nms(mlvl_bboxes, mlvl_scores,
                                                    cfg.score_thr, cfg.nms,
                                                    cfg.max_per_img, return_inds=False)
            return det_bboxes, det_labels
        else:
            return mlvl_bboxes, mlvl_scores


    def get_targets(self,
                    proposal_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    stage='init',
                    label_channels=1,
                    unmap_outputs=True):
        """Get targets for GFL head.

        This method is almost the same as `AnchorHead.get_targets()`. Besides
        returning the targets as the parent method does, it also returns the
        anchors as the first element of the returned tuple.
        """
        assert stage in ['init', 'refine']
        num_imgs = len(img_metas)
        assert len(proposal_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_proposals = [proposals.size(0) for proposals in proposal_list[0]]
        num_level_proposals_list = [num_level_proposals] * num_imgs

        # concat all level anchors and flags to a single tensor
        for i in range(num_imgs):
            assert len(proposal_list[i]) == len(valid_flag_list[i])
            proposal_list[i] = torch.cat(proposal_list[i])
            valid_flag_list[i] = torch.cat(valid_flag_list[i])

        # compute targets for each image
        if gt_bboxes_ignore_list is None:
            gt_bboxes_ignore_list = [None for _ in range(num_imgs)]
        if gt_labels_list is None:
            gt_labels_list = [None for _ in range(num_imgs)]
        (all_proposals, all_labels, all_label_weights, all_bbox_targets,
         all_bbox_weights, pos_inds_list, neg_inds_list) = multi_apply(
             self._get_target_single,
             proposal_list,
             valid_flag_list,
             num_level_proposals_list,
             gt_bboxes_list,
             gt_bboxes_ignore_list,
             gt_labels_list,
             img_metas,
             label_channels=label_channels,
             stage=stage,
             unmap_outputs=unmap_outputs)
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels
        proposals_list = images_to_levels(all_proposals, num_level_proposals)
        labels_list = images_to_levels(all_labels, num_level_proposals)
        label_weights_list = images_to_levels(all_label_weights,
                                              num_level_proposals)
        bbox_targets_list = images_to_levels(all_bbox_targets,
                                             num_level_proposals)
        bbox_weights_list = images_to_levels(all_bbox_weights,
                                             num_level_proposals)
        return (proposals_list, labels_list, label_weights_list,
                bbox_targets_list, bbox_weights_list, num_total_pos,
                num_total_neg)

    def _get_target_single(self,
                           flat_proposals,
                           valid_flags,
                           num_level_proposals,
                           gt_bboxes,
                           gt_bboxes_ignore,
                           gt_labels,
                           img_meta,
                           label_channels=1,
                           stage='init',
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
        # inside_flags = anchor_inside_flags(flat_anchors, valid_flags,
        #                                    img_meta['img_shape'][:2],
        #                                    self.train_cfg.allowed_border)

        inside_flags = valid_flags
        if not inside_flags.any():
            return (None, ) * 7
        # assign gt and sample anchors
        proposals = flat_proposals[inside_flags, :]

        if stage == 'init':
            assigner = self.init_assigner
            pos_weight = self.train_cfg.init.pos_weight
        elif stage == 'refine':
            assigner = self.refine_assigner
            pos_weight = self.train_cfg.refine.pos_weight
        else:
            raise ValueError('Invalid stage!')

        num_level_proposals_inside = self.get_num_level_proposals_inside(
            num_level_proposals, inside_flags)

        if stage == 'init':
            # assign_result = assigner.assign(proposals, 
            #                                 gt_bboxes, gt_bboxes_ignore, 
            #                                 gt_labels)
            assign_result = assigner.assign(proposals, num_level_proposals_inside,
                                            gt_bboxes, gt_bboxes_ignore,
                                            gt_labels)
        elif stage == 'refine':
            assign_result = assigner.assign(proposals, num_level_proposals_inside,
                                            gt_bboxes, gt_bboxes_ignore,
                                            gt_labels)
            # assign_result = assigner.assign(proposals,
            #                                 gt_bboxes, gt_bboxes_ignore,
            #                                 gt_labels)
        else:
            raise ValueError('Invalid stage!')

        sampling_result = self.sampler.sample(assign_result, proposals,
                                              gt_bboxes)

        num_valid_proposals = proposals.shape[0]
        bbox_targets = proposals.new_zeros([num_valid_proposals, 4])
        bbox_weights = proposals.new_zeros([num_valid_proposals, 4])
        labels = proposals.new_full((num_valid_proposals, ),
                                  self.num_classes,
                                  dtype=torch.long)
        label_weights = proposals.new_zeros(num_valid_proposals, dtype=torch.float)

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
            if pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = pos_weight
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # map up to original set of anchors
        if unmap_outputs:
            num_total_proposals = flat_proposals.size(0)
            proposals = unmap(proposals, num_total_proposals, inside_flags)
            labels = unmap(
                labels, num_total_proposals, inside_flags, fill=self.num_classes)
            label_weights = unmap(label_weights, num_total_proposals,
                                  inside_flags)
            bbox_targets = unmap(bbox_targets, num_total_proposals, inside_flags)
            bbox_weights = unmap(bbox_weights, num_total_proposals, inside_flags)

        return (proposals, labels, label_weights, bbox_targets, bbox_weights,
                pos_inds, neg_inds)

    def get_num_level_proposals_inside(self, num_level_proposals, inside_flags):
        split_inside_flags = torch.split(inside_flags, num_level_proposals)
        num_level_proposals_inside = [
            int(flags.sum()) for flags in split_inside_flags
        ]
        return num_level_proposals_inside


def topk_unsorted(x, k=1, dim=0, largest=True):
    val, idx = torch.topk(x, k, dim=dim, largest=largest)
    sorted_idx, new_idx = torch.sort(idx, dim=dim)
    val = torch.gather(val, dim=dim, index=new_idx)

    return val, sorted_idx
