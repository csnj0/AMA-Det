
from turtle import forward
from mmcv.cnn.utils.weight_init import constant_init
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, Scale, normal_init, build_norm_layer, bias_init_with_prob
from mmcv.runner import force_fp32
from mmcv.ops import DeformConv2d
from mmdet.core import (anchor_inside_flags, bbox, bbox_overlaps,
                        build_assigner, build_sampler, distance2bbox,
                        images_to_levels, multi_apply, multiclass_nms, post_processing,
                        reduce_mean, unmap, PointGenerator)
from ..builder import HEADS, build_loss
from .anchor_free_head import AnchorFreeHead
from mmcv.ops.nms import batched_nms
import numpy as np


class Integral(nn.Module):
    def __init__(self, start=-2.0, stop=2.0, bins=5, with_softmax=True):
        super(Integral, self).__init__()
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
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, groups=1, deform_groups=1, norm_cfg=None):
        super(DeformConv2dModule, self).__init__()
        self.conv = DeformConv2d(in_channels, out_channels, kernel_size,
                                 stride=stride, padding=padding,
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


class FeatureExtractionModule(nn.Module):
    def __init__(self, in_channels, feat_channels, stacked_convs=3, conv_cfg=None, norm_cfg=None, dcn_on_last_conv=False):
        super(FeatureExtractionModule, self).__init__()
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(stacked_convs):
            chn = in_channels if i == 0 else feat_channels
            if dcn_on_last_conv and (i == stacked_convs - 1):
                conv_cfg = dict(type="DCNv2")
            else:
                conv_cfg = conv_cfg
            self.cls_convs.append(
                ConvModule(
                    chn,
                    feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                )
            )
            self.reg_convs.append(
                ConvModule(
                    chn,
                    feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                )
            )
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)

    def forward(self, x):
        cls_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)

        merge_feat = cls_feat + x
        reg_feat = merge_feat
        cls_feat = merge_feat

        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)

        return cls_feat, reg_feat


class FeatureAggregationModule(nn.Module):
    def __init__(self, feat_channels, num_points=9, norm_cfg=None):
        super(FeatureAggregationModule, self).__init__()
        assert num_points % 2 == 1
        self.base_offset = self.get_base_offset(num_points)
        self.cls_conv = DeformConv2dModule(feat_channels, feat_channels, [1, num_points],
                                           stride=1, padding=[0, num_points//2], norm_cfg=norm_cfg)
        self.reg_conv = DeformConv2dModule(feat_channels, feat_channels, [1, num_points],
                                           stride=1, padding=[0, num_points//2], norm_cfg=norm_cfg)
        self.offset_conv = ConvModule(
            feat_channels, feat_channels, 3, padding=1, norm_cfg=norm_cfg)
        self.offset_out = nn.Conv2d(feat_channels, num_points*2, 1)

        normal_init(self.offset_conv.conv, std=0.01)
        normal_init(self.offset_out, std=0.01)

    def get_base_offset(self, num_points):
        dcn_base_x = np.arange(-((num_points-1)//2),
                               num_points//2+1).astype(np.float64)
        dcn_base_y = dcn_base_x * 0.0
        dcn_base_offset = np.stack(
            [dcn_base_y, dcn_base_x], axis=1).reshape((-1))
        dcn_base_offset = torch.tensor(dcn_base_offset).view(1, -1, 1, 1)
        return dcn_base_offset

    def forward(self, cls_feat, reg_feat, scale):
        base_offset = self.base_offset.type_as(reg_feat)
        pts = scale(self.offset_out(self.offset_conv(reg_feat)))
        cls_feat = self.cls_conv(cls_feat, pts - base_offset)
        reg_feat = self.reg_conv(reg_feat, pts - base_offset)
        return cls_feat, reg_feat, pts


class ResultsGenerationModule(nn.Module):
    def __init__(self, feat_channels, cls_out_channels=80, num_points=9, loc_type='points', GFLV2_cfg=None):
        super(ResultsGenerationModule, self).__init__()
        self.num_points = num_points
        self.cls_out_channels = cls_out_channels

        self.loc_type = loc_type

        self.GFLV2_cfg = GFLV2_cfg
        self.with_GFLV2 = GFLV2_cfg is not None

        if self.with_GFLV2:
            self.cls_out = nn.Conv2d(feat_channels, self.cls_out_channels, 1)
            if self.loc_type == 'points':
                self.reg_out = nn.Conv2d(
                    feat_channels, self.num_points * 2 * self.GFLV2_cfg['bins_full'], 1)
            elif self.loc_type == 'bbox':
                self.reg_out = nn.Conv2d(
                    feat_channels, 4 * self.GFLV2_cfg['bins_full'], 1)
            else:
                raise ValueError("loc_type is not valid.")

            self.reg_conf = nn.Sequential(
                nn.Conv2d(
                    4 * self.GFLV2_cfg['bins_select'], self.GFLV2_cfg['channels'], 1),
                nn.GroupNorm(8, self.GFLV2_cfg["channels"]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.GFLV2_cfg["channels"], 1, 1),
                nn.Sigmoid(),
            )
            for m in self.reg_conf:
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)
        else:
            self.cls_out = nn.Conv2d(feat_channels, self.cls_out_channels, 1)
            if self.loc_type == 'points':
                self.reg_out = nn.Conv2d(feat_channels, self.num_points*2, 1)
            elif self.loc_type == 'bbox':
                self.reg_out = nn.Conv2d(feat_channels, 4, 1)

        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.cls_out, std=0.01, bias=bias_cls)
        normal_init(self.reg_out, std=0.01)

        self.integral = Integral(
            0, self.GFLV2_cfg['bins_full']-1, bins=self.GFLV2_cfg['bins_full'], with_softmax=False)

    def points2distance(self, pts, return_index=False):
        if pts.dim() == 4:
            pts_reshape = pts.view(pts.shape[0], -1, 2, *pts.shape[2:])
            pts_y = pts_reshape[:, :, 0, ...]
            pts_x = pts_reshape[:, :, 1, ...]
        elif pts.dim() == 2:
            pts_reshape = pts.view(pts.shape[0], -1, 2)
            pts_y = pts_reshape[:, :, 0]
            pts_x = pts_reshape[:, :, 1]

        bbox_left, index_left = pts_x.min(dim=1, keepdim=True)
        bbox_up, index_up = pts_y.min(dim=1, keepdim=True)
        bbox_right, index_right = pts_x.max(dim=1, keepdim=True)
        bbox_bottom, index_bottom = pts_y.max(dim=1, keepdim=True)
        bbox = torch.cat(
            [-bbox_left, -bbox_up, bbox_right, bbox_bottom], dim=1)

        if return_index:
            return bbox, index_left, index_up, index_right, index_bottom
        else:
            return bbox

    def forward(self, cls_feat, reg_feat, pts_initial, scale):
        N, _, H, W = cls_feat.shape
        if self.with_GFLV2:
            cls_score = self.cls_out(cls_feat)
            prob_refine = scale(self.reg_out(reg_feat))
            prob_refine = prob_refine.reshape(
                N, -1, self.GFLV2_cfg['bins_full'], H, W).softmax(dim=2)
            if self.loc_type == 'points':
                pts_refine = self.integral(prob_refine.reshape(N, -1, H, W))
                pts_pred = pts_initial + pts_refine
                bbox_pred, index_left, index_up, index_right, index_bottom = self.points2distance(
                    pts_pred, return_index=True)
                prob_refine = prob_refine.reshape(
                    N, -1, 2, self.GFLV2_cfg['bins_full'], H, W)
                prob_refine_y = prob_refine[:, :, 0, ...].reshape(N, -1, H, W)
                prob_refine_x = prob_refine[:, :, 1, ...].reshape(N, -1, H, W)
                prob_left = torch.gather(
                    prob_refine_x, dim=1, index=index_left*self.GFLV2_cfg['bins_full']+torch.arange(self.GFLV2_cfg['bins_full']).type_as(index_left)[None, :, None, None])
                prob_up = torch.gather(prob_refine_y, dim=1, index=index_up*self.GFLV2_cfg['bins_full']+torch.arange(
                    self.GFLV2_cfg['bins_full']).type_as(index_up)[None, :, None, None])
                prob_right = torch.gather(
                    prob_refine_x, dim=1, index=index_right*self.GFLV2_cfg['bins_full']+torch.arange(self.GFLV2_cfg['bins_full']).type_as(index_right)[None, :, None, None])
                prob_bottom = torch.gather(
                    prob_refine_y, dim=1, index=index_bottom*self.GFLV2_cfg['bins_full']+torch.arange(self.GFLV2_cfg['bins_full']).type_as(index_bottom)[None, :, None, None])
                prob_refine = torch.stack(
                    [prob_left, prob_up, prob_right, prob_bottom], dim=1)
            elif self.loc_type == 'bbox':
                bbox_initial = self.points2distance(
                    pts_initial, return_index=False)
                bbox_refine = self.integral(prob_refine.reshape(N, -1, H, W))
                bbox_pred = bbox_initial + bbox_refine
            if self.GFLV2_cfg['is_sorted']:
                prob_topk = torch.topk(
                    prob_refine, k=self.GFLV2_cfg['bins_select'], dim=2)[0]
            else:
                prob_topk = topk_unsorted(
                    prob_refine, k=self.GFLV2_cfg['bins_select'], dim=2)[0]
            cls_score = cls_score.sigmoid() * \
                self.reg_conf(prob_topk.reshape(N, -1, H, W))
        else:
            cls_score = self.cls_out(cls_feat)
            if self.loc_type == 'points':
                pts_initial = scale(self.reg_out(reg_feat))
                pts_pred = pts_initial + pts_refine
                bbox_pred = self.points2distance(
                    pts_pred, return_index=False)
            elif self.loc_type == 'bbox':
                bbox_initial = self.points2distance(
                    pts_initial, return_index=False)
                bbox_refine = scale(self.reg_out(reg_feat))
                bbox_pred = bbox_initial + bbox_refine
        return cls_score, bbox_pred


@HEADS.register_module()
class AMAHead(AnchorFreeHead):
    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=3,
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 point_strides=[8, 16, 32, 64, 128],
                 point_base_scale=8,
                 dcn_on_last_conv=False,
                 num_points=9,
                 loc_type='points',
                 GFLV2_cfg=None,
                 loss_cls=None,
                 loss_bbox_initial=None,
                 loss_bbox=None,
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.point_strides = point_strides
        self.point_base_scale = point_base_scale
        self.dcn_on_last_conv = dcn_on_last_conv
        self.num_points = num_points
        self.loc_type = loc_type
        self.GFLV2_cfg = GFLV2_cfg
        super(AMAHead, self).__init__(num_classes, in_channels, **kwargs)

        self.loss_cls_cfg = loss_cls
        self.loss_bbox_initial_cfg = loss_bbox_initial
        self.loss_bbox_cfg = loss_bbox

        self.loss_cls = build_loss(self.loss_cls_cfg)
        if self.loss_bbox_initial_cfg is not None:
            self.loss_bbox_initial = build_loss(self.loss_bbox_initial_cfg)

        self.loss_bbox = build_loss(self.loss_bbox_cfg)

        self.score_with_sigmoid = (loss_cls.type == 'QualityFocalLoss')

        self.sampling = False
        if self.train_cfg:
            if self.train_cfg.type == 'ATSS':
                self.assigner = build_assigner(self.train_cfg.assigner)
                self.pos_weight = self.train_cfg.pos_weight
            elif self.train_cfg.type in ['RepPoints', 'RepPoints-ATSS']:
                self.init_assigner = build_assigner(
                    self.train_cfg.init.assigner)
                self.assigner = build_assigner(
                    self.train_cfg.refine.assigner)
                self.init_pos_weight = self.train_cfg.init.pos_weight
                self.pos_weight = self.train_cfg.refine.pos_weight
            # SSD sampling=False so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

        self.point_generators = [PointGenerator() for _ in self.point_strides]
        self.scales_initial = nn.ModuleList(
            [Scale(1.0) for _ in self.point_strides])
        self.scales_refine = nn.ModuleList(
            [Scale(1.0) for _ in self.point_strides])

    def _init_layers(self):
        self.feature_extractor = FeatureExtractionModule(
            self.in_channels, self.feat_channels, self.stacked_convs,
            self.conv_cfg, self.norm_cfg, self.dcn_on_last_conv)
        self.feature_aggregator = FeatureAggregationModule(
            self.feat_channels, self.num_points, self.norm_cfg)
        self.result_generator = ResultsGenerationModule(
            self.feat_channels, self.cls_out_channels, self.num_points,
            self.loc_type, self.GFLV2_cfg)

    def init_weights(self):
        pass

    def forward(self, feats):
        return multi_apply(self.forward_single, feats, self.scales_initial, self.scales_refine)

    def forward_single(self, x, scale_initial, scale_refine):
        cls_feat, reg_feat = self.feature_extractor(x)
        cls_feat, reg_feat, pts_initial = self.feature_aggregator(
            cls_feat, reg_feat, scale_initial)
        cls_score, bbox_pred = self.result_generator(
            cls_feat, reg_feat, pts_initial, scale_refine)
        return cls_score, pts_initial, bbox_pred

    def loss_single(self, point, cls_score, pts_initial, bbox_pred,
                    labels_initial, bbox_targets_initial,
                    labels_refine, label_weights_refine, bbox_targets_refine,
                    num_total_samples_refine):

        point_xy, point_stride = point.reshape(-1, 3).split([2, 1], dim=1)

        cls_score = cls_score.permute(
            0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        bbox_initial = self.points2distance(pts_initial)
        bbox_initial = bbox_initial.permute(0, 2, 3, 1).reshape(-1, 4)
        bbox_pred = bbox_pred.permute(0, 2, 3, 1).reshape(-1, 4)

        labels_refine = labels_refine.reshape(-1)
        label_weights_refine = label_weights_refine.reshape(-1)
        bbox_targets_refine = bbox_targets_refine.reshape(-1, 4)

        bg_class_ind = self.num_classes
        pos_inds = ((labels_refine >= 0)
                    & (labels_refine < bg_class_ind)).nonzero(as_tuple=False).squeeze(-1)
        score = label_weights_refine.new_zeros(labels_refine.shape)
        if len(pos_inds) > 0:
            pos_bbox_targets = bbox_targets_refine[pos_inds] / \
                point_stride[pos_inds]
            pos_bbox_preds = distance2bbox(point_xy[pos_inds] / point_stride[pos_inds],
                                           bbox_pred[pos_inds])
            weight_targets = cls_score.detach(
            ) if self.score_with_sigmoid else cls_score.detach().sigmoid()
            weight_targets = weight_targets.max(dim=1)[0][pos_inds]

            score[pos_inds] = bbox_overlaps(
                pos_bbox_preds.detach(),
                pos_bbox_targets,
                is_aligned=True).clamp(min=0.0)

            loss_bbox = self.loss_bbox(
                pos_bbox_preds,
                pos_bbox_targets,
                weight=weight_targets,
                avg_factor=1)
        else:
            loss_bbox = bbox_pred.sum() * 0.0
            weight_targets = torch.tensor(0.0).cuda()

        if self.loss_cls_cfg.type == 'QualityFocalLoss':
            loss_cls = self.loss_cls(
                cls_score, (labels_refine, score),
                weight=label_weights_refine,
                avg_factor=num_total_samples_refine)
        elif self.loss_bbox_cfg.type == 'FocalLoss':
            loss_cls = self.loss_cls(
                cls_score, labels_refine,
                weight=label_weights_refine,
                avg_factor=num_total_samples_refine)

        if self.loss_bbox_initial_cfg is None:
            return loss_cls, loss_bbox, weight_targets.sum()
        else:
            labels_initial = labels_initial.reshape(-1)
            bbox_targets_initial = bbox_targets_initial.reshape(-1, 4)
            bg_class_ind = self.num_classes
            pos_inds = ((labels_initial >= 0)
                        & (labels_initial < bg_class_ind)).nonzero(as_tuple=False).squeeze(-1)
            if len(pos_inds) > 0:
                pos_bbox_targets = bbox_targets_initial[pos_inds] / \
                    point_stride[pos_inds]
                pos_bbox_preds = distance2bbox(point_xy[pos_inds] / point_stride[pos_inds],
                                               bbox_initial[pos_inds])
                weight_targets = cls_score.detach(
                ) if self.score_with_sigmoid else cls_score.detach().sigmoid()
                weight_targets = weight_targets.max(dim=1)[0][pos_inds]
                loss_bbox_initial = self.loss_bbox_initial(
                    pos_bbox_preds,
                    pos_bbox_targets,
                    weight=weight_targets,
                    avg_factor=1)
            else:
                loss_bbox_initial = bbox_initial.sum() * 0.0
            return loss_cls, loss_bbox_initial, loss_bbox, weight_targets.sum()

    def points2distance(self, pts, return_index=False):
        if pts.dim() == 4:
            pts_reshape = pts.view(pts.shape[0], -1, 2, *pts.shape[2:])
            pts_y = pts_reshape[:, :, 0, ...]
            pts_x = pts_reshape[:, :, 1, ...]
        elif pts.dim() == 2:
            pts_reshape = pts.view(pts.shape[0], -1, 2)
            pts_y = pts_reshape[:, :, 0]
            pts_x = pts_reshape[:, :, 1]

        bbox_left, index_left = pts_x.min(dim=1, keepdim=True)
        bbox_up, index_up = pts_y.min(dim=1, keepdim=True)
        bbox_right, index_right = pts_x.max(dim=1, keepdim=True)
        bbox_bottom, index_bottom = pts_y.max(dim=1, keepdim=True)
        bbox = torch.cat(
            [-bbox_left, -bbox_up, bbox_right, bbox_bottom], dim=1)

        if return_index:
            return bbox, index_left, index_up, index_right, index_bottom
        else:
            return bbox

    def points_to_bboxes(self, point_list, distance):
        bbox_list = []
        assert isinstance(distance, float) or len(distance) == len(point_list)

        for i_img, point in enumerate(point_list):
            bbox = []
            for i_lvl in range(len(self.point_strides)):
                if isinstance(distance, float):
                    bbox_distance = distance * self.point_strides[i_lvl]
                else:
                    bbox_distance = self.points2distance(
                        distance[i_lvl]).detach() * self.point_strides[i_lvl]
                bbox_shift = bbox_distance * \
                    torch.tensor([-1.0, -1.0, 1.0, 1.0]
                                 ).type_as(point[0]).view(1, 4)

                bbox_center = torch.cat(
                    [point[i_lvl][:, :2], point[i_lvl][:, :2]], dim=1)
                bbox.append(bbox_center + bbox_shift)
            bbox_list.append(bbox)
        return bbox_list

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

    @force_fp32(apply_to=('cls_scores', 'pts_initials', 'bbox_preds'))
    def loss(self,
             cls_scores,
             pts_initials,
             bbox_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        device = cls_scores[0].device
        label_channels = 1

        point_list, valid_flag_list = self.get_points(
            featmap_sizes, img_metas, device=device)
        # list{batch}-list{level}-tensor(num, 4)

        bbox_list = self.points_to_bboxes(
            point_list, self.point_base_scale*0.5)

        # if self.with_initial_assigner:
        if self.train_cfg.type in ['RepPoints', 'RepPoints-ATSS']:
            cls_reg_targets_initial = self.get_targets(
                bbox_list,
                valid_flag_list,
                gt_bboxes,
                img_metas,
                gt_bboxes_ignore_list=gt_bboxes_ignore,
                gt_labels_list=gt_labels,
                assigner=self.init_assigner,
                pos_weight=self.init_pos_weight,
                label_channels=label_channels)
            (_, labels_initial_list, label_weights_initial_list,
             bbox_targets_initial_list, bbox_weights_initial_list, num_total_pos_initial,
             num_total_neg_initial) = cls_reg_targets_initial
            num_total_samples_initial = reduce_mean(
                torch.tensor(num_total_pos_initial, dtype=torch.float,
                             device=device)).item()
            num_total_samples_initial = max(num_total_samples_initial, 1.0)

        if self.train_cfg.anchor_guiding:
            bbox_list = self.points_to_bboxes(point_list, pts_initials)
        else:
            bbox_list = self.points_to_bboxes(
                point_list, self.point_base_scale*0.5)

        cls_reg_targets_refine = self.get_targets(
            bbox_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            assigner=self.assigner,
            pos_weight=self.pos_weight,
            label_channels=label_channels)
        (_, labels_refine_list, label_weights_refine_list,
         bbox_targets_refine_list, bbox_weights_refine_list, num_total_pos_refine,
         num_total_neg_refine) = cls_reg_targets_refine

        num_total_samples_refine = reduce_mean(
            torch.tensor(num_total_pos_refine, dtype=torch.float,
                         device=device)).item()
        num_total_samples_refine = max(num_total_samples_refine, 1.0)

        point_lvl_list = []
        for i_lvl in range(len(point_list[0])):
            point_batch = []
            for i_img in range(len(point_list)):
                point_batch.append(point_list[i_img][i_lvl])
            point_lvl_list.append(torch.cat(point_batch, dim=0))

        if self.loss_bbox_initial_cfg is None:
            losses_cls, losses_bbox, \
                avg_factor = multi_apply(
                    self.loss_single,
                    point_lvl_list,
                    cls_scores,
                    pts_initials,
                    bbox_preds,
                    labels_refine_list, bbox_targets_refine_list,
                    labels_refine_list, label_weights_refine_list,
                    bbox_targets_refine_list, num_total_samples_refine=num_total_pos_refine
                )
            avg_factor = sum(avg_factor)
            avg_factor = reduce_mean(avg_factor).item()
            losses_bbox = list(map(lambda x: x / avg_factor, losses_bbox))
            return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)
        else:
            losses_cls, losses_bbox_initial, losses_bbox, \
                avg_factor = multi_apply(
                    self.loss_single,
                    point_lvl_list,
                    cls_scores,
                    pts_initials,
                    bbox_preds,
                    labels_initial_list, bbox_targets_initial_list,
                    labels_refine_list, label_weights_refine_list,
                    bbox_targets_refine_list, num_total_samples_refine=num_total_pos_refine
                )
            avg_factor = sum(avg_factor)
            avg_factor = reduce_mean(avg_factor).item()
            losses_bbox_initial = list(
                map(lambda x: x / avg_factor, losses_bbox_initial))
            losses_bbox = list(
                map(lambda x: x / avg_factor, losses_bbox))
            return dict(loss_cls=losses_cls,
                        loss_bbox_initial=losses_bbox_initial,
                        loss_bbox=losses_bbox)

    @force_fp32(apply_to=('cls_scores', 'pts_initials', 'bbox_preds'))
    def get_bboxes(self,
                   cls_scores,
                   pts_initials,
                   bbox_preds,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        assert len(cls_scores) == len(pts_initials) == len(bbox_preds)
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
            cls_scores_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            pts_initials_list = [
                pts_initials[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_preds_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]

            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            if with_nms:
                # some heads don't support with_nms argument
                proposals = self._get_bboxes_single(cls_scores_list,
                                                    pts_initials_list,
                                                    bbox_preds_list,
                                                    mlvl_points,
                                                    img_shape,
                                                    scale_factor, cfg, rescale)
            else:
                proposals = self._get_bboxes_single(cls_scores_list,
                                                    pts_initials_list,
                                                    bbox_preds_list,
                                                    mlvl_points,
                                                    img_shape,
                                                    scale_factor, cfg, rescale,
                                                    with_nms)
            result_list.append(proposals)
        return result_list

    def _get_bboxes_single(self,
                           cls_scores,
                           pts_initials,
                           bbox_preds,
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
        assert len(cls_scores) == len(pts_initial) == len(
            bbox_pred) == len(mlvl_points)
        mlvl_bboxes = []
        mlvl_centers = []
        mlvl_pts_initials = []
        mlvl_scores = []
        for cls_score, pts_initial, bbox_pred, stride, points in zip(
                cls_scores, pts_initials, bbox_preds, self.strides,
                mlvl_points):
            assert cls_score.size(
            )[-2:] == pts_initial.size()[-2:] == bbox_pred.size()[-2:]

            scores = cls_score.permute(
                1, 2, 0).reshape(-1, self.cls_out_channels)
            scores = scores if self.score_with_sigmoid else scores.sigmoid()

            pts_initial = pts_initial.permute(
                1, 2, 0).reshape(-1, self.num_points*2)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            bbox_pred = bbox_pred * \
                torch.tensor([-1.0, -1.0, 1.0, 1.0]
                             ).type_as(bbox_pred)[None, :]

            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = scores.max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                pts_initial = pts_initial[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]

            bbox_pos_center = torch.cat([points[:, :2], points[:, :2]], dim=1)
            bboxes = bbox_pred * stride + bbox_pos_center
            x1 = bboxes[:, 0].clamp(min=0, max=img_shape[1])
            y1 = bboxes[:, 1].clamp(min=0, max=img_shape[0])
            x2 = bboxes[:, 2].clamp(min=0, max=img_shape[1])
            y2 = bboxes[:, 3].clamp(min=0, max=img_shape[0])
            bboxes = torch.stack([x1, y1, x2, y2], dim=-1)

            pts_initial = pts_initial.reshape(-1, self.num_points, 2)
            pts_initial_y = pts_initial[:, :, 0] * stride + points[:, 1:2]
            pts_initial_y = pts_initial_y.clamp(min=0, max=img_shape[0])
            pts_initial_x = pts_initial[:, :, 1] * stride + points[:, 0:1]
            pts_initial_x = pts_initial_x.clamp(min=0, max=img_shape[1])
            pts_initial = torch.stack(
                [pts_initial_x, pts_initial_y], dim=-1).reshape(-1, self.num_points*2)

            mlvl_bboxes.append(bboxes)
            mlvl_centers.append(points[:, :2])
            mlvl_pts_initials.append(pts_initial)
            mlvl_scores.append(scores)

        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_centers = torch.cat(mlvl_centers)
        mlvl_pts_initials = torch.cat(mlvl_pts_initials)

        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
            mlvl_centers /= mlvl_centers.new_tensor(scale_factor[0:2])
            mlvl_pts_initials /= mlvl_pts_initials.new_tensor(
                scale_factor[0:2].repeat(self.num_points))

        # Add a dummy background class to the backend when using sigmoid
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)

        # self.show_results = True
        self.show_results = False

        if self.show_results:
            if with_nms:
                det_bboxes, det_labels, det_centers, det_pts_init = multiclass_nms_expanded(
                    mlvl_bboxes, mlvl_scores,
                    mlvl_centers, mlvl_pts_initials,
                    cfg.score_thr, cfg.nms,
                    cfg.max_per_img)
                return det_bboxes, det_centers, det_pts_init, det_labels
            else:
                return mlvl_bboxes, mlvl_centers, mlvl_pts_initials, mlvl_scores
        else:
            if with_nms:
                det_bboxes, det_labels = multiclass_nms(mlvl_bboxes, mlvl_scores,
                                                        cfg.score_thr, cfg.nms,
                                                        cfg.max_per_img, return_inds=False)
                return det_bboxes, det_labels
            else:
                return mlvl_bboxes, mlvl_scores

    @force_fp32(apply_to=('cls_scores', 'pts_initial', 'bbox_pred'))
    def get_results(self,
                    cls_scores,
                    pts_initial,
                    bbox_pred,
                    gt_bboxes,
                    gt_labels,
                    img_metas):

        assert len(cls_scores) == len(pts_initial) == len(bbox_pred)
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
            if self.score_with_sigmoid:
                cls_scores_list = [
                    cls_scores[i][img_id].detach().permute(1, 2, 0).reshape(-1, self.cls_out_channels
                                                                            ) for i in range(num_levels)]
            else:
                cls_scores_list = [
                    cls_scores[i][img_id].detach().permute(1, 2, 0).reshape(-1, self.cls_out_channels
                                                                            ).sigmoid() for i in range(num_levels)]
            pts_initials_list = [
                pts_initial[i][img_id].detach().permute(1, 2, 0).reshape(-1, self.num_points*2
                                                                         ) for i in range(num_levels)]
            bbox_preds_list = [
                bbox_pred[i][img_id].detach().permute(1, 2, 0).reshape(-1, 4
                                                                       ) for i in range(num_levels)]

            conf, iou = self._get_result_single(
                cls_scores_list,
                pts_initials_list,
                bbox_preds_list,
                mlvl_points,
                self.strides,
                featmap_sizes,
                gt_bboxes[img_id],
                gt_labels[img_id],
                img_metas[img_id]['img_shape'])
            result_list.append([conf, iou])

        return result_list

    def _get_result_single(self,
                           cls_scores,
                           pts_initials,
                           bbox_preds,
                           points,
                           strides,
                           featmap_sizes,
                           gt_bboxes,
                           gt_labels,
                           img_shape):

        bbox_list = []
        for score, pts_initial, bbox_pred, point, stride in zip(
                cls_scores, pts_initials, bbox_preds, points, strides):
            bbox_pred = bbox_pred * \
                torch.tensor([-1.0, -1.0, 1.0, 1.0]
                             ).type_as(bbox_pred)[None, :]

            bbox_center = torch.cat([point[:, :2], point[:, :2]], dim=1)
            bbox = bbox_pred * stride + bbox_center
            x1 = bbox[:, 0].clamp(min=0, max=img_shape[1])
            y1 = bbox[:, 1].clamp(min=0, max=img_shape[0])
            x2 = bbox[:, 2].clamp(min=0, max=img_shape[1])
            y2 = bbox[:, 3].clamp(min=0, max=img_shape[0])
            bbox = torch.stack([x1, y1, x2, y2], dim=-1)
            bbox_list.append(bbox)

        conf_list = []
        iou_list = []
        for gt_bbox, gt_label in zip(gt_bboxes, gt_labels):
            conf_lvl_list = []
            iou_lvl_list = []
            for score, bbox, feat_size in zip(cls_scores, bbox_list, featmap_sizes):
                conf = score[:, gt_label].reshape(feat_size)
                iou = bbox_overlaps(
                    gt_bbox[None, :], bbox, is_aligned=False).reshape(feat_size)
                conf_lvl_list.append(conf)
                iou_lvl_list.append(iou)
            conf_list.append(conf_lvl_list)
            iou_list.append(iou_lvl_list)

        # conf_list: list[gt]-list[lvl]-tensor[H, W, 1]
        # iou_list: list[gt]-list[lvl]-tensor[H, W, 1]
        return conf_list, iou_list

    def get_targets(self,
                    proposal_list,
                    valid_flag_list,
                    gt_bboxes_list,
                    img_metas,
                    gt_bboxes_ignore_list=None,
                    gt_labels_list=None,
                    assigner=None,
                    pos_weight=-1,
                    label_channels=1,
                    unmap_outputs=True):
        """Get targets for head.

        This method is almost the same as `AnchorHead.get_targets()`. Besides
        returning the targets as the parent method does, it also returns the
        anchors as the first element of the returned tuple.
        """
        num_imgs = len(img_metas)
        assert len(proposal_list) == len(valid_flag_list) == num_imgs

        # anchor number of multi levels
        num_level_proposals = [proposals.size(
            0) for proposals in proposal_list[0]]
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
             assigner=assigner,
             pos_weight=pos_weight,
             unmap_outputs=unmap_outputs)
        # no valid anchors
        if any([labels is None for labels in all_labels]):
            return None
        # sampled anchors of all images
        num_total_pos = sum([max(inds.numel(), 1) for inds in pos_inds_list])
        num_total_neg = sum([max(inds.numel(), 1) for inds in neg_inds_list])
        # split targets to a list w.r.t. multiple levels

        proposals_list = images_to_levels(
            all_proposals, num_level_proposals_list[0])
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
                           assigner=None,
                           pos_weight=-1,
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
        inside_flags = valid_flags
        if not inside_flags.any():
            return (None, ) * 7
        # assign gt and sample proposals
        proposals = flat_proposals[inside_flags, :]

        num_level_proposals_inside = self.get_num_level_proposals_inside(
            num_level_proposals, inside_flags)

        # assign_result = assigner.assign(proposals,
        #                                 gt_bboxes, gt_bboxes_ignore,
        #                                 gt_labels)
        assign_result = assigner.assign(proposals, num_level_proposals_inside,
                                        gt_bboxes, gt_bboxes_ignore,
                                        gt_labels)

        sampling_result = self.sampler.sample(assign_result, proposals,
                                              gt_bboxes)

        num_valid_proposals = proposals.shape[0]
        bbox_targets = proposals.new_zeros([num_valid_proposals, 4])
        bbox_weights = proposals.new_zeros([num_valid_proposals, 4])
        labels = proposals.new_full((num_valid_proposals, ),
                                    self.num_classes,
                                    dtype=torch.long)
        label_weights = proposals.new_zeros(
            num_valid_proposals, dtype=torch.float)

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
            bbox_targets = unmap(
                bbox_targets, num_total_proposals, inside_flags)
            bbox_weights = unmap(
                bbox_weights, num_total_proposals, inside_flags)

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


def multiclass_nms_expanded(multi_bboxes,
                            multi_scores,
                            mlvl_points, mlvl_pts_init,
                            score_thr,
                            nms_cfg,
                            max_num=-1,
                            score_factors=None,
                            return_inds=False):
    num_classes = multi_scores.size(1) - 1
    # exclude background category
    if multi_bboxes.shape[1] > 4:
        bboxes = multi_bboxes.view(multi_scores.size(0), -1, 4)
    else:
        bboxes = multi_bboxes[:, None].expand(
            multi_scores.size(0), num_classes, 4)

    points = mlvl_points[:, None].expand(
        mlvl_points.size(0), num_classes, mlvl_points.size(1))
    pts_init = mlvl_pts_init[:, None].expand(
        mlvl_pts_init.size(0), num_classes, mlvl_pts_init.size(1))

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

    # remove low scoring boxes
    valid_mask = scores > score_thr
    inds = valid_mask.nonzero(as_tuple=False).squeeze(1)
    bboxes, scores, labels = bboxes[inds], scores[inds], labels[inds]
    points, pts_init = points[inds], pts_init[inds]

    if inds.numel() == 0:
        if torch.onnx.is_in_onnx_export():
            raise RuntimeError('[ONNX Error] Can not record NMS '
                               'as it has not been executed this time')
        if return_inds:
            return bboxes, labels, points, pts_init, inds
        else:
            return bboxes, labels, points, pts_init

    # TODO: add size check before feed into batched_nms
    dets, keep = batched_nms(bboxes, scores, labels, nms_cfg)

    if max_num > 0:
        dets = dets[:max_num]
        keep = keep[:max_num]

    if return_inds:
        return dets, labels[keep], points[keep], pts_init[keep], keep
    else:
        return dets, labels[keep], points[keep], pts_init[keep]
