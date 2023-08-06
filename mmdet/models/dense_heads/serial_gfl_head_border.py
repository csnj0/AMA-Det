from mmcv.ops.deform_conv import DeformConv2dPack
from torch.nn.modules import padding
from mmdet.core.bbox.transforms import bbox2distance
from mmcv.cnn.utils.weight_init import constant_init
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, Scale, bias_init_with_prob, normal_init
from mmcv.runner import force_fp32
from mmcv.ops import DeformConv2d
from torch.autograd import Function


from mmdet.core import (anchor_inside_flags, bbox_overlaps,
                        build_assigner, build_sampler, distance2bbox,
                        images_to_levels, multi_apply, multiclass_nms,
                        reduce_mean, unmap)
from ..builder import HEADS, build_loss
from .anchor_head import AnchorHead

import numpy as np


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


class DeformConv2dModule(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(DeformConv2dModule, self).__init__()
        self.conv = DeformConv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.norm = nn.GroupNorm(32, out_channels)
        self.act = nn.ReLU()

    def forward(self, x, offset):
        x = self.conv(x, offset)
        x = self.norm(x)
        x = self.act(x)
        return x


class AttentionDeformModule(nn.Module):
    def __init__(self, feat_channels, bins=5, points=1):
        super(AttentionDeformModule, self).__init__()

        self.bins = bins
        self.points = points

        base_x = np.arange(-(bins//2), bins//2 + 1).astype(np.float64)
        base_y = base_x * 0.0
        base_offset = np.stack([base_y, base_x], axis=1).reshape((-1))
        self.base_offset = torch.tensor(base_offset).view(1, -1, 1, 1)

        # self.heatmap = nn.Conv2d(feat_channels, 1, 1)
        self.expand_feat = nn.Conv2d(feat_channels, self.points*feat_channels, 1)
        # self.heatmap = nn.Conv2d(self.points*feat_channels, self.points, 1, groups=self.points)
        self.heatmap = nn.Sequential(
            nn.Conv2d(self.points*feat_channels, self.points*feat_channels, 1, groups=self.points),
            nn.ReLU(),
            nn.Conv2d(self.points*feat_channels, self.points, 1, groups=self.points)
        )


        self.dconv_feat = DeformConv2d(feat_channels, 
                                       feat_channels, 
                                       [1, self.bins], 
                                       stride=[1,1], 
                                       padding=[0, self.bins//2], 
                                       groups=feat_channels)
        constant_init(self.dconv_feat, 1.0)
        self.dconv_feat.weight.detach_()

        self.dconv_map = DeformConv2d(1, 1, [1, self.bins], stride=[1,1], padding=[0, self.bins//2])
        constant_init(self.dconv_map, 1.0)
        self.dconv_map.weight.detach_()


        self.conv_merge = nn.Conv2d(self.points * feat_channels, feat_channels, 1)
        self.norm = nn.GroupNorm(32, feat_channels)
        self.act = nn.ReLU()

    def forward(self, x, offsets):
        
        self.base_offset = self.base_offset.type_as(x)

        assert offsets.dim() == 4, "offsets must be (B, C, H, W)"
        assert offsets.size(1) == self.points * self.bins * 2
        offsets = torch.chunk(offsets, self.points, dim=1)

        expand_feats = self.expand_feat(x)
        heatmaps = self.heatmap(expand_feats).exp().chunk(self.points, dim=1)
        expand_feats = expand_feats.chunk(self.points, dim=1)

        outs = []
        for offset, heatmap, expand_feat in zip(offsets, heatmaps, expand_feats):
            offset = offset.contiguous()

            out = self.dconv_feat(expand_feat * heatmap.contiguous(), offset - self.base_offset)

            out = out / (self.dconv_map(heatmap.contiguous(), offset - self.base_offset) + 1E-6)

            outs.append(out)

        
        outs = torch.cat(outs, dim=1)
        outs = self.conv_merge(outs)
        outs = self.norm(outs)
        outs = self.act(outs)


        return outs




'''
class Unfold(nn.Module):
    def __init__(self, in_channels, bins):
        super(Unfold, self).__init__()
        self.bins = bins
        self.conv1x = nn.Conv2d(in_channels, 1, 3, 1, 1)
        self.conv1y = nn.Conv2d(in_channels, 1, 3, 1, 1)

        self.conv2 = nn.Conv2d(bins, bins, 1, 1, 0)
    def forward(self, feat):
        feat = self.conv1x(feat)
        x = F.unfold(feat, [1, self.bins], stride=[1,1], padding=[0, self.bins//2]).permute(0,2,1)

        feat = self.conv1y(feat)
        y = F.unfold(feat, [self.bins, 1], stride=[1,1], padding=[self.bins//2, 0]).permute(0,2,1)
        x = F.interpolate(x, [self.bins*2]).permute(0,2,1).reshape(feat.size(0), self.bins*2, feat.size(2), feat.size(3))
        y = F.interpolate(y, [self.bins*2]).permute(0,2,1).reshape(feat.size(0), self.bins*2, feat.size(2), feat.size(3))


        l = self.conv2(x[:, 0:self.bins, ...])
        t = self.conv2(y[:, 0:self.bins, ...])
        r = self.conv2(y[:, self.bins:, ...])
        b = self.conv2(y[:, self.bins:, ...])
        out = torch.cat([l, t, r, b], dim=1)
        return out
'''


@HEADS.register_module()
class SERIALGFLHead(AnchorHead):

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 bins=16,
                 points=9,
                 DGQP_cfg=dict(channels=64),
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.bins = bins
        self.points = points
        # self.y_num = points // 2

        self.DGQP_cfg = DGQP_cfg
        self.with_DGQP = DGQP_cfg is not None


        '''
        dcn_base_x = np.arange(-(self.points//2),
                             self.points//2 + 1).astype(np.float64)
        dcn_base_y = dcn_base_x * 0.0
        dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=1).reshape((-1))
        self.dcn_base_offset = torch.tensor(dcn_base_offset).view(1, -1, 1, 1)
        '''


        self.dcn_kernel = int(np.sqrt(self.points))
        self.dcn_pad = int((self.dcn_kernel - 1) / 2)
        assert self.dcn_kernel * self.dcn_kernel == self.points, \
            'The points number should be a square number.'
        assert self.dcn_kernel % 2 == 1, \
            'The points number should be an odd square number.'
        dcn_base = np.arange(-self.dcn_pad,
                             self.dcn_pad + 1).astype(np.float64)
        dcn_base_y = np.repeat(dcn_base, self.dcn_kernel)
        dcn_base_x = np.tile(dcn_base, self.dcn_kernel)
        dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=1).reshape((-1))
        self.dcn_base_offset = torch.tensor(dcn_base_offset).view(1, -1, 1, 1)


        super(SERIALGFLHead, self).__init__(num_classes, in_channels, **kwargs)

        self.sampling = False
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            # SSD sampling=False so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

        self.integral = Integral(-(self.bins-1.0)/2.0, (self.bins-1.0)/2.0, self.bins)
        self.loss_bbox = build_loss(dict(type='GIoULoss', loss_weight=2.0))

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

        self.reg_initial = nn.Conv2d(self.feat_channels, 4, 3, stride=1, padding=1)
        # self.reg_initial = nn.Conv2d(self.feat_channels, self.points * 2, 3, stride=1, padding=1)
        self.cls_initial = nn.Conv2d(self.feat_channels, self.points * 2, 3, stride=1, padding=1)

        # self.reg_refine = DeformConv2dModule(self.feat_channels, self.feat_channels, [3,3], stride=[1,1], padding=[1,1])
        self.cls_refine = DeformConv2dModule(self.feat_channels, self.feat_channels, [3,3], stride=[1,1], padding=[1,1])

        self.reg_refine = AttentionDeformModule(self.feat_channels, bins=5, points=4)
        # self.cls_refine = AttentionDeformModule(self.feat_channels, bins=25, points=1)


        assert self.num_anchors == 1, 'anchor free version'
        self.gfl_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.gfl_reg = nn.Conv2d(
            self.feat_channels, 4 * self.bins, 3, padding=1)

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

        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.gfl_cls, std=0.01, bias=bias_cls)
        normal_init(self.gfl_reg, std=0.01)
        normal_init(self.reg_initial, std=0.01)
        normal_init(self.cls_initial, std=0.01)


        if self.with_DGQP:
            for m in self.reg_conf:
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)


    def forward(self, feats):
        return multi_apply(self.forward_single, feats)
    

    
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

        
    def forward_single(self, x):
        dcn_base_offset = self.dcn_base_offset.type_as(x)

        cls_feat = x
        reg_feat = x

        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)

        # offset_reg = self.reg_initial(reg_feat)
        # bbox_pred_initial = self.points2distance(offset_reg)
        bbox_pred_initial = self.reg_initial(reg_feat).exp()
        


        # offset_cls = self.cls_initial(cls_feat)

        # bbox_pred_refine = self.gfl_reg(self.reg_refine(reg_feat, offset_reg - dcn_base_offset))
        # cls_score = self.gfl_cls(self.cls_refine(cls_feat, offset_cls - dcn_base_offset)).sigmoid()

        l = bbox_pred_initial[:, 0:1, ...].detach()
        t = bbox_pred_initial[:, 1:2, ...].detach()
        r = bbox_pred_initial[:, 2:3, ...].detach()
        b = bbox_pred_initial[:, 3:, ...].detach()


        offset_reg_0_y = torch.arange(0, 5).type_as(x)[None, :, None, None] * (t+b)/(5-1) - t
        offset_reg_0_x = -l.repeat(1, 5, 1, 1)
        offset_reg_0 = torch.stack([offset_reg_0_y, offset_reg_0_x], dim=2).reshape(x.size(0), -1, x.size(2), x.size(3))

        offset_reg_1_y = -t.repeat(1, 5, 1, 1)
        offset_reg_1_x = torch.arange(0, 5).type_as(x)[None, :, None, None] * (l+r)/(5-1) - l
        offset_reg_1 = torch.stack([offset_reg_1_y, offset_reg_1_x], dim=2).reshape(x.size(0), -1, x.size(2), x.size(3))

        offset_reg_2_y = torch.arange(0, 5).type_as(x)[None, :, None, None] * (t+b)/(5-1) - t
        offset_reg_2_x = r.repeat(1, 5, 1, 1)
        offset_reg_2 = torch.stack([offset_reg_2_y, offset_reg_2_x], dim=2).reshape(x.size(0), -1, x.size(2), x.size(3))


        offset_reg_3_y = b.repeat(1, 5, 1, 1)
        offset_reg_3_x = torch.arange(0, 5).type_as(x)[None, :, None, None] * (l+r)/(5-1) - l
        offset_reg_3 = torch.stack([offset_reg_3_y, offset_reg_3_x], dim=2).reshape(x.size(0), -1, x.size(2), x.size(3))


        offset_reg = torch.cat([
            offset_reg_0,
            offset_reg_1,
            offset_reg_2,
            offset_reg_3
        ], dim=1)

        # offset_cls_y = torch.arange(0, 5).type_as(x)[None, :, None, None] * (t+b) / (5-1) - t
        # offset_cls_y = offset_cls_y[:, :, None, :, :].repeat(1, 1, 5, 1, 1).reshape(x.size(0), -1, x.size(2), x.size(3))
        # offset_cls_x = torch.arange(0, 5).type_as(x)[None, :, None, None] * (l+r) / (5-1) - l
        # offset_cls_x = offset_cls_x[:, None, :, :, :].repeat(1, 5, 1, 1, 1).reshape(x.size(0), -1, x.size(2), x.size(3))
        # offset_cls = torch.stack([offset_cls_y, offset_cls_x], dim=2).reshape(x.size(0), -1, x.size(2), x.size(3))


        bbox_pred_refine = self.gfl_reg(self.reg_refine(reg_feat, offset_reg))


        offset_cls = self.cls_initial(cls_feat)
        cls_score = self.gfl_cls(self.cls_refine(cls_feat, offset_cls)).sigmoid()


        if self.with_DGQP:
            N, _, H, W = bbox_pred_refine.size()
            prob = F.softmax(bbox_pred_refine.reshape(N, 4, self.bins, H, W), dim=2)
            cls_score = cls_score * self.reg_conf(prob.reshape(N, -1, H, W))

        return cls_score, bbox_pred_initial, bbox_pred_refine
    

    def anchor_center(self, anchors):
        anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
        anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2
        return torch.stack([anchors_cx, anchors_cy], dim=-1)


    def loss_single(self, anchors, cls_score, bbox_pred_initial, bbox_pred_refine, labels, label_weights,
                    bbox_targets, stride, num_total_samples):

        assert stride[0] == stride[1], 'h stride is not equal to w stride!'
        anchors = anchors.reshape(-1, 4)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        bbox_pred_initial = bbox_pred_initial.permute(0, 2, 3, 1).reshape(-1, 4)
        bbox_pred_refine = bbox_pred_refine.permute(0, 2, 3, 1).reshape(-1, 4 * self.bins)


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
            pos_anchors = anchors[pos_inds]
            pos_anchor_centers = self.anchor_center(pos_anchors) / stride[0]

            weight_targets = cls_score.detach()
            weight_targets = weight_targets.max(dim=1)[0][pos_inds]

            pos_bbox_pred_refine_corners = self.integral(pos_bbox_pred_refine)
            pos_decode_bbox_pred = distance2bbox(pos_anchor_centers,
                                                 pos_bbox_pred_initial_corners + pos_bbox_pred_refine_corners)

            pos_decode_bbox_targets = pos_bbox_targets / stride[0]

            score[pos_inds] = bbox_overlaps(
                pos_decode_bbox_pred.detach(),
                pos_decode_bbox_targets,
                is_aligned=True).clamp(min=0.0)

            loss_bbox = self.loss_bbox(
                    distance2bbox(pos_anchor_centers,
                                  pos_bbox_pred_initial_corners + pos_bbox_pred_refine_corners),
                    pos_decode_bbox_targets,
                    weight=weight_targets,
                    avg_factor=1.0)
        else:
            loss_bbox = bbox_pred_initial.sum() * 0.0
            weight_targets = torch.tensor(0).cuda()

        loss_cls = self.loss_cls(
            cls_score, (labels, score),
            weight=label_weights,
            avg_factor=num_total_samples)

        return loss_cls, loss_bbox, weight_targets.sum()



    @force_fp32(apply_to=('cls_scores', 'bbox_preds_initial', 'bbox_preds_refine'))
    def loss(self,
             cls_scores,
             bbox_preds_initial,
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

        losses_cls,  losses_bbox, \
            avg_factor = multi_apply(
                self.loss_single,
                anchor_list,
                cls_scores,
                bbox_preds_initial,
                bbox_preds_refine,
                labels_list,
                label_weights_list,
                bbox_targets_list,
                self.anchor_generator.strides,
                num_total_samples=num_total_samples)

        avg_factor = sum(avg_factor) 
        avg_factor = reduce_mean(avg_factor).item()
        losses_bbox = list(map(lambda x: x / avg_factor, losses_bbox))

        return dict(loss_cls=losses_cls, loss_bbox=losses_bbox)


    @force_fp32(apply_to=('cls_scores', 'bbox_preds_initial', 'bbox_preds_refine'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds_initial,
                   bbox_preds_refine,
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
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            if with_nms:
                # some heads don't support with_nms argument
                proposals = self._get_bboxes_single(cls_score_list,
                                                    bbox_pred_initial_list,
                                                    bbox_pred_refine_list,
                                                    mlvl_anchors, img_shape,
                                                    scale_factor, cfg, rescale)
            else:
                proposals = self._get_bboxes_single(cls_score_list,
                                                    bbox_pred_initial_list,
                                                    bbox_pred_refine_list,
                                                    mlvl_anchors, img_shape,
                                                    scale_factor, cfg, rescale,
                                                    with_nms)
            result_list.append(proposals)
        return result_list

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds_initial,
                           bbox_preds_refine,
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
        mlvl_scores = []
        for cls_score, bbox_pred_initial, bbox_pred_refine, stride, anchors in zip(
                cls_scores, bbox_preds_initial, bbox_preds_refine, self.anchor_generator.strides,
                mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred_initial.size()[-2:] == bbox_pred_refine.size()[-2:]
            assert stride[0] == stride[1]

            scores = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels)
            
            bbox_pred = (bbox_pred_initial.permute(1, 2, 0).reshape(-1, 4) + self.integral(bbox_pred_refine.permute(1, 2, 0))) * stride[0]

            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = scores.max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]

            bboxes = distance2bbox(
                self.anchor_center(anchors), bbox_pred, max_shape=img_shape)
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
                                                    cfg.max_per_img)
            return det_bboxes, det_labels
        else:
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
