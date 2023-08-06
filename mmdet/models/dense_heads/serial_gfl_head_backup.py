from mmcv.cnn.utils.weight_init import constant_init
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, Scale, bias_init_with_prob, normal_init
from mmcv.runner import force_fp32
from mmcv.ops import DeformConv2d
from torch.autograd import Function


from mmdet.core import (anchor_inside_flags, bbox2distance, bbox_overlaps,
                        build_assigner, build_sampler, distance2bbox,
                        images_to_levels, multi_apply, multiclass_nms,
                        reduce_mean, unmap)
from mmdet.models.losses.smooth_l1_loss import smooth_l1_loss
from ..builder import HEADS, build_loss
from .anchor_head import AnchorHead

import numpy as np

import math


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


@HEADS.register_module()
class SERIALGFLHead(AnchorHead):
    """Generalized Focal Loss: Learning Qualified and Distributed Bounding
    Boxes for Dense Object Detection.

    GFL head structure is similar with ATSS, however GFL uses
    1) joint representation for classification and localization quality, and
    2) flexible General distribution for bounding box locations,
    which are supervised by
    Quality Focal Loss (QFL) and Distribution Focal Loss (DFL), respectively

    https://arxiv.org/abs/2006.04388

    Args:
        num_classes (int): Number of categories excluding the background
            category.
        in_channels (int): Number of channels in the input feature map.
        stacked_convs (int): Number of conv layers in cls and reg tower.
            Default: 4.
        conv_cfg (dict): dictionary to construct and config conv layer.
            Default: None.
        norm_cfg (dict): dictionary to construct and config norm layer.
            Default: dict(type='GN', num_groups=32, requires_grad=True).
        loss_qfl (dict): Config of Quality Focal Loss (QFL).
        reg_max (int): Max value of integral set :math: `{0, ..., reg_max}`
            in QFL setting. Default: 16.
    Example:
        >>> self = GFLHead(11, 7)
        >>> feats = [torch.rand(1, 7, s, s) for s in [4, 8, 16, 32, 64]]
        >>> cls_quality_score, bbox_pred = self.forward(feats)
        >>> assert len(cls_quality_score) == len(self.scales)
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 stacked_convs=4,
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 loss_dfl=dict(type='DistributionFocalLoss', loss_weight=0.25),
                 bins=16,
                 DGQP_cfg=dict(topk=4, sorted=False, channels=64, add_mean=False),
                 RGQP_cfg=dict(),
                 **kwargs):
        self.stacked_convs = stacked_convs
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.bins = bins

        self.DGQP_cfg = DGQP_cfg
        self.with_DGQP = DGQP_cfg is not None

        self.RGQP_cfg = RGQP_cfg
        self.with_RGQP = RGQP_cfg is not None


        super(SERIALGFLHead, self).__init__(num_classes, in_channels, **kwargs)

        self.sampling = False
        if self.train_cfg:
            self.assigner = build_assigner(self.train_cfg.assigner)
            # SSD sampling=False so use PseudoSampler
            sampler_cfg = dict(type='PseudoSampler')
            self.sampler = build_sampler(sampler_cfg, context=self)

        self.integral_initial = Integral(0.0, 16.0, 5)
        self.integral_refine = Integral(-2.0, 2.0, self.bins)

        self.loss_dfl = build_loss(loss_dfl)
        self.loss_bbox_initial = build_loss(dict(type='GIoULoss', loss_weight=2.0))
        self.loss_bbox_refine = build_loss(dict(type='GIoULoss', loss_weight=2.0))

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

        for i in range(self.stacked_convs - 1):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        

        self.reg_initial = nn.Conv2d(self.feat_channels, 9 * 2, 3, stride=1, padding=1) # l, t, r, b
        self.reg_refine = DeformConv2d(self.feat_channels, self.feat_channels, 3, stride=1, padding=1)

        self.reg_refine_affine = nn.Sequential(
            nn.GroupNorm(32, self.feat_channels),
            nn.ReLU()
        )


        self.cls_refine = DeformConv2d(self.feat_channels, self.feat_channels, 3, stride=1, padding=1)
        self.cls_refine_affine = nn.Sequential(
            nn.GroupNorm(32, self.feat_channels),
            nn.ReLU()
        )



        assert self.num_anchors == 1, 'anchor free version'
        self.gfl_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)
        self.gfl_reg = nn.Conv2d(
            self.feat_channels, 4 * self.bins, 3, padding=1)
        self.scales = nn.ModuleList(
            [Scale(1.0) for _ in self.anchor_generator.strides])

        
        if self.with_DGQP:
            total_dim = self.DGQP_cfg['topk']

            if self.DGQP_cfg['add_mean']:
                total_dim += 1
            self.reg_dconf = nn.Sequential(
                nn.Conv2d(4 * total_dim, self.DGQP_cfg['channels'], 1),
                self.relu,
                nn.Conv2d(self.DGQP_cfg['channels'], 1, 1),
                nn.Sigmoid()
            )
        

        self.cls_reweight = nn.Sequential(
            nn.Conv2d(4, 64, 1),
            self.relu,
            nn.Conv2d(64, 1, 1)
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

        if self.with_DGQP:
            for m in self.reg_dconf:
                if isinstance(m, nn.Conv2d):
                    normal_init(m, std=0.01)
        
        for m in self.cls_reweight:
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)



    def forward(self, feats):
        """Forward features from the upstream network.

        Args:
            feats (tuple[Tensor]): Features from the upstream network, each is
                a 4D-tensor.

        Returns:
            tuple: Usually a tuple of classification scores and bbox prediction
                cls_scores (list[Tensor]): Classification and quality (IoU)
                    joint scores for all scale levels, each is a 4D-tensor,
                    the channel number is num_classes.
                bbox_preds (list[Tensor]): Box distribution logits for all
                    scale levels, each is a 4D-tensor, the channel number is
                    4*(n+1), n is max value of integral set.
        """
        return multi_apply(self.forward_single, feats, self.scales)
    
    def points2distance(self, pts, y_first=False):
        pts_reshape = pts.view(pts.shape[0], -1, 2, *pts.shape[2:])
        pts_y = pts_reshape[:, :, 0, ...] if y_first else pts_reshape[:, :, 1,
                                                                      ...]
        pts_x = pts_reshape[:, :, 1, ...] if y_first else pts_reshape[:, :, 0,
                                                                      ...]

        bbox_left = pts_x.min(dim=1, keepdim=True)[0]
        bbox_right = pts_x.max(dim=1, keepdim=True)[0]
        bbox_up = pts_y.min(dim=1, keepdim=True)[0]
        bbox_bottom = pts_y.max(dim=1, keepdim=True)[0]
        bbox = torch.cat([bbox_left, bbox_up, bbox_right, bbox_bottom],
                            dim=1)
        return bbox

    def forward_single(self, x, scale):
        """Forward feature of a single scale level.

        Args:
            x (Tensor): Features of a single scale level.
            scale (:obj: `mmcv.cnn.Scale`): Learnable scale module to resize
                the bbox prediction.

        Returns:
            tuple:
                cls_score (Tensor): Cls and quality joint scores for a single
                    scale level the channel number is num_classes.
                bbox_pred (Tensor): Box distribution logits for a single scale
                    level, the channel number is 4*(n+1), n is max value of
                    integral set.
        """
        # dcn_base_offset = self.dcn_base_offset.type_as(x)


        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)

        # cls_score = self.gfl_cls(cls_feat).sigmoid()

        
        offset = self.reg_initial(reg_feat)
        bbox_pred_initial = self.points2distance(offset).abs()
        bbox_pred_refine_dist = self.gfl_reg(self.reg_refine_affine(self.reg_refine(reg_feat, offset)))



        cls_score = self.gfl_cls(self.cls_refine_affine(self.cls_refine(cls_feat, offset))).sigmoid()



        # bbox_pred_refine_dist = self.gfl_reg(self.reg_refine_affine(self.reg_refine(reg_feat)))
        # bbox_pred_refine_dist = self.gfl_reg(self.reg_refine_affine(self.reg_refine(reg_feat)))
        # bbox_pred_refine_dist = self.gfl_reg(reg_feat)

        N, C, H, W = bbox_pred_refine_dist.size()
        bbox_pred_refine = self.integral_refine(bbox_pred_refine_dist.permute(0, 2, 3, 1).reshape(-1, 4 * self.bins)).reshape(N, H, W, 4).permute(0, 3, 1, 2).float()

        if self.with_DGQP:
            N, C, H, W = bbox_pred_refine_dist.size()
            prob = F.softmax(bbox_pred_refine_dist.reshape(N, 4, self.bins, H, W), dim=2)
            if self.DGQP_cfg['sorted']:
                prob_topk, _ = prob.topk(self.DGQP_cfg['topk'], dim=2)
            else:
                prob_topk, _ = topk_unsorted(prob, k=self.DGQP_cfg['topk'], dim=2)
            if self.DGQP_cfg['add_mean']:
                stat = torch.cat([prob_topk, prob_topk.mean(dim=2, keepdim=True)],
                                dim=2)
            else:
                stat = prob_topk

            cls_score = cls_score * self.reg_dconf(stat.reshape(N, -1, H, W))


        prob_topk_2, _ = cls_score.topk(4, dim=1)
        cls_reweight = self.cls_reweight(prob_topk_2.detach()).exp()


        if self.training:
            return cls_score, bbox_pred_initial, bbox_pred_refine, bbox_pred_refine_dist, cls_reweight
        else:
            return cls_score, bbox_pred_initial, bbox_pred_refine


    def anchor_center(self, anchors):
        """Get anchor centers from anchors.

        Args:
            anchors (Tensor): Anchor list with shape (N, 4), "xyxy" format.

        Returns:
            Tensor: Anchor centers with shape (N, 2), "xy" format.
        """
        anchors_cx = (anchors[:, 2] + anchors[:, 0]) / 2
        anchors_cy = (anchors[:, 3] + anchors[:, 1]) / 2
        return torch.stack([anchors_cx, anchors_cy], dim=-1)

    def loss_single(self, anchors, cls_score, bbox_pred_initial, bbox_pred_refine, bbox_pred_refine_dist, cls_reweight, labels, label_weights,
                    bbox_targets, stride, num_total_samples):
        """Compute loss of a single scale level.

        Args:
            anchors (Tensor): Box reference for each scale level with shape
                (N, num_total_anchors, 4).
            cls_score (Tensor): Cls and quality joint scores for each scale
                level has shape (N, num_classes, H, W).
            bbox_pred (Tensor): Box distribution logits for each scale
                level with shape (N, 4*(n+1), H, W), n is max value of integral
                set.
            labels (Tensor): Labels of each anchors with shape
                (N, num_total_anchors).
            label_weights (Tensor): Label weights of each anchor with shape
                (N, num_total_anchors)
            bbox_targets (Tensor): BBox regression targets of each anchor wight
                shape (N, num_total_anchors, 4).
            stride (tuple): Stride in this scale level.
            num_total_samples (int): Number of positive samples that is
                reduced over all GPUs.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        assert stride[0] == stride[1], 'h stride is not equal to w stride!'
        anchors = anchors.reshape(-1, 4)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        bbox_pred_initial = bbox_pred_initial.permute(0, 2, 3, 1).reshape(-1, 4)
        bbox_pred_refine = bbox_pred_refine.permute(0, 2, 3, 1).reshape(-1, 4)
        bbox_pred_refine_dist = bbox_pred_refine_dist.permute(0, 2, 3, 1).reshape(-1, 4 * self.bins)

        cls_reweight = cls_reweight.permute(0, 2, 3, 1).reshape(-1)

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
            # pos_bbox_pred_initial_corners = self.integral_initial(bbox_pred_initial[pos_inds])
            pos_bbox_pred_refine = bbox_pred_refine[pos_inds]
            pos_anchors = anchors[pos_inds]
            pos_anchor_centers = self.anchor_center(pos_anchors) / stride[0]

            weight_targets = cls_score.detach()
            weight_targets = weight_targets.max(dim=1)[0][pos_inds]
            pos_bbox_pred_refine_corners = pos_bbox_pred_refine
            # pos_bbox_pred_refine_corners = self.integral_refine(pos_bbox_pred_refine)
            pos_decode_bbox_pred = distance2bbox(pos_anchor_centers,
                                                 pos_bbox_pred_initial_corners + pos_bbox_pred_refine_corners)

            pos_decode_bbox_targets = pos_bbox_targets / stride[0]


            score[pos_inds] = bbox_overlaps(
                pos_decode_bbox_pred.detach(),
                pos_decode_bbox_targets,
                is_aligned=True).clamp(min=0.0)


            '''
            corners_initial = torch.stack([pos_bbox_pred_initial_corners, target_corners], dim=-1)
            corners_refine = torch.stack([pos_bbox_pred_initial_corners.detach() + pos_bbox_pred_refine_corners, target_corners], dim=-1)

            loss_bbox = 0.5 * (1.0 - (corners_initial.min(-1)[0] / corners_initial.max(-1)[0])) + 0.5 * (1.0 - (corners_refine.min(-1)[0] / corners_refine.max(-1)[0]))
            # loss_bbox = 1.0 - (corners_refine.min(-1)[0] / corners_refine.max(-1)[0])
            loss_bbox = loss_bbox.sum(-1) * weight_targets
            loss_bbox = loss_bbox.sum()
            '''


            '''
            loss_bbox = self.loss_bbox(
                pos_decode_bbox_pred,
                pos_decode_bbox_targets,
                weight=weight_targets,
                avg_factor=1.0)
            '''

            '''
            loss_bbox_initial = self.loss_bbox_initial(
                    distance2bbox(pos_anchor_centers, 
                                  pos_bbox_pred_initial_corners),
                    pos_decode_bbox_targets,
                    weight=weight_targets,
                    avg_factor=1.0) * 0.1
            '''

            loss_bbox_initial = bbox_pred_initial.sum() * 0.0


            '''
            corners = torch.stack([
                pos_bbox_pred_initial_corners,
                bbox2distance(pos_anchor_centers,
                              pos_decode_bbox_targets,
                              16)], dim=-1)

            loss_bbox_initial = 1.0 - (corners.min(-1)[0] / corners.max(-1)[0])
            loss_bbox_initial = loss_bbox_initial.sum(-1).mul(weight_targets).sum() * 0.0
            '''




            '''
            corners = torch.stack([
                pos_bbox_pred_initial_corners,
                bbox2distance(pos_anchor_centers,
                              pos_decode_bbox_targets,
                              16)], dim=-1)

            loss_bbox_initial = 1.0 - (corners.min(-1)[0] / (corners.max(-1)[0] + 1E-8))
            loss_bbox_initial = loss_bbox_initial.sum(-1).mul(weight_targets).sum()
            '''
            
            

            '''
            # loss_bbox_initial = pos_bbox_pred_refine_corners.abs().sum(-1).mul(weight_targets).sum() * 0.001
            loss_bbox_initial = smooth_l1_loss(
                        pos_bbox_pred_refine_corners, 
                        torch.zeros_like(pos_bbox_pred_refine_corners), 
                        weight=weight_targets[:, None].expand(-1, 4),
                        avg_factor=4.0) * 1.0
            '''

            # print(
            #     "mean={}, std={}, max={}, min={}".format(
            #         pos_bbox_pred_refine_corners.mean(),
            #         pos_bbox_pred_refine_corners.std(),
            #         pos_bbox_pred_refine_corners.max(),
            #         pos_bbox_pred_refine_corners.min()
            #     ))
            

            '''
            corners = torch.stack([
                pos_bbox_pred_initial_corners,
                bbox2distance(pos_anchor_centers,
                              pos_decode_bbox_targets,
                              16)], dim=-1)

            loss_bbox_initial = 1.0 - (corners.min(-1)[0] / (corners.max(-1)[0] + 1E-8))
            # loss_bbox_initial = corners.max(-1)[0] - corners.min(-1)[0]
            loss_bbox_initial = loss_bbox_initial.sum(-1).mul(weight_targets).sum() * 0.1
            '''

            # loss_bbox_initial = bbox_pred_initial.sum() * 0.0



            

            # diff = pos_bbox_pred_initial_corners - bbox2distance(pos_anchor_centers, pos_decode_bbox_targets, 16)
            # print(torch.histc(diff, 9, min=-4, max=4))

            '''
            loss_bbox_initial = self.loss_bbox_initial(
                    distance2bbox(pos_anchor_centers,
                                  pos_bbox_pred_initial_corners),
                    pos_decode_bbox_targets,
                    weight=weight_targets,
                    avg_factor=1.0) * 0.0
            '''



            loss_bbox_refine = self.loss_bbox_refine(
                    distance2bbox(pos_anchor_centers,
                                  pos_bbox_pred_initial_corners + pos_bbox_pred_refine_corners),
                    pos_decode_bbox_targets,
                    weight=weight_targets,
                    avg_factor=1.0)
            



            # print(self.loss_bbox_refine(
            #         distance2bbox(pos_anchor_centers,
            #                       pos_bbox_pred_initial_corners + pos_bbox_pred_refine_corners),
            #         pos_decode_bbox_targets,
            #         reduction_override='none').size())
            # print(cls_reweight[pos_inds].size())
            # exit(0)

            


            '''
            loss_bbox_refine = self.loss_bbox_refine(
                    distance2bbox(pos_anchor_centers,
                                  pos_bbox_pred_initial_corners + pos_bbox_pred_refine_corners),
                    pos_decode_bbox_targets,
                    weight=cls_reweight[pos_inds],
                    avg_factor=1.0) - 0.05 * (cls_reweight[pos_inds] + 1E-8).log().sum()
            '''



            '''
            corners = torch.stack([
                pos_bbox_pred_initial_corners + pos_bbox_pred_refine_corners,
                bbox2distance(pos_anchor_centers,
                              pos_decode_bbox_targets,
                              16)], dim=-1)

            loss_bbox_refine = 1.0 - (corners.min(-1)[0] / (corners.max(-1)[0] + 1E-8))
            loss_bbox_refine = loss_bbox_refine.sum(-1) * cls_reweight[pos_inds] - 0.4 * (cls_reweight[pos_inds] + 1E-8).log()
            loss_bbox_refine = loss_bbox_refine.sum() / num_total_samples
            print(loss_bbox_refine)
            exit(0)
            '''
            

            
            # loss_bbox_refine = bbox_pred_initial.sum() * 0.0

            

            
            # loss_dfl = F.softmax(bbox_pred_refine_dist[pos_inds].reshape(-1, 4, self.bins), dim=2)[:, :, self.bins//2].add(1E-8).log().neg().mean(-1) * weight_targets

            
            '''
            pred_dfl = bbox_pred_refine_dist[pos_inds].reshape(-1, self.bins)
            loss_dfl = weight_targets[:, None].expand(-1, 4).reshape(-1) * F.cross_entropy(
                pred_dfl,
                torch.zeros(pred_dfl.size(0)).fill_(self.bins//2).long().cuda(),
                reduction='none') / 4.0
            loss_dfl = loss_dfl.sum() * 0.0
            '''
            


            # print(
            #     "mean={} std={} mid={}".format(
            #         F.softmax(bbox_pred_refine_dist[pos_inds].reshape(-1, 4, self.bins), dim=2).mean(dim=2).mean(),
            #         F.softmax(bbox_pred_refine_dist[pos_inds].reshape(-1, 4, self.bins), dim=2).std(dim=2).mean(),
            #         F.softmax(bbox_pred_refine_dist[pos_inds].reshape(-1, 4, self.bins), dim=2)[:, :, self.bins//2].mean()
            #     )
            # )


            # print(
            #         F.softmax(bbox_pred_refine_dist[pos_inds].reshape(-1, 4, self.bins), dim=2)[0, 0, :]
            # )


            loss_dfl = bbox_pred_initial.sum() * 0.0


        else:
            loss_bbox_initial = bbox_pred_initial.sum() * 0.0
            loss_bbox_refine = bbox_pred_initial.sum() * 0.0
            loss_dfl = bbox_pred_initial.sum() * 0.0
            weight_targets = torch.tensor(0).cuda()

        # cls (qfl) loss
        loss_cls = self.loss_cls(
            cls_score, (labels, score),
            weight=label_weights,
            avg_factor=num_total_samples)

        return loss_cls, loss_bbox_initial, loss_bbox_refine, loss_dfl, weight_targets.sum() + 1E-8


    @force_fp32(apply_to=('cls_scores', 'bbox_preds_initial', 'bbox_preds_refine', 'bbox_preds_refine_dist', 'cls_reweight'))
    def loss(self,
             cls_scores,
             bbox_preds_initial,
             bbox_preds_refine,
             bbox_preds_refine_dist,
             cls_reweight,
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

        losses_cls, losses_bbox_initial, losses_bbox_refine, losses_dfl, \
            avg_factor = multi_apply(
                self.loss_single,
                anchor_list,
                cls_scores,
                bbox_preds_initial,
                bbox_preds_refine,
                bbox_preds_refine_dist,
                cls_reweight,
                labels_list,
                label_weights_list,
                bbox_targets_list,
                self.anchor_generator.strides,
                num_total_samples=num_total_samples)

        avg_factor = sum(avg_factor) 
        avg_factor = reduce_mean(avg_factor).item()
        losses_bbox_initial = list(map(lambda x: x / avg_factor, losses_bbox_initial))
        losses_bbox_refine = list(map(lambda x: x / avg_factor, losses_bbox_refine))
        # losses_bbox_refine = list(map(lambda x: x / num_total_samples, losses_bbox_refine))

        losses_dfl = list(map(lambda x: x / avg_factor, losses_dfl))

        return dict(
            loss_cls=losses_cls, loss_bbox_initial=losses_bbox_initial, loss_bbox_refine=losses_bbox_refine, loss_dfl=losses_dfl)


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
            
            # bbox_pred = (self.integral_initial(bbox_pred_initial.permute(1, 2, 0)) + self.integral_refine(bbox_pred_refine.permute(1, 2, 0))) * stride[0]
            # bbox_pred = (bbox_pred_initial.permute(1, 2, 0).reshape(-1, 4)
            #  + self.integral_refine(bbox_pred_refine.permute(1, 2, 0))) * stride[0]
            bbox_pred = (bbox_pred_initial.permute(1, 2, 0).reshape(-1, 4) + bbox_pred_refine.permute(1, 2, 0).reshape(-1, 4)) * stride[0]
            # bbox_pred = bbox_pred_initial.permute(1, 2, 0).reshape(-1, 4) * stride[0]


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
