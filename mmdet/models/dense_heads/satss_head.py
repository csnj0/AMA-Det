import torch
import torch.nn as nn
import torch.distributed as dist

from mmcv.cnn import ConvModule, Scale, bias_init_with_prob, normal_init, constant_init
from mmcv.runner import force_fp32

from mmdet.core import (anchor_inside_flags, build_assigner, build_sampler,
                        images_to_levels, multi_apply, multiclass_nms,
                        reduce_mean, reduce_sum, unmap, bbox_overlaps, distance2bbox)
from ..builder import HEADS, build_loss

from .base_dense_head import BaseDenseHead

EPS = 1e-12
INF = 1e+5


@HEADS.register_module()
class SATSSHead(BaseDenseHead):
    """Separated Adaptive Training Sample Selection.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 stacked_convs=4,
                 strides=(8, 16, 32, 64, 128),
                 anchor_scale=4,
                 topk=9,
                 conv_cfg=None,
                 norm_cfg=dict(type='GN', num_groups=32, requires_grad=True),
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox=dict(type='GIoULoss', loss_weight=1.0),
                 loss_centerness=dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 train_cfg=None,
                 test_cfg=None,
                 **kwargs):
        super(SATSSHead, self).__init__()

        self.num_classes = num_classes
        self.cls_out_channels = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.strides = strides
        self.anchor_scale = anchor_scale
        self.topk = topk
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox = build_loss(loss_bbox)
        self.loss_centerness = build_loss(loss_centerness)
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

        self._init_layers()

    def _init_layers(self):
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
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
        self.atss_cls = nn.Conv2d(
            self.feat_channels,
            self.cls_out_channels,
            3,
            padding=1)
        self.atss_reg = nn.Conv2d(
            self.feat_channels, 4, 3, padding=1)
        self.atss_centerness = nn.Conv2d(
            self.feat_channels, 1, 3, padding=1)
        self.scales = nn.ModuleList(
            [Scale(1.0) for _ in self.strides])

    def init_weights(self):
        """Initialize weights of the head."""
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.atss_cls, std=0.01, bias=bias_cls)
        # normal_init(self.atss_reg, std=0.01)
        constant_init(self.atss_reg, 0.0, bias=self.anchor_scale / 2.0)
        normal_init(self.atss_centerness, std=0.01)

    def forward(self, feats):
        return multi_apply(self.forward_single, feats, self.scales, self.strides)

    def forward_single(self, x, scale, stride):
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)
        cls_score = self.atss_cls(cls_feat)
        bbox_pred = scale(self.atss_reg(reg_feat)).float().relu() * stride / 2.0
        centerness = self.atss_centerness(reg_feat)
        return cls_score, bbox_pred, centerness


    def _get_target_single(self, 
                           gt_bboxes, 
                           gt_labels,
                           points,
                           anchors,
                           num_points_per_lvl):
        '''
        Args:
            gt_bboxes: tensor(num_gts, 4)
            gt_labels: tensor(num_gts)
            points: tensor(lvl*h*w, 2)
            anchors: tensor(lvl*h*w, 4)
            num_points_per_lvl: list(lvl)
        Return:
        '''
        num_points = points.size(0)
        num_gts = gt_labels.size(0)
        device = points.device
        
        if num_gts == 0:
            return gt_labels.new_full((num_points,), self.num_classes), \
                   gt_bboxes.new_zeros((num_points, 4))

        # Step 1: get the top-k closest points per level.
        gt_cx = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2.0
        gt_cy = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2.0
        gt_points = torch.stack((gt_cx, gt_cy), dim=1)
        distances = (points[:, None, :] - gt_points[None, :, :]).pow(2).sum(-1).sqrt()
        distances_lvl_list = distances.split(num_points_per_lvl, dim=0)
        k_inds_list = [distances_lvl.topk(self.topk, dim=0, largest=False).indices
                       for distances_lvl in distances_lvl_list]
        is_topk_list = [distances.new_zeros([num_points_, num_gts], device=device,
                                            dtype=torch.bool).scatter_(0, k_inds_, True)
                        for (num_points_, k_inds_) in zip(num_points_per_lvl, k_inds_list)]
        is_topk = torch.cat(is_topk_list, dim=0)

        # Step 2: filter candidate points using mean+std.
        overlaps = bbox_overlaps(anchors, gt_bboxes)
        overlaps_lvl_list = overlaps.split(num_points_per_lvl, dim=0)
        overlaps_topk_list = [overlaps_[k_inds_, torch.arange(num_gts)]
                              for (k_inds_, overlaps_) in zip(k_inds_list, overlaps_lvl_list)]
        overlaps_topk = torch.cat(overlaps_topk_list, dim=0)
        overlap_mean_per_gt = overlaps_topk.mean(dim=0)
        overlap_std_per_gt = overlaps_topk.std(dim=0)
        overlap_thr_per_gt = overlap_mean_per_gt + overlap_std_per_gt
        is_pos = overlaps.gt(overlap_thr_per_gt)
        is_pos = is_pos * is_topk

        # Step 3: get the points inside gts
        xs, ys = points[:, 0], points[:, 1]
        left = xs[:, None] - gt_bboxes[:, 0][None, :]
        right = gt_bboxes[:, 2][None, :] - xs[:, None]
        top = ys[:, None] - gt_bboxes[:, 1][None, :]
        bottom = gt_bboxes[:, 3][None, :] - ys[:, None]
        bbox_targets = torch.stack((left, top, right, bottom), -1)

        inside_matrix = bbox_targets.min(-1).values.gt(0.0)
        inside_flags = inside_matrix.sum(1).gt(0)
        inside_inds = torch.nonzero(inside_flags, as_tuple=False).reshape(-1)

        is_pos = is_pos * inside_matrix
        pos_flags = is_pos.sum(1).gt(0)
        pos_inds = torch.nonzero(pos_flags, as_tuple=False).reshape(-1)


        # Step 4: get the max IoU for overlap points
        overlaps_pos = torch.where(is_pos, overlaps, overlaps.new_tensor(-INF))
        (_, assigned_inds) = overlaps_pos.max(dim=1)
        assigned_gt_matrix = torch.zeros_like(is_pos)
        assigned_gt_matrix[pos_inds, assigned_inds[pos_inds]] = True
        # assigned_gt_inds = assigned_gt_matrix.nonzero()[:, 1]
        assigned_gt_inds = torch.nonzero(assigned_gt_matrix, as_tuple=False)[:, 1]


        # Step 5: bbox_targets
        pos_bbox_targets = bbox_targets[pos_inds, assigned_gt_inds, :]
        bbox_targets = unmap(pos_bbox_targets, num_points, pos_flags, fill=0.0)

        # Step 6: labels
        labels = gt_labels.new_full((num_points,), self.num_classes)
        labels[pos_inds] = gt_labels[assigned_gt_inds]

        return labels, bbox_targets
    
    def get_targets(self, points, gt_bboxes_list, gt_labels_list):
        '''
        Args;
            points: list(lvl)-tensor(h*w, 2)
            gt_bboxes_list: list(bs)-tensor(num_gts, 4)
            gt_labels_list: list(bs)-tensor(num_gts, N)
        '''
        num_imgs = len(gt_labels_list)
        num_points_per_lvl = [point.size(0) for point in points]
        offsets = [points[i].new_full(points[i].size(), self.anchor_scale * stride / 2)
                   for (i, stride) in enumerate(self.strides)]
        anchors_tl = [(point - offset) for (point, offset) in zip(points, offsets)]
        anchors_br = [(point + offset) for (point, offset) in zip(points, offsets)]
        anchors = [torch.cat([tl, br], dim=1) for (tl, br) in zip(anchors_tl, anchors_br)]
        anchors = torch.cat(anchors, dim=0)
        points = torch.cat(points, dim=0)

        labels_list, bbox_targets_list = multi_apply(
            self._get_target_single,
            gt_bboxes_list,
            gt_labels_list,
            points=points,
            anchors=anchors,
            num_points_per_lvl=num_points_per_lvl
        )

        return (labels_list, bbox_targets_list)


    def centerness_target(self, pos_bbox_targets):
        """Compute centerness targets.

        Args:
            pos_bbox_targets (Tensor): BBox targets of positive bboxes in shape
                (num_pos, 4)

        Returns:
            Tensor: Centerness target.
        """
        # only calculate pos centerness targets, otherwise there may be nan
        left_right = pos_bbox_targets[:, [0, 2]]
        top_bottom = pos_bbox_targets[:, [1, 3]]
        centerness_targets = (
            left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * (
                top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)
    

    def get_points_single(self, featmap_size, stride, dtype, device):
        h, w = featmap_size
        x_range = torch.arange(
            0, w * stride, stride, dtype=dtype, device=device)
        y_range = torch.arange(
            0, h * stride, stride, dtype=dtype, device=device)
        y, x = torch.meshgrid(y_range, x_range)
        points = torch.stack(
            (x.reshape(-1), y.reshape(-1)), dim=-1) + stride / 2.0
        return points

    def get_points(self, featmap_sizes, dtype, device):
        """Get points according to feature map sizes.

        Args:
            featmap_sizes: list(lvl, 2)
        Returns:
            list(lvl)-tensor(h*w, 2)
        """
        mlvl_points = []
        for i in range(len(featmap_sizes)):
            mlvl_points.append(
                self.get_points_single(featmap_sizes[i], self.strides[i],
                                       dtype, device))
        return mlvl_points


    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def loss(self,
             cls_scores,
             bbox_preds,
             centernesses,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        assert len(cls_scores) == len(bbox_preds) == len(centernesses)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]

        points = self.get_points(featmap_sizes, bbox_preds[0].dtype, bbox_preds[0].device)
        (labels_list, bbox_targets_list) = self.get_targets(points, gt_bboxes, gt_labels)
        points = torch.cat(points, dim=0)

        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(cls_score.size(0), -1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1)

        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(bbox_pred.size(0), -1, 4)
            for bbox_pred in bbox_preds
        ]
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)

        flatten_centerness = [
            centerness.permute(0, 2, 3, 1).reshape(centerness.size(0), -1)
            for centerness in centernesses
        ]
        flatten_centerness = torch.cat(flatten_centerness, dim=1)

        num_total_pos = 0.0
        num_total_bbox = 0.0
        loss_cls_list = []
        loss_bbox_list = []
        loss_centerness_list = []

        for (labels, bbox_targets, cls_scores, bbox_preds, centernesses) in zip(
                                                        labels_list,
                                                        bbox_targets_list,
                                                        flatten_cls_scores,
                                                        flatten_bbox_preds,
                                                        flatten_centerness):
            bg_class_ind = self.num_classes
            pos_inds = ((labels >= 0) & (labels < bg_class_ind)).nonzero(as_tuple=False).reshape(-1)
            
            num_pos = len(pos_inds)
            num_total_pos += num_pos

            loss_cls = self.loss_cls(cls_scores, labels, reduction_override='none')
            loss_cls_list.append(loss_cls.sum())

            if num_pos > 0:
                pos_decoded_bbox_preds = distance2bbox(points[pos_inds], 
                                                       bbox_preds[pos_inds])
                pos_decoded_bbox_targets = distance2bbox(points[pos_inds],
                                                         bbox_targets[pos_inds])
                pos_centerness_targets = self.centerness_target(bbox_targets[pos_inds])

                loss_bbox = self.loss_bbox(
                                pos_decoded_bbox_preds, 
                                pos_decoded_bbox_targets,
                                weight=pos_centerness_targets,
                                reduction_override='none')

                num_total_bbox += pos_centerness_targets.sum()
                loss_bbox_list.append(loss_bbox.sum())
                
                loss_centerness = self.loss_centerness(
                                        centernesses[pos_inds],
                                        pos_centerness_targets,
                                        reduction_override='none')
                loss_centerness_list.append(loss_centerness.sum())

        num_total_pos = reduce_mean(torch.tensor(num_total_pos).cuda()).item()
        num_total_bbox = reduce_mean(torch.tensor(num_total_bbox).cuda()).item()

        loss_cls = torch.stack(loss_cls_list).sum() / max(num_total_pos, 1)
        loss_centerness = torch.stack(loss_centerness_list).sum() / max(num_total_pos, 1)
        loss_bbox = torch.stack(loss_bbox_list).sum() / max(num_total_bbox, 1E-5)

        return dict(
            loss_cls=loss_cls,
            loss_cent=loss_centerness,
            loss_bbox=loss_bbox
        )


    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'centernesses'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   centernesses,
                   img_metas,
                   cfg=None,
                   rescale=False,
                   with_nms=True):
        """Transform network output for a batch into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for each scale level
                with shape (N, num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for each scale
                level with shape (N, num_points * 4, H, W).
            centernesses (list[Tensor]): Centerness for each scale level with
                shape (N, num_points * 1, H, W).
            img_metas (list[dict]): Meta information of each image, e.g.,
                image size, scaling factor, etc.
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used. Default: None.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            list[tuple[Tensor, Tensor]]: Each item in result_list is 2-tuple.
                The first item is an (n, 5) tensor, where the first 4 columns
                are bounding box positions (tl_x, tl_y, br_x, br_y) and the
                5-th column is a score between 0 and 1. The second item is a
                (n,) tensor where each item is the predicted class label of the
                corresponding box.
        """
        assert len(cls_scores) == len(bbox_preds)
        num_levels = len(cls_scores)

        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        mlvl_points = self.get_points(featmap_sizes, bbox_preds[0].dtype,
                                      bbox_preds[0].device)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            centerness_pred_list = [
                centernesses[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            det_bboxes = self._get_bboxes_single(
                cls_score_list, bbox_pred_list, centerness_pred_list,
                mlvl_points, img_shape, scale_factor, cfg, rescale, with_nms)
            result_list.append(det_bboxes)
        return result_list

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           centernesses,
                           mlvl_points,
                           img_shape,
                           scale_factor,
                           cfg,
                           rescale=False,
                           with_nms=True):
        """Transform outputs for a single batch item into bbox predictions.

        Args:
            cls_scores (list[Tensor]): Box scores for a single scale level
                with shape (num_points * num_classes, H, W).
            bbox_preds (list[Tensor]): Box energies / deltas for a single scale
                level with shape (num_points * 4, H, W).
            centernesses (list[Tensor]): Centerness for a single scale level
                with shape (num_points * 4, H, W).
            mlvl_points (list[Tensor]): Box reference for a single scale level
                with shape (num_total_points, 4).
            img_shape (tuple[int]): Shape of the input image,
                (height, width, 3).
            scale_factor (ndarray): Scale factor of the image arrange as
                (w_scale, h_scale, w_scale, h_scale).
            cfg (mmcv.Config | None): Test / postprocessing configuration,
                if None, test_cfg would be used.
            rescale (bool): If True, return boxes in original image space.
                Default: False.
            with_nms (bool): If True, do nms before return boxes.
                Default: True.

        Returns:
            tuple(Tensor):
                det_bboxes (Tensor): BBox predictions in shape (n, 5), where
                    the first 4 columns are bounding box positions
                    (tl_x, tl_y, br_x, br_y) and the 5-th column is a score
                    between 0 and 1.
                det_labels (Tensor): A (n,) tensor where each item is the
                    predicted class label of the corresponding box.
        """
        cfg = self.test_cfg if cfg is None else cfg
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points)
        mlvl_bboxes = []
        mlvl_scores = []
        mlvl_centerness = []
        for cls_score, bbox_pred, centerness, points in zip(
                cls_scores, bbox_preds, centernesses, mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels).sigmoid()
            centerness = centerness.permute(1, 2, 0).reshape(-1).sigmoid()

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = (scores * centerness[:, None]).max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
                centerness = centerness[topk_inds]
            bboxes = distance2bbox(points, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_centerness.append(centerness)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)
        mlvl_centerness = torch.cat(mlvl_centerness)

        if with_nms:
            det_bboxes, det_labels = multiclass_nms(
                mlvl_bboxes,
                mlvl_scores,
                cfg.score_thr,
                cfg.nms,
                cfg.max_per_img,
                score_factors=mlvl_centerness)
            return det_bboxes, det_labels
        else:
            return mlvl_bboxes, mlvl_scores, mlvl_centerness

