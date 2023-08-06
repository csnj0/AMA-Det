import torch
import torch.nn as nn
import torch.distributed as dist
import torch.nn.functional as F

from mmcv.cnn import ConvModule, Scale, bias_init_with_prob, normal_init, constant_init
from mmcv.runner import force_fp32

from mmdet.core import (anchor_inside_flags, build_assigner, build_sampler,
                        images_to_levels, multi_apply, multiclass_nms,
                        reduce_mean, reduce_sum, unmap, bbox_overlaps, distance2bbox)
from ..builder import HEADS, build_loss

from .base_dense_head import BaseDenseHead

EPS = 1e-8
INF = 1e+8


class Integral(nn.Module):
    def __init__(self, reg_max=16):
        super(Integral, self).__init__()
        self.reg_max = reg_max
        self.register_buffer('project',
                             torch.linspace(0, self.reg_max, self.reg_max + 1))
    def forward(self, x):

        # x = F.softmax(x.reshape(-1, self.reg_max + 1), dim=1)
        # x = F.linear(x, self.project.type_as(x)).reshape(-1, 4)

        x = F.softmax(x, dim=-1) # [N, 4, bins]
        x = torch.sum(x * self.project[None, None, :].type_as(x), dim=-1)
        return x



@HEADS.register_module()
class ATSS2BHead(BaseDenseHead):
    """Separated Adaptive Training Sample Selection.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 stacked_convs=4,
                 strides=(8, 16, 32, 64, 128),
                 anchor_scale=6,
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
                 train_cfg=None,
                 test_cfg=None,
                 **kwargs):
        super(ATSS2BHead, self).__init__()

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
        
        self.reg_max = 16
        self.atss_reg = nn.Conv2d(
            self.feat_channels, 
            4 * (self.reg_max + 1),
            3, 
            padding=1)


        self.integral = Integral(self.reg_max)

        self.scales = nn.ModuleList(
            [Scale(1.0) for _ in self.strides])


        '''
        self.bbox_reweight = nn.Sequential(
            nn.Conv2d(5*4, 256, 1),
            self.relu,
            nn.Conv2d(256, 1, 1),
            nn.Sigmoid()
        )

        self.reg_conf = nn.Sequential(
            nn.Conv2d(self.feat_channels, 1, 3, padding=1),
            nn.Sigmoid()
        )
        '''
        
        self.reg_topk = 4
        self.total_dim = self.reg_topk + 1
        self.reg_channels = 64
        self.reg_conf = nn.Sequential(
            nn.Conv2d(4 * self.total_dim, self.reg_channels, 1),
            self.relu,
            nn.Conv2d(self.reg_channels, 1, 1),
            nn.Sigmoid()
        )


    def init_weights(self):
        """Initialize weights of the head."""
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.atss_cls, std=0.01, bias=bias_cls)


        nn.init.constant_(self.atss_reg.weight, 0.0)
        nn.init.constant_(self.atss_reg.bias, -2.0)
        for i in range(4):
            nn.init.constant_(self.atss_reg.bias[4 + 9 * i], 2.0)


        for m in self.reg_conf:
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01)


        '''
        for m in self.bbox_reweight:
            if isinstance(m, nn.Conv2d):
                normal_init(m, std=0.01, bias=bias_init_with_prob(0.99))
        '''


    def forward(self, feats):
        return multi_apply(self.forward_single, feats, self.scales, self.strides)

    def forward_single(self, x, scale, stride):
        cls_feat = x
        reg_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            reg_feat = reg_conv(reg_feat)

        # cls_score = self.atss_cls(cls_feat).float().sigmoid() * self.reg_conf(reg_feat).float()
        cls_score = self.atss_cls(cls_feat).float().sigmoid()
        bbox_pred = self.atss_reg(reg_feat).float()


        '''
        N, C, H, W = bbox_pred.size()
        prob = F.softmax(bbox_pred.reshape(N, 4, self.reg_max+1, H, W), dim=2)
        prob_topk, _ = topk_unsorted(prob, k=self.reg_topk, dim=2, largest=True)
        stat = torch.cat([prob_topk, prob_topk.mean(dim=2, keepdim=True)], dim=2)
        quality_score = self.reg_conf(stat.reshape(N, -1, H, W))
        cls_score = self.atss_cls(cls_feat).float().sigmoid() * quality_score
        '''


        size_ = [bbox_pred.size(0), 1, bbox_pred.size(2), bbox_pred.size(3)]
        scale_factor = bbox_pred.new_ones(size_, requires_grad=False)
        scale_factor = scale(scale_factor) * stride
        
        #  * self.anchor_scale

        return cls_score, bbox_pred, scale_factor


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

        inside_matrix = bbox_targets.min(-1).values.gt(0.01)
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

        return labels, bbox_targets, assigned_gt_matrix
    
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

        labels_list, bbox_targets_list, assigned_gt_matrix_list = multi_apply(
            self._get_target_single,
            gt_bboxes_list,
            gt_labels_list,
            points=points,
            anchors=anchors,
            num_points_per_lvl=num_points_per_lvl
        )

        return (labels_list, bbox_targets_list, assigned_gt_matrix_list)

    
    def points_prob_target(self, bbox_ious, gt_labels, num_points):
        # bbox_ious = bbox_ious.t_()
        # t1 = 0.6
        # t2 = bbox_ious.max(dim=1, keepdim=True).values.clamp(min=t1 + 1e-12)
        # gts_points_prob = torch.clamp((bbox_ious - t1) / (t2 - t1), min=0.0, max=1.0)

        gts_points_prob = bbox_ious.t_()

        indices = torch.stack(
            [torch.arange(gt_labels.size(0)).type_as(gt_labels), gt_labels],
            dim=0
        )
        gts_cls_points_prob = torch.sparse.LongTensor(
            indices, gts_points_prob,
            torch.Size([gt_labels.size(0), self.cls_out_channels, num_points])
        ).cuda()
        points_cls_prob = torch.sparse.sum(gts_cls_points_prob, dim=0).to_dense()
        indices = torch.nonzero(points_cls_prob, as_tuple=False).t_()

        if indices.numel() == 0:
            points_prob = gts_points_prob.new_zeros(
                num_points, self.cls_out_channels)
        else:
            nonzero_points_prob = torch.where(
                (gt_labels.unsqueeze(dim=-1)) == indices[0],
                gts_points_prob[:, indices[1]],
                gts_points_prob.new_zeros([1])
            ).max(dim=0).values
            points_prob = torch.sparse.FloatTensor(
                indices.flip([0]),
                nonzero_points_prob,
                torch.Size([num_points, self.cls_out_channels])
            ).to_dense()
        return points_prob
    

    def get_points_single(self, featmap_size, stride, dtype, device):
        h, w = featmap_size
        x_range = torch.arange(
            0, w * stride, stride, dtype=dtype, device=device)
        y_range = torch.arange(
            0, h * stride, stride, dtype=dtype, device=device)
        y, x = torch.meshgrid(y_range, x_range)
        points = torch.stack(
            (x.reshape(-1), y.reshape(-1)), dim=-1) + stride // 2
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


    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'scale_factors'))
    def loss(self,
             cls_scores,
             bbox_preds,
             scale_factors,
             gt_bboxes,
             gt_labels,
             img_metas,
             gt_bboxes_ignore=None):
        assert len(cls_scores) == len(bbox_preds)
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]

        points = self.get_points(featmap_sizes, bbox_preds[0].dtype, bbox_preds[0].device)
        (labels_list, bbox_targets_list, assigned_gt_matrix_list) = self.get_targets(points, gt_bboxes, gt_labels)
        points = torch.cat(points, dim=0)

        flatten_cls_scores = [
            cls_score.permute(0, 2, 3, 1).reshape(cls_score.size(0), -1, self.cls_out_channels)
            for cls_score in cls_scores
        ]
        flatten_cls_scores = torch.cat(flatten_cls_scores, dim=1)

        flatten_bbox_preds = [
            bbox_pred.permute(0, 2, 3, 1).reshape(bbox_pred.size(0), -1, 4, self.reg_max+1)
            for bbox_pred in bbox_preds
        ]
        flatten_bbox_preds = torch.cat(flatten_bbox_preds, dim=1)

        flatten_scale_factors = [
            scale_factor.permute(0, 2, 3, 1).reshape(scale_factor.size(0), -1, 1)
            for scale_factor in scale_factors
        ]
        flatten_scale_factors = torch.cat(flatten_scale_factors, dim=1)


        num_total_pos = 0.0
        num_total_bbox = 0.0
        loss_cls_list = []
        loss_bbox_list = []
        loss_conf_list = []

        for (labels, bbox_targets, assigned_gt_matrix, cls_scores, bbox_preds, scale_factors) in zip(
                                                        labels_list,
                                                        bbox_targets_list,
                                                        assigned_gt_matrix_list,
                                                        flatten_cls_scores,
                                                        flatten_bbox_preds,
                                                        flatten_scale_factors,
                                                        ):
            cls_scores = cls_scores.float()
            bbox_preds = bbox_preds.float()
            scale_factors = scale_factors.float()

            bg_class_ind = self.num_classes
            pos_inds = ((labels >= 0) & (labels < bg_class_ind)).nonzero(as_tuple=False).reshape(-1)
            
            num_pos = len(pos_inds)
            num_total_pos += num_pos

            pos_bbox_preds = self.integral(bbox_preds[pos_inds]) * scale_factors[pos_inds]
            pos_decoded_bbox_preds = distance2bbox(
                                        points[pos_inds], 
                                        pos_bbox_preds
                                        )
            pos_decoded_bbox_targets = distance2bbox(
                                        points[pos_inds], 
                                        bbox_targets[pos_inds]
                                        )

            points_probs = cls_scores.new_zeros([cls_scores.size(0), 1])
            points_probs[pos_inds, 0] =  bbox_overlaps(
                                            pos_decoded_bbox_preds, 
                                            pos_decoded_bbox_targets,
                                            is_aligned=True
                                        )

            cls_mask = cls_scores.new_zeros(cls_scores.size())
            cls_mask[pos_inds, labels[pos_inds]] = 1.0
            cls_weight = torch.where(
                                cls_mask.bool(),
                                (points_probs.detach() - cls_scores).pow(2.0),
                                cls_scores.pow(2.0)
                            )
            loss_cls = cls_weight * F.binary_cross_entropy(
                            cls_scores, 
                            points_probs.detach() * cls_mask,
                            reduction='none'
                        )
            loss_cls_list.append(loss_cls.sum())
        


            if num_pos > 0:
                
                bbox_weight = cls_scores[pos_inds, labels[pos_inds]].detach()

                loss_bbox =  bbox_weight * self.loss_bbox(
                                pos_decoded_bbox_preds,
                                pos_decoded_bbox_targets,
                                reduction_override='none')
                loss_bbox_list.append(loss_bbox.sum())

                num_total_bbox += bbox_weight.sum().item()
                


                rev_pos_bbox_targets = bbox_targets[pos_inds] / (scale_factors[pos_inds].detach() + 1E-8) # [N, 4]

                rev_pos_bbox_targets = rev_pos_bbox_targets.clamp(max=self.reg_max-0.01)
                label_left  = rev_pos_bbox_targets.long()
                label_right = rev_pos_bbox_targets.long() + 1


                loss_conf = (rev_pos_bbox_targets.reshape(-1) - label_left.reshape(-1))  * F.cross_entropy(bbox_preds[pos_inds].reshape([-1, self.reg_max+1]), label_right.reshape([-1]), reduction='none') + \
                            (label_right.reshape(-1) - rev_pos_bbox_targets.reshape(-1)) * F.cross_entropy(bbox_preds[pos_inds].reshape([-1, self.reg_max+1]), label_left.reshape([-1]), reduction='none')
                loss_conf = bbox_weight[:, None].expand(-1, 4).reshape(-1) * loss_conf
                loss_conf_list.append(loss_conf.sum() / 4.0)



                '''
                weight_matrix = assigned_gt_matrix.new_zeros(assigned_gt_matrix.size()).float()
                assigned_gt_inds = torch.nonzero(assigned_gt_matrix, as_tuple=False)[:, 1]
                weight_matrix[pos_inds, assigned_gt_inds] = 1.0 / (1.0 - bbox_reweights[pos_inds] + 1E-8) 
                weight_matrix = weight_matrix / (weight_matrix.sum(1, keepdim=True) + 1E-8)

                loss_bbox_matrix = assigned_gt_matrix.new_zeros(assigned_gt_matrix.size()).float()


                loss_bbox_matrix[pos_inds, assigned_gt_inds] =  self.loss_bbox(
                                                                pos_decoded_bbox_preds,
                                                                pos_decoded_bbox_targets,
                                                                reduction_override='none')
                loss_bbox_matrix *=  weight_matrix
                loss_bbox2_list.append(loss_bbox_matrix.sum())
                num_total_gts += assigned_gt_matrix.size(1)
                '''
                

            else:
                loss_bbox = bbox_preds.sum() * 0.0
                loss_bbox_list.append(loss_bbox)

                loss_conf = bbox_preds.sum() * 0.0
                loss_conf_list.append(loss_conf)


        num_total_pos = reduce_sum(torch.tensor(num_total_pos).cuda()).item()
        num_total_bbox = reduce_sum(torch.tensor(num_total_bbox).cuda()).item()


        loss_cls = torch.stack(loss_cls_list).sum() * dist.get_world_size() / max(num_total_pos, 1)
        loss_bbox = torch.stack(loss_bbox_list).sum() * dist.get_world_size() / max(num_total_bbox, 1E-8)
        loss_conf = torch.stack(loss_conf_list).sum() * dist.get_world_size() / max(num_total_bbox, 1E-8)
        # loss_conf = torch.stack(loss_conf_list).sum() * dist.get_world_size() / max(num_total_pos, 1)



        return dict(
            loss_cls=loss_cls,
            loss_bbox=2.0 * loss_bbox,
            loss_conf=0.25 * loss_conf
        )


    @force_fp32(apply_to=('cls_scores', 'bbox_preds', 'scales'))
    def get_bboxes(self,
                   cls_scores,
                   bbox_preds,
                   scales,
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
            scale_list = [
                scales[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            det_bboxes = self._get_bboxes_single(
                cls_score_list, bbox_pred_list, scale_list,
                mlvl_points, img_shape, scale_factor, cfg, rescale, with_nms)
            result_list.append(det_bboxes)
        return result_list

    def _get_bboxes_single(self,
                           cls_scores,
                           bbox_preds,
                           scales,
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
        for cls_score, bbox_pred, scale, points in zip(
                cls_scores, bbox_preds, scales, mlvl_points):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            # scores = cls_score.permute(1, 2, 0).reshape(
            #     -1, self.cls_out_channels).sigmoid()

            scores = cls_score.permute(1, 2, 0).reshape(
                -1, self.cls_out_channels)

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4, self.reg_max+1)
        
            scale = scale.permute(1, 2, 0).reshape(-1, 1)
            bbox_pred = self.integral(bbox_pred) * scale

            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                max_scores, _ = scores.max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
            bboxes = distance2bbox(points, bbox_pred, max_shape=img_shape)
            mlvl_bboxes.append(bboxes.float())
            mlvl_scores.append(scores.float())
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
        # remind that we set FG labels to [0, num_class-1] since mmdet v2.0
        # BG cat_id: num_class
        mlvl_scores = torch.cat([mlvl_scores, padding], dim=1)

        if with_nms:
            det_bboxes, det_labels = multiclass_nms(
                mlvl_bboxes,
                mlvl_scores,
                cfg.score_thr,
                cfg.nms,
                cfg.max_per_img)
            return det_bboxes, det_labels
        else:
            return mlvl_bboxes, mlvl_scores


def topk_unsorted(x, k=1, dim=0, largest=True):
    val, idx = torch.topk(x, k, dim=dim, largest=largest)
    sorted_idx, new_idx = torch.sort(idx, dim=dim)
    val = torch.gather(val, dim=dim, index=new_idx)

    return val, sorted_idx

