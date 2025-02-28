from __future__ import division

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import normal_init
from mmdet.core import (AnchorGeneratorRotated, anchor_target,
                        build_bbox_coder, delta2bbox_rotated, force_fp32,
                        images_to_levels, multi_apply, multiclass_nms_rotated)
from mmdet.models.losses import FocalLoss
from mmdet.models.losses.cross_entropy_loss import binary_cross_entropy

from ...ops import DeformConv
from ...ops.orn import ORConv2d, RotationInvariantPooling
from ..builder import build_loss
from ..registry import HEADS
from ..utils import ConvModule, bias_init_with_prob

#from ..plugins import AdaptiveAttention, SqueezeExcitationSpatialAttention
import torch.nn.functional as F
DEBUG = False


class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, alpha=.25, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha]).cuda()
        self.gamma = gamma

    def forward(self, inputs, targets, weight):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets.float(), weight, reduction='none')
        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        print(targets.shape)
        print(at.shape)
        print(pt.shape)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        return F_loss.mean()


@HEADS.register_module
class S2ANetHeadRC05(nn.Module):

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 stacked_convs=2,
                 with_orconv=True,
                 anchor_scales=[4],
                 anchor_ratios=[1.0],
                 anchor_strides=[8, 16, 32, 64, 128],
                 anchor_base_sizes=None,
                 target_means=(.0, .0, .0, .0, .0),
                 target_stds=(1.0, 1.0, 1.0, 1.0, 1.0),
                 loss_fam_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_fam_bbox=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                 loss_odm_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_odm_xwh_bbox=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0)):
        super(S2ANetHeadRC05, self).__init__()
        self.num_classes = num_classes
        self.in_channels = in_channels
        self.feat_channels = feat_channels
        self.stacked_convs = stacked_convs
        self.with_orconv = with_orconv
        self.anchor_scales = anchor_scales
        self.anchor_ratios = anchor_ratios
        self.anchor_strides = anchor_strides
        self.anchor_base_sizes = list(
            anchor_strides) if anchor_base_sizes is None else anchor_base_sizes
        self.target_means = target_means
        self.target_stds = target_stds

        self.use_sigmoid_cls = loss_odm_cls.get('use_sigmoid', False)
        self.sampling = loss_odm_cls['type'] not in ['FocalLoss', 'GHMC']
        if self.use_sigmoid_cls:
            self.cls_out_channels = num_classes - 1
        else:
            self.cls_out_channels = num_classes

        if self.cls_out_channels <= 0:
            raise ValueError('num_classes={} is too small'.format(num_classes))
        self.loss_fam_cls = build_loss(loss_fam_cls)
        # start edit
        self.loss_fam_bbox = build_loss(loss_fam_bbox)
        #self.loss_fam_theta_bbox = self._loss_dcl_single
        # end edit
        self.loss_odm_cls = build_loss(loss_odm_cls)
        # start edit
        self.loss_odm_xwh_bbox = build_loss(loss_odm_xwh_bbox)
        self.loss_odm_theta_bbox = self._loss_dcl_single
        self.xwh_lambda = 1
        self.theta_lambda = 0.1
        self.cls_lambda = 1
        self.theta_range = (3*np.pi/4, -np.pi/4)
        # end edit
        self.fp16_enabled = False
        # start edit
        self.theta_num_classes = 8
        self.theta_granulaty = np.pi / (2**self.theta_num_classes)
        self.aspect_ratio_thresh = 1.5
        # end edit
        self.anchor_generators = []
        for anchor_base in self.anchor_base_sizes:
            self.anchor_generators.append(
                AnchorGeneratorRotated(anchor_base, anchor_scales, anchor_ratios))
        # training mode
        self.training = True
        # anchor cache
        self.base_anchors = dict()
        self._init_layers()

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=True)
        self.fam_reg_convs = nn.ModuleList()
        #self.fam_reg_adaptive_attention = nn.ModuleList()
        #self.fam_reg_spatial_attention = nn.ModuleList()
        self.fam_cls_convs = nn.ModuleList()
        #self.fam_cls_attention = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.fam_reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1))
            # self.fam_reg_spatial_attention.append(
            #     SqueezeExcitationSpatialAttention(self.feat_channels)
            # )
            # self.fam_reg_adaptive_attention.append(
            #     AdaptiveAttention(self.feat_channels)
            # )
            self.fam_cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1))
            # self.fam_cls_attention.append(
            #     AdaptiveAttention(self.feat_channels)
            # )
        # start edit
        self.fam_reg = nn.Conv2d(self.feat_channels, 5, 1)
        # self.fam_theta_cls = nn.Conv2d(
        #     in_channels=self.feat_channels,
        #     # 8 bit encoding, so we've 256 angle
        #     # omega = 180/256 = 0.703125
        #     out_channels=self.theta_num_classes,
        #     kernel_size=1
        # )
        # end edit
        self.fam_cls = nn.Conv2d(self.feat_channels, self.cls_out_channels, 1)

        self.align_conv = AlignConv(
            self.feat_channels, self.feat_channels, kernel_size=3)

        if self.with_orconv:
            self.or_conv = ORConv2d(self.feat_channels, int(
                self.feat_channels / 8), kernel_size=3, padding=1, arf_config=(1, 8))
        else:
            self.or_conv = nn.Conv2d(
                self.feat_channels, self.feat_channels, 3, padding=1)
        self.or_pool = RotationInvariantPooling(256, 8)

        self.odm_reg_convs = nn.ModuleList()
        # self.odm_reg_adaptive_attention = nn.ModuleList()
        # self.odm_reg_spatial_attention = nn.ModuleList()
        self.odm_cls_convs = nn.ModuleList()
        # self.odm_cls_attention = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = int(self.feat_channels /
                      8) if i == 0 and self.with_orconv else self.feat_channels
            self.odm_reg_convs.append(
                ConvModule(
                    self.feat_channels,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1))
            # self.odm_reg_spatial_attention.append(
            #     SqueezeExcitationSpatialAttention(self.feat_channels)
            # )
            # self.odm_reg_adaptive_attention.append(
            #     AdaptiveAttention(self.feat_channels)
            # )
            self.odm_cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1))
            # self.odm_cls_attention.append(
            #     AdaptiveAttention(self.feat_channels)
            # )

        self.odm_cls = nn.Conv2d(
            self.feat_channels, self.cls_out_channels, 3, padding=1)
        # start edit
        self.odm_xwh_reg = nn.Conv2d(self.feat_channels, 4, 3, padding=1)
        self.odm_theta_cls = nn.Conv2d(
            in_channels=self.feat_channels,
            # 8 bit encoding, so we've 256 angle
            # omega = 180/256 = 0.703125
            out_channels=self.theta_num_classes,
            kernel_size=3,
            padding=1
        )
        # end edit

    def init_weights(self):
        for m in self.fam_reg_convs:
            normal_init(m.conv, std=0.01)
        for m in self.fam_cls_convs:
            normal_init(m.conv, std=0.01)
        # for m in self.fam_reg_adaptive_attention:
        #     m.init_weights()
        # for m in self.fam_reg_spatial_attention:
        #     m.init_weights()
        # for m in self.fam_cls_attention:
        #     m.init_weights()
        bias_cls = bias_init_with_prob(0.01)
        # start edit
        normal_init(self.fam_reg, std=0.01)
        # normal_init(self.fam_theta_cls, std=0.01, bias=bias_cls)
        # end edit
        normal_init(self.fam_cls, std=0.01, bias=bias_cls)

        self.align_conv.init_weights()

        normal_init(self.or_conv, std=0.01)
        for m in self.odm_reg_convs:
            normal_init(m.conv, std=0.01)
        for m in self.odm_cls_convs:
            normal_init(m.conv, std=0.01)
        # for m in self.odm_reg_adaptive_attention:
        #     m.init_weights()
        # for m in self.odm_reg_spatial_attention:
        #     m.init_weights()
        # for m in self.odm_cls_attention:
        #     m.init_weights()
        normal_init(self.odm_cls, std=0.01, bias=bias_cls)
        # start edit
        normal_init(self.odm_xwh_reg, std=0.01)
        normal_init(self.odm_theta_cls, std=0.01, bias=bias_cls)
        # end edit

    # start edit
    def _theta_decode(self, cls_theta):
        # Here we subtract eps from cls_theta because when
        # we have 0s and 1s only (no need for sigmoid) we
        # will get only 1s when we round the results
        # EPS = 1e-4
        # with open('/content/cls_theta_for_decode.txt', 'a') as f:
        #     f.write(f'cls_theta:{cls_theta}\n\n')
        theta_prob = torch.sigmoid(cls_theta)
        theta_round = torch.round(theta_prob)
        pred_int = self._bin_to_int(theta_round)
        # with open('/content/pred_int.txt', 'a') as f:
        #     f.write(f'theta_round: {theta_round} -> pred_int: {pred_int}\n\n')
        reg_theta = self.theta_range[0] - self.theta_granulaty * pred_int
        # with open('/content/reg_theta.txt', 'a') as f:
        #     f.write(f'reg_theta: {reg_theta} <- pred_int: {pred_int}\n\n')
        if len(reg_theta.shape) != 4:
          reg_theta = reg_theta.unsqueeze(-1)
        return reg_theta

    def _theta_encode(self, reg_theta):
        int_theta = (reg_theta-self.theta_range[0])/self.theta_granulaty
        # with open('/content/int_theta.txt', 'a') as f:
        #     f.write(f'reg_theta: {reg_theta} -> int_theta: {int_theta}\n\n')
        int_theta = - torch.round(int_theta)
        cls_theta = self._int_to_bin(int_theta)
        # with open('/content/cls_theta.txt', 'a') as f:
        #     f.write(f'int_theta: {int_theta} -> cls_theta: {cls_theta}\n\n')
        return cls_theta

    def _int_to_bin(self, pred_int):
        mask = 2 ** torch.arange(
            self.theta_num_classes - 1, -1, -1
        ).to(pred_int.device, torch.int)
        return ((pred_int.unsqueeze(-1).int())&(mask)).ne(0).long()

    def _bin_to_int(self, pred_bin):
        mask = 2 ** torch.arange(
            self.theta_num_classes - 1, -1, -1
        ).to(pred_bin.device, pred_bin.dtype)
        return torch.sum(mask * pred_bin, -1).long()

    def _w_adarsw(self, gt_theta, gt_h, gt_w, pred_theta):
        reg_theta = self._theta_decode(pred_theta)
        # with open('int_to_bin.txt', 'a') as f:
        #     f.write(f'reg_theata: {reg_theta} <- bin_theta: {pred_theta}\n\n')
        diff = gt_theta-reg_theta
        #alpha = 1 if (gt_h/gt_w) <= self.aspect_ratio_thresh else 2
        # NOTE: we switched w and h here beacase we know that
        # is the long side not h (as in paper).
        #print(gt_w.shape)
        #print(gt_h.shape)
        w_h = torch.stack((gt_w, gt_h), dim=-1)
        #print(w_h.shape)
        w = torch.max(w_h, dim=-1).values
        h = torch.min(w_h, dim=-1).values
        assert w.shape == gt_w.shape
        assert h.shape == gt_h.shape
        alpha = ((w/h) <= self.aspect_ratio_thresh).int() + 1
        # alpha = 1
        # with open('/content/weight_log.txt', 'a') as f:
        #     f.write(f'diff: {diff}, alpha: {alpha}\n')
        w = torch.abs(torch.sin(alpha*diff))
        return w

    def _loss_dcl_single(self, gt_theta, gt_h, gt_w, pred_theta, weight, avg_factor):
        # print(f'gt_theta: {gt_theta}')
        # focal_loss = FocalLoss(loss_weight=weight)
        gt_bin_theta = self._theta_encode(gt_theta)
        # with open('/content/int_to_bin.txt', 'a') as f:
        #     f.write(f'reg_theata: {gt_theta} -> bin_theta: {gt_bin_theta}\n\n')
        gt_bin_theta = gt_bin_theta.squeeze(1)
        # print(f'gt_bin_theta: {gt_bin_theta}')
        # print(f'pred_theta: {pred_theta}')
        # f_loss = focal_loss(pred_theta, gt_bin_theta)

        # NOTE: we've moved to bce instead of focal loss
        # because focal loss seems to implement ce not bce
        # and also here we don't have background class so no need for it

        # cls_weight = 2**torch.arange(
        #     self.theta_num_classes - 1, -1, -1
        # ).to(pred_theta.device, torch.int).unsqueeze(0)
        # cls_weight = cls_weight / torch.sum(cls_weight)
        # cls_weight = cls_weight*weight

        # print(f'cls_weight.shape: {cls_weight.shape}')
        # print(f'pred_theta.shape: {pred_theta.shape}')
        # print(f'gt_bin_theta.shape: {gt_bin_theta.shape}')

        # NOTE: here we tried to increase the loss for theta to see if
        # it will train, and it did, but when we will train on the whole
        # dataset we will let it as it is.
        #focal_loss = WeightedFocalLoss()
        #f_loss = self.focal_loss(pred_theta, gt_bin_theta.float())
        #f_loss = f_loss*weight
        #f_loss = f_loss.mean()
        f_loss = binary_cross_entropy(
            pred_theta, gt_bin_theta, weight=weight,
            avg_factor=avg_factor
        )
        #f_loss = binary_cross_entropy(pred_theta, gt_bin_theta, weight=weight,
        #                              reduction='sum')
        #non_zero_count = torch.sum(weight==1) + 1
        #f_loss = f_loss / non_zero_count
        w = self._w_adarsw(gt_theta, gt_h, gt_w, pred_theta)

        # with open('/content/dcl_loss_log.txt', 'a') as f:
        #     f.write(f'f_loss: {f_loss}\n')
        #     f.write(f'pred_theta: {torch.round(torch.sigmoid(pred_theta))}\n')
        #     f.write(f'gt_bin_theta: {gt_bin_theta}\n\n\n')
        loss = w * f_loss
        #loss = f_loss
        return loss
    # end edit


    def forward_single(self, x, stride):
        fam_reg_feat = x
        for i, fam_reg_conv in enumerate(self.fam_reg_convs):
            fam_reg_feat = fam_reg_conv(fam_reg_feat)
            #attention = self.fam_reg_spatial_attention[i](fam_reg_feat)  # added this
            #fam_reg_feat = fam_reg_feat * attention  # added this
            #attention = self.fam_reg_adaptive_attention[i](fam_reg_feat)  # added this
            #fam_reg_feat = fam_reg_feat * attention  # added this
        # start edit
        fam_bbox_pred = self.fam_reg(fam_reg_feat)
        # with open('/content/debug.txt', 'a') as f:      
        #     f.write(f'1- fam_reg_feat.shape: {fam_reg_feat.shape}\n')
        #     f.write(f'1- fam_reg_feat: {fam_reg_feat}\n')
        # fam_bbox_theta_pred = self.fam_theta_cls(fam_reg_feat)
        # with open('/content/debug.txt', 'a') as f:      
        #     f.write(f'2- fam_bbox_theta_pred.shape: {fam_bbox_theta_pred.shape}\n')
        #     f.write(f'2- fam_bbox_theta_pred: {fam_bbox_theta_pred}\n')
        # fam_bbox_theta_pred = fam_bbox_theta_pred.permute((0, 2, 3, 1))
        # fam_bbox_theta_decoded_pred = self._theta_decode(fam_bbox_theta_pred)

        # fam_bbox_theta_decoded_pred = fam_bbox_theta_decoded_pred.permute((0, 3, 1, 2))
        # with open('/content/debug.txt', 'a') as f:
        #     f.write(f'1- Theta after fam.shape: {fam_bbox_theta_decoded_pred.shape}\n')
        #     f.write(f'1- Theta after fam: {fam_bbox_theta_decoded_pred}\n')
        if DEBUG:
            print(f'fam_bbox_theta_decoded_pred.shape: {fam_bbox_theta_decoded_pred.shape}')
            print(f'fam_bbox_xwh_pred.shape: {fam_bbox_xwh_pred.shape}')
        # fam_bbox_pred = torch.cat((fam_bbox_xwh_pred, fam_bbox_theta_decoded_pred), 1)
        
        if DEBUG:
            print(f'fam_bbox_pred.shape: {fam_bbox_pred.shape}')
        # end edit

        # only forward during training
        if self.training:
            fam_cls_feat = x
            for i, fam_cls_conv in enumerate(self.fam_cls_convs):
                fam_cls_feat = fam_cls_conv(fam_cls_feat)
                #attention = self.fam_cls_attention[i](fam_cls_feat)  # added this
                #fam_cls_feat = fam_cls_feat * attention  # added this
            fam_cls_score = self.fam_cls(fam_cls_feat)
        else:
            fam_cls_score = None

        num_level = self.anchor_strides.index(stride)
        featmap_size = fam_bbox_pred.shape[-2:]
        if (num_level, featmap_size) in self.base_anchors:
            init_anchors = self.base_anchors[(num_level, featmap_size)]
        else:
            device = fam_bbox_pred.device
            init_anchors = self.anchor_generators[num_level].grid_anchors(
                featmap_size, self.anchor_strides[num_level], device=device)
            self.base_anchors[(num_level, featmap_size)] = init_anchors

        refine_anchor = bbox_decode(
            fam_bbox_pred.detach(),
            init_anchors,
            self.target_means,
            self.target_stds)
        if DEBUG:
            print(f'refine_anchor.shape {refine_anchor.shape}')
        align_feat = self.align_conv(x, refine_anchor.clone(), stride)

        or_feat = self.or_conv(align_feat)
        odm_reg_feat = or_feat
        if self.with_orconv:
            odm_cls_feat = self.or_pool(or_feat)
        else:
            odm_cls_feat = or_feat
        
        for i, odm_reg_conv in enumerate(self.odm_reg_convs):
            odm_reg_feat = odm_reg_conv(odm_reg_feat)
            #attention = self.odm_reg_spatial_attention[i](odm_reg_feat)  # added this
            #odm_reg_feat = odm_reg_feat * attention  # added this
            #attention = self.odm_reg_adaptive_attention[i](odm_reg_feat)  # added this
            #odm_reg_feat = odm_reg_feat * attention  # added this
        for i, odm_cls_conv in enumerate(self.odm_cls_convs):
            odm_cls_feat = odm_cls_conv(odm_cls_feat)
            #attention = self.odm_cls_attention[i](odm_cls_feat)  # added this
            #odm_cls_feat = odm_cls_feat * attention  # added this
        odm_cls_score = self.odm_cls(odm_cls_feat)
        # start edit
        odm_bbox_xwh_pred = self.odm_xwh_reg(odm_reg_feat)
        odm_bbox_theta_pred = self.odm_theta_cls(odm_reg_feat)
        # with open('/content/debug.txt', 'a') as f:  
        #     f.write(f'2- Theta after odm.shape: {odm_bbox_theta_pred.shape}\n')
        #     f.write(f'2- Theta after odm: {odm_bbox_theta_pred}\n')

        if DEBUG:
            print(f'odm_bbox_xwh_pred.shape: {odm_bbox_xwh_pred.shape}')
            print(f'odm_bbox_theta_pred.shape: {odm_bbox_theta_pred.shape}')
        return (
            fam_cls_score, fam_bbox_pred,
            refine_anchor, odm_cls_score,
            odm_bbox_xwh_pred, odm_bbox_theta_pred
        )
        # end edit

    def forward(self, feats):
        return multi_apply(self.forward_single, feats, self.anchor_strides)

    def get_init_anchors(self,
                         featmap_sizes,
                         img_metas,
                         device='cuda'):
        """Get anchors according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.
            device (torch.device | str): device for returned tensors

        Returns:
            tuple: anchors of each image, valid flags of each image
        """
        num_imgs = len(img_metas)
        num_levels = len(featmap_sizes)

        # since feature map sizes of all images are the same, we only compute
        # anchors for one time
        multi_level_anchors = []
        for i in range(num_levels):
            anchors = self.anchor_generators[i].grid_anchors(
                featmap_sizes[i], self.anchor_strides[i], device=device)
            multi_level_anchors.append(anchors)
        anchor_list = [multi_level_anchors for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level anchors
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = []
            for i in range(num_levels):
                anchor_stride = self.anchor_strides[i]
                feat_h, feat_w = featmap_sizes[i]
                h, w, _ = img_meta['pad_shape']
                valid_feat_h = min(int(np.ceil(h / anchor_stride)), feat_h)
                valid_feat_w = min(int(np.ceil(w / anchor_stride)), feat_w)
                flags = self.anchor_generators[i].valid_flags(
                    (feat_h, feat_w), (valid_feat_h, valid_feat_w),
                    device=device)
                multi_level_flags.append(flags)
            valid_flag_list.append(multi_level_flags)
        return anchor_list, valid_flag_list

    def get_refine_anchors(self,
                           featmap_sizes,
                           refine_anchors,
                           img_metas,
                           is_train=True,
                           device='cuda'):
        num_levels = len(featmap_sizes)

        refine_anchors_list = []
        for img_id, img_meta in enumerate(img_metas):
            mlvl_refine_anchors = []
            for i in range(num_levels):
                refine_anchor = refine_anchors[i][img_id].reshape(-1, 5)
                mlvl_refine_anchors.append(refine_anchor)
            refine_anchors_list.append(mlvl_refine_anchors)

        valid_flag_list = []
        if is_train:
            for img_id, img_meta in enumerate(img_metas):
                multi_level_flags = []
                for i in range(num_levels):
                    anchor_stride = self.anchor_strides[i]
                    feat_h, feat_w = featmap_sizes[i]
                    h, w, _ = img_meta['pad_shape']
                    valid_feat_h = min(int(np.ceil(h / anchor_stride)), feat_h)
                    valid_feat_w = min(int(np.ceil(w / anchor_stride)), feat_w)
                    flags = self.anchor_generators[i].valid_flags(
                        (feat_h, feat_w), (valid_feat_h, valid_feat_w),
                        device=device)
                    multi_level_flags.append(flags)
                valid_flag_list.append(multi_level_flags)
        return refine_anchors_list, valid_flag_list

    @force_fp32(apply_to=(
        'fam_cls_scores',
        'fam_bbox_preds',
        # 'fam_bbox_theta_pred',
        'odm_cls_scores',
        'odm_bbox_xwh_preds',
        'odm_bbox_theta_preds'))
    def loss(self,
             fam_cls_scores,
             fam_bbox_preds,
             # fam_bbox_theta_preds,
             refine_anchors,
             odm_cls_scores,
             odm_bbox_xwh_preds,
             odm_bbox_theta_preds,
             gt_bboxes,
             gt_labels,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in odm_cls_scores]
        assert len(featmap_sizes) == len(self.anchor_generators)
        device = odm_cls_scores[0].device

        anchor_list, valid_flag_list = self.get_init_anchors(
            featmap_sizes, img_metas, device=device)

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0) for anchors in anchor_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i in range(len(anchor_list)):
            concat_anchor_list.append(torch.cat(anchor_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)

        # Feature Alignment Module
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = anchor_target(
            anchor_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            self.target_means,
            self.target_stds,
            cfg.fam_cfg,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            sampling=self.sampling)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

        losses_fam_cls, losses_fam_bbox = multi_apply(
            self.loss_fam_single,
            fam_cls_scores,
            fam_bbox_preds,
            # fam_bbox_theta_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples,
            cfg=cfg.fam_cfg)

        # Oriented Detection Module targets
        refine_anchors_list, valid_flag_list = self.get_refine_anchors(
            featmap_sizes, refine_anchors, img_metas, device=device)

        # anchor number of multi levels
        num_level_anchors = [anchors.size(0)
                             for anchors in refine_anchors_list[0]]
        # concat all level anchors and flags to a single tensor
        concat_anchor_list = []
        for i in range(len(refine_anchors_list)):
            concat_anchor_list.append(torch.cat(refine_anchors_list[i]))
        all_anchor_list = images_to_levels(concat_anchor_list,
                                           num_level_anchors)

        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1
        cls_reg_targets = anchor_target(
            refine_anchors_list,
            valid_flag_list,
            gt_bboxes,
            img_metas,
            self.target_means,
            self.target_stds,
            cfg.odm_cfg,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            sampling=self.sampling)
        if cls_reg_targets is None:
            return None
        (labels_list, label_weights_list, bbox_targets_list, bbox_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos + num_total_neg if self.sampling else num_total_pos)

        losses_odm_cls, losses_odm_bbox = multi_apply(
            self.loss_odm_single,
            odm_cls_scores,
            odm_bbox_xwh_preds,
            odm_bbox_theta_preds,
            all_anchor_list,
            labels_list,
            label_weights_list,
            bbox_targets_list,
            bbox_weights_list,
            num_total_samples=num_total_samples,
            cfg=cfg.odm_cfg)

        return dict(loss_fam_cls=losses_fam_cls,
                    loss_fam_bbox=losses_fam_bbox,
                    loss_odm_cls=losses_odm_cls,
                    loss_odm_bbox=losses_odm_bbox)

    def loss_fam_single(self,
                        fam_cls_score,
                        fam_bbox_pred,
                        anchors,
                        labels,
                        label_weights,
                        bbox_targets,
                        bbox_weights,
                        num_total_samples,
                        cfg):
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        fam_cls_score = fam_cls_score.permute(
            0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        loss_fam_cls = self.loss_fam_cls(
            fam_cls_score, labels, label_weights, avg_factor=num_total_samples)
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 5)
        bbox_weights = bbox_weights.reshape(-1, 5)
        fam_bbox_pred = fam_bbox_pred.permute(0, 2, 3, 1).reshape(-1, 5)

        reg_decoded_bbox = cfg.get('reg_decoded_bbox', False)
        if reg_decoded_bbox:
            # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            # is applied directly on the decoded bounding boxes, it
            # decodes the already encoded coordinates to absolute format.
            bbox_coder_cfg = cfg.get('bbox_coder', '')
            if bbox_coder_cfg == '':
                bbox_coder_cfg = dict(type='DeltaXYWHBBoxCoder')
            bbox_coder = build_bbox_coder(bbox_coder_cfg)
            anchors = anchors.reshape(-1, 5)
            fam_bbox_pred = bbox_coder.decode(anchors, fam_bbox_pred)
        loss_fam_bbox = self.loss_fam_bbox(
            fam_bbox_pred,
            bbox_targets,
            bbox_weights,
            avg_factor=num_total_samples)
        return loss_fam_cls, loss_fam_bbox


    def loss_odm_single(self,
                        odm_cls_score,
                        odm_bbox_xwh_pred,
                        odm_bbox_theta_pred,
                        anchors,
                        labels,
                        label_weights,
                        bbox_targets,
                        bbox_weights,
                        num_total_samples,
                        cfg):
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        odm_cls_score = odm_cls_score.permute(0, 2, 3,
                                              1).reshape(-1, self.cls_out_channels)
        loss_odm_cls = self.loss_odm_cls(
            odm_cls_score, labels, label_weights, avg_factor=num_total_samples
        ) * self.cls_lambda
        # regression loss
        bbox_targets = bbox_targets.reshape(-1, 5)
        bbox_xwh_targets = bbox_targets[:, :-1]
        bbox_theta_targets = bbox_targets[:, -1:]
        bbox_weights = bbox_weights.reshape(-1, 5)
        bbox_xwh_weights = bbox_weights[:, :-1]
        bbox_theta_weights = bbox_weights[:, -1:]

        odm_bbox_xwh_pred = odm_bbox_xwh_pred.permute((0, 2, 3, 1)).reshape(-1, 4)
        odm_bbox_theta_pred = odm_bbox_theta_pred.permute((0, 2, 3, 1)).reshape(
            -1, self.theta_num_classes
        )

        reg_decoded_bbox = cfg.get('reg_decoded_bbox', False)
        anchors = anchors.reshape(-1, 5)
        if reg_decoded_bbox:
            # When the regression loss (e.g. `IouLoss`, `GIouLoss`)
            # is applied directly on the decoded bounding boxes, it
            # decodes the already encoded coordinates to absolute format.
            bbox_coder_cfg = cfg.get('bbox_coder', '')
            if bbox_coder_cfg == '':
                bbox_coder_cfg = dict(type='DeltaXYWHBBoxCoder')
            bbox_coder = build_bbox_coder(bbox_coder_cfg)
            odm_bbox_xwh_pred = bbox_coder.decode(anchors[..., :-1], odm_bbox_xwh_pred)
        loss_odm_xwh_bbox = self.loss_odm_xwh_bbox(
            odm_bbox_xwh_pred,
            bbox_xwh_targets,
            bbox_xwh_weights,
            avg_factor=num_total_samples
        ) * self.xwh_lambda
        loss_odm_theta_bbox = self.loss_odm_theta_bbox(
            bbox_theta_targets,
            bbox_targets[..., 2],
            bbox_targets[..., 1],
            odm_bbox_theta_pred,
            bbox_theta_weights,
            avg_factor=num_total_samples
        ) * self.theta_lambda
        loss_odm_bbox = loss_odm_xwh_bbox + loss_odm_theta_bbox
        return loss_odm_cls, loss_odm_bbox

    @force_fp32(apply_to=(
        'fam_cls_scores',
        'fam_bbox_pred',
        # 'fam_bbox_theta_pred',
        'odm_cls_scores',
        'odm_bbox_xwh_pred',
        'odm_bbox_theta_pred'))
    def get_bboxes(self,
                   fam_cls_scores,
                   fam_bbox_pred,
                   # fam_bbox_theta_pred,
                   refine_anchors,
                   odm_cls_scores,
                   odm_bbox_xwh_pred,
                   odm_bbox_theta_pred,
                   img_metas,
                   cfg,
                   rescale=False):
        odm_bbox_preds = []
        for i in range(len(odm_bbox_theta_pred)):
            xwh = odm_bbox_xwh_pred[i]
            theta = odm_bbox_theta_pred[i]
            theta = theta.permute((0, 2, 3, 1))
            theta_decoded = self._theta_decode(theta)
            theta_decoded = theta_decoded.permute((0, 3, 1, 2))
            pred = torch.cat((xwh, theta_decoded), 1)
            odm_bbox_preds.append(pred)

        assert len(odm_cls_scores) == len(odm_bbox_preds)

        featmap_sizes = [featmap.size()[-2:] for featmap in odm_cls_scores]
        num_levels = len(odm_cls_scores)
        device = odm_cls_scores[0].device

        refine_anchors = self.get_refine_anchors(
            featmap_sizes, refine_anchors, img_metas, is_train=False, device=device)
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                odm_cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                odm_bbox_preds[i][img_id].detach() for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self.get_bboxes_single(cls_score_list, bbox_pred_list,
                                               refine_anchors[0][0], img_shape,
                                               scale_factor, cfg, rescale)

            result_list.append(proposals)
        return result_list

    def get_bboxes_single(self,
                          cls_score_list,
                          bbox_pred_list,
                          mlvl_anchors,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False):
        """
        Transform outputs for a single batch item into labeled boxes.
        """
        assert len(cls_score_list) == len(bbox_pred_list) == len(mlvl_anchors)
        mlvl_bboxes = []
        mlvl_scores = []
        for cls_score, bbox_pred, anchors in zip(cls_score_list,
                                                 bbox_pred_list, mlvl_anchors):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            cls_score = cls_score.permute(
                1, 2, 0).reshape(-1, self.cls_out_channels)

            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)

            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 5)
            # anchors = rect2rbox(anchors)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                # Get maximum scores for foreground classes.
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    max_scores, _ = scores[:, 1:].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                anchors = anchors[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                scores = scores[topk_inds, :]
            bboxes = delta2bbox_rotated(anchors, bbox_pred, self.target_means,
                                        self.target_stds, img_shape)
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        if rescale:
            mlvl_bboxes[..., :4] /= mlvl_bboxes.new_tensor(scale_factor)
        mlvl_scores = torch.cat(mlvl_scores)
        if self.use_sigmoid_cls:
            # Add a dummy background class to the front when using sigmoid
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
        det_bboxes, det_labels = multiclass_nms_rotated(mlvl_bboxes,
                                                        mlvl_scores,
                                                        cfg.score_thr, cfg.nms,
                                                        cfg.max_per_img)
        return det_bboxes, det_labels


def bbox_decode(
        bbox_preds,
        anchors,
        means=[0, 0, 0, 0, 0],
        stds=[1, 1, 1, 1, 1]):
    """
    Decode bboxes from deltas
    :param bbox_preds: [N,5,H,W]
    :param anchors: [H*W,5]
    :param means: mean value to decode bbox
    :param stds: std value to decode bbox
    :return: [N,H,W,5]
    """
    num_imgs, _, H, W = bbox_preds.shape
    bboxes_list = []
    for img_id in range(num_imgs):
        bbox_pred = bbox_preds[img_id]
        # bbox_pred.shape=[5,H,W]
        bbox_delta = bbox_pred.permute(1, 2, 0).reshape(-1, 5)
        bboxes = delta2bbox_rotated(
            anchors, bbox_delta, means, stds, wh_ratio_clip=1e-6)
        bboxes = bboxes.reshape(H, W, 5)
        bboxes_list.append(bboxes)
    return torch.stack(bboxes_list, dim=0)


class AlignConv(nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 deformable_groups=1):
        super(AlignConv, self).__init__()
        self.kernel_size = kernel_size
        self.deform_conv = DeformConv(in_channels,
                                      out_channels,
                                      kernel_size=kernel_size,
                                      padding=(kernel_size - 1) // 2,
                                      deformable_groups=deformable_groups)
        # self.relu = nn.ReLU(inplace=True)
        self.relu = nn.LeakyReLU(
            negative_slope=0.2, inplace=True
        )

    def init_weights(self):
        normal_init(self.deform_conv, std=0.01)

    @torch.no_grad()
    def get_offset(self, anchors, featmap_size, stride):
        dtype, device = anchors.dtype, anchors.device
        feat_h, feat_w = featmap_size
        pad = (self.kernel_size - 1) // 2
        idx = torch.arange(-pad, pad + 1, dtype=dtype, device=device)
        yy, xx = torch.meshgrid(idx, idx)
        xx = xx.reshape(-1)
        yy = yy.reshape(-1)

        # get sampling locations of default conv
        xc = torch.arange(0, feat_w, device=device, dtype=dtype)
        yc = torch.arange(0, feat_h, device=device, dtype=dtype)
        yc, xc = torch.meshgrid(yc, xc)
        xc = xc.reshape(-1)
        yc = yc.reshape(-1)
        x_conv = xc[:, None] + xx
        y_conv = yc[:, None] + yy

        # get sampling locations of anchors
        x_ctr, y_ctr, w, h, a = torch.unbind(anchors, dim=1)
        x_ctr, y_ctr, w, h = x_ctr / stride, y_ctr / stride, w / stride, h / stride
        cos, sin = torch.cos(a), torch.sin(a)
        dw, dh = w / self.kernel_size, h / self.kernel_size
        x, y = dw[:, None] * xx, dh[:, None] * yy
        xr = cos[:, None] * x - sin[:, None] * y
        yr = sin[:, None] * x + cos[:, None] * y
        x_anchor, y_anchor = xr + x_ctr[:, None], yr + y_ctr[:, None]
        # get offset filed
        offset_x = x_anchor - x_conv
        offset_y = y_anchor - y_conv
        # x, y in anchors is opposite in image coordinates,
        # so we stack them with y, x other than x, y
        offset = torch.stack([offset_y, offset_x], dim=-1)
        # NA,ks*ks*2
        offset = offset.reshape(anchors.size(
            0), -1).permute(1, 0).reshape(-1, feat_h, feat_w)
        return offset

    def forward(self, x, anchors, stride):
        num_imgs, H, W = anchors.shape[:3]
        offset_list = [
            self.get_offset(anchors[i].reshape(-1, 5), (H, W), stride)
            for i in range(num_imgs)
        ]
        offset_tensor = torch.stack(offset_list, dim=0)
        #print(offset_tensor.shape)
        #print(offset_tensor.dtype)
        #print(x.shape)
        #print(x.dtype)
        x = self.relu(self.deform_conv(x, offset_tensor))
        return x