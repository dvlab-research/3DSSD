import tensorflow as tf
import numpy as np
from functools import partial

from core.config import cfg
from utils.anchor_encoder import *
from utils.anchor_decoder import *

class EncoderDecoder:
    def __init__(self, stage):
        """
        stage: 0/1, first stage / second stage
        For anchors sent to EncoderDecoder, all have shape as [bs, -1, 7]
        """
        if stage == 0:
            self.encoder_cfg = cfg.MODEL.FIRST_STAGE.REGRESSION_METHOD
        elif stage == 1:
            self.encoder_cfg = cfg.MODEL.SECOND_STAGE.REGRESSION_METHOD
        else: raise Exception('Not Implementation Error')

        self.regression_method = self.encoder_cfg.TYPE
        
        self.encoder_dict = {
            'Log-Anchor': encode_log_anchor,
            'Dist-Anchor': encode_dist_anchor,
            'Dist-Anchor-free': encode_dist_anchor_free,
            'Bin-Anchor': partial(encode_bin_anchor,
                              half_bin_search_range=self.encoder_cfg.HALF_BIN_SEARCH_RANGE, 
                              bin_num_class=self.encoder_cfg.BIN_CLASS_NUM,),
        }

        self.decoder_dict = {
            'Log-Anchor': decode_log_anchor,
            'Dist-Anchor': decode_dist_anchor,
            'Dist-Anchor-free': decode_dist_anchor_free,
            'Bin-Anchor': partial(decode_bin_anchor,
                              half_bin_search_range=self.encoder_cfg.HALF_BIN_SEARCH_RANGE, 
                              bin_num_class=self.encoder_cfg.BIN_CLASS_NUM,),
        }

        self.encoder = self.encoder_dict[self.regression_method]
        self.decoder = self.decoder_dict[self.regression_method]

    def encode(self, center_xyz, assigned_gt_boxes, batch_anchors_3d):
        """
        center_xyz: [bs, points_num, 3], points location
        assigned_gt_boxes: [bs, points_num, cls_num, 7]
        batch_anchors_3d: [bs, points_num, cls_num, 7]
        """
        bs, points_num, cls_num, _ = assigned_gt_boxes.get_shape().as_list()
        gt_offset = assigned_gt_boxes[:, :, :, :-1]
        gt_offset = tf.reshape(gt_offset, [bs, points_num * cls_num, 6])
        reshape_anchors_3d = tf.reshape(batch_anchors_3d, [bs, points_num * cls_num, -1])

        gt_ctr, gt_size = tf.split(gt_offset, num_or_size_splits=2, axis=-1)
        if self.regression_method == 'Dist-Anchor-free':
            encoded_ctr, encoded_offset = self.encoder(gt_ctr, gt_size, center_xyz)
            gt_angle = assigned_gt_boxes[:, :, :, -1]
        else: # anchor-based method
            anchor_ctr, anchor_size = tf.split(reshape_anchors_3d[:, :, :-1], num_or_size_splits=2, axis=-1)
            encoded_ctr, encoded_offset = self.encoder(gt_ctr, gt_size, anchor_ctr, anchor_size) 
            gt_angle = assigned_gt_boxes[:, :, :, -1] - batch_anchors_3d[:, :, :, -1] 

        encoded_ctr = tf.reshape(encoded_ctr, [bs, points_num, cls_num, -1])
        encoded_offset = tf.reshape(encoded_offset, [bs, points_num, cls_num, -1])

        # bs, points_num, cls_num
        encoded_angle_cls, encoded_angle_res = encode_angle2class_tf(gt_angle, cfg.MODEL.ANGLE_CLS_NUM) 

        # bs, points_num, cls_num, 8 (bin-loss) / 6 (residual-loss)
        target = tf.concat([encoded_ctr, encoded_offset], axis=-1)
        return target, encoded_angle_cls, encoded_angle_res
        

    def decode(self, center_xyz, det_offset, det_angle_cls, det_angle_res, is_training, batch_anchors_3d):
        """
        center_xyz: [bs, points_num, 3], points location
        det_offset: [bs, points_num, cls_num, 6]
        det_angle_cls/det_angle_res: [bs, points_num, cls_num, num_angle]
        batch_anchors_3d: [bs, points_num, cls_num, 7]
        """
        bs, points_num, cls_num, _ = det_offset.get_shape().as_list()
        det_offset = tf.reshape(det_offset, [bs, points_num * cls_num, -1])
        det_angle_cls = tf.reshape(det_angle_cls, [bs, points_num * cls_num, cfg.MODEL.ANGLE_CLS_NUM])
        det_angle_res = tf.reshape(det_angle_res, [bs, points_num * cls_num, cfg.MODEL.ANGLE_CLS_NUM])
        batch_anchors_3d = tf.reshape(batch_anchors_3d, [bs, points_num * cls_num, -1])

        if self.regression_method == 'Dist-Anchor-free':
            pred_anchors_3d = self.decoder(center_xyz, det_offset, det_angle_cls, det_angle_res, is_training)
        else:
            pred_anchors_3d = self.decoder(det_offset, det_angle_cls, det_angle_res, batch_anchors_3d, is_training)

        pred_anchors_3d = tf.reshape(pred_anchors_3d, [bs, points_num, cls_num, 7])
        return pred_anchors_3d
