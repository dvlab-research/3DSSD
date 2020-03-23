import tensorflow as tf
import numpy as np

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
            self.encoder_cfg = cfg.MODEL.FIRST_STAGE
        elif stage == 1:
            self.encoder_cfg = cfg.MODEL.SECOND_STAGE
        else: raise Exception('Not Implementation Error')

        self.regression_method = self.encoder_cfg.REGRESSION_METHOD
        
        self.encoder_dict = {
            'Log-Anchor': encode_log_anchor,
            'Dist-Anchor': encode_dist_anchor,
            'Dist-Anchor-free': encode_dist_anchor_free,
        }

        self.decoder_dict = {
            'Log-Anchor': decode_log_anchor,
            'Dist-Anchor': decode_dist_anchor,
            'Dist-Anchor-free': decode_dist_anchor_free,
        }

        self.encoder = self.encoder_dict[self.regression_method]
        self.decoder = self.decoder_dict[self.regression_method]

    def encode(self, center_xyz, gt_offset, batch_anchors_3d):
        """
        center_xyz: [bs, points_num, 3], points location
        gt_offset: [bs, points_num, cls_num, 7]
        batch_anchors_3d: [bs, points_num, cls_num, 6]
        """
        bs, points_num, cls_num, _ = gt_offset.get_shape().as_list()
        gt_offset = gt_offset[:, :, :, :-1]
        gt_offset = tf.reshape(gt_offset, [bs, points_num * cls_num, 6])
        batch_anchors_3d = tf.reshape(batch_anchors_3d, [bs, points_num * cls_num, -1])

        gt_ctr, gt_size = tf.split(gt_offset, num_or_size_splits=2, axis=-1)
        if self.regression_method == 'Dist-Anchor-free':
            encoded_ctr, encoded_offset = self.encoder(gt_ctr, gt_size, center_xyz)
        else:
            anchor_ctr, anchor_size = tf.slice(batch_anchors_3d[:, :, :-1], num_or_size_splits=2, axis=-1)
            encoded_ctr, encoded_offset = self.encoder(gt_ctr, gt_size, anchor_ctr, anchor_size) 

        encoded_ctr = tf.reshape(encoded_ctr, [bs, points_num, cls_num, 3])
        encoded_offset = tf.reshape(encoded_offset, [bs, points_num, cls_num, 3])

        # bs, points_num, cls_num, 6
        target = tf.concat([encoded_ctr, encoded_offset], axis=-1)
        return target
        

    def decode(self, center_xyz, det_offset, det_angle_cls, det_angle_res, is_training, batch_anchors_3d):
        """
        center_xyz: [bs, points_num, 3], points location
        det_offset: [bs, points_num, cls_num, 6]
        det_angle_cls/det_angle_res: [bs, points_num, cls_num, num_angle]
        batch_anchors_3d: [bs, points_num, cls_num, 7]
        """
        bs, points_num, cls_num, _ = det_offset.get_shape().as_list()
        det_offset = tf.reshape(det_offset, [bs, points_num * cls_num, 6])
        det_angle_cls = tf.reshape(det_angle_cls, [bs, points_num * cls_num, cfg.MODEL.ANGLE_CLS_NUM])
        det_angle_res = tf.reshape(det_angle_res, [bs, points_num * cls_num, cfg.MODEL.ANGLE_CLS_NUM])
        batch_anchors_3d = tf.reshape(batch_anchors_3d, [bs, points_num * cls_num, -1])

        if self.regression_method == 'Dist-Anchor-free':
            pred_anchors_3d = self.decoder(center_xyz, det_offset, det_angle_cls, det_angle_res, is_training)
        else:
            det_ctr, det_offset = tf.split(det_offset, num_or_size_splits=2, axis=-1)
            pred_anchors_3d = self.decoder(det_ctr, det_offset, det_angle_cls, det_angle_res, batch_anchors_3d, is_training)

        pred_anchors_3d = tf.reshape(pred_anchors_3d, [bs, points_num, cls_num, 7])
        return pred_anchors_3d
