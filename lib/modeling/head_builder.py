import tensorflow as tf
import numpy as np

from core.config import cfg
import utils.tf_util as tf_util
import dataset.maps_dict as maps_dict

class HeadBuilder:
    def __init__(self, anchor_num, head_idx, head_cfg, is_training):
        self.is_training = is_training
        self.head_idx = head_idx
        self.anchor_num = anchor_num

        cur_head = head_cfg

        self.xyz_index = cur_head[0]
        self.feature_index = cur_head[1]
        self.mlp_list = cur_head[2]
        self.bn = cur_head[3]
        self.scope = cur_head[4]

        if head_idx == 0: # stage 1
            self.head_cfg = cfg.MODEL.FIRST_STAGE
        elif head_idx == 1: # stage 2
            self.head_cfg = cfg.MODEL.SECOND_STAGE
        else: raise Exception('Not Implementation Error!!!') # stage 3

        # determine channel number
        if self.head_cfg.CLS_ACTIVATION == 'Sigmoid':
            self.pred_cls_channel = self.anchor_num
        elif self.head_cfg.CLS_ACTIVATION == 'Softmax':
            self.pred_cls_channel = self.anchor_num + 1

        pred_reg_base_num = {
            'Dist-Anchor': self.anchor_num,
            'Log-Anchor': self.anchor_num,
            'Dist-Anchor-free': 1,
        } 
        self.pred_reg_base_num = pred_reg_base_num[self.head_cfg.REGRESSION_METHOD]

    def build_layer(self, xyz_list, feature_list, bn_decay, output_dict):
        xyz_input = []
        for xyz_index in self.xyz_index:
            xyz_input.append(xyz_list[xyz_index])
        xyz_input = tf.concat(xyz_input, axis=1) # bs, npoint, 3
        
        feature_input = []
        for feature_index in self.feature_index:
            feature_input.append(feature_list[feature_index])
        feature_input = tf.concat(feature_input, axis=1) # bs, npoint, c

        bs, points_num, _ = xyz_input.get_shape().as_list()

        with tf.variable_scope(self.scope) as sc:
            for i, channel in enumerate(self.mlp_list):
                feature_input = tf_util.conv1d(feature_input, channel, 1, padding='VALID', stride=1, bn=self.bn, scope='conv1d_%d'%(i), bn_decay=bn_decay, is_training=self.is_training)

            # classification
            pred_cls = tf_util.conv1d(feature_input, 128, 1, padding='VALID', bn=self.bn, is_training=self.is_training, scope='pred_cls_base', bn_decay=bn_decay)
            pred_cls = tf_util.conv1d(pred_cls, self.pred_cls_channel, 1, padding='VALID', activation_fn=None, scope='pred_cls')

            # recognition
            pred_reg = tf_util.conv1d(feature_input, 128, 1, padding='VALID', bn=self.bn, is_training=self.is_training, scope='pred_reg_base', bn_decay=bn_decay)
            pred_reg = tf_util.conv1d(pred_reg, self.pred_reg_base_num * (6 + cfg.MODEL.ANGLE_CLS_NUM * 2), 1, padding='VALID', activation_fn=None, scope='pred_reg')
            pred_reg = tf.reshape(pred_reg, [bs, points_num, self.pred_reg_base_num, 6 + cfg.MODEL.ANGLE_CLS_NUM * 2])

            if self.head_cfg.PREDICT_ATTRIBUTE_AND_VELOCITY: # velocity and attribute
                pred_attr = tf_util.conv1d(feature_input, 128, 1, padding='VALID', bn=self.bn, is_training=self.is_training, scope='pred_attr_base', bn_decay=bn_decay)
                pred_attr = tf_util.conv1d(pred_attr, self.pred_reg_base_num * 8, 1, padding='VALID', activation_fn=None, scope='pred_attr')
                pred_attr = tf.reshape(pred_attr, [bs, points_num, self.pred_reg_base_num, 8])

                pred_velo = tf_util.conv1d(feature_input, 128, 1, padding='VALID', bn=self.bn, is_training=self.is_training, scope='pred_velo_base', bn_decay=bn_decay)
                pred_velo = tf_util.conv1d(pred_velo, self.pred_reg_base_num * 2, 1, padding='VALID', activation_fn=None, scope='pred_velo')
                pred_velo = tf.reshape(pred_velo, [bs, points_num, self.pred_reg_base_num, 2])

                output_dict[maps_dict.PRED_ATTRIBUTE].append(pred_attr)
                output_dict[maps_dict.PRED_VELOCITY].append(pred_velo)

        output_dict[maps_dict.KEY_OUTPUT_XYZ].append(xyz_input)
        output_dict[maps_dict.KEY_OUTPUT_FEATURE].append(feature_input)

        output_dict[maps_dict.PRED_CLS].append(pred_cls)
        output_dict[maps_dict.PRED_OFFSET].append(tf.slice(pred_reg, [0, 0, 0, 0], [-1, -1, -1, 6]))
        output_dict[maps_dict.PRED_ANGLE_CLS].append(tf.slice(pred_reg, [0, 0, 0, 6], [-1, -1, -1, cfg.MODEL.ANGLE_CLS_NUM]))
        output_dict[maps_dict.PRED_ANGLE_RES].append(tf.slice(pred_reg, [0, 0, 0, 6+cfg.MODEL.ANGLE_CLS_NUM], [-1, -1, -1, -1]))

        return pred_cls, pred_reg, xyz_input, feature_input
