import tensorflow as tf
import numpy as np

from core.config import cfg
from utils.head_util import *
from functools import partial

import dataset.maps_dict as maps_dict


class HeadBuilder:
    def __init__(self, batch_size, anchor_num, head_idx, head_cfg, is_training):
        self.is_training = is_training
        self.head_idx = head_idx
        self.anchor_num = anchor_num
        self.batch_size = batch_size 

        cur_head = head_cfg

        self.xyz_index = cur_head[0]
        self.feature_index = cur_head[1]
        self.op_type = cur_head[2]
        self.mlp_list = cur_head[3]
        self.bn = cur_head[4]
        self.layer_type = cur_head[5]
        self.scope = cur_head[6]

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
        if self.layer_type == 'IoU':
            self.pred_cls_channel = self.anchor_num

        self.reg_method = self.head_cfg.REGRESSION_METHOD.TYPE 
        anchor_type = self.reg_method.split('-')[-1] # Anchor & free

        pred_reg_base_num = {
            'Anchor': self.anchor_num,
            'free': 1,
        } 
        self.pred_reg_base_num = pred_reg_base_num[anchor_type]

        pred_reg_channel_num = {
            'Dist-Anchor': 6,
            'Log-Anchor': 6,
            'Dist-Anchor-free': 6,
            # bin_x/res_x/bin_z/res_z/res_y/res_size
            'Bin-Anchor': self.head_cfg.REGRESSION_METHOD.BIN_CLASS_NUM * 4 + 4,
        } 
        self.pred_reg_channel_num = pred_reg_channel_num[self.reg_method]

        self.conv_op = select_conv_op(self.op_type)
        # build the head predictor
        layer_type_dict = {
            'Det': box_regression_head,
            'IoU': iou_regression_head,
        }
        self.head_predictor = partial(
            layer_type_dict[self.layer_type],
            pred_cls_channel=self.pred_cls_channel, 
            bn=self.bn,
            is_training=self.is_training,
            conv_op=select_conv_op('conv1d'),
        ) 
        if self.layer_type == 'Det':
            self.head_predictor = partial(
                self.head_predictor, 
                pred_reg_base_num=self.pred_reg_base_num,
                pred_reg_channel_num=self.pred_reg_channel_num,
                pred_attr_velo=self.head_cfg.PREDICT_ATTRIBUTE_AND_VELOCITY,
            )

    def build_layer(self, xyz_list, feature_list, bn_decay, output_dict):
        xyz_input = []
        for xyz_index in self.xyz_index:
            xyz_input.append(xyz_list[xyz_index])
        xyz_input = tf.concat(xyz_input, axis=1) # bs, npoint, 3
        
        feature_input = []
        for feature_index in self.feature_index:
            feature_input.append(feature_list[feature_index])
        feature_input = tf.concat(feature_input, axis=1) # bs, npoint, c
        feature_input = self.format_input(feature_input)


        with tf.variable_scope(self.scope) as sc:
            for i, channel in enumerate(self.mlp_list):
                feature_input = self.conv_op(feature_input, channel, bn=self.bn, scope='conv1d_%d'%(i), bn_decay=bn_decay, is_training=self.is_training)


            # format feature and xyz format to [bs, npoint, c]
            xyz_shape = xyz_input.get_shape().as_list()
            feature_shape = feature_input.get_shape().as_list()
            if len(xyz_shape) != 3 or len(feature_shape) != 3:
                # pooled_features with shape [bs * proposal_num, c]
                xyz_input = tf.reshape(xyz_input, [self.batch_size, -1, 3])            
                feature_input = tf.reshape(feature_input, [self.batch_size, -1, feature_shape[-1]])

            self.head_predictor(feature_input=feature_input, bn_decay=bn_decay, output_dict=output_dict)

        if self.layer_type == 'Det': # only add xyz and feature in 'Det' mode
            output_dict[maps_dict.KEY_OUTPUT_XYZ].append(xyz_input)
            output_dict[maps_dict.KEY_OUTPUT_FEATURE].append(feature_input)

        return xyz_input, feature_input

    
    def format_input(self, feature):
        """ Enforce the feature can be operated by self.conv_op
        e.g: feature: [bs, pts_num, c] and self.conv_op == 'fc', 
             ---> format_feature: [bs, pts_num * c]
        """ 
        feature_shape = feature.get_shape().as_list()
        format_feature = feature
        if (self.op_type == 'fc') and (len(feature_shape) != 2):
            format_feature = tf.reshape(feature, [feature_shape[0], -1]) 
        if (self.op_type == 'conv1d') and (len(feature_shape) != 3): 
            format_feature = tf.reshape(feature, [self.batch_size, -1, feature_shape[-1]]) 
        if (self.op_type == 'conv2d') and (len(feature_shape) != 4):
            raise Exception('Implementation Error!!!')
        return format_feature 
