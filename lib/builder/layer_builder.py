import tensorflow as tf
import numpy as np

import utils.tf_util as tf_util

from core.config import cfg
from utils.layers_util import *

import dataset.maps_dict as maps_dict

class LayerBuilder:
    def __init__(self, layer_idx, is_training, layer_cfg):
        self.layer_idx = layer_idx
        self.is_training = is_training

        self.layer_architecture = layer_cfg[self.layer_idx]

        self.xyz_index = self.layer_architecture[0]
        self.feature_index = self.layer_architecture[1]
        self.radius_list = self.layer_architecture[2]
        self.nsample_list = self.layer_architecture[3]
        self.mlp_list = self.layer_architecture[4]
        self.bn = self.layer_architecture[5]

        self.fps_sample_range_list = self.layer_architecture[6]
        self.fps_method_list = self.layer_architecture[7]
        self.npoint_list = self.layer_architecture[8]
        assert len(self.fps_sample_range_list) == len(self.fps_method_list)
        assert len(self.fps_method_list) == len(self.npoint_list)

        self.former_fps_idx = self.layer_architecture[9]
        self.use_attention = self.layer_architecture[10]
        self.layer_type = self.layer_architecture[11]
        self.scope = self.layer_architecture[12] 
        self.dilated_group = self.layer_architecture[13]
        self.vote_ctr_index = self.layer_architecture[14]
        self.aggregation_channel = self.layer_architecture[15]

        if self.layer_type in ['SA_Layer', 'Vote_Layer']:
            assert len(self.xyz_index) == 1
        elif self.layer_type == 'FP_Layer':
            assert len(self.xyz_index) == 2
        else: raise Exception('Not Implementation Error!!!')

    def build_layer(self, xyz_list, feature_list, fps_idx_list, bn_decay, output_dict):
        """
        Build layers
        """
        xyz_input = []
        for xyz_index in self.xyz_index:
            xyz_input.append(xyz_list[xyz_index])
 
        feature_input = []
        for feature_index in self.feature_index:
            feature_input.append(feature_list[feature_index])

        if self.former_fps_idx != -1:
            former_fps_idx = fps_idx_list[self.former_fps_idx]
        else:
            former_fps_idx = None

        if self.vote_ctr_index != -1:
            vote_ctr = xyz_list[self.vote_ctr_index]
        else: vote_ctr = None

        if self.layer_type == 'SA_Layer':
            new_xyz, new_points, new_fps_idx = pointnet_sa_module_msg(xyz_input[0], feature_input[0], self.radius_list, self.nsample_list, 
                                                                      self.mlp_list, self.is_training, bn_decay, self.bn, 
                                                                      self.fps_sample_range_list, self.fps_method_list, self.npoint_list, 
                                                                      former_fps_idx, self.use_attention, self.scope, 
                                                                      self.dilated_group, vote_ctr, self.aggregation_channel)
            xyz_list.append(new_xyz)
            feature_list.append(new_points)
            fps_idx_list.append(new_fps_idx)

        elif self.layer_type == 'FP_Layer':
            new_points = pointnet_fp_module(xyz_input[0], xyz_input[1], feature_input[0], feature_input[1], self.mlp_list, self.is_training, bn_decay, self.scope, self.bn)
            feature_list.append(new_points)
        
        elif self.layer_type == 'Vote_Layer':
            new_xyz, new_points = vote_layer(xyz_input[0], feature_input[0], self.mlp_list, self.is_training, bn_decay, self.bn, self.scope)
            output_dict[maps_dict.PRED_VOTE_BASE].append(xyz_input[0])
            output_dict[maps_dict.PRED_VOTE_OFFSET].append(new_xyz)
            xyz_list.append(new_xyz)
            feature_list.append(new_points)
            fps_idx_list.append(None)

        return xyz_list, feature_list, fps_idx_list
         
