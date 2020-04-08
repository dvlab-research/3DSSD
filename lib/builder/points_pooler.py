import tensorflow as tf
import numpy as np

from core.config import cfg
from utils.tf_ops.grouping.tf_grouping import query_boxes_3d_points, group_point, query_boxes_3d_mask
from utils.tf_ops.points_pooling.points_pooling import points_pooling
from utils.rotation_util import rotate_points
from utils.pool_utils import *

class PointsPooler:
    def __init__(self, pool_cfg):
        self.pool_type = pool_cfg[0]
        self.pool_info_keys = pool_cfg[1] 
        self.pool_channel_list = pool_cfg[2]
        self.sample_pts_num = pool_cfg[3]
        self.context_range = pool_cfg[4]
        self.l, self.h, self.w, self.sample_num = pool_cfg[5]
        self.vfe_channel_list = pool_cfg[6]
        self.bn = pool_cfg[7]
        self.scope = pool_cfg[8]

        pool_function_dict = {
            'RegionPool': self.region_pool,
            'PointsPool': self.points_pool, 
        }
        self.pool = pool_function_dict[self.pool_type] 

    def get_valid_mask(self, base_xyz, pred_boxes_3d):
        """
        base_xyz: [bs, pts_num, 3]
        pred_boxes_3d: [bs, proposal_num, 7]
        """
        expand_pred_boxes_3d = self.expand_proposals_context(pred_boxes_3d)
        mask = query_boxes_3d_mask(base_xyz, expand_pred_boxes_3d) # bs, proposal_num, pts_num
        mask = tf.reduce_max(mask, axis=-1) # bs, proposal_num
        mask = tf.expand_dims(mask, axis=-1)
        return mask
        

    def region_pool(self, base_xyz, base_feature, base_mask, pred_boxes_3d, is_training, bn_decay):
        """
        base_xyz: [bs, pts_num, 3], xyz values
        base_feature: [bs, pts_num, c]
        pred_boxes_3d: [bs, proposal_num, 7]
        """ 
        expand_pred_boxes_3d = self.expand_proposals_context(pred_boxes_3d)

        pool_xyz, pool_info, pool_feature, pool_mask = self.gather_interior_xyz_feature(base_xyz, base_feature, base_mask, expand_pred_boxes_3d)

        # then canonical pool_xyz
        canonical_xyz = self.canonical_xyz(pool_xyz, expand_pred_boxes_3d)

        additional_info = tf.concat([canonical_xyz, pool_info], axis=-1)
        pool_feature = self.align_info_and_feature(additional_info, pool_feature, is_training, bn_decay)
        
        pool_output = tf.concat([canonical_xyz, pool_feature], axis=-1) # bs, proposal_num, nsample, c

        bs, proposal_num, nsample, c = pool_output.get_shape().as_list() 
        pool_output = tf.reshape(pool_output, [bs * proposal_num, nsample, c])

        return pool_output, pool_mask


    def points_pool(self, base_xyz, base_feature, base_mask, pred_boxes_3d, is_training, bn_decay):
        """PointsPool cast sparse feature to dense grids
        """
        expand_pred_boxes_3d = self.expand_proposals_context(pred_boxes_3d)

        pool_xyz, pool_info, pool_feature, pool_mask = self.gather_interior_xyz_feature(base_xyz, base_feature, base_mask, expand_pred_boxes_3d)

        # then canonical pool_xyz
        canonical_xyz = self.canonical_xyz(pool_xyz, expand_pred_boxes_3d)

        # put canonical_xyz back to the center of each proposal
        local_canonical_xyz = canonical_xyz + tf.expand_dims(expand_pred_boxes_3d[:, :, :3], axis=2)

        additional_info = tf.concat([local_canonical_xyz, canonical_xyz, pool_info], axis=-1)
        additional_info_channel = additional_info.get_shape().as_list()[-1]
        pool_feature = tf.concat([additional_info, pool_feature], axis=-1)

        dense_feature, idx, voxel_pts_num, voxel_ctrs = points_pooling(pool_feature, expand_pred_boxes_3d[:, :, :-1], local_canonical_xyz, l=self.l, h=self.h, w=self.w, sample_num=self.sample_num) # bs, proposal_num, l, h, w, sample_num, c 
     
        bs, proposal_num, _, _, _, _, channel_num = dense_feature.get_shape().as_list()
        dense_feature = tf.reshape(dense_feature, 
            [bs * proposal_num, self.l * self.h * self.w, self.sample_num, channel_num])
        dense_feature_mask = tf.reshape(voxel_pts_num, [bs * proposal_num, self.l * self.h * self.w, 1])
        dense_feature_mask = tf.greater(dense_feature_mask, 0)
        dense_feature_mask = tf.cast(dense_feature_mask, tf.float32) 
        voxel_ctrs = tf.reshape(voxel_ctrs, [bs * proposal_num, self.l * self.h * self.w, 1, 3])

        dense_local_xyz = tf.slice(dense_feature, [0, 0, 0, 0], [-1, -1, -1, 3])
        dense_canonical_xyz = tf.slice(dense_feature, [0, 0, 0, 3], [-1, -1, -1, 3])
        dense_info = tf.slice(dense_feature, [0, 0, 0, 6], [-1, -1, -1, additional_info_channel - 6])
        dense_feature = tf.slice(dense_feature, [0, 0, 0, additional_info_channel], [-1, -1, -1, -1])

        dense_pillars_info = dense_local_xyz - voxel_ctrs
        dense_additional_info = tf.concat([dense_canonical_xyz, dense_info, dense_pillars_info], axis=-1) # [bs * proposal_num, l*h*w, sample_num, c] 
        dense_feature = self.align_info_and_feature(dense_additional_info, dense_feature, is_training, bn_decay) 

        # finally VFE layer
        dense_feature = align_channel_network(dense_feature, self.vfe_channel_list, self.bn, is_training, bn_decay, '%s/vfe'%self.scope) 
        dense_feature = tf.reduce_max(dense_feature, axis=2)
        dense_feature = dense_feature * dense_feature_mask

        dense_voxel_ctrs = tf.reshape(voxel_ctrs, [bs * proposal_num, self.l * self.h * self.w,3])
        dense_feature = tf.concat([dense_voxel_ctrs, dense_feature], axis=-1)
        return dense_feature, pool_mask

        
    def align_info_and_feature(self, additional_info, pool_feature, is_training, bn_decay):
        """
        Encode additional information, and concatenate with pool_feature
        """
        encoded_info = align_channel_network(additional_info, self.pool_channel_list, self.bn, is_training, bn_decay, self.scope)
        pool_feature = tf.concat([encoded_info, pool_feature], axis=-1)
        return pool_feature


    def gather_interior_xyz_feature(self, base_xyz, base_feature, base_mask, pred_boxes_3d):
        """
        Gather interior points among proposals and generate their features 
        """
        pool_idx, pool_cnt = query_boxes_3d_points(self.sample_pts_num, base_xyz, pred_boxes_3d) 
        # [bs, proposal, 1]
        pool_mask = tf.expand_dims(tf.cast(tf.greater(pool_cnt, 0), tf.int32), axis=-1)
        pool_idx = pool_mask * pool_idx

        pool_xyz = group_point(base_xyz, pool_idx) # bs, proposal_num, nsample, 3
        pool_feature = group_point(base_feature, pool_idx)

        pool_info_list = []
        pool_info_dict = {
            'mask': group_point(base_mask, pool_idx),
            'dist': tf.norm(pool_xyz, axis=-1, keep_dims=True), 
        }
        for key in self.pool_info_keys:
            pool_info_list.append(pool_info_dict[key]) 
        pool_info = tf.concat(pool_info_list, axis=-1) # [bs, proposal_num, nsample, c1]
        
        return pool_xyz, pool_info, pool_feature, pool_mask
      
        
    def canonical_xyz(self, pool_xyz, proposals):
        """
        apply canonical transformation on pool_xyz
        pool_xyz: [bs, proposal_num, nsample, 3]
        proposals: [bs, proposal_num, 7]

        canonical_xyz: [bs, proposal_num, nsample, 3]
        """
        # first normalize pool_xyz by proposals_center 
        normalized_xyz = pool_xyz - tf.expand_dims(proposals, axis=2)[:, :, :, :3]

        # then rotate normalized_xyz
        canonical_xyz = rotate_points(normalized_xyz, -proposals[:, :, -1]) 
        return canonical_xyz


    def expand_proposals_context(self, pred_boxes_3d):
        """
        Expand proposals with context range
        pred_boxes_3d: [bs, proposal_num, 7]
        """
        pred_boxes_ctr = tf.slice(pred_boxes_3d, [0, 0, 0], [-1, -1, 3])
        pred_boxes_size = tf.slice(pred_boxes_3d, [0, 0, 3], [-1, -1, 3])
        pred_boxes_ry = tf.slice(pred_boxes_3d, [0, 0, 6], [-1, -1, -1])

        pred_boxes_size += self.context_range 
        expand_pred_boxes_3d = tf.concat([pred_boxes_ctr, pred_boxes_size, pred_boxes_ry], axis=-1)
        return expand_pred_boxes_3d
