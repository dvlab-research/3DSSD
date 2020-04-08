import tensorflow as tf
import numpy as np
import utils.tf_util as tf_util
import utils.model_util as model_util

from utils.tf_ops.grouping.tf_grouping import *
from utils.tf_ops.sampling.tf_sampling import *
from utils.tf_ops.interpolation.tf_interpolate import *
from core.config import cfg


def vote_layer(xyz, points, mlp_list, is_training, bn_decay, bn, scope):
    """
    Voting layer
    """
    with tf.variable_scope(scope) as sc:
        for i, channel in enumerate(mlp_list):
            points = tf_util.conv1d(points, channel, 1, padding='VALID', stride=1, bn=bn, scope='vote_layer_%d'%i, bn_decay=bn_decay, is_training=is_training)
        ctr_offsets = tf_util.conv1d(points, 3, 1, padding='VALID', stride=1, bn=False, activation_fn=None, scope='vote_offsets')

        min_offset = tf.reshape(cfg.MODEL.MAX_TRANSLATE_RANGE, [1, 1, 3])
        limited_ctr_offsets = tf.minimum(tf.maximum(ctr_offsets, min_offset), -min_offset)
        xyz = xyz + limited_ctr_offsets 
    return xyz, points, ctr_offsets


def pointnet_sa_module(xyz, points, mlp,
                       is_training, bn_decay, bn, 
                       scope):
    ''' PointNet Set Abstraction (SA) Module (Last Layer)
        Sample all points within the point cloud and extract a global feature
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            mlp_list: list of int32 -- output size for MLP on each point
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, mlp[-1] or mlp2[-1]) TF tensor
            idx: (batch_size, npoint, nsample) int32 -- indices for local regions
    '''
    with tf.variable_scope(scope) as sc:
        grouped_points = tf.concat([xyz, points], axis=-1) # [bs, npoint, 3+c] 

        for j, num_out_channel in enumerate(mlp):
            grouped_points = tf_util.conv1d(grouped_points, 
                                            num_out_channel, 
                                            1,
                                            padding='VALID', 
                                            bn=bn, 
                                            is_training=is_training,
                                            scope='conv%d' % j, 
                                            bn_decay=bn_decay)
        # bs, num_out_channel
        new_points = tf.reduce_max(grouped_points, axis=1)
    return new_points
    


def pointnet_sa_module_msg(xyz, points, radius_list, nsample_list, 
                           mlp_list, is_training, bn_decay, bn, 
                           fps_sample_range_list, fps_method_list, npoint_list, 
                           former_fps_idx, use_attention, scope,
                           dilated_group, vote_ctr=None, aggregation_channel=None,
                           debugging=False,
                           epsilon=1e-5):
    ''' PointNet Set Abstraction (SA) module with Multi-Scale Grouping (MSG)
        Input:
            xyz: (batch_size, ndataset, 3) TF tensor
            points: (batch_size, ndataset, channel) TF tensor
            npoint: int -- points sampled in farthest point sampling
            radius_list: list of float32 -- search radius in local region
            nsample_list: list of int32 -- how many points in each local region
            mlp_list: list of list of int32 -- output size for MLP on each point
            fps_method: 'F-FPS', 'D-FPS', 'FS'
            fps_start_idx: 
        Return:
            new_xyz: (batch_size, npoint, 3) TF tensor
            new_points: (batch_size, npoint, \sum_k{mlp[k][-1]}) TF tensor
    '''
    bs = xyz.get_shape().as_list()[0]
    with tf.variable_scope(scope) as sc:
        cur_fps_idx_list = []
        last_fps_end_index = 0
        for fps_sample_range, fps_method, npoint in zip(fps_sample_range_list, fps_method_list, npoint_list):
            tmp_xyz = tf.slice(xyz, [0, last_fps_end_index, 0], [-1, fps_sample_range, -1]) 
            tmp_points = tf.slice(points, [0, last_fps_end_index, 0], [-1, fps_sample_range, -1])
            if npoint == 0: 
                last_fps_end_index += fps_sample_range 
                continue
            if vote_ctr is not None:
                npoint = vote_ctr.get_shape().as_list()[1]
                fps_idx = tf.tile(tf.reshape(tf.range(npoint), [1, npoint]), [bs, 1])
            elif fps_method == 'FS':
                features_for_fps = tf.concat([tmp_xyz, tmp_points], axis=-1)
                features_for_fps_distance = model_util.calc_square_dist(features_for_fps, features_for_fps, norm=False) 
                fps_idx_1 = farthest_point_sample_with_distance(npoint, features_for_fps_distance)
                fps_idx_2 = farthest_point_sample(npoint, tmp_xyz)
                fps_idx = tf.concat([fps_idx_1, fps_idx_2], axis=-1) # [bs, npoint * 2]
            elif npoint == tmp_xyz.get_shape().as_list()[1]:
                fps_idx = tf.tile(tf.reshape(tf.range(npoint), [1, npoint]), [bs, 1])
            elif fps_method == 'F-FPS':
                features_for_fps = tf.concat([tmp_xyz, tmp_points], axis=-1)
                features_for_fps_distance = model_util.calc_square_dist(features_for_fps, features_for_fps, norm=False) 
                fps_idx = farthest_point_sample_with_distance(npoint, features_for_fps_distance)
            else: # D-FPS
                fps_idx = farthest_point_sample(npoint, tmp_xyz)

            fps_idx = fps_idx + last_fps_end_index 
            cur_fps_idx_list.append(fps_idx)
            last_fps_end_index += fps_sample_range 
        fps_idx = tf.concat(cur_fps_idx_list, axis=-1)

        if former_fps_idx is not None:
            fps_idx = tf.concat([fps_idx, former_fps_idx], axis=-1) 

        if vote_ctr is not None:
            new_xyz = gather_point(vote_ctr, fps_idx)
        else:
            new_xyz = gather_point(xyz, fps_idx)

        # if deformed_xyz is not None, then no attention model
        if use_attention:
            # first gather the points out 
            new_points = gather_point(points, fps_idx) # [bs, npoint, c] 

            # choose farthest feature to center points
            # [bs, npoint, ndataset]
            relation = model_util.calc_square_dist(new_points, points)
            # choose these points with largest distance to center_points
            _, relation_idx = tf.nn.top_k(relation, k=relation.shape.as_list()[-1])
 
        idx_list, pts_cnt_list = [], []
        cur_radius_list = []
        for i in range(len(radius_list)):
            radius = radius_list[i]
            nsample = nsample_list[i]
            if dilated_group:
                # cfg.POINTNET.DILATED_GROUPING
                if i == 0: min_radius = 0.
                else: min_radius = radius_list[i - 1]
                idx, pts_cnt = query_ball_point_dilated(min_radius, radius, nsample, xyz, new_xyz)
            elif use_attention:
                idx, pts_cnt = query_ball_point_withidx(radius, nsample, xyz, new_xyz, relation_idx)
            else:
                idx, pts_cnt = query_ball_point(radius, nsample, xyz, new_xyz)
            idx_list.append(idx)
            pts_cnt_list.append(pts_cnt)

        # debugging
        debugging_list = []
        new_points_list = []
        for i in range(len(radius_list)):
            nsample = nsample_list[i]
            idx, pts_cnt = idx_list[i], pts_cnt_list[i]
            radius = radius_list[i]

            pts_cnt_mask = tf.cast(tf.greater(pts_cnt, 0), tf.int32) # [bs, npoint]
            pts_cnt_fmask = tf.cast(pts_cnt_mask, tf.float32)
            idx = idx * tf.expand_dims(pts_cnt_mask, axis=2)  # [bs, npoint, nsample]
            grouped_xyz = group_point(xyz, idx)
            original_xyz = grouped_xyz
            grouped_xyz -= tf.expand_dims(new_xyz, 2)
            grouped_points = group_point(points, idx)

            grouped_points = tf.concat([grouped_points, grouped_xyz], axis=-1)

            for j, num_out_channel in enumerate(mlp_list[i]):
                grouped_points = tf_util.conv2d(grouped_points, 
                                                num_out_channel, 
                                                [1, 1],
                                                padding='VALID', 
                                                stride=[1, 1], 
                                                bn=bn, 
                                                is_training=is_training,
                                                scope='conv%d_%d' % (i, j), 
                                                bn_decay=bn_decay)

            new_points = tf.reduce_max(grouped_points, axis=[2])

            new_points *= tf.expand_dims(pts_cnt_fmask, axis=-1)
            new_points_list.append(new_points)
        if len(new_points_list) > 0:
            new_points_concat = tf.concat(new_points_list, axis=-1)
            if cfg.MODEL.NETWORK.AGGREGATION_SA_FEATURE:
                new_points_concat = tf_util.conv1d(new_points_concat, aggregation_channel, 1, padding='VALID', bn=bn, is_training=is_training, scope='ensemble', bn_decay=bn_decay) 
        else: 
            new_points_concat = gather_point(points, fps_idx) 

    return new_xyz, new_points_concat, fps_idx


def pointnet_fp_module(xyz1, xyz2, points1, points2, mlp, is_training, bn_decay, scope, bn=True):
    ''' PointNet Feature Propogation (FP) Module
        Input:
            the unknown features 13
            xyz1: (batch_size, ndataset1, 3) TF tensor
            the known features 14
            xyz2: (batch_size, ndataset2, 3) TF tensor, sparser than xyz1
            points1: (batch_size, ndataset1, nchannel1) TF tensor
            points2: (batch_size, ndataset2, nchannel2) TF tensor
            mlp: list of int32 -- output size for MLP on each point
        Return:
            new_points: (batch_size, ndataset1, mlp[-1]) TF tensor
    '''
    with tf.variable_scope(scope) as sc:
        dist, idx = three_nn(xyz1, xyz2)
        dist = tf.maximum(dist, 1e-10)
        norm = tf.reduce_sum((1.0 / dist), axis=2, keep_dims=True)
        norm = tf.tile(norm, [1, 1, 3])
        weight = (1.0 / dist) / norm
        interpolated_points = three_interpolate(points2, idx, weight)

        if points1 is not None:
            new_points1 = tf.concat(axis=2, values=[interpolated_points, points1])  # B,ndataset1,nchannel1+nchannel2
        else:
            new_points1 = interpolated_points
        new_points1 = tf.expand_dims(new_points1, 2)
        for i, num_out_channel in enumerate(mlp):
            new_points1 = tf_util.conv2d(new_points1, num_out_channel, [1, 1],
                                         padding='VALID', stride=[1, 1],
                                         bn=bn, is_training=is_training,
                                         scope='conv_%d' % (i), bn_decay=bn_decay)
        new_points1 = tf.squeeze(new_points1, [2])  # B,ndataset1,mlp[-1]
        return new_points1

