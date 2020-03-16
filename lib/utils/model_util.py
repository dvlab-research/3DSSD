import numba
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import tensorflow.contrib.slim as slim
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import utils.tf_util as tf_util
from utils.tf_ops.sampling.tf_sampling import farthest_point_sample, gather_point
from utils.tf_ops.grouping.tf_grouping import query_corners_point, query_ball_point, group_point

from core.config import cfg

# -----------------
# Global Constants
# -----------------
################# Points RCNN #################
# original IPOD setting
g_type_mean_size = {'Kitti_Car': np.array([3.88311640418,1.62856739989,1.52563191462]),
                    'Kitti_Van': np.array([5.06763659,1.9007158,2.20532825]),
                    'Kitti_Truck': np.array([10.13586957,2.58549199,3.2520595]),
                    'Kitti_Pedestrian': np.array([0.84422524,1.76255119,0.66068622]),
                    'Kitti_Person_sitting': np.array([0.80057803,1.27450867,0.5983815]),
                    'Kitti_Cyclist': np.array([1.76282397,1.73698127,0.59706367]),
                    'Kitti_Tram': np.array([16.17150617,2.53246914,3.53079012]),
                    'Kitti_Misc': np.array([3.64300781,1.54298177,1.92320313]),
                    # original nuscenes
                    'NuScenes_child': np.array([0.527759, 1.376287, 0.513003]), # l, h, w 
                    'NuScenes_barrier': np.array([0.494674, 0.988850, 2.512046]),
                    'NuScenes_bicycle': np.array([1.698427, 1.293067, 0.604398]),
                    'NuScenes_bus': np.array([11.180965, 3.495353, 2.94905]),
                    'NuScenes_car': np.array([4.619270, 1.735112, 1.960518]),
                    'NuScenes_construction_vehicle': np.array([6.479316, 3.174820, 2.820066]),
                    'NuScenes_motorcycle': np.array([2.110251, 1.464422, 0.776560]),
                    'NuScenes_pedestrian': np.array([0.727708, 1.772415, 0.669095]),
                    'NuScenes_traffic_cone': np.array([0.414219, 1.076862, 0.408734]),
                    'NuScenes_trailer': np.array([12.283108, 3.865766, 2.922243]),
                    'NuScenes_truck': np.array([6.885711, 2.826359, 2.509883]),
                    # original lyft
                    'Lyft_car': np.array([4.756137, 1.718259, 1.922855]),
                    'Lyft_pedestrian': np.array([0.798200, 1.777827, 0.770559]),
                    'Lyft_animal': np.array([0.775029, 0.573300, 0.385750]),
                    'Lyft_other_vehicle': np.array([8.217489, 3.234986, 2.790774]),
                    'Lyft_bus': np.array([12.328907, 3.433031, 2.950655]),
                    'Lyft_motorcycle': np.array([2.368642, 1.583713, 0.978719]),
                    'Lyft_truck': np.array([10.333140, 3.463256, 2.843518]),
                    'Lyft_emergency_vehicle': np.array([5.758920, 2.294880, 2.304800]),
                    'Lyft_bicycle': np.array([1.753566, 1.444639, 0.630577]),
                    }

# -----------------
# TF Functions Helpers
# -----------------
def tf_random_choose_points(points_mask, npoints):
    """
    Randomly choose npoints point from points_mask whose value == 1
    points_mask: [b, num_proposals, num_points]

    return: [b, num_proposals, npoints], inside npoints is the index
    """
    bs = points_mask.shape[0]
    num_proposals = points_mask.shape[1]
    num_points = points_mask.shape[2]

    ret_result = np.zeros((bs, num_proposals, npoints), dtype=np.int32)
    useful_mask = np.ones((bs, num_proposals), dtype=np.int32)
    for i in range(bs):
        for j in range(num_proposals):
            cur_points_mask = points_mask[i, j, :]
            useful_idx = np.where(cur_points_mask >= 1)[0]
            useful_length = len(useful_idx)
            if useful_length == 0: 
                useful_mask[i, j] = 0
                continue
            if useful_length >= npoints:
                chosen_idx = np.random.choice(useful_idx, npoints, replace=False)
            else: # useful_length < npoints
                chosen_idx = np.random.choice(useful_idx, npoints, replace=True)
            ret_result[i, j, :] = chosen_idx
    return ret_result, useful_mask


def tf_gather_object_pc(point_cloud, mask, npoints=2048, is_training=True):
    ''' Gather object point clouds according to predicted masks.
        Rather than random choice, we use a FPS sampling
    Input:
        point_cloud: TF tensor in shape (B,N,C)
        mask: TF tensor in shape (B,N) of 0 (not pick) or 1 (pick)
        npoints: int scalar, maximum number of points to keep (default: 512)
    Output:
        object_pc: TF tensor in shape (B,npoint,C)
        indices: TF int tensor in shape (B,npoint,2)
    '''
    def mask_to_indices(mask, npoints):
        indices = np.zeros((mask.shape[0], npoints, 2), dtype=np.int32)
        for i in range(mask.shape[0]):
            pos_indices = np.where(mask[i,:]>0.5)[0]
            # skip cases when pos_indices is empty
            if len(pos_indices) > 0: 
                if len(pos_indices) >= npoints:
                    choice = np.random.choice(len(pos_indices),
                        npoints, replace=False)
                else:
                    choice = np.random.choice(len(pos_indices),
                        npoints-len(pos_indices), replace=True)
 
                    # testing
                    # if is_training:
                    #     rand_pos_indices = np.arange(len(pos_indices))
                    #     np.random.shuffle(rand_pos_indices)
                    # else:
                    #     rand_pos_indices = np.arange(len(pos_indices))

                    # origin
                    rand_pos_indices = np.arange(len(pos_indices))
                    choice = np.concatenate((rand_pos_indices, choice))

                # if is_training:
                #     if len(pos_indices) >= npoints:
                #         choice = np.random.choice(len(pos_indices), npoints, replace=False)
                #     else:
                #         choice = np.random.choice(len(pos_indices), npoints, replace=True)
                # else:
                #     # not is training
                #     choice = np.zeros([npoints])
                #     choice_length = len(pos_indices)
     
                if is_training:
                    np.random.shuffle(choice)
              
                indices[i,:,1] = pos_indices[choice]
            indices[i,:,0] = i
        return indices

    npoints = tf.constant(npoints)
    indices = tf.py_func(mask_to_indices, [mask, npoints], tf.int32)  
        
    # bs = tf.shape(point_cloud)[0]
    # points_num = tf.shape(point_cloud)[1]
    # channels = tf.shape(point_cloud)[2]
    # zero_points_cloud = tf.zeros([bs, 1, channels], dtype=point_cloud.dtype)
    # point_cloud_gather = tf.concat([zero_points_cloud, point_cloud], axis=1)

    object_pc = tf.gather_nd(point_cloud, indices)
    return object_pc, indices




def focal_loss_producer(prediction_tensor, target_tensor, weights=None, class_indices=None, _gamma=2.0, _alpha=0.25):
    """Compute loss function.
      Args:
        prediction_tensor: A float tensor of shape [batch_size, num_anchors,
          num_classes] representing the predicted logits for each class
        target_tensor: A float tensor of shape [batch_size, num_anchors,
          num_classes] representing one-hot encoded classification targets
        weights: a float tensor of shape [batch_size, num_anchors]
        class_indices: (Optional) A 1-D integer tensor of class indices.
          If provided, computes loss only for the specified class indices.
      Returns:
        loss: a float tensor of shape [batch_size, num_anchors, num_classes]
          representing the value of the loss function.
    """
    if class_indices is not None:
      weights = tf.expand_dims(weights, 2)
      weights *= tf.reshape(
          ops.indices_to_dense_vector(class_indices,
                                      tf.shape(prediction_tensor)[2]),
          [1, 1, -1])
    per_entry_cross_ent = (tf.nn.sigmoid_cross_entropy_with_logits(
        labels=target_tensor, logits=prediction_tensor))
    prediction_probabilities = tf.sigmoid(prediction_tensor)
    p_t = ((target_tensor * prediction_probabilities) +
           ((1 - target_tensor) * (1 - prediction_probabilities)))
    modulating_factor = 1.0
    if _gamma:
      modulating_factor = tf.pow(1.0 - p_t, _gamma)
    alpha_weight_factor = 1.0
    if _alpha is not None:
      alpha_weight_factor = (target_tensor * _alpha +
                             (1 - target_tensor) * (1 - _alpha))
    focal_cross_entropy_loss = (modulating_factor * alpha_weight_factor *
                                per_entry_cross_ent)
    return focal_cross_entropy_loss

def bg_loss_producer(prediction_tensor, target_tensor, _gamma=2.):
    """
    Classification score at 0.5 is the most valuable.
    """
    per_entry_cross_ent = (tf.nn.sigmoid_cross_entropy_with_logits(
        labels=target_tensor, logits=prediction_tensor))
    prediction_probabilities = tf.sigmoid(prediction_tensor)
    p_t = prediction_probabilities * (1 - prediction_probabilities)
    # then normalize p_t to 1.
    p_t = p_t / 0.25
    modulating_factor = 1.
    if _gamma:
        modulating_factor = tf.pow(p_t, _gamma)
    bg_cross_entropy_loss = modulating_factor * per_entry_cross_ent
    return bg_cross_entropy_loss


def exp_regression_yasminasun_loss(iou_pred, iou_label, norm_param):
    assert len(cfg.MODEL.EXP_REGRESSION_THRESHOLD) == 1, 'Only support one class until now !!!'
    threshold = float(cfg.MODEL.EXP_REGRESSION_THRESHOLD[0])
    gamma = float(cfg.MODEL.EXP_REGRESSION_GAMMA)
    ###################### original yasminasun loss ###########################
    loss_base = -1 * (iou_pred - threshold) * (iou_label - threshold) * gamma
    loss_base = tf.exp(loss_base)
    ###################### original yasminasun loss ###########################

    ###################### focal loss ###########################
    # loss_base = tf.pow(iou_pred - iou_label, gamma)
    ###################### focal loss ###########################

    loss_weight = tf.stop_gradient(loss_base)
    l2_loss = loss_weight * tf.square(iou_pred - iou_label) / 2.
    l2_loss = tf.reduce_sum(l2_loss) / norm_param
    return l2_loss 


def hard_sample_miner(cls_loss, loc_loss, pmask, nmask, pred_box, iou_thresh=0.7, num_hard_sample=128, loss_type='both', cls_loss_weights=0.05, loc_loss_weights=0.06, max_negatives_per_positive=None, min_negatives_per_image=16):
    if loss_type is 'loc':
        image_loss = loc_loss
    elif loss_type is 'cls':
        image_loss = cls_loss
    elif loss_type is 'both':
        image_loss = cls_loss_weights * cls_loss + loc_loss_weights * loc_loss
    else:
        raise ValueError('loss_type must enum in loc, cls or both')

    nms_index = tf.image.non_max_suppression(pred_box, image_loss, num_hard_sample, iou_threshold=iou_thresh)
    if max_negatives_per_positive is not None:
        nms_index = subsample_to_desire_ratio(nms_index, pmask, nmask, max_negatives_per_positive, min_negatives_per_image)


    hard_sample_fmask = tf.cast(indices_to_dense_vector(nms_index, tf.shape(cls_loss)[0]), tf.float32)

    debug_mask = tf.equal(hard_sample_fmask, 1)
    cnt = tf.reduce_sum(hard_sample_fmask)

    return cls_loss, loc_loss, hard_sample_fmask

def subsample_to_desire_ratio(nms_index, pmask, nmask, max_negatives_per_positive, min_negatives_per_image):
    positives_indicator = tf.gather(pmask, nms_index)
    negatives_indicator = tf.gather(nmask, nms_index)

    positive_num = tf.reduce_sum(tf.to_int32(positives_indicator))
    max_negatives = tf.maximum(min_negatives_per_image,
                               tf.to_int32(max_negatives_per_positive *
                                           tf.to_float(positive_num)))

    topk_negatives_indicator = tf.less_equal(
        tf.cumsum(tf.to_int32(negatives_indicator)), max_negatives)
    subsampled_selection_indices = tf.where(
        tf.logical_or(positives_indicator, topk_negatives_indicator))
    num_negatives = tf.size(subsampled_selection_indices) - positive_num

    return tf.reshape(tf.gather(nms_index, subsampled_selection_indices), [-1])


def huber_loss(error, delta):
    abs_error = tf.abs(error)
    quadratic = tf.minimum(abs_error, delta)
    linear = (abs_error - quadratic)
    losses = 0.5 * quadratic**2 + delta * linear
    return losses


def point_cloud_masking(point_cloud, logits, end_points, xyz_only=True, is_training=True):
    ''' Select point cloud with predicted 3D mask,
    translate coordinates to the masked points centroid.
    
    Input:
        point_cloud: TF tensor in shape (B,N,C)
        logits: TF tensor in shape (B,N,2)
        end_points: dict
        xyz_only: boolean, if True only return XYZ channels
    Output:
        object_point_cloud: TF tensor in shape (B,M,3)
            for simplicity we only keep XYZ here
            M = NUM_OBJECT_POINT as a hyper-parameter
        mask_xyz_mean: TF tensor in shape (B,3)
    '''
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    mask = tf.slice(logits,[0,0,0],[-1,-1,1]) < \
        tf.slice(logits,[0,0,1],[-1,-1,1])
    # [batch_size, num_points, 1]
    mask = tf.to_float(mask) # BxNx1
    mask_count = tf.tile(tf.reduce_sum(mask,axis=1,keep_dims=True),
        [1,1,3]) # Bx1x3
    point_cloud_xyz = tf.slice(point_cloud, [0,0,0], [-1,-1,3]) # BxNx3
    # the sum of xyz in each point in a point cloud in the segmentation res
    mask_xyz_mean = tf.reduce_sum(tf.tile(mask, [1,1,3])*point_cloud_xyz,
        axis=1, keep_dims=True) # Bx1x3
    mask = tf.squeeze(mask, axis=[2]) # BxN
    end_points['mask'] = mask
    # normalize the centroid of the segmentation points
    # figure4(b) to figure4(c)
    mask_xyz_mean = mask_xyz_mean/tf.maximum(mask_count,1) # Bx1x3

    # Translate to masked points' centroid
    # figure4(c)
    point_cloud_xyz_stage1 = point_cloud_xyz - \
        tf.tile(mask_xyz_mean, [1,num_point,1])
    # point_cloud_xyz_stage1 = point_cloud_xyz

    if xyz_only:
        point_cloud_stage1 = point_cloud_xyz_stage1
    else:
        point_cloud_features = tf.slice(point_cloud, [0,0,3], [-1,-1,-1])
        point_cloud_stage1 = tf.concat(\
            [point_cloud_xyz_stage1, point_cloud_features], axis=-1)
    num_channels = point_cloud_stage1.get_shape()[2].value

    # NUM_OBJECT_POINT: by default set 512
    # gather the object_point_cloud out
    # and then we reshape it to [batch_size, num_points(512), 3 or 4]
    NUM_OBJECT_POINT = 512 
    object_point_cloud, _ = tf_gather_object_pc(point_cloud_stage1,
        mask, NUM_OBJECT_POINT, is_training=is_training)
    object_point_cloud.set_shape([batch_size, NUM_OBJECT_POINT, num_channels])

    return object_point_cloud, tf.squeeze(mask_xyz_mean, axis=1), end_points


def points_rcnn_align_layer(proposals_prior_info, mlp, is_training, bn_decay, scope, bn=True, reuse=False, activation_final=True):
    info_dim = len(proposals_prior_info.get_shape().as_list())
    if info_dim == 3:
        kernel_size = 1
        conv_func = tf_util.conv1d
    elif info_dim == 4:
        kernel_size = [1, 1]
        conv_func = tf_util.conv2d
    else: raise Exception('Not implement in PointsRCNNAlignLayer')
    with tf.variable_scope(scope, reuse=reuse) as sc:
        for i, num_channel in enumerate(mlp):
            if i == len(mlp) - 1:
                if activation_final:
                    tmp_bn=bn
                    activation_fn=tf.nn.relu
                else:
                    tmp_bn=False
                    activation_fn = None
            else:
                tmp_bn=bn
                activation_fn=tf.nn.relu
            proposals_prior_info = conv_func(proposals_prior_info, num_channel, kernel_size, padding='VALID', bn=tmp_bn, is_training=is_training, scope='conv%d'%i, bn_decay=bn_decay, activation_fn=activation_fn)
    return proposals_prior_info


def vfe_merge_layer(points_pooled_feature, mlp, is_training, bn_decay, scope, bn=True, pooling='max', reuse=False, points_num_mask=None):
    print(points_pooled_feature)
    bs, voxel_num, sample_num, c = points_pooled_feature.get_shape().as_list()
    data_format = 'NHWC'
        
    with tf.variable_scope(scope, reuse=reuse) as sc:
        # first cast feature to [b, -1, sample_num, c]
        for i, num_out_channel in enumerate(mlp):
            points_pooled_feature = tf_util.conv2d(points_pooled_feature, num_out_channel, [1, 1],
                                                   padding='VALID', stride=[1, 1],
                                                   bn=bn, is_training=is_training,
                                                   scope='conv%d' % (i), bn_decay=bn_decay,
                                                   data_format=data_format) 
        if points_num_mask is not None:
            # then multi num_points_per_voxel and generate final points pool feature
            # points_num_mask: [bs, voxel_num, points_num, 1]
            points_pooled_feature = points_pooled_feature * points_num_mask 
            
        # [b, -1, sample_num, num_out_channel]
        if pooling == 'max':
            new_points_pooled_feature = tf.reduce_max(points_pooled_feature, axis=2, keep_dims=True, name='maxpool')
        elif pooling == 'avg':
            new_points_pooled_feature = tf.reduce_mean(points_pooled_feature, axis=2, keep_dims=True, name='avgpool')
        new_points_pooled_feature = tf.squeeze(new_points_pooled_feature, axis=2)
    return new_points_pooled_feature

        

def calc_point_similarity(net, sim_thresh=0.1):
    """ Calculating similarity between different points 

    Args:
        net: [b, n, c]: calc the feature similarity
    """
    similarity = tf.matmul(net, tf.transpose(net, [0, 2, 1]))
    similarity = tf.nn.softmax(similarity) # calc similarity
    # similarity = tf.greater_equal(similarity, sim_thresh)
    return similarity

def calc_cosine_similarity(a, b):
    """ Calculate cosine similarity between different set of points
    a: [bs, npoint, c]
    b: [bs, npoint, nsample, c]
    return: [bs, npoint, nsample]
    """
    a = tf.expand_dims(a, axis=2) # [bs, npoint, 1, c]  
    b = tf.expand_dims(b, axis=2) # [bs, npoint, 1, c]
    # then calc cosine similarity between a and b
    a_norm = tf.nn.l2_normalize(a, dim=-1) # [bs, npoint, 1, c]
    b_norm = tf.nn.l2_normalize(b, dim=-1) # [bs, npoint, 1, c]
    b_norm_transpose = tf.transpose(b_norm, [0, 1, 3, 2]) # [bs, npoint, c, 1]
    cos_sim = tf.matmul(a_norm, b_norm_transpose) # [bs, npoint, 1, 1]
    cos_sim = tf.squeeze(cos_sim, axis=[2, 3]) # [bs, npoint]
    return cos_sim


def calc_square_dist(a, b, norm=True):
    """
    Calculating square distance between a and b
    a: [bs, npoint, c]
    b: [bs, ndataset, c]
    """
    a = tf.expand_dims(a, axis=2) # [bs, npoint, 1, c]
    b = tf.expand_dims(b, axis=1) # [bs, 1, ndataset, c]
    a_square = tf.reduce_sum(tf.square(a), axis=-1) # [bs, npoint, 1]
    b_square = tf.reduce_sum(tf.square(b), axis=-1) # [bs, 1, ndataset]
    a = tf.squeeze(a, axis=2) # [bs, npoint,c]
    b = tf.squeeze(b, axis=1) # [bs, ndataset, c]
    if norm:
        dist = tf.sqrt(a_square + b_square - 2 * tf.matmul(a, tf.transpose(b, [0, 2, 1]))) / tf.cast(tf.shape(a)[-1], tf.float32) # [bs, npoint, ndataset]
    else:
        dist = a_square + b_square - 2 * tf.matmul(a, tf.transpose(b, [0, 2, 1])) # [bs, npoint, ndataset]
    return dist

def SigmRound(x):
    """
    Straight-Through Estimator
    Rounds a tensor whose values are in [-inf,inf] to a tensor with values in {0, 1},
    using the straight through estimator for the gradient.
    """
    g = tf.get_default_graph()

    with ops.name_scope("SigmRound") as name:
        with g.gradient_override_map({"Sign": "Identity"}):
            # in forward, cast the value to {0, 1}
            # and in backward, using tf.sigmoid to calc the backward
            return (tf.sign(x, name=name)+1)/2


def pass_gradient_to_other(x, y):
    """
    Straight-Through Estimator
    during training, only use y, 
    during gradient, only use x
    """
    g = tf.get_default_graph()

    with ops.name_scope("STE_Grad") as name:
        with g.gradient_override_map({"Sign": "Identity"}):
            # in forward, cast the value to {0, 1}
            # and in backward, using tf.sigmoid to calc the backward
            return tf.sign(x) - 1 + y


def get_interior_pts_mask(pts, gt_boxes_3d):
    """
    Get interior points mask
    pts: [bs, npoint, 3]
    gt_boxes_3d: [bs, gt_num, -1]
    """
    bs, _, _ = gt_boxes_3d.get_shape().as_list()
    gt_num = tf.shape(gt_boxes_3d)[1]
    gt_boxes_3d = tf.reshape(gt_boxes_3d, [bs * gt_num, 7])
    gt_boxes_corners = get_box3d_corners_helper(gt_boxes_3d[:, :3], gt_boxes_3d[:, -1], gt_boxes_3d[:, 3:-1] + 0.1)
    gt_boxes_corners = tf.reshape(gt_boxes_corners, [bs, gt_num, 8, 3])
    sem_labels_gt = query_corners_point(pts, gt_boxes_corners) # [bs, gt_num, pts_num]
    sem_labels = tf.cast(tf.reduce_max(sem_labels_gt, axis=1), tf.float32) # [bs, pts_num]
    argmax_sem_labels_gt = tf.cast(tf.argmax(sem_labels_gt, axis=1), tf.int32) # [bs, pts_num]
    gt_boxes_3d = tf.reshape(gt_boxes_3d, [bs, gt_num, 7])
    assigned_boxes_3d = gather_point(gt_boxes_3d, argmax_sem_labels_gt) # [bs, pts_num, 7]
    return assigned_boxes_3d, sem_labels

def gumbel_sample(shape, eps=1e-20):
    """
    Sample a variable from Gumbel distribution
    """
    u = tf.random_uniform(shape, minval=0, maxval=1) # draw from uniform distribution
    return -tf.log(-tf.log(u + eps) + eps)


def gumbel_softmax_sample(logits, temp):
    """
    add gumbel random variable to former logits
    for every last dimension, the max value is 1, and sum is one
    return:
        one_hot varaible like logits: [bs, npoint, nsample, ndataset]
    """
    # y = logits + gumbel_sample(tf.shape(logits))
    # return tf.nn.softmax( y / temp)
    y = logits # just choose the max angle
    return tf.nn.softmax(y)


def gumbel_softmax(logits, temp=0.5, scope='', hard=True):
    """
    Gumbel softNMS, a differetiate one_hot(argmax())
    logits: [bs, npoint, nsample], the distance between nsample-chosen point and ndataset
    temp: default param in gumbel_softnms, by default is 0.5
    hard: if true, then the maximum value is 1, else close to 1
    """
    with tf.variable_scope(scope) as sc:
        y = gumbel_softmax_sample(logits, temp)
        if hard:
            # then make it to one
            y_hard = tf.cast(tf.equal(y,tf.reduce_max(y,-1,keep_dims=True)),y.dtype)
            y = tf.stop_gradient(y_hard - y) + y # the max is 1 now
    return y

