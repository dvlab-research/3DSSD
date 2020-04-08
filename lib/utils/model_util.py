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


####################################################################
# Some Useful Loss Function 
####################################################################
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



####################################################################
# Some Useful Tools
####################################################################
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


def merge_head_prediction(start_idx, output_dict, keys):
    """
    Merge items in output dict from start_idx to -1, and add the merged result to output_dict
    for each item in output_dict, it has a shape of [bs, npoint, ...]
    merge them on npoint dimension
    """
    for key in keys:
        cur_output = output_dict[key][start_idx:]
        if len(cur_output) == 0: continue
        merged_output = tf.concat(cur_output, axis=1) 
        output_dict[key].append(merged_output)
    return 


def cast_bottom_to_center(boxes_3d):
    """ Cast the xyz location of a boxes_3d from bottom point to center point
    boxes_3d: [..., 7]
    """
    cx, by, cz, l, h, w, ry = tf.unstack(boxes_3d, axis=-1)

    # cast bottom point to center
    cy = by - h / 2.

    ctr_boxes_3d = tf.stack([cx, cy, cz, l, h, w, ry], axis=-1)
    return ctr_boxes_3d
