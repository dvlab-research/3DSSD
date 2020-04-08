import numpy as np
import tensorflow as tf

from core.config import cfg

###########################################################
# Encode Angle 
###########################################################
def encode_angle2class_np(angle, num_class):
    angle = np.mod(angle, (2 * np.pi))
    assert np.all(np.logical_and(angle>=0, angle<=2*np.pi))
    angle_per_class = 2*np.pi/float(num_class)
    shifted_angle = (angle+angle_per_class/2)%(2*np.pi)
    class_id = (shifted_angle/angle_per_class).astype(np.int)
    residual_angle = shifted_angle - \
        (class_id * angle_per_class + angle_per_class/2)
    # finally normalize residual angle to [0, 1]
    residual_angle = residual_angle / (2 * np.pi / num_class) 
    return class_id, residual_angle


def encode_angle2class_tf(angle, num_class):
    angle = tf.mod(angle, (2 * np.pi))

    angle_per_class = 2*np.pi/float(num_class)
    shifted_angle = tf.mod(angle+angle_per_class/2, 2*np.pi)
    class_id_f = tf.floor(shifted_angle/angle_per_class)
    class_id = tf.cast(class_id_f, tf.int32)
    residual_angle = shifted_angle - \
        (class_id_f * angle_per_class + angle_per_class/2)
    # finally normalize residual angle to [0, 1]
    residual_angle = residual_angle / angle_per_class
    return class_id, residual_angle


###########################################################
# Encode PointRCNN Bin
###########################################################
def encode_bin_residual(assigned_res, half_bin_search_range, bin_num_class):
    """ Encode assigned_residual to bin_class and bin_residual
    """
    bin_interval_range = half_bin_search_range * 2 / bin_num_class
    bin_cls = tf.floor((assigned_res + half_bin_search_range) / bin_interval_range)
    bin_cls = tf.minimum(tf.maximum(0., bin_cls), float(bin_num_class - 1))

    bin_res = assigned_res + half_bin_search_range \
             - (bin_cls * bin_interval_range + bin_interval_range / 2.)
    bin_res = bin_res / bin_interval_range
    return bin_cls, bin_res
    


###########################################################
# Log-Anchor
###########################################################
# tf
def encode_log_anchor(gt_ctr, gt_offset, anchor_ctr, anchor_offset):
    """
    Encoding part-a2 center
    :param:
        gt_ctr: [bs, points_num, 3]
        gt_offset: [bs, points_num, 3]
        anchor_ctr: [bs, points_num, 3]
        anchor_offset: [bs, points_num, 3]
    :return:
        encoded_ctr: [bs, points_num, 3]
        encoded_offset: [bs, points_num, 3]
    """
    gt_l, gt_h, gt_w = tf.unstack(gt_offset, axis=-1) # [bs, points_num]
    anchor_l, anchor_h, anchor_w = tf.unstack(anchor_offset, axis=-1) # [bs, points_num]

    anchor_d = tf.stack([anchor_l, anchor_w], axis=-1)  # [bs, points_num, 2]
    anchor_d = tf.norm(anchor_d, axis=-1) # [bs, points_num]

    gt_x, gt_y, gt_z = tf.unstack(gt_ctr, axis=-1)
    anchor_x, anchor_y, anchor_z = tf.unstack(anchor_ctr, axis=-1)

    encode_x = (gt_x - anchor_x) / anchor_d
    encode_y = (gt_y - anchor_y) / anchor_h
    encode_z = (gt_z - anchor_z) / anchor_d

    encode_l = tf.log(gt_l / anchor_l)
    encode_h = tf.log(gt_h / anchor_h)
    encode_w = tf.log(gt_w / anchor_w)
    
    encoded_ctr = tf.stack([encode_x, encode_y, encode_z], axis=-1)
    encoded_offset = tf.stack([encode_l, encode_h, encode_w], axis=-1)

    return encoded_ctr, encoded_offset 

# numpy
def encode_log_anchor_np(gt_ctr, gt_offset, anchor_ctr, anchor_offset):
    """
    Encoding part-a2 center
    :param:
        gt_ctr: [bs, points_num, 3]
        gt_offset: [bs, points_num, 3]
        anchor_ctr: [bs, points_num, 3]
        anchor_offset: [bs, points_num, 3]
    :return:
        encoded_ctr: [bs, points_num, 3]
        encoded_offset: [bs, points_num, 3]
    """
    gt_l, gt_h, gt_w = np.split(gt_offset, 3, axis=-1) # [bs, points_num, 1]
    anchor_l, anchor_h, anchor_w = np.split(anchor_offset, 3, axis=-1) # [bs, points_num, 1]

    anchor_d = np.concatenate([anchor_l, anchor_w], axis=-1)  # [bs, points_num, 2]
    anchor_d = np.linalg.norm(anchor_d, axis=-1, keepdims=True) # [bs, points_num, 1]

    gt_x, gt_y, gt_z = np.split(gt_ctr, 3, axis=-1) # [bs, points_num, 1]
    anchor_x, anchor_y, anchor_z = np.split(anchor_ctr, 3, axis=-1) # [bs, points_num, 1]

    encode_x = (gt_x - anchor_x) / anchor_d
    encode_y = (gt_y - anchor_y) / anchor_h
    encode_z = (gt_z - anchor_z) / anchor_d

    encode_l = np.log(gt_l / anchor_l)
    encode_h = np.log(gt_h / anchor_h)
    encode_w = np.log(gt_w / anchor_w)
    
    encoded_ctr = np.concatenate([encode_x, encode_y, encode_z], axis=-1)
    encoded_offset = np.concatenate([encode_l, encode_h, encode_w], axis=-1)

    return encoded_ctr, encoded_offset 


###########################################################
# Dist-Anchor
###########################################################
def encode_dist_anchor(gt_ctr, gt_offset, anchor_ctr, anchor_offset):
    """
    Encoding Distance anchors 
    :param:
        gt_ctr: [bs, points_num, 3]
        gt_offset: [bs, points_num, 3]
        anchor_ctr: [bs, points_num, 3]
        anchor_offset: [bs, points_num, 3]
    :return:
        encoded_ctr: [bs, points_num, 3]
        encoded_offset: [bs, points_num, 3]
    """
    encoded_ctr = gt_ctr - anchor_ctr
    encoded_offset = (gt_offset - anchor_offset) / anchor_offset
    return encoded_ctr, encoded_offset


###########################################################
# Dist-Anchor-Free
###########################################################
# tf
def encode_dist_anchor_free(gt_ctr, gt_offset, anchor_ctr, anchor_offset=None):
    """
    3DSSD anchor-free encoder
    :param:
        gt_ctr: [bs, points_num, 3]
        gt_offset: [bs, points_num, 3]
        anchor_ctr: [bs, points_num, 3]
        anchor_offset: [bs, points_num, 3]
    :return:
        encoded_ctr: [bs, points_num, 3]
        encoded_offset: [bs, points_num, 3]
    """
    
    target_ctr_half = gt_offset / 2.

    # translate to center
    padding_half_height = target_ctr_half[:, :, 1]
    padding_zeros = tf.zeros_like(padding_half_height)
    padding_translate = tf.stack([padding_zeros, padding_half_height, padding_zeros], axis=-1) # [bs, points_num, 3]

    encoded_ctr = gt_ctr - padding_translate # to object center
    encoded_ctr = encoded_ctr - anchor_ctr
    return encoded_ctr, target_ctr_half

# np 
def encode_dist_anchor_free_np(gt_ctr, gt_offset, anchor_ctr, anchor_offset=None):
    """
    3DSSD anchor-free encoder
    :param:
        gt_ctr: [bs, points_num, 3]
        gt_offset: [bs, points_num, 3]
        anchor_ctr: [bs, points_num, 3]
        anchor_offset: [bs, points_num, 3]
    :return:
        encoded_ctr: [bs, points_num, 3]
        encoded_offset: [bs, points_num, 3]
    """
    
    target_ctr_half = gt_offset / 2.

    # translate to center
    padding_half_height = target_ctr_half[:, :, 1]
    padding_zeros = np.zeros_like(padding_half_height)
    padding_translate = np.stack([padding_zeros, padding_half_height, padding_zeros], axis=-1) # [bs, points_num, 3]

    encoded_ctr = gt_ctr - padding_translate # to object center
    encoded_ctr = encoded_ctr - anchor_ctr
    return encoded_ctr, target_ctr_half
    

###########################################################
# Bin-Anchor
###########################################################
# tf
def encode_bin_anchor(gt_ctr, gt_offset, anchor_ctr, anchor_offset, half_bin_search_range, bin_num_class):
    """
    Encoding PointRCNN bin label 
    :param:
        gt_ctr: [bs, points_num, 3]
        gt_offset: [bs, points_num, 3]
        anchor_ctr: [bs, points_num, 3]
        anchor_offset: [bs, points_num, 3]
    :return:
        encoded_ctr: [bs, points_num, 3]
        encoded_offset: [bs, points_num, 3]
    """
    gt_x, gt_y, gt_z = tf.unstack(gt_ctr, axis=-1)
    anchor_x, anchor_y, anchor_z = tf.unstack(anchor_ctr, axis=-1)

    x_bin, x_res = encode_bin_residual(gt_x - anchor_x, half_bin_search_range, bin_num_class)
    z_bin, z_res = encode_bin_residual(gt_z - anchor_z, half_bin_search_range, bin_num_class)

    y_res = gt_y - anchor_y
    y_res = tf.expand_dims(y_res, axis=-1)
    offset = gt_offset - anchor_offset
     
    encoded_ctr = tf.stack([x_bin, x_res, z_bin, z_res], axis=-1)
    encoded_offset = tf.concat([y_res, offset], axis=-1)

    return encoded_ctr, encoded_offset 
