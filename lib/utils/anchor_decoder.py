import numpy as np
import tensorflow as tf

from core.config import cfg

def decode_class2angle(pred_cls, pred_res_norm, bin_size, bin_interval, bin_offset=0.):
    """
    decode the angle from the predicted class
    """
    pred_cls_onehot = tf.cast(tf.one_hot(pred_cls, depth=bin_size, on_value=1, off_value=0, axis=-1), tf.float32)
    pred_angle_res_norm = tf.reduce_sum(pred_cls_onehot * pred_res_norm, axis=-1)
    f_pred_cls = tf.cast(pred_cls, tf.float32)
    pred_res = (f_pred_cls + pred_angle_res_norm + bin_offset) * bin_interval
    return pred_res


def decode_log_anchor(det_residual, det_angle_cls, det_angle_res, batch_anchors_3d, is_training):
    """
    Decode bin loss anchors:
    Args:
        det_residual: [bs, points_num, 6]
        det_angle_cls: [bs, points_num, -1]
        det_angle_res: [bs, points_num, -1]
        batch_anchors_3d: [bs, points_num, 7]
    Return:
        pred_anchors_3d: [bs, points_num, 7]
    """
    det_ctr, det_offset = tf.split(det_residual, num_or_size_splits=2, axis=-1) # [bs, anchors_num, 3] * 2

    a_l, a_h, a_w = tf.unstack(batch_anchors_3d[:, :, 3:-1], axis=-1) # [bs, anchors_num]
    a_d = tf.norm(tf.stack([a_l, a_w], axis=-1), axis=-1) 

    a_x, a_y, a_z = tf.unstack(batch_anchors_3d[:, :, :3], axis=-1) # [bs, anchors_num]
    p_x, p_y, p_z = tf.unstack(det_ctr, axis=-1)
    p_l, p_h, p_w = tf.unstack(det_offset, axis=-1)

    pred_x = p_x * a_d + a_x
    pred_y = p_y * a_h + a_y
    pred_z = p_z * a_d + a_z
    pred_l = tf.exp(p_l) * a_l
    pred_h = tf.exp(p_h) * a_h
    pred_w = tf.exp(p_w) * a_w

    
    det_angle_cls = tf.argmax(det_angle_cls, axis=-1)
    pred_angle = decode_class2angle(det_angle_cls, det_angle_res, bin_size=cfg.MODEL.ANGLE_CLS_NUM, bin_interval=2 * np.pi / cfg.MODEL.ANGLE_CLS_NUM)
    anchor_angle = batch_anchors_3d[:, :, -1]
    pred_angle = anchor_angle + pred_angle # bs, anchor_num
 
    pred_ctr = tf.stack([pred_x, pred_y, pred_z], axis=-1)
    pred_offset = tf.maximum(tf.stack([pred_l, pred_h, pred_w], axis=-1), 0.1)
    pred_angle = tf.expand_dims(pred_angle, axis=-1)

    pred_anchors_3d = tf.concat([pred_ctr, pred_offset, pred_angle], axis=-1)
    return pred_anchors_3d


def decode_dist_anchor(det_residual, det_angle_cls, det_angle_res, batch_anchors_3d, is_training):
    """
    Decode bin loss anchors:
    Args:
        det_residual: [bs, points_num, 6]
        det_angle_cls: [bs, points_num, -1]
        det_angle_res: [bs, points_num, -1]
        batch_anchors_3d: [bs, points_num, 7]
    Return:
        pred_anchors_3d: [bs, points_num, 7]
    """
    det_ctr, det_offset = tf.split(det_residual, num_or_size_splits=2, axis=-1)

    pred_ctr = batch_anchors_3d[:, :, :3] + det_ctr

    pred_offset = batch_anchors_3d[:, :, 3:6] + det_offset * batch_anchors_3d[:, :, 3:6]
    pred_offset = tf.maximum(pred_offset, 0.1)
    
    det_angle_cls = tf.argmax(det_angle_cls, axis=-1)
    pred_angle = decode_class2angle(det_angle_cls, det_angle_res, bin_size=cfg.MODEL.ANGLE_CLS_NUM, bin_interval=2 * np.pi / cfg.MODEL.ANGLE_CLS_NUM)
    anchor_angle = batch_anchors_3d[:, :, -1]
    pred_angle = anchor_angle + pred_angle # bs, anchor_num
    pred_angle = tf.expand_dims(pred_angle, axis=-1)

    pred_anchors_3d = tf.concat([pred_ctr, pred_offset, pred_angle], axis=-1)
    return pred_anchors_3d


def decode_dist_anchor_free(center_xyz, det_forced_6_distance, det_angle_cls, det_angle_res, is_training):
    """
    Decode the predicted box 3d from FCOS loss
    Args:
        center_xyz: [bs, points_num, 3]
        det_forced_6_distance: [bs, points_num, 6], distance to 6 surfaces
        original_xyz: [bs, ndataset, 4], original input points, byd default [bs, 16384, 4]
        original_anchor_size: [1, 1, cls_num, 3]
    """
    bs, points_num, _ = center_xyz.get_shape().as_list()

    det_angle_cls = tf.argmax(det_angle_cls, axis=-1)
    pred_angle = decode_class2angle(det_angle_cls, det_angle_res, bin_size=cfg.MODEL.ANGLE_CLS_NUM, bin_interval=2 * np.pi / cfg.MODEL.ANGLE_CLS_NUM)
    pred_angle = tf.expand_dims(pred_angle, axis=-1)

    translate_vector = det_forced_6_distance[:, :, :3]
    half_distance = det_forced_6_distance[:, :, 3:6]
    ctr_xyz = center_xyz + translate_vector
    # then add half translate_vector to ctr_xyz
    padding_half_height = half_distance[:, :, 1]
    padding_zeros = tf.zeros_like(padding_half_height)
    padding_translate = tf.stack([padding_zeros, padding_half_height, padding_zeros], axis=-1)
    ctr_xyz += padding_translate
    lhw = tf.maximum(half_distance * 2., 0.1)

    pred_anchors_3d = tf.concat([ctr_xyz, lhw, pred_angle], axis=-1)
    return pred_anchors_3d


def decode_bin_anchor(det_residual, det_angle_cls, det_angle_res, batch_anchors_3d, is_training,
                      half_bin_search_range, bin_num_class):
    """
    Decode bin loss anchors:
    Args:
        det_residual: [bs, points_num, xbin/xres/zbin/zres/yres/offset]
        det_angle_cls: [bs, points_num, -1]
        det_angle_res: [bs, points_num, -1]
        batch_anchors_3d: [bs, points_num, 7]
    Return:
        pred_anchors_3d: [bs, points_num, 7]
    """
    x_bin = tf.slice(det_residual, [0, 0, bin_num_class * 0], [-1, -1, bin_num_class])
    x_res = tf.slice(det_residual, [0, 0, bin_num_class * 1], [-1, -1, bin_num_class])
    z_bin = tf.slice(det_residual, [0, 0, bin_num_class * 2], [-1, -1, bin_num_class])
    z_res = tf.slice(det_residual, [0, 0, bin_num_class * 3], [-1, -1, bin_num_class])
    det_offset = tf.slice(det_residual, [0, 0, bin_num_class * 4], [-1, -1, -1])

    anchor_x, anchor_y, anchor_z = tf.unstack(batch_anchors_3d[:, :, :3], axis=-1)


    x_bin = tf.argmax(x_bin, axis=-1)
    decode_x_res = decode_class2angle(x_bin, x_res, bin_size=bin_num_class, \
                                      bin_interval=(half_bin_search_range * 2 / bin_num_class), \
                                      bin_offset=0.5) 
    pred_x = anchor_x - half_bin_search_range + decode_x_res

    z_bin = tf.argmax(z_bin, axis=-1)
    decode_z_res = decode_class2angle(z_bin, z_res, bin_size=bin_num_class, \
                                      bin_interval=(half_bin_search_range * 2 / bin_num_class), \
                                      bin_offset=0.5) 
    pred_z = anchor_z - half_bin_search_range + decode_z_res 
    
    det_y_res = det_offset[:, :, 0]
    pred_y = anchor_y + det_y_res

    pred_ctr = tf.stack([pred_x, pred_y, pred_z], axis=-1)

    det_size_res = det_offset[:, :, 1:]
    pred_offset = batch_anchors_3d[:, :, 3:6] + det_size_res 
    pred_offset = tf.maximum(pred_offset, 0.1)
    
    det_angle_cls = tf.argmax(det_angle_cls, axis=-1)
    pred_angle = decode_class2angle(det_angle_cls, det_angle_res, bin_size=cfg.MODEL.ANGLE_CLS_NUM, bin_interval=2 * np.pi / cfg.MODEL.ANGLE_CLS_NUM)
    anchor_angle = batch_anchors_3d[:, :, -1]
    pred_angle = anchor_angle + pred_angle # bs, anchor_num
    pred_angle = tf.expand_dims(pred_angle, axis=-1)

    pred_anchors_3d = tf.concat([pred_ctr, pred_offset, pred_angle], axis=-1)
    return pred_anchors_3d
