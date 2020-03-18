import numpy as np
import tensorflow as tf

from core.config import cfg
import utils.anchor_encoder as anchor_encoder

def decode_class2angle(pred_cls, pred_res_norm, bin_size, bin_interval):
    """
    decode the angle from the predicted class
    """
    pred_cls_onehot = tf.cast(tf.one_hot(pred_cls, depth=bin_size, on_value=1, off_value=0, axis=-1), tf.float32)
    pred_angle_res_norm = tf.reduce_sum(pred_cls_onehot * pred_res_norm, axis=-1)
    f_pred_cls = tf.cast(pred_cls, tf.float32)
    pred_res = (f_pred_cls + pred_angle_res_norm) * bin_interval
    return pred_res


def decode_log_anchor(det_ctr, det_offset, det_angle_cls, det_angle_res, batch_anchors_3d, is_training):
    """
    Decode bin loss anchors:
    Args:
        det_ctr: [bs, points_num, 3]
        det_offset: [bs, points_num, 3]
        det_angle_cls: [bs, points_num, -1]
        det_angle_res: [bs, points_num, -1]
        batch_anchors_3d: [bs, points_num, 7]
    Return:
        pred_anchors_3d: [bs, points_num, 7]
    """
    a_l, a_h, a_w = tf.unstack(batch_anchors_3d[:, :, 3:-1], axis=-1) # [bs, anchors_num]
    a_d = tf.norm(tf.stack([a_l, a_w], axis=-1), axis=-1) # [bs, anchors_nu]

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

    pred_ctr = tf.stack([pred_x, pred_y, pred_z], axis=-1)
    pred_offset = tf.maximum(tf.stack([pred_l, pred_h, pred_w], axis=-1), 0.1)
    pred_angle = tf.expand_dims(pred_angle, axis=-1)

    pred_anchors_3d = tf.concat([pred_ctr, pred_offset, pred_angle], axis=-1)
    # finally stop gradient here
    return pred_anchors_3d


def decode_dist_anchor(det_ctr, det_offset, det_angle_cls, det_angle_res, batch_anchors_3d, is_training):
    """
    Decode bin loss anchors:
    Args:
        det_ctr: [bs, points_num, 3]
        det_offset: [bs, points_num, 3]
        det_angle_cls: [bs, points_num, -1]
        det_angle_res: [bs, points_num, -1]
        batch_anchors_3d: [bs, points_num, 7]
    Return:
        pred_anchors_3d: [bs, points_num, 7]
    """
    pred_ctr = batch_anchors_3d[:, :, :3] + det_ctr

    pred_offset = batch_anchors_3d[:, :, 3:6] + det_offset * batch_anchors_3d[:, :, 3:6]
    pred_offset = tf.maximum(pred_offset, 0.1)
    
    det_angle_cls = tf.argmax(det_angle_cls, axis=-1)
    pred_angle = decode_class2angle(det_angle_cls, det_angle_res, bin_size=cfg.MODEL.ANGLE_CLS_NUM, bin_interval=2 * np.pi / cfg.MODEL.ANGLE_CLS_NUM)
    pred_angle = tf.expand_dims(pred_angle, axis=-1)

    pred_anchors_3d = tf.concat([pred_ctr, pred_offset, pred_angle], axis=-1)
    # finally stop gradient here
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
