import tensorflow as tf
import numpy as np

from core.config import cfg

def generate_3d_anchors_by_point(points, anchor_sizes):
    '''
        Generate 3d anchors by points
        points: [b, n, 3]
        anchors_size: [cls_num, 3], l, h, w

        Return [b, n, cls_num, 7]
    '''
    bs, points_num, _ = points.shape
    anchor_sizes = np.array(anchor_sizes) # [cls_num, 3]
    anchors_num = len(anchor_sizes)
   
    # then generate anchors for each points 
    ctr = np.tile(np.reshape(points, [bs, points_num, 1, 3]), [1, 1, anchors_num, 1]) # [x, y, z]

    offset = np.tile(np.reshape(anchor_sizes, [1, 1, anchors_num, 3]), [bs, points_num, 1, 1]) # [l, h, w]

    # then sub y to force anchors on the center
    ctr[:, :, :, 1] += offset[:, :, :, 1] / 2.
    ry = np.zeros([bs, points_num, anchors_num, 1], dtype=ctr.dtype)
    
    all_anchor_boxes_3d = np.concatenate([ctr, offset, ry], axis=-1)
    all_anchor_boxes_3d = np.reshape(all_anchor_boxes_3d, [bs, points_num, anchors_num, 7])

    return all_anchor_boxes_3d


def generate_3d_anchors_by_point_tf(points, anchor_sizes):
    '''
        Generate 3d anchors by points
        points: [bs, n, 3], xyz, the location of this points
        anchor_sizes: [cls_num, 3], lhw
    
        Return [b, n, cls_num, 7]
    '''
    bs, points_num, _ = points.get_shape().as_list()

    anchor_sizes = np.array(anchor_sizes).astype(np.float32)
    anchors_num = len(anchor_sizes)
    
    # then generate anchors for each points 
    x, y, z = tf.unstack(points, axis=-1) # [bs, points_num]
    x = tf.tile(tf.reshape(x, [bs, points_num, 1, 1]), [1, 1, anchors_num, 1])
    y = tf.tile(tf.reshape(y, [bs, points_num, 1, 1]), [1, 1, anchors_num, 1])
    z = tf.tile(tf.reshape(z, [bs, points_num, 1, 1]), [1, 1, anchors_num, 1])


    # then sub y_ctr by the anchor_size
    l, h, w = tf.unstack(anchor_sizes, axis=-1) # [anchors_num]
    l = tf.tile(tf.reshape(l, [1, 1, anchors_num, 1]), [bs, points_num, 1, 1])
    h = tf.tile(tf.reshape(h, [1, 1, anchors_num, 1]), [bs, points_num, 1, 1])
    w = tf.tile(tf.reshape(w, [1, 1, anchors_num, 1]), [bs, points_num, 1, 1])

    ry = tf.zeros_like(l) # [bs, points_num, anchors_num]

    y = y + h / 2.

    all_anchor_boxes_3d = tf.concat([x, y, z, l, h, w, ry], axis=-1)
    all_anchor_boxes_3d = tf.reshape(all_anchor_boxes_3d, [bs, points_num, anchors_num, 7])
    return all_anchor_boxes_3d 
