import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
points_nms_module=tf.load_op_library(os.path.join(BASE_DIR, 'tf_nms_so.so'))

def points_nms(iou_matrix, points_sample, merge_function, iou_thresh):
    return points_nms_module.points_nms(iou_matrix, points_sample, merge_function, iou_thresh) 
ops.NoGradient('PointsNms')

def points_iou(points_sample_mask):
    return points_nms_module.points_iou(points_sample_mask)
ops.NoGradient('PointsIou')

def points_nms_block(points_sample, merge_function, iou_thresh, num_to_keep):
    return points_nms_module.points_nms_block(points_sample, merge_function, iou_thresh, num_to_keep)
ops.NoGradient('PointsNmsBlock')

def points_inside_boxes(points, anchors):
    # points: [-1, 3], anchors: [-1, 6]
    # return: [boxes_num, points_num]
    return points_nms_module.points_inside_boxes(points, anchors)
ops.NoGradient('PointsInsideBoxes')

# test for points inside boxes:
# if __name__ == '__main__':
#     # debugging
#     points = tf.constant([[0, 0, 0], [1, 0, 0], [2, 0, 0], [3, 0, 0], [-1, 0, 0], [0.5, 0, 0]], tf.float32)
#     anchors = tf.constant([[0, 1, 0, 2, 2, 2], [0, 2, 0, 4, 4, 4]], tf.float32)
#     print(points)
#     print(anchors)
#     points_sample_mask = points_inside_boxes(points, anchors)
#     sess = tf.Session()
#     print(sess.run(points_sample_mask))

# test for points nms
if __name__ == '__main__':
    # debugging
    a = tf.constant([1, 0, 0, 0, 1, 1, 1, 0], dtype=tf.int32)
    b = tf.constant([1, 1, 1, 1, 0, 0, 0, 0], dtype=tf.int32)
    c = tf.constant([1, 0, 1, 1, 0, 1, 0, 0], dtype=tf.int32)
    d = tf.constant([0, 1, 0, 0, 0, 0, 1, 1], dtype=tf.int32)

    # debugging for points iou cacl
    # points_sample_mask = tf.stack([a, b, c, d], axis=0)
    # iou_matrix = points_iou(points_sample_mask)
    # sess = tf.Session()
    # res, mask = sess.run([iou_matrix, points_sample_mask])
    # print(res)
    # print('----------------------') 
    # print(mask)

    # debugging the points_nms_block
    # points_sample_mask = tf.stack([a, b, c, d], axis=0)
    points_sample_mask = tf.ones([60000, 10000], tf.int32)
    keep_inds, nmsed_points_sample = points_nms_block(points_sample_mask, 2, 0.5, 2)
    sess = tf.Session()
    print('-------------------------')
    print(sess.run(keep_inds))
    print('-------------------------')
    print(sess.run(nmsed_points_sample)) 
    

    # debugging for points merge
    # initialize the iou_matrix
    # pc_matrix_1 = tf.cast(tf.reshape(tf.stack([a, b, c, d], axis=0), [1, 4, 8]), tf.bool)
    # pc_matrix_2 = tf.cast(tf.reshape(tf.stack([a, b, c, d], axis=0), [4, 1, 8]), tf.bool)
    # # [4, 4, 8]
    # intersection = tf.cast(tf.logical_and(pc_matrix_1, pc_matrix_2), tf.float32)
    # intersection = tf.reduce_sum(intersection, axis=-1)

    # union = tf.cast(tf.logical_or(pc_matrix_1, pc_matrix_2), tf.float32)
    # union = tf.reduce_sum(union, axis=-1)

    # iou = intersection / union

    # pc_matrix_1 = tf.cast(pc_matrix_1, tf.int32)
    # points_num = tf.reduce_sum(pc_matrix_1, axis=-1)
    # points_num = tf.reshape(points_num, [4])
    # idx = tf.nn.top_k(points_num, 4).indices

    # points_sample = tf.reshape(pc_matrix_1, [4, 8])
    # points_sample_ordered = tf.gather(points_sample, idx)
    # iou_matrix_ordered = tf.gather(iou, idx)

    # keep_inds, nmsed_points_sample = points_nms(iou_matrix_ordered, points_sample_ordered, 0, 0.5)
    # sess = tf.Session()
    # print(sess.run(iou_matrix_ordered))
    # print('-----------------------')
    # print(sess.run(points_sample_ordered))
    # print('-----------------------')
    # print(sess.run(keep_inds))
    # print('-----------------------')
    # print(sess.run(nmsed_points_sample))
