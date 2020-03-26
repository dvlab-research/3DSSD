import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
points_pooling_module=tf.load_op_library(os.path.join(BASE_DIR, 'tf_points_pooling_so.so'))

def points_pooling(pc, box_3d, pc_loc, l=7, h=7, w=7, sample_num=35):
    """
    param pc: the whole point cloud, [bs, proposal_num, pts_sample_num, c]
    param box_3d: [bs, proposal, 6]: x, y, z, l, h, w
    param pc_loc: [bs, proposal_num, pts_sample_num, 3], [xyz] location of each points
    return: 
      out_features: [bs, proposal_num, l, h, w, sample_num, c]
      out_idx: [bs, proposal_num, l, h, w, sample_num], index
      out_points_num: [bs, proposal_num, l, h, w]
      pillars: [bs, proposal_num, l, h, w, 3] # center of each pillar
    """
    return points_pooling_module.points_pooling(pc, box_3d, pc_loc, l, h, w, sample_num)
@tf.RegisterGradient('PointsPooling')
def _points_pooling_grad(op, features_grad, _1, _2, _3):
    # features_grad: [n, l, h, w, sampling_num, c]
    pc = op.inputs[0]
    out_idx = op.outputs[1]
    sampled_num_lists = op.outputs[2]

    return[points_pooling_module.points_pooling_grad(pc, out_idx, sampled_num_lists, features_grad), None, None, None]

def calc_anchors_pillar_center(proposals, l=7, h=7, w=7, anchor_offset=[0.5, 0.5, 0.5]):
    """
    calc the proposals' center location of each grid 

    proposals: [bs, 6]: [xmin, xmax, ymin, ymax, zmin, zmax]
    return:
        anchors_pillars: [batch_size, l, h, w, 3]
    """
    proposal_unstack = tf.unstack(proposals, axis=0)
    anchors_pillars = []
    anchor_offset = np.array(anchor_offset, dtype=np.float32)
    for proposal in proposal_unstack: 
        proposal = tf.reshape(proposal, [3, 2])
        xdim = proposal[0, 1] - proposal[0, 0]
        ydim = proposal[1, 1] - proposal[1, 0]
        zdim = proposal[2, 1] - proposal[2, 0]

        xmin, ymin, zmin = proposal[0, 0], proposal[1, 0], proposal[2, 0]

        xdim = xdim / l
        ydim = ydim / h
        zdim = zdim / w

        x, y, z = tf.meshgrid(tf.range(0, l, dtype=tf.float32), tf.range(0, h, dtype=tf.float32), tf.range(0, w, dtype=tf.float32), indexing='ij') # [l, h, w]
        x = xmin + (x + anchor_offset[0]) * xdim
        y = ymin + (y + anchor_offset[1]) * ydim
        z = zmin + (z + anchor_offset[2]) * zdim

        anchors_pillar = tf.stack([x, y, z], axis=-1)
        anchors_pillar = tf.reshape(anchors_pillar, [-1, 3]) # [l, h, w, 3]
        anchors_pillars.append(anchors_pillar)
    anchors_pillars = tf.stack(anchors_pillars, axis=0) # [bs, -1, 3]
    return anchors_pillars

 

if __name__ == '__main__':
    # pc = np.reshape(np.arange(10), [1, 10, 1])
    # pc = np.tile(pc, [2, 1, 100])

    # proposals_1 = [0, 1, 0, 2, 2, 2]
    # proposals_2 = [0.5, 0.85, 0.5, 3, 0.7, 0.7]
    # proposals = np.stack([proposals_1, proposals_2], axis=0)

    # pc_location = np.ones([2, 10, 3]) * -2.0
    # 
    # out_features, out_idx, out_points_num, points_pillars = points_pooling(pc, proposals, pc_location, l=4, h=4, w=4, sample_num=5)
    # sess = tf.Session()
    # op_feature, op_idx, op_num, op_pillars = sess.run([out_features, out_idx, out_points_num, points_pillars])
    # print(op_pillars.shape)
    # print(op_pillars)
    # print(op_num)
    # print(op_idx)
    # print(op_feature[0, 2, 2, 2])
    # print(op_feature.shape)

    a = np.array([[0, 6, 0, 6, 0, 6], [1, 6, 2, 7, 3, 8]])
    l=3
    h=5
    w=7
 
    anchors_pillars = calc_anchors_pillar_center(a, l, h, w)
    print(anchors_pillars[0])
