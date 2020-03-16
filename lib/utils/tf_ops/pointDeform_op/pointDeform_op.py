"""
Author: Jiang Mingyang
email: jmydurant@sjtu.edu.cn
pointSIFT module op, do not modify it !!!
"""

import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os
import numpy as np
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

pointDeform_module = tf.load_op_library(os.path.join(BASE_DIR, 'tf_pointDeform_so.so'))

def pointDeform_select(xyz, radius):
    """
    find the nearest num points
    :param xyz: (b, n, 3) float
    :param radius: float
    :param num: the chosen num
    :return: (b, n, num) int
    """
    idx = pointDeform_module.nearest_select(xyz, radius)
    return idx
ops.NoGradient('NearestSelect')

def add_offset(offset, group_xyz, idx):
    """
    offset: [batch_size, points_num, 3 * 8]
    group_xyz: [batch_size, points_num, 8, 3]
    idx: [batch_size, points_num, 8]
    return:
      group_xyz_out: [batch_size, points_num, 8, 3]
    """
    return pointDeform_module.add_offset(offset, group_xyz, idx)
@tf.RegisterGradient('AddOffset')
def _add_offset_grad(op, grad_out):
    # return offset_grad: [batch_size, points_num, 3*8]
    offset = op.inputs[0]
    group_xyz = op.inputs[1]
    idx = op.inputs[2]

    return [pointDeform_module.add_offset_grad(offset, group_xyz, idx, grad_out), None, None]


if __name__ == '__main__':
    # the testing code
    points = np.array([[0, 0, 0], [0, 0, 1], [0, 1, 0], [1, 0, 0], [1, 0, 1], [1, 1, 0], [0, 1, 1], [1, 1, 1], [0.5, 0.5, 0.5]]) 
    points = np.reshape(points, [1, -1, 3])
    print(points.shape)

    idx = pointDeform_select(points, 1)
    print(idx.shape)
    sess = tf.Session()
    print(sess.run(idx))

