import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os

# debugging
import numpy as np
from core.config import cfg

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
grouping_module=tf.load_op_library(os.path.join(BASE_DIR, 'tf_grouping_so.so'))


def query_boxes_3d_mask(xyz, boxes_3d):
    """ Calculate whether the points inside the box_3d 
    Input
        xyz: [b, n, 3]
        boxes_3d: [b, num_proposals, 7] 
    Return:
        mask: [b, num_proposals, n], whether inside the corners or not
    """
    return grouping_module.query_boxes3d_mask(xyz, boxes_3d)
ops.NoGradient('QueryBoxes3dMask')

def query_points_iou(xyz, anchors_3d, gt_boxes_3d, iou_matrix):
    """ Calculate the PointsIoU between anchors_3d and gt_boxes_3d
    Input
        xyz: [b, n, 3]
        anchors_3d: [b, anchors_num, 7]
        gt_boxes_3d: [b, gt_num, 7]
        iou_matrix: [b, anchors_num, gt_num]
    Return:
        iou_points: [b, anchors_num, gt_num]
    """
    return grouping_module.query_points_iou(xyz, anchors_3d, gt_boxes_3d, iou_matrix)
ops.NoGradient('QueryPointsIou')

def query_boxes_3d_points(nsample, xyz, proposals):
    """
    Input:
        nsample: int32, number of points selected in each boxes
        xyz: [bs, pts_num, 3]
        proposals: [bs, proposal_num, 7]
    Return:
        idx: [bs, proposal_num, nsample]
        pts_cnt: [bs, proposal_num]
    """
    return grouping_module.query_boxes3d_points(xyz, proposals, nsample)
ops.NoGradient('QueryBoxes3dPoints')


def query_ball_point(radius, nsample, xyz1, xyz2):
    '''
    Input:
        radius: float32, ball search radius
        nsample: int32, number of points selected in each ball region
        xyz1: (batch_size, ndataset, 3) float32 array, input points
        xyz2: (batch_size, npoint, 3) float32 array, query points
    Output:
        idx: (batch_size, npoint, nsample) int32 array, indices to input points
        pts_cnt: (batch_size, npoint) int32 array, number of unique points in each local region
    '''
    return grouping_module.query_ball_point(xyz1, xyz2, radius, nsample)

ops.NoGradient('QueryBallPoint')

def query_ball_point_dilated(min_radius, max_radius, nsample, xyz1, xyz2):
    """
    dilated_ball_query: dilated pointnet++
    Input:
        min_radius: float32, ball search min radius
        max_radius: float32, ball search max radius
        nsample: int32, number of points selected in each ball region
        xyz1: (batch_size, ndataset, 3) float32 array, input points
        xyz2: (batch_size, npoint, 3) float32 array, query points
    Output:
        idx: (batch_size, npoint, nsample) int32 array, indices to input points
        pts_cnt: (batch_size, npoint) int32 array, number of unique points in each local region
    """
    return grouping_module.query_ball_point_dilated(xyz1, xyz2, min_radius, max_radius, nsample)

ops.NoGradient('QueryBallPointDilated')

def query_ball_point_withidx(radius, nsample, xyz1, xyz2, sort_idx):
    """
    IdXBallQuery Operation
    Input: 
        radius: float32, ball query radius
        nsample: int32, number of points selected in each ball region
        xyz1: (batch_size, ndataset, 3) float32 array, input points
        xyz2: (batch_size, npoint, 3) float32 array, query points
        idx: (batch_size, npoint, ndataset), the argsort from self attention
    Output:
        idx: (batch_size, npoint, nsample) int32 array, indices to input points
        pts_cnt: (batch_size, npoint) int32 array, number of unique points in each local region
    """
    return grouping_module.query_ball_point_withidx(xyz1, xyz2, sort_idx, radius, nsample)

ops.NoGradient('QueryBallPointWithidx')


def select_top_k(k, dist):
    '''
    Input:
        k: int32, number of k SMALLEST elements selected
        dist: (b,m,n) float32 array, distance matrix, m query points, n dataset points
    Output:
        idx: (b,m,n) int32 array, first k in n are indices to the top k
        dist_out: (b,m,n) float32 array, first k in n are the top k
    '''
    return grouping_module.selection_sort(dist, k)
ops.NoGradient('SelectionSort')
def group_point(points, idx):
    '''
    Input:
        points: (batch_size, ndataset, channel) float32 array, points to sample from
        idx: (batch_size, npoint, nsample) int32 array, indices to points
    Output:
        out: (batch_size, npoint, nsample, channel) float32 array, values sampled from points
    '''
    return grouping_module.group_point(points, idx)
@tf.RegisterGradient('GroupPoint')

def _group_point_grad(op, grad_out):
    points = op.inputs[0]
    idx = op.inputs[1]
    return [grouping_module.group_point_grad(points, idx, grad_out), None]

def knn_point(k, xyz1, xyz2):
    '''
    Input:
        k: int32, number of k in k-nn search
        xyz1: (batch_size, ndataset, c) float32 array, input points
        xyz2: (batch_size, npoint, c) float32 array, query points
    Output:
        val: (batch_size, npoint, k) float32 array, L2 distances
        idx: (batch_size, npoint, k) int32 array, indices to input points
    '''
    b = tf.shape(xyz1)[0]
    n = tf.shape(xyz1)[1]
    c = tf.shape(xyz1)[2]
    m = tf.shape(xyz2)[1]
    b, n, c = xyz1.get_shape().as_list()
    _, m, _ = xyz2.get_shape().as_list()

    xyz1 = tf.tile(tf.reshape(xyz1, (b,1,n,c)), [1,m,1,1])
    xyz2 = tf.tile(tf.reshape(xyz2, (b,m,1,c)), [1,1,n,1])
    dist = tf.reduce_sum((xyz1-xyz2)**2, -1)

    outi, out = select_top_k(k, dist)
    idx = tf.slice(outi, [0,0,0], [-1,-1,k])
    val = tf.slice(out, [0,0,0], [-1,-1,k])

    #val, idx = tf.nn.top_k(-dist, k=k) # ONLY SUPPORT CPU
    return val, idx

