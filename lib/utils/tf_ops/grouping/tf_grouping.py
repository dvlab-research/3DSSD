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

def calculate_points_iou(batch_points, batch_anchors_corners, batch_label_corners):
    """ Calculate the points_iou between anchors and labels
    Args:
        batch_points: [bs, points_num, 3] 
        batch_anchors_corners: [bs, anchors_num, 8, 3]
        batch_label_corners: [bs, gt_num, 8, 3]
    Return:
        points_iou: [bs, anchors_num, gt_num]
    """
    raise Exception('Not Implementation Error!!!')
    return grouping_module.calculate_points_iou(batch_points, batch_anchors_corners, batch_label_corners)
ops.NoGradient('CalculatePointsIou')

def query_corners_point(batch_points, batch_anchors_corners):
    """ Calculate whether the points inside the 
        Args:
            batch_points: [b, n, 3]
            batch_anchors_corners: [b, num_proposals, 8, 3] 
        Return:
            result_out: [b, num_proposals, n], whether inside the corners or not
    """
    return grouping_module.query_corners_point(batch_points, batch_anchors_corners)
ops.NoGradient('QueryCornersPoint')

def query_cube_point(radius, nsample, subcube_num, xyz1, xyz2):
    '''
    Input: 
        radius: float32, half length of cube
        nsample: int32, number of points selected in each cube region
        subcube_num: int32, number of subcube per cube
        xyz1: (batch_size, ndataset, 3), float32, input points
        xyz2: (batch_size, npoint, 3), float32, query points
    Output:
        idx: (batch_size, npoint, nsample)
        pts_cnt: (batch_size, npoint, subcube_num), pts_cnt per sub_cube num
        subcube_location: (batch_size, npoint, subcube_num, 3), center
    '''
    return grouping_module.query_cube_point(xyz1, xyz2, radius, nsample, subcube_num)

ops.NoGradient('QueryCubePoint')

# query points with dynamic shape
def query_ball_point_dynamic_shape(nsample, xyz1, xyz2, radius):
    '''
    Input:
        nsample: int32, number of points selected in each region
        xyz1: (batch_size, ndataset, 3) float32 array, input points
        xyz2: (batch_size, npoint, 3) float32 array, query points
        radius: (batch_size, npoint, split_bin_num) float32 array, the length of each split bin of xyz2
    Output:
        idx: (batch_size, npoint, nsample) int32 array, indices to input points
        pts_cnt: (batch_size, npoint) int32 array, number of unique points in each local region
    '''
    return grouping_module.query_ball_point_dynamic_shape(xyz1, xyz2, radius, nsample)

ops.NoGradient('QueryBallPointDynamicShape')

# query their dynamic radius
def query_dynamic_radius_for_points(xyz1, radius):
    '''
    Query the dynamic radius for points in xyz1
    Input:
        xyz1: (batch_size, npoint, nsample, 3) float32 array, input points
        radius: (batch_size, npoint, split_bin_num) float32 array, the length of each split bin of xyz2
    Output:
        radius_idx: (batch_size, npoint, nsample, 2) radius idx
        radius_rate: (batch_size, npoint, nsample, 2) radius_rate
    '''
    return grouping_module.query_dynamic_radius_for_points(xyz1, radius)
ops.NoGradient('QueryDynamicRadiusForPoints')

# query the distance among each angle
def query_target_distance_for_points(split_bin_num, xyz1, gt_boxes_3d):
    '''
    Query the distance from xyz1 to its assigned gt_boxes_3d
    Note that, xyz1 has to be normalized by the angle from gt_boxes_3d
    Input:
        xyz1: [bs, npoint, 3]
        gt_boxes_3d: [bs, npoint, 7]
    Return:
        target_dist: [bs, npoint, split_bin_num]
    '''
    return grouping_module.query_target_distance_for_points(xyz1, gt_boxes_3d, split_bin_num)
ops.NoGradient('QueryTargetDistanceForPoints')


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


def query_ball_point_dynamic_radius(nsample, xyz1, xyz2, radius):
    '''
    Input:
        nsample: int32, number of points selected in each ball region
        xyz1: (batch_size, ndataset, 3) float32 array, input points
        xyz2: (batch_size, npoint, 3) float32 array, query points
        radius: (batch_size, npoint) float32 array, different group_radius for different xyz2
    Output:
        idx: (batch_size, npoint, nsample) int32 array, indices to input points
        pts_cnt: (batch_size, npoint) int32 array, number of unique points in each local region
    '''
    return grouping_module.query_ball_point_dynamic_radius(xyz1, xyz2, radius, nsample)

ops.NoGradient('QueryBallPointDynamicRadius')

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

if __name__ == '__main__':
    # test whether can calc the points inside 
    xyz1 = np.array([[0, 0, 0], [0.2, 0.3, 0.34], [-0.2, 0.3, 0.34], [-0.34, -0.3, -0.2], [0.9, 0.9, 0.9], [0.8, 0.3, -0.00001]], dtype=np.float32)
    xyz2 = np.array([[0, 0, 0]], dtype=np.float32)
    xyz1 = tf.reshape(xyz1, [1, 6, 3])
    xyz2 = tf.reshape(xyz2, [1, 1, 3])

    bin_split_num = 32
    radius = np.ones([bin_split_num], dtype=np.float32)
    radius = tf.cast(tf.reshape(radius, [1, 1, bin_split_num]), tf.float32)
    idx, pts_cnt, radius_idx, radius_rate = query_ball_point_dynamic_shape(10, xyz1, xyz2, radius)
    sess = tf.Session()
    idx_op, pts_cnt_op, radius_idx_op, radius_rate_op = sess.run([idx, pts_cnt, radius_idx,radius_rate])
    print(idx_op, idx_op.shape)
    print(pts_cnt_op, pts_cnt_op.shape)
    print(radius_idx_op, radius_idx_op.shape)
    print(radius_rate_op, radius_rate_op.shape)

    grouped_xyz = group_point(xyz1, idx)
    radius_idx, radius_rate = query_dynamic_radius_for_points(grouped_xyz, radius)
    grouped_xyz_op, radius_idx_op, radius_rate_op = sess.run([grouped_xyz, radius_idx, radius_rate])
    print(grouped_xyz_op, grouped_xyz_op.shape)
    print(radius_idx_op, radius_idx_op.shape)
    print(radius_rate_op, radius_rate_op.shape)
