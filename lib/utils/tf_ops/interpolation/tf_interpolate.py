import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
interpolate_module=tf.load_op_library(os.path.join(BASE_DIR, 'tf_interpolate_so.so'))
def three_nn(xyz1, xyz2):
    '''
    Input:
        xyz1: (b,n,3) float32 array, unknown points
        xyz2: (b,m,3) float32 array, known points
    Output:
        dist: (b,n,3) float32 array, distances to known points
        idx: (b,n,3) int32 array, indices to known points
    '''
    return interpolate_module.three_nn(xyz1, xyz2)
ops.NoGradient('ThreeNN')

def deformable_three_nn(xyz1, xyz2):
    '''
    dist: the smallest three distance for each point in xyz1 to xyz2
    idx: the index of the smallest distance in xyz2
    Input:
        xyz1: (b,n,3) float32 array, unknown points
        xyz2: (b,m,3) float32 array, known points
    Output:
        dist: (b,n,3) float32 array, distances to known points
        idx: (b,n,3) int32 array, indices to known points
    '''
    return interpolate_module.deformable_three_nn(xyz1, xyz2)

@tf.RegisterGradient('DeformableThreeNN')
def _deformable_three_nn_grad(op, grad_out, _):
    '''
    back propogation
    the gradients are passed to xyz1 only
    xyz1: [b, n, 3]
    grad_out: [b, n, 3]
    the first grad_out is to the distance
    and the second is to the idx, but we dont care idx to be honest
    '''
    xyz1 = op.inputs[0]
    xyz2 = op.inputs[1]

    dist = op.outputs[0]
    idx = op.outputs[1]
    print('OOOOOOOOOOOOOOOOOOOOPSSSSSSSSSSSSSS')
    print(grad_out)
    print(_)
    print('OOOOOOOOOOOOOOOOOOOOPSSSSSSSSSSSSSS')
    return [interpolate_module.deformable_three_nn_grad(xyz1, xyz2, idx, grad_out), None]


def deformable_three_interpolate(points, weight, idx):
    '''
    Input:
        points: (b,m,c) float32 array, known points
        weight: (b,n,3) float32 array, weights on known points
        idx: (b,n,3) int32 array, indices to known points
    Output:
        out: (b,n,c) float32 array, interpolated point values
    '''
    return interpolate_module.deformable_three_interpolate(points, weight, idx)
@tf.RegisterGradient('DeformableThreeInterpolate')
def _deformable_three_interpolate_grad(op, grad_out):
    # return the grad_points and the grad_weights
    points = op.inputs[0]
    weight = op.inputs[1]
    idx = op.inputs[2]
    point_grad, weight_grad = interpolate_module.deformable_three_interpolate_grad(points, weight, idx, grad_out)
    return [point_grad, weight_grad, None]

def three_interpolate(points, idx, weight):
    '''
    Input:
        points: (b,m,c) float32 array, known points
        idx: (b,n,3) int32 array, indices to known points
        weight: (b,n,3) float32 array, weights on known points
    Output:
        out: (b,n,c) float32 array, interpolated point values
    '''
    return interpolate_module.three_interpolate(points, idx, weight)
@tf.RegisterGradient('ThreeInterpolate')
def _three_interpolate_grad(op, grad_out):
    points = op.inputs[0]
    idx = op.inputs[1]
    weight = op.inputs[2]
    return [interpolate_module.three_interpolate_grad(points, idx, weight, grad_out), None, None]


def k_interpolate(points, idx, weight):
    '''
    Input:
        points: (b,m,c) float32 array, known points
        idx: (b,n,k) int32 array, indices to known points
        weight: (b,n,k) float32 array, weights on known points
    Output:
        out: (b,n,c) float32 array, interpolated point values
    '''
    return interpolate_module.k_interpolate(points, idx, weight)
@tf.RegisterGradient('KInterpolate')
def _k_interpolate_grad(op, grad_out):
    points = op.inputs[0]
    idx = op.inputs[1]
    weight = op.inputs[2]
    return [interpolate_module.k_interpolate_grad(points, idx, weight, grad_out), None, None]
