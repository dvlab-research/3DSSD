import numpy as np
import tensorflow as tf

def rotate_points(points, rys):
    """
    Rotate rys
    points: [..., n, 3]
    rys: [...]
    """
    if isinstance(points, tf.Tensor):
        lib_name = tf
    else:
        lib_name = np
    # transpose points from [..., n, 3]->[..., 3, n]
    shape_length = len(rys.get_shape().as_list())
    transpose_vector = list(np.arange(shape_length))
    transpose_vector = transpose_vector + [shape_length+1, shape_length]

    c = lib_name.cos(rys) # [...]
    s = lib_name.sin(rys)

    points = lib_name.transpose(points, transpose_vector) # [..., 3, n]
    ones = lib_name.ones_like(c)
    zeros = lib_name.zeros_like(c)
    row1 = lib_name.stack([c,zeros,s], axis=-1) # [...,3]
    row2 = lib_name.stack([zeros,ones,zeros], axis=-1)
    row3 = lib_name.stack([-s,zeros,c], axis=-1)
    R = lib_name.stack([row1, row2, row3], axis=-2) # (...,3,3)
    canonical_points = lib_name.matmul(R, points) # [..., 3, n]
    canonical_points = lib_name.transpose(canonical_points, transpose_vector) # [b, n, 3]
    return canonical_points

def symmetric_rotate_points(points, rys):
    """
    flip original points from left to right
    First rotate points by -rys, then translate points by negative z and finally rotate by rys
    points: [b, n, 3]
    rys: [b]
    """
    if isinstance(points, tf.Tensor):
        lib_name = tf
        b = tf.shape(points)[0]
    else:
        lib_name = np
        b = points.shape[0]

    c = lib_name.cos(rys)
    s = lib_name.sin(rys)
    c_2 = lib_name.square(c)
    s_2 = lib_name.square(s)

    points = lib_name.transpose(points, [0, 2, 1]) # [b, 3, n]
    ones = lib_name.ones([b], dtype=np.float32)
    zeros = lib_name.zeros([b], dtype=np.float32)
    row1 = lib_name.stack([c_2 - s_2,zeros,-2*c*s], axis=1) # (b,3)
    row2 = lib_name.stack([zeros,ones,zeros], axis=1)
    row3 = lib_name.stack([-2*c*s,zeros,s_2 - c_2], axis=1)
    R = lib_name.stack([row1, row2, row3], axis=1) # (N,3,3)
    canonical_points = lib_name.matmul(R, points) # [b, 3, n]
    canonical_points = lib_name.transpose(canonical_points, [0, 2, 1]) # [b, n, 3]
    return canonical_points


# some rotation matrix
def rotx(t):
    ''' 3D Rotation about the x-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[1,  0,  0],
                     [0,  c, -s],
                     [0,  s,  c]])


def roty(t):
    ''' Rotation about the y-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c,  0,  s],
                     [0,  1,  0],
                     [-s, 0,  c]])


def inv_roty(t):
    ''' Inverse matrix of the y-axis rotation matrix '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, 0, -s, 0],
                     [0, 1, 0, 0],
                     [s, 0, c, 0],
                     [0, 0, 0, 1]])


def rotz(t):
    ''' Rotation about the z-axis. '''
    c = np.cos(t)
    s = np.sin(t)
    return np.array([[c, -s,  0],
                     [s,  c,  0],
                     [0,  0,  1]])
