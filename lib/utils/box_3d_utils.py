import numpy as np
import tensorflow as tf


def object_label_to_box_3d(obj_label):
    """Turns an ObjectLabel into an box_3d

    Args:
        obj_label: ObjectLabel

    Returns:
        anchor: 3D box in box_3d format [x, y, z, l, w, h, ry]
    """
    box_3d = np.zeros(7)

    box_3d[0:3] = obj_label.t
    box_3d[3] = obj_label.l
    box_3d[4] = obj_label.h
    box_3d[5] = obj_label.w
    box_3d[6] = obj_label.ry

    return box_3d

############# Cast label_boxes_3d to anchors ###############
def box_3d_to_anchor(boxes_3d, ortho_rotate=False):
    """Converts a box_3d tensor to anchor format by ortho rotating it.
    This is similar to 'box_3d_to_anchor' above however it takes
    a tensor as input.

    Args:
        boxes_3d: N x 7 tensor of box_3d in the format [x, y, z, l, w, h, ry]

    Returns:
        anchors: N x 6 tensor of anchors in anchor form ->
            [x, y, z, dim_x, dim_y, dim_z]
    """
    if isinstance(boxes_3d, tf.Tensor):
        lib_name = tf
        concat = tf.concat
    else: 
        lib_name = np
        concat = np.concatenate
    x, y, z, l, h, w, ry = lib_name.split(boxes_3d, 7, axis=-1)

    # Ortho rotate
    if ortho_rotate:
        half_pi = np.pi / 2
        box_ry = lib_name.round(ry / half_pi) * half_pi
    else: box_ry = ry
    cos_ry = lib_name.abs(lib_name.cos(box_ry))
    sin_ry = lib_name.abs(lib_name.sin(box_ry))

    dimx = l * cos_ry + w * sin_ry
    dimy = h
    dimz = w * cos_ry + l * sin_ry

    anchors = concat([x,y,z,dimx,dimy,dimz], axis=-1)

    return anchors

############# Cast label_boxes_3d to corners###############
def get_box3d_corners_helper_np(centers, headings, sizes):
    ''' Input: (N, 3), (N, ), (N, 3), Output: [N, 8, 3]'''
    N = centers.shape[0]
    l = sizes[:, 0]
    h = sizes[:, 1]
    w = sizes[:, 2]

    z = np.zeros_like(l)
    x_corners = np.stack([l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2], axis=1) # (N,8)
    y_corners = np.stack([z,z,z,z,-h,-h,-h,-h], axis=1) # (N,8)
    z_corners = np.stack([w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2], axis=1) # (N,8)
    corners = np.concatenate([np.expand_dims(x_corners,1), np.expand_dims(y_corners,1), np.expand_dims(z_corners,1)], axis=1) # (N,3,8)
    #print x_corners, y_corners, z_corners
    c = np.cos(headings)
    s = np.sin(headings)
    ones = np.ones([N], dtype=np.float32)
    zeros = np.zeros([N], dtype=np.float32)
    row1 = np.stack([c,zeros,s], axis=1) # (N,3)
    row2 = np.stack([zeros,ones,zeros], axis=1)
    row3 = np.stack([-s,zeros,c], axis=1)
    R = np.concatenate([np.expand_dims(row1,1), np.expand_dims(row2,1), np.expand_dims(row3,1)], axis=1) # (N,3,3)
    #print row1, row2, row3, R, N
    corners_3d = np.matmul(R, corners) # (N,3,8)
    corners_3d += np.tile(np.expand_dims(centers,2), [1,1,8]) # (N,3,8)
    corners_3d = np.transpose(corners_3d, [0,2,1]) # (N,8,3)
    return corners_3d

def get_box3d_corners_helper(centers, headings, sizes):
    """ TF layer. Input: (N,3), (N,), (N,3), Output: (N,8,3) """
    #print '-----', centers
    N = tf.shape(centers)[0]
    l = tf.slice(sizes, [0,0], [-1,1]) # (N,1)
    h = tf.slice(sizes, [0,1], [-1,1]) # (N,1)
    w = tf.slice(sizes, [0,2], [-1,1]) # (N,1)
    z = tf.zeros_like(l)
    #print l,w,h
    x_corners = tf.concat([l/2,l/2,-l/2,-l/2,l/2,l/2,-l/2,-l/2], axis=1) # (N,8)
    y_corners = tf.concat([z,z,z,z,-h,-h,-h,-h], axis=1) # (N,8)
    z_corners = tf.concat([w/2,-w/2,-w/2,w/2,w/2,-w/2,-w/2,w/2], axis=1) # (N,8)
    corners = tf.concat([tf.expand_dims(x_corners,1), tf.expand_dims(y_corners,1), tf.expand_dims(z_corners,1)], axis=1) # (N,3,8)
    #print x_corners, y_corners, z_corners
    c = tf.cos(headings)
    s = tf.sin(headings)
    ones = tf.ones([N], dtype=tf.float32)
    zeros = tf.zeros([N], dtype=tf.float32)
    row1 = tf.stack([c,zeros,s], axis=1) # (N,3)
    row2 = tf.stack([zeros,ones,zeros], axis=1)
    row3 = tf.stack([-s,zeros,c], axis=1)
    R = tf.concat([tf.expand_dims(row1,1), tf.expand_dims(row2,1), tf.expand_dims(row3,1)], axis=1) # (N,3,3)
    #print row1, row2, row3, R, N
    corners_3d = tf.matmul(R, corners) # (N,3,8)
    corners_3d += tf.tile(tf.expand_dims(centers,2), [1,1,8]) # (N,3,8)
    corners_3d = tf.transpose(corners_3d, perm=[0,2,1]) # (N,8,3)
    return corners_3d


def transfer_box3d_to_corners(boxes_3d):
    """
    Transfer box_3d with any shape to corners
    boxes_3d: [..., 7]

    Return: [..., 8, 3]
    """
    boxes_3d_shape = tf.shape(boxes_3d)
    new_shape = tf.concat([boxes_3d_shape[:-1], [8, 3]], axis=0)
    boxes_3d_reshape = tf.reshape(boxes_3d, [-1, 7])
    corners = get_box3d_corners_helper(boxes_3d_reshape[:, :3], boxes_3d_reshape[:, -1], boxes_3d_reshape[:, 3:-1]) # [-1, 8, 3]
    corners = tf.reshape(corners, new_shape)
    return corners 
