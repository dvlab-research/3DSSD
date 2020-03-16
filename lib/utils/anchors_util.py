import tensorflow as tf
import numpy as np

import utils.kitti_util as kitti_util
from utils.box_3d_utils import *
from core.config import cfg
import utils.anchor_encoder as anchor_encoder


############ Project label_anchors to bev_anchors ##############
def project_to_bev(anchors):
    """
    Projects an array of 3D anchors into bird's eye view

    Args:
        anchors: list of anchors in anchor format (N x 6):
            N x [x, y, z, dim_x, dim_y, dim_z],
            can be a numpy array or tensor
        bev_extents: xz extents of the 3d area
            [[min_x, max_x], [min_z, max_z]]

    Returns:
          box_corners_norm: corners as a percentage of the map size, in the
            format N x [x1, y1, x2, y2]. Origin is the top left corner
    """
    tensor_format = isinstance(anchors, tf.Tensor)
    if not tensor_format:
        lib_name = np
        anchors = np.array(anchors)
        concat = np.concatenate
    else: # tensor_format
        lib_name = tf
        concat = tf.concat
   
    # we project the anchors to the bev maps
    x, y, z, l, h, w = lib_name.split(anchors, 6, axis=-1)
    half_dim_x = l / 2.0
    half_dim_z = w / 2.0
    
    # first get the x min and x max
    x_min = x - half_dim_x
    x_max = x + half_dim_x
    
    z_min = z - half_dim_z
    z_max = z + half_dim_z
    # thus now, we get the project anchor
    bev_anchors = concat([x_min, z_min, x_max, z_max], axis=-1)

    return bev_anchors



############ Project to image space ##############
def project_to_image_space_corners(anchors_corners, stereo_calib_p2, img_shape=(375, 1242)):
    """
    Projects 3D anchors into image space

    Args:
        anchors: list of anchors in anchor format N x 8 x 3 
        stereo_calib_p2: stereo camera calibration p2 matrix 3x4
        image_shape: dimensions of the image [h, w]

    Returns:
        box_corners: corners in image space - N x [x1, y1, x2, y2]
        box_corners_norm: corners as a percentage of the image size -
            N x [x1, y1, x2, y2]
    """
    if anchors_corners.shape[1] != 8:
        raise ValueError("Invalid shape for anchors {}, should be "
                         "(N, 8, 3)".format(anchors.shape[1]))

    # Apply the 2D image plane transformation
    anchors_corners = np.reshape(anchors_corners, [-1, 3])
    pts_2d = kitti_util.project_to_image(anchors_corners, stereo_calib_p2) # [-1, 2]

    pts_2d = np.reshape(pts_2d, [-1, 8, 2]) 

    h, w = img_shape

    # Get the min and maxes of image coordinates
    i_axis_min_points = np.minimum(np.maximum(np.amin(pts_2d[:, :, 0], axis=1), 0), w)
    j_axis_min_points = np.minimum(np.maximum(np.amin(pts_2d[:, :, 1], axis=1), 0), h)

    i_axis_max_points = np.minimum(np.maximum(np.amax(pts_2d[:, :, 0], axis=1), 0), w)
    j_axis_max_points = np.minimum(np.maximum(np.amax(pts_2d[:, :, 1], axis=1), 0), h)

    box_corners = np.stack([i_axis_min_points, j_axis_min_points,
                            i_axis_max_points, j_axis_max_points], axis=-1)


    return np.array(box_corners, dtype=np.float32)


def tf_project_to_image_space_corners(anchors_corners, stereo_calib_p2, img_shape=(375, 1242)):
    """
    Projects 3D anchors into image space

    Args:
        anchors: list of anchors in anchor format N x 8 x 3 
        stereo_calib_p2: stereo camera calibration p2 matrix
        image_shape: dimensions of the image [h, w]

    Returns:
        box_corners: corners in image space - N x [x1, y1, x2, y2]
        box_corners_norm: corners as a percentage of the image size -
            N x [x1, y1, x2, y2]
    """
    if anchors_corners.get_shape().as_list()[1] != 8:
        raise ValueError("Invalid shape for anchors {}, should be "
                         "(N, 8, 3)".format(anchors.shape[1]))

    # Apply the 2D image plane transformation
    anchors_corners = tf.reshape(anchors_corners, [-1, 3])

    # [-1, 2]
    pts_2d = kitti_util.tf_project_to_image_tensor(anchors_corners, stereo_calib_p2)

    pts_2d = tf.reshape(pts_2d, [-1, 8, 2]) 

    h, w = img_shape

    # Get the min and maxes of image coordinates
    i_axis_min_points = tf.minimum(tf.maximum(tf.reduce_min(pts_2d[:, :, 0], axis=1), 0), w)
    j_axis_min_points = tf.minimum(tf.maximum(tf.reduce_min(pts_2d[:, :, 1], axis=1), 0), h)

    i_axis_max_points = tf.minimum(tf.maximum(tf.reduce_max(pts_2d[:, :, 0], axis=1), 0), w)
    j_axis_max_points = tf.minimum(tf.maximum(tf.reduce_max(pts_2d[:, :, 1], axis=1), 0), h)

    box_corners = tf.stack([i_axis_min_points, j_axis_min_points,
                            i_axis_max_points, j_axis_max_points], axis=-1)

    return tf.cast(box_corners, tf.float32)



def reorder_projected_boxes(box_corners):
    """Helper function to reorder image corners.

    This reorders the corners from [x1, y1, x2, y2] to
    [y1, x1, y2, x2] which is required by the tf.crop_and_resize op.

    Args:
        box_corners: tensor image corners in the format
            N x [x1, y1, x2, y2]

    Returns:
        box_corners_reordered: tensor image corners in the format
            N x [y1, x1, y2, x2]
    """
    boxes_reordered = tf.stack([box_corners[:, 1],
                                box_corners[:, 0],
                                box_corners[:, 3],
                                box_corners[:, 2]],
                               axis=1)
    return boxes_reordered


