import copy

import numpy as np
from utils.rotation_util import *


def flip_image(image):
    """Flips an image horizontally
    """
    flipped_image = np.fliplr(image)
    return flipped_image


def flip_points(points):
    """Flips a list of points (N, 3)
    """
    flipped_points = np.copy(points)
    flipped_points[:, 0] = -points[:, 0]
    return flipped_points


def flip_label_in_3d_only(obj_label):
    """Flips only the 3D position of an object label. The 2D bounding box is
    not flipped to save time since it is not used.

    Args:
        obj_label: ObjectLabel

    Returns:
        A flipped object
    """

    flipped_label = copy.deepcopy(obj_label)

    # Flip the rotation
    if obj_label.ry >= 0:
        flipped_label.ry = np.pi - obj_label.ry
    else:
        flipped_label.ry = -np.pi - obj_label.ry

    # Flip the t.x sign, t.y and t.z remains the unchanged
    flipped_t = (-flipped_label.t[0], flipped_label.t[1], flipped_label.t[2])
    flipped_label.t = flipped_t

    return flipped_label


def flip_boxes_3d(boxes_3d, flip_ry=True):
    """Flips boxes_3d

    Args:
        boxes_3d: List of boxes in box_3d format
        flip_ry bool: (optional) if False, rotation is not flipped to save on
            computation (useful for flipping anchors)

    Returns:
        flipped_boxes_3d: Flipped boxes in box_3d format
    """

    flipped_boxes_3d = np.copy(boxes_3d)

    if flip_ry:
        # Flip the rotation
        above_zero = boxes_3d[:, 6] >= 0
        below_zero = np.logical_not(above_zero)
        flipped_boxes_3d[above_zero, 6] = np.pi - boxes_3d[above_zero, 6]
        flipped_boxes_3d[below_zero, 6] = -np.pi - boxes_3d[below_zero, 6]

    # Flip the t.x sign, t.y and t.z remains the unchanged
    flipped_boxes_3d[:, 0] = -boxes_3d[:, 0]

    return flipped_boxes_3d


def flip_ground_plane(ground_plane):
    """Flips the ground plane by negating the x coefficient
        (ax + by + cz + d = 0)

    Args:
        ground_plane: ground plane coefficients

    Returns:
        Flipped ground plane coefficients
    """
    flipped_ground_plane = np.copy(ground_plane)
    flipped_ground_plane[0] = -ground_plane[0]
    return flipped_ground_plane


def flip_stereo_calib_p2(calib_p2, image_shape):
    """Flips the stereo calibration matrix to correct the projection back to
    image space. Flipping the image can be seen as a movement of both the
    camera plane, and the camera itself. To account for this, the instrinsic
    matrix x0 value is flipped with respect to the image width, and the
    extrinsic matrix t1 value is negated.

    Args:
        calib_p2: 3 x 4 stereo camera calibration matrix
        image_shape: (h, w) image shape

    Returns:
        'Flipped' calibration p2 matrix with shape (3, 4)
    """
    flipped_p2 = np.copy(calib_p2)
    flipped_p2[0, 2] = image_shape[1] - calib_p2[0, 2]
    flipped_p2[0, 3] = -calib_p2[0, 3]

    return flipped_p2


def scale_multi_stereo_calib_p2(calib_p2, scale):
    """ Scale the calib_p2 in order to let same point projected to same pixel in image

    Args:
        calib_p2: [3, 4] matrix
        scale: the enlarge scale
    """
    inv_scale = 1 / scale
    calib_p2_1 = calib_p2[:, :3]
    calib_p2_2 = calib_p2[:, 3]
    calib_p2_1 *= inv_scale
    calib_p2 = np.concatenate([calib_p2_1, calib_p2_2[:, np.newaxis]], axis=-1)
    return calib_p2


def rotate_stereo_calib_p2(calib_p2, angle):
    """ rotate the calib_p2 in order to let same point projected to same pixel

    Args:
        calib_p2: [3, 4] matrix
        angle: rotate angle
    """
    inv_roty = inv_roty(angle) # [4, 4]
    calib_p2 = np.matmul(calib_p2, inv_roty)
    return calib_p2


