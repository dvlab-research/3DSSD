'''
filter the points method
'''

import numpy as np
import tensorflow as tf

def get_point_filter(point_cloud, extents):
    """
    Creates a point filter using the 3D extents and ground plane

    :param point_cloud: Point cloud in the form [N, 3](x, y, z)
    :param extents: 3D area in the form
        [[min_x, max_x], [min_y, max_y], [min_z, max_z]]
    :param ground_plane: Optional, coefficients of the ground plane
        (a, b, c, d)
    :param offset_dist: If ground_plane is provided, removes points above
        this offset from the ground_plane
    :return: A binary mask for points within the extents and offset plane
    """
    point_cloud = np.array(point_cloud)

    x_extents = extents[0]
    y_extents = extents[1]
    z_extents = extents[2]

    extents_filter = (point_cloud[:, 0] > x_extents[0]) & \
                     (point_cloud[:, 0] < x_extents[1]) & \
                     (point_cloud[:, 1] > y_extents[0]) & \
                     (point_cloud[:, 1] < y_extents[1]) & \
                     (point_cloud[:, 2] > z_extents[0]) & \
                     (point_cloud[:, 2] < z_extents[1])

    point_filter = extents_filter

    return point_filter


def get_point_filter_in_image(point_cloud, calib, height, width):
    ''' get point filter in image '''
    # first we exchange the point_cloud to image coord
    img_coord = calib.project_rect_to_image(point_cloud)
    # [:, 0] x-coord; [:, 1] y-coord
    point_filter = ((img_coord[:, 0] >= 0) & \
                    (img_coord[:, 0] < width) & \
                    (img_coord[:, 1] >= 0) & \
                    (img_coord[:, 1] < height))
    # and also filter these z < 0
    z_filter = point_cloud[:, 2] >= 0
    point_filter = np.logical_and(point_filter, z_filter)
    return point_filter
        
