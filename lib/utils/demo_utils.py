import os, sys
import tensorflow as tf
import numpy as np
import cv2

import utils.anchors_util as anchors_util
    

def show_corners(points, calib, corners_anchors, ctr_points=None, sv_img_path=None):
    """
    Input:
        points: [N, 3]
        calib: a calib object
        anchors: [N, 8, 3]
    """
    if 'mlab' not in sys.modules: import mayavi.mlab as mlab
    from viz_util import draw_lidar_simple, draw_lidar, draw_gt_boxes3d

    # first project anchors back to the velo location
    points = calib.project_rect_to_velo(points)

    # first project it back to velo
    corners_anchors = np.reshape(corners_anchors, [-1, 3])
    corners_anchors = calib.project_rect_to_velo(corners_anchors)
    corners_anchors = np.reshape(corners_anchors, [-1, 8, 3])

    # now, we get the corners_anchors, then, we only have to draw them
    fig = mlab.figure(figure=None, bgcolor=(0,0,0),
        fgcolor=None, engine=None, size=(1000, 500))
    draw_lidar(points, fig=fig)
    if ctr_points is not None:
        draw_lidar(ctr_points, fig=fig, pts_scale=0.10, pts_color=(1.0, 0.0, 0.0))
    draw_gt_boxes3d(corners_anchors, fig=fig) 
    if sv_img_path is not None:
        mlab.savefig(sv_img_path, figure=fig)
    else:
        mlab.show(1)
