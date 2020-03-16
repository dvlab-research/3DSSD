import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os
import numpy as np

# used for debugging
import utils.kitti_object as kitti_object
from utils.points_filter import get_point_filter_in_image, get_point_filter
import cv2

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
points_fuse_module=tf.load_op_library(os.path.join(BASE_DIR, 'tf_points_fuse_so.so'))

def points_fuse(points, img_feature, calib, down_sample_rate):
    """ A module that generates the 3D points' 2D feature
        param points: [batch_size, -1, 3]
        param img_feature: [batch_size, h, w, c]
        param calib: [batch_size, 3, 4]
        param down_sample_rate: int, down sample rate from original img feature

        Return:
            2D points location, [batch_size, -1, 2]
            2D points feature, [batch_size, -1, c]
    """
    return points_fuse_module.points_fuse(points, img_feature, calib, down_sample_rate)
@tf.RegisterGradient('PointsFuse')
def _points_fuse_grad(op, location_grad, feature_grad):
    """ A module generates the grad
        location_grad: [batch_size, -1, 2]
        feature_grad: [batch_size, -1, c]
    """
    img_feature = op.inputs[1] # [batch_size, h, w, c]
    down_sample_rate = op.get_attr("down_sample_rate")
    img_pc = op.outputs[0] # [batch_size, -1, 2]
    
    return [None, points_fuse_module.points_fuse_grad(img_feature, img_pc, feature_grad, down_sample_rate), None]

if __name__ == '__main__':
    # Debugging
    training_object = kitti_object.kitti_object('/cephfs/group/youtu/person/tomztyang/3D_detection/3D_ssd/dataset/KITTI/object') 

    lidar_point = training_object.get_lidar(0)
    img = training_object.get_image(0)
    calib = training_object.get_calibration(0)
    lidar_point = lidar_point[:, :3]

    # first cast these points outside img
    rect_point = calib.project_velo_to_rect(lidar_point)
    img_point = calib.project_rect_to_image(rect_point)

    img_points_filter = get_point_filter_in_image(rect_point, calib, img.shape[0], img.shape[1])
    rect_point = rect_point[img_points_filter]
    img_point = img_point[img_points_filter]

    dbg_rect_point = np.expand_dims(rect_point, axis=0)
    dbg_img_feature = cv2.resize(img, None, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_LINEAR) 
    dbg_img_feature = np.expand_dims(dbg_img_feature, axis=0).astype(np.float32)
    calib_p2 = calib.P
    calib_p2 = np.expand_dims(calib_p2, axis=0)
    tf_location, tf_feature = points_fuse(dbg_rect_point, dbg_img_feature, calib_p2, 4.)

    sess = tf.Session()
    dbg_location, dbg_feature = sess.run([tf_location, tf_feature])

    print(img_point / 4.)
    print('----')
    print(dbg_location)
    print('----')
    print(dbg_img_feature[0, 90, 152, :], dbg_img_feature[0, 90, 153, :], dbg_img_feature[0, 91, 152, :], dbg_img_feature[0, 91, 153, :])
    print('----')
    print(dbg_img_feature.shape)
    print(dbg_location[0, -1, :])
    print(dbg_feature[0, -1, :])
