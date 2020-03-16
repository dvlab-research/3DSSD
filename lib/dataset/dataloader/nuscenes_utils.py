import sys, os
import numpy as np
import itertools
import pickle

from pyquaternion import Quaternion
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box

# Casting label format from NuScenes Format to KITTI format
def cast_points_to_kitti(points):
    """
    cast points label to kitti format
    points: [-1, n]
    """
    points_xyz = points[:, :3]
    # cast points_xyz to kitti format
    points_xyz = points_xyz[:, [0, 2, 1]] # lhw
    points_xyz[:, 1] = -points_xyz[:, 1]
    points[:, :3] = points_xyz
    return points

def cast_box_3d_to_kitti_format(cur_boxes):
    """
    box_3d: [-1, 7], cast box_3d to kitti_format
    """
    # then cast boxes and velocity to kitti format
    cur_boxes_size = cur_boxes[:, 3:-1]
    cur_boxes_size = cur_boxes_size[:, [1, 2, 0]]
    cur_boxes_center = cur_boxes[:, :3]
    cur_boxes_center = cur_boxes_center[:, [0, 2, 1]] # lhw
    cur_boxes_center[:, 1] = -cur_boxes_center[:, 1]
    cur_boxes_center[:, 1] += cur_boxes_size[:, 1] / 2.
    cur_boxes = np.concatenate([cur_boxes_center, cur_boxes_size, cur_boxes[:, -1][:, np.newaxis]], axis=-1)
    return cur_boxes


# Casting prediction format from KITTI Format to NuScenes format
def cast_kitti_format_to_nusc_box_3d(cur_boxes, cur_score, cur_class, cur_attribute=None, cur_velocity=None, classes=None):
    """
    cur_boxes: [-1, 7], kitti format box
    cur_score: [-1]
    cur_class: [-1]
    cur_velocity: [-1, 3]
    """
    cur_boxes_ctr = cur_boxes[:, :3]
    cur_boxes_size = cur_boxes[:, 3:-1]
    cur_boxes_angle = cur_boxes[:, -1]

    cur_boxes_ctr[:, 1] -= cur_boxes_size[:, 1] / 2.
    cur_boxes_ctr[:, 1] = -cur_boxes_ctr[:, 1] # l, h, w
    cur_boxes_ctr = cur_boxes_ctr[:, [0, 2, 1]] # lwh

    # from lhw to wlh
    cur_boxes_size = cur_boxes_size[:, [2, 0, 1]]

    # finally cast angle
    cur_boxes_angle = -cur_boxes_angle

    box_list = []
    for i in range(cur_boxes.shape[0]):
        quat = Quaternion(axis=[0, 0, 1], radians=cur_boxes_angle[i])
        if cur_velocity is not None:
            velocity = (*cur_velocity[i, :], 0.0)
        else:
            velocity = (np.nan, np.nan, np.nan)
        if cur_attribute is not None:
            attribute = cur_attribute[i] # 8
            cur_class_name = classes[cur_class[i]]
            if cur_class_name in ['car', 'truck', 'bus', 'trailer', 'construction_vehicle']:
                attribute = np.argmax(attribute[:3])
            elif cur_class_name in ['pedestrian']:
                attribute = np.argmax(attribute[5:])
            elif cur_class_name in ['motorcycle', 'bicycle']:
                attribute = np.argmax(attribute[3:5])
            elif cur_class_name in ['traffic_cone', 'barrier']:
                attribute = -1
        else:
            attribute = -1
        box = Box(
              cur_boxes_ctr[i],
              cur_boxes_size[i],
              quat,
              label=cur_class[i],
              score=cur_score[i],
              velocity=velocity,
              name=attribute,
              )
        box_list.append(box)
    return box_list

def _lidar_nusc_box_to_global(info, boxes, classes, eval_version="cvpr_2019"):
    import pyquaternion
    box_list = []
    for box in boxes:
        # Move box to ego vehicle coord system
        box.rotate(pyquaternion.Quaternion(info['lidar2ego_rotation']))
        box.translate(np.array(info['lidar2ego_translation']))
        from nuscenes.eval.detection.config import eval_detection_configs
        # filter det in ego.
        cls_range_map = eval_detection_configs[eval_version]["class_range"]
        radius = np.linalg.norm(box.center[:2], 2)
        det_range = cls_range_map[classes[box.label]]
        if radius > det_range:
            continue
        # Move box to global coord system
        box.rotate(pyquaternion.Quaternion(info['ego2global_rotation']))
        box.translate(np.array(info['ego2global_translation']))
        box_list.append(box)
    return box_list
