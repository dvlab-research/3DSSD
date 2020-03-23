import tensorflow as tf 
import numpy as np

from core.config import cfg
from dataset import maps_dict

class PlaceHolders:
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.placeholders = dict()
        
        get_placeholders = {
            'KITTI': self.get_placeholders_kitti,
            'NuScenes': self.get_placeholders_nuscenes,
        }

        self.get_placeholders = get_placeholders[cfg.DATASET.TYPE] 

    def _add_placeholder(self, dtype, shape, name):
        placeholder = tf.placeholder(dtype, shape, name)
        self.placeholders[name] = placeholder
        return placeholder

    def get_placeholders_kitti(self):
        with tf.variable_scope('points_input'):
            self._add_placeholder(tf.float32, [self.batch_size, cfg.MODEL.POINTS_NUM_FOR_TRAINING, 4], maps_dict.PL_POINTS_INPUT)

        with tf.variable_scope('pl_labels'):
            self._add_placeholder(tf.float32, [self.batch_size, None, 7],
                                  maps_dict.PL_LABEL_BOXES_3D)
            self._add_placeholder(tf.int32, [self.batch_size, None],
                                  maps_dict.PL_LABEL_CLASSES)
            self._add_placeholder(tf.int32, [self.batch_size, None], maps_dict.PL_ANGLE_CLS)
            self._add_placeholder(tf.float32, [self.batch_size, None], maps_dict.PL_ANGLE_RESIDUAL)

            self._add_placeholder(tf.int32, [self.batch_size, None], maps_dict.PL_LABEL_SEMSEGS)
            self._add_placeholder(tf.float32, [self.batch_size, None], maps_dict.PL_LABEL_DIST)

            self._add_placeholder(tf.float32, [self.batch_size, 3, 4], maps_dict.PL_CALIB_P2)
         
    def get_placeholders_nuscenes(self):
        with tf.variable_scope('points_input'):
            self._add_placeholder(tf.float32, [self.batch_size, cfg.DATASET.NUSCENE.MAX_NUMBER_OF_VOXELS, cfg.DATASET.MAX_NUMBER_OF_POINT_PER_VOXEL, cfg.DATASET.NUSCENES.INPUT_FEATURE_CHANNEL], maps_dict.PL_POINTS_INPUT)

        with tf.variable_scope('pl_labels'):
            self._add_placeholder(tf.float32, [self.batch_size, None, 7],
                                  maps_dict.PL_LABEL_BOXES_3D)
            self._add_placeholder(tf.int32, [self.batch_size, None],
                                  maps_dict.PL_LABEL_CLASSES)
            self._add_placeholder(tf.int32, [self.batch_size, None],
                                  maps_dict.PL_LABEL_ATTRIBUTES)
            self._add_placeholder(tf.float32, [self.batch_size, None, 2],
                                  maps_dict.PL_LABEL_VELOCITY)
            self._add_placeholder(tf.int32, [self.batch_size, None], 
                                  maps_dict.PL_ANGLE_CLS)
            self._add_placeholder(tf.float32, [self.batch_size, None], 
                                  maps_dict.PL_ANGLE_RESIDUAL)

            self._add_placeholder(tf.float32, [self.batch_size, cfg.DATASET.NUSCENE.MAX_CUR_SAMPLE_POINTS_NUM, cfg.DATASET.MAX_NUMBER_OF_POINT_PER_VOXEL, cfg.DATASET.NUSCENES.INPUT_FEATURE_CHANNEL], maps_dict.PL_CUR_SWEEP_POINTS_INPUT)
            self._add_placeholder(tf.float32, [self.batch_size, cfg.DATASET.NUSCENE.MAX_NUMBER_OF_VOXELS - cfg.DATASET.NUSCENE.MAX_CUR_SAMPLE_POINTS_NUM, cfg.DATASET.MAX_NUMBER_OF_POINT_PER_VOXEL, cfg.DATASET.NUSCENES.INPUT_FEATURE_CHANNEL], maps_dict.PL_OTHER_SWEEP_POINTS_INPUT)
            self._add_placeholder(tf.int32, [self.batch_size, cfg.DATASET.POINTS_NUM_FOR_TRAINING], maps_dict.PL_POINTS_NUM_PER_VOXEL)

