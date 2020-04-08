import numpy as np
import tensorflow as tf
from core.config import cfg

import utils.generate_anchors as generate_anchors
from utils.model_util import g_type_mean_size 

class Anchors:
    def __init__(self, stage, class_list):
        """
        The anchor class is targeted on generating anchors, assigning anchors and regressing anchors
        class_list: ['Car', 'Pedestrian', 'Cyclist'] for KITTI dataset
        prefix: 'Kitti', 'NuScenes', 'Lyft'
        """ 
        self.class_list = class_list 
        if cfg.DATASET.TYPE == 'KITTI':
            prefix = 'Kitti'
        elif cfg.DATASET.TYPE == 'NuScenes':
            prefix = 'NuScenes'
        elif cfg.DATASET.TYPE == 'Lyft':
            prefix = 'Lyft'
        self.class_size_keys = ["%s_%s"%(prefix, cls) for cls in class_list]

        self.anchor_sizes = [g_type_mean_size[cls_name] for cls_name in self.class_size_keys]
        self.anchors_num = len(self.anchor_sizes)

        if stage == 0:
            self.anchor_cfg = cfg.MODEL.FIRST_STAGE
        elif stage == 1:
            self.anchor_cfg = cfg.MODEL.SECOND_STAGE

        generate_function = {
            'Anchor': self.generate_anchors,
            'free': self.generate_anchors_free,
        }
        reg_method = self.anchor_cfg.REGRESSION_METHOD.TYPE
        anchor_type = reg_method.split('-')[-1]
        self.generate = generate_function[anchor_type]


    def generate_anchors(self, points):
        """
        generate anchors based on points
        bs, npoint, cls_num, 7
        """
        if isinstance(points, tf.Tensor):
            anchors_3d = generate_anchors.generate_3d_anchors_by_point_tf(points, self.anchor_sizes)
        else: # numpy
            anchors_3d = generate_anchors.generate_3d_anchors_by_point(points, self.anchor_sizes)
        return anchors_3d

    def generate_anchors_free(self, points):
        """
        generate anchors based on points
        bs, npoint, 1, 3 -> bs, npoint, cls_num, 7
        """
        return tf.expand_dims(points, axis=2)
        

    
