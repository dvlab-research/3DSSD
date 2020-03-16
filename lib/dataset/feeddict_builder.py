import tensorflow
import numpy as np

from core.config import cfg
from dataset import maps_dict

class FeedDictCreater:
    def __init__(self, dataset_iter, model_list, batch_size):
        self.dataset_iter = dataset_iter
        self.model_list = model_list
        self.gpu_num = len(self.model_list)
        self.batch_size = batch_size

        # then create feed_dict
        placeholders_list = []
        for index, model in self.model_list:
            placeholders_list.append(model.placeholders)
        self.placeholders_list = placeholders_list

        if cfg.DATASET.TYPE == 'KITTI':
            self.create_feed_dict = self.kitti_create_feed_dict
        elif cfg.DATASET.TYPE == 'NuScenes':
            self.create_feed_dict = self.nuscenes_create_feed_dict
        else: raise Exception('Not Implementation Error!!!')

    def kitti_create_feed_dict(self):
        sample = next(self.dataset_iter, None)
        points, sem_labels, sem_dists, label_boxes_3d, ry_cls_label, residual_angle, label_classes, calib_P, sample_name = sample
        self.info = [calib_P, sample_name]

        feed_dict = dict()
        for i in range(self.gpu_num):
            cur_placeholder = self.placeholders_list[i] 
            begin_idx = i * self.batch_size
            end_idx = (i+1) * self.batch_size

            feed_dict[cur_placeholder[maps_dict.PL_POINTS_INPUT]] = points[begin_idx:end_idx]

            feed_dict[cur_placeholder[maps_dict.PL_LABEL_SEMSEGS]] = sem_labels[begin_idx:end_idx]
            feed_dict[cur_placeholder[maps_dict.PL_LABEL_DIST]] = sem_dists[begin_idx:end_idx] 

            feed_dict[cur_placeholder[maps_dict.PL_LABEL_BOXES_3D]] = label_boxes_3d[begin_idx:end_idx]
            feed_dict[cur_placeholder[maps_dict.PL_LABEL_CLASSES]] = label_classes[begin_idx:end_idx] 
            feed_dict[cur_placeholder[maps_dict.PL_ANGLE_CLS]] = ry_cls_label[begin_idx:end_idx] 
            feed_dict[cur_placeholder[maps_dict.PL_ANGLE_RESIDUAL]] = residual_angle[begin_idx:end_idx] 

            feed_dict[cur_placeholder[maps_dict.PL_CALIB_P2]] = calib_P[begin_idx:end_idx] 

        return feed_dict
    

    def nuscenes_create_feed_dict(self):
        sample = next(self.dataset_iter, None)
        rect_point_cloud, cur_sample_points, other_sample_points, label_boxes_3d, label_y_angle, label_res_angle, label_classes, label_attributes, label_velocity, sample_name, cur_transformation_matrix, sweeps, original_cur_sweep_points = sample
        self.info = [sample_name, cur_transformation_matrix, sweeps] 

        feed_dict = dict()
        for i in range(self.gpu_num):
            cur_placeholder = self.placeholders_list[i] 
            begin_idx = i * self.batch_size
            end_idx = (i+1) * self.batch_size

            feed_dict[cur_placeholder[maps_dict.PL_POINTS_INPUT]] = rect_point_cloud[begin_idx:end_idx] 
            feed_dict[cur_placeholder[maps_dict.PL_CUR_SWEEP_POINTS_INPUT]] = cur_sample_points[begin_idx:end_idx] 
            feed_dict[cur_placeholder[maps_dict.PL_OTHER_SWEEP_POINTS_INPUT]] = other_sample_points[begin_idx:end_idx] 

            feed_dict[cur_placeholder[maps_dict.PL_LABEL_BOXES_3D]] = label_boxes_3d[begin_idx:end_idx]
            feed_dict[cur_placeholder[maps_dict.PL_LABEL_CLASSES]] = label_classes[begin_idx:end_idx]
            feed_dict[cur_placeholder[maps_dict.PL_LABEL_ATTRIBUTES]] = label_attributes[begin_idx:end_idx]
            feed_dict[cur_placeholder[maps_dict.PL_LABEL_VELOCITY]] = label_velocity[begin_idx:end_idx]
            feed_dict[cur_placeholder[maps_dict.PL_ANGLE_CLS]] = label_y_angle[begin_idx:end_idx]
            feed_dict[cur_placeholder[maps_dict.PL_ANGLE_RESIDUAL]] = label_res_angle[begin_idx:end_idx]

        return feed_dict
