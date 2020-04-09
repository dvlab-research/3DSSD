import tensorflow as tf
import sys, os
import numpy as np
import cv2
import itertools
import pickle
import time
import json
import subprocess
import tqdm

from core.config import cfg
import utils.kitti_aug as kitti_aug
import utils.box_3d_utils as box_3d_utils
import dataset.maps_dict as maps_dict

from dataset.dataloader.nuscenes_utils import * 
from dataset.dataloader.nuscenes_split import *
from utils.voxelnet_aug import check_inside_points
from utils.anchor_encoder import encode_angle2class_np
from builder.voxel_generator.voxel_generator import VoxelGenerator
from builder.data_augmentor import DataAugmentor
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud, RadarPointCloud, Box
from pyquaternion import Quaternion
from dataset.data_provider.data_provider import DataFromList, MultiProcessMapData, BatchDataNuscenes
from dataset.dataloader.dataloader import Dataset

class NuScenesDataset(Dataset):
    """
    NuScenes dataset loader and producer
    """
    def __init__(self, mode, split='training', img_list='trainval', is_training=True, workers_num=1):
        """
        mode: 'loading', 'preprocessing'
        """
        self.mode = mode
        self.dataset_dir = os.path.join(cfg.ROOT_DIR, cfg.DATASET.KITTI.BASE_DIR_PATH)
        self.max_sweeps = cfg.DATASET.NUSCENES.NSWEEPS
        self.is_training = is_training
        self.img_list = img_list
        self.workers_num = workers_num

        # cast labels from NuScenes name to useful name
        self.useful_cls_dict = {'animal': 'ignore',
                           'human.pedestrian.personal_mobility': 'ignore',
                           'human.pedestrian.stroller': 'ignore',
                           'human.pedestrian.wheelchair': 'ignore',
                           'movable_object.debris': 'ignore',
                           'movable_object.pushable_pullable': 'ignore',
                           'static_object.bicycle_rack': 'ignore',
                           'vehicle.emergency.ambulance': 'ignore',
                           'vehicle.emergency.police': 'ignore',
                           'movable_object.barrier': 'barrier',
                           'vehicle.bicycle': 'bicycle',
                           'vehicle.bus.bendy': 'bus',
                           'vehicle.bus.rigid': 'bus',
                           'vehicle.car': 'car',
                           'vehicle.construction': 'construction_vehicle',
                           'vehicle.motorcycle': 'motorcycle',
                           'human.pedestrian.adult': 'pedestrian',
                           'human.pedestrian.child': 'pedestrian',
                           'human.pedestrian.construction_worker': 'pedestrian',
                           'human.pedestrian.police_officer': 'pedestrian',
                           'movable_object.trafficcone': 'traffic_cone',
                           'vehicle.trailer': 'trailer',
                           'vehicle.truck': 'truck'}
        # cast attribute to index
        self.attribute_idx_list = {'vehicle.moving': 0,
                                 'vehicle.stopped': 1,
                                 'vehicle.parked': 2,
                                 'cycle.with_rider': 3,
                                 'cycle.without_rider': 4,
                                 'pedestrian.sitting_lying_down': 5,
                                 'pedestrian.standing': 6,
                                 'pedestrian.moving': 7,
                                 'default': -1,
                                 }
        self.idx_attribute_list = dict([(v, k) for k, v in self.attribute_idx_list.items()])
        self.AttributeIdxLabelMapping = {
            "car": ['vehicle.moving', 'vehicle.stopped', 'vehicle.parked'],
            "truck": ['vehicle.moving', 'vehicle.stopped', 'vehicle.parked'],
            "bus": ['vehicle.moving', 'vehicle.stopped', 'vehicle.parked'],
            "trailer": ['vehicle.moving', 'vehicle.stopped', 'vehicle.parked'],
            "construction_vehicle": ['vehicle.moving', 'vehicle.stopped', 'vehicle.parked'],
            "pedestrian": ['pedestrian.sitting_lying_down', 'pedestrian.standing', 'pedestrian.moving'],
            "motorcycle": ['cycle.with_rider', 'cycle.without_rider', ''],
            "bicycle": ['cycle.with_rider', 'cycle.without_rider', ''],
            "traffic_cone": ['', '', ''],
            "barrier": ['', '', ''],
        }

        self.DefaultAttribute = {
            "car": "vehicle.parked",
            "pedestrian": "pedestrian.moving",
            "trailer": "vehicle.parked",
            "truck": "vehicle.parked",
            "bus": "vehicle.parked",
            "motorcycle": "cycle.without_rider",
            "construction_vehicle": "vehicle.parked",
            "bicycle": "cycle.without_rider",
            "barrier": "",
            "traffic_cone": "",
        }

        self.cls_list = cfg.DATASET.KITTI.CLS_LIST
        self.idx2cls_dict = dict([(idx+1, cls) for idx, cls in enumerate(self.cls_list)])
        self.cls2idx_dict = dict([(cls, idx+1) for idx, cls in enumerate(self.cls_list)])

        self.sv_npy_path = os.path.join(cfg.ROOT_DIR, cfg.DATASET.KITTI.SAVE_NUMPY_PATH, 'NuScenes', '{}_{}'.format(img_list, self.max_sweeps))
        self.train_list = os.path.join(self.sv_npy_path, 'infos.pkl')

        self.voxel_generator = VoxelGenerator()

        self.test_mode = cfg.TEST.TEST_MODE
        if self.test_mode == 'mAP':
            self.evaluation = self.evaluate_map
            self.logger_and_select_best = self.logger_and_select_best_map
        elif self.test_mode == 'Recall':
            self.evaluation = self.evaluate_recall
            self.logger_and_select_best = self.logger_and_select_best_recall
        else: raise Exception('No other evaluation mode.') 

        if mode == 'loading':
            # data loader
            with open(self.train_list, 'rb') as f:
                self.train_npy_list = pickle.load(f)
            self.sample_num = len(self.train_npy_list)
            if self.is_training:
                self.data_augmentor = DataAugmentor('NuScenes', workers_num=self.workers_num)

        elif mode == 'preprocessing':
            # preprocess raw data
            if img_list == 'train':
                self.nusc = NuScenes(dataroot=self.dataset_dir, version='v1.0-trainval')
                self.scenes = [scene for scene in self.nusc.scene if scene['name'] in train_scene]
            elif img_list == 'val':
                self.nusc = NuScenes(dataroot=self.dataset_dir, version='v1.0-trainval')
                self.scenes = [scene for scene in self.nusc.scene if scene['name'] in val_scene]
            else: # test
                self.nusc = NuScenes(dataroot=self.dataset_dir, version='v1.0-test')
                self.scenes = self.nusc.scene

            self.sample_data_token_list = OrderedDict()
            sample_num = 0
            for scene in self.scenes:
                # static the sample num, and save all sample_data_token
                self.sample_data_token_list[scene['token']] = []
                all_sample = self.nusc.field2token('sample', 'scene_token', scene['token'])
                sample_num += len(all_sample)
                for sample in all_sample: # all sample token
                    sample = self.nusc.get('sample', sample)
                    cur_token = sample['token']
                    cur_data_token = sample['data']['LIDAR_TOP']
                    self.sample_data_token_list[scene['token']].append(cur_data_token)

            self.sample_num = sample_num

            self.extents = cfg.DATASET.POINT_CLOUD_RANGE
            self.extents = np.reshape(self.extents, [3, 2])
            if not os.path.exists(self.sv_npy_path): os.makedirs(self.sv_npy_path)

            # also calculate the mean size here
            self.cls_size_dict = dict([(cls, np.array([0, 0, 0], dtype=np.float32)) for cls in self.cls_list])
            self.cls_num_dict = dict([(cls, 0) for cls in self.cls_list])

            # the save path for MixupDB
            if self.img_list in ['train', 'val', 'trainval'] and cfg.TEST.WITH_GT and cfg.TRAIN.AUGMENTATIONS.MIXUP.OPEN:
                self.mixup_db_cls_path = dict()
                self.mixup_db_trainlist_path = dict()
                self.mixup_db_class = cfg.TRAIN.AUGMENTATIONS.MIXUP.CLASS
                for cls in self.mixup_db_class:
                    mixup_db_cls_path = os.path.join(cfg.ROOT_DIR, cfg.DATASET.KITTI.SAVE_NUMPY_PATH, cfg.TRAIN.AUGMENTATIONS.MIXUP.SAVE_NUMPY_PATH, cfg.TRAIN.AUGMENTATIONS.MIXUP.PC_LIST, '{}'.format(cls))
                    mixup_db_trainlist_path = os.path.join(mixup_db_cls_path, 'train_list.txt')
                    if not os.path.exists(mixup_db_cls_path): os.makedirs(mixup_db_cls_path)
                    self.mixup_db_cls_path[cls] = mixup_db_cls_path
                    self.mixup_db_trainlist_path[cls] = mixup_db_trainlist_path

    def __len__(self):
        return self.sample_num

    def load_samples(self, sample_idx, pipename):
        """ load data per thread """
        pipename = int(pipename)
        biggest_label_num = 0
        sample_dict = self.train_npy_list[sample_idx] 

        points_path = sample_dict[maps_dict.KEY_POINT_CLOUD]
        sweeps = sample_dict[maps_dict.KEY_SWEEPS]
        sample_name = sample_dict[maps_dict.KEY_SAMPLE_NAME]
        cur_transformation_matrix = sample_dict[maps_dict.KEY_TRANSFORMRATION_MATRIX]
        ts = sample_dict[maps_dict.KEY_TIMESTAMPS] / 1e6

        # then first read points and stack points from multiple frame
        points =  np.fromfile(points_path, dtype=np.float32)
        points = points.reshape((-1, 5))
        points = cast_points_to_kitti(points)
        points[:, 3] /= 255
        points[:, 4] = 0
        sweep_points_list = [points]
        original_cur_sweep_points = points
        cur_sweep_points_num = points.shape[0]
        for sweep in sweeps:
            points_sweep = np.fromfile(sweep['lidar_path'], dtype=np.float32)
            points_sweep = points_sweep.reshape((-1, 5))
            sweep_ts = sweep['timestamp'] / 1e6
            points_sweep[:, 3] /= 255
            points_sweep[:, :3] = points_sweep[:, :3] @ sweep['sweep2lidar_rotation'].T
            points_sweep[:, :3] += sweep['sweep2lidar_translation']
            points_sweep[:, 4] = ts - sweep_ts
            points_sweep = cast_points_to_kitti(points_sweep)
            sweep_points_list.append(points_sweep)
        if cfg.DATASET.NUSCENES.INPUT_FEATURE_CHANNEL == 4:
            points = np.concatenate(sweep_points_list, axis=0)[:, [0, 1, 2, 4]]
        else:
            points = np.concatenate(sweep_points_list, axis=0)

        # then read groundtruth file if have
        if self.is_training or cfg.TEST.WITH_GT:
            label_boxes_3d = sample_dict[maps_dict.KEY_LABEL_BOXES_3D]
            label_boxes_3d = cast_box_3d_to_kitti_format(label_boxes_3d)

            label_classes_name = sample_dict[maps_dict.KEY_LABEL_CLASSES]
            label_classes = np.array([self.cls2idx_dict[label_class] for label_class in label_classes_name])

            label_attributes = sample_dict[maps_dict.KEY_LABEL_ATTRIBUTES]
            label_velocity = sample_dict[maps_dict.KEY_LABEL_VELOCITY] # [-1, 2]

            ry_cls_label, residual_angle = encode_angle2class_np(label_boxes_3d[:, -1], cfg.MODEL.ANGLE_CLS_NUM)
        else: # not is_training and no_gt
            label_boxes_3d = np.zeros([1, 7], np.float32)
            label_classes = np.zeros([1], np.int32)
            label_attributes = np.zeros([1], np.int32)
            label_velocity = np.zeros([1, 2], np.float32)
            ry_cls_label = np.zeros([1], np.int32)
            residual_angle = np.zeros([1], np.float32)

        if self.is_training: # data augmentation
            points, label_boxes_3d, label_classes, label_attributes, label_velocity, cur_sweep_points_num = self.data_augmentor.nuscenes_forward(points, label_boxes_3d, label_classes, pipename, label_attributes, label_velocity, cur_sweep_points_num)   
            ry_cls_label, residual_angle = encode_angle2class_np(label_boxes_3d[:, -1], cfg.MODEL.ANGLE_CLS_NUM)
        cur_label_num = len(label_boxes_3d)

        # then randomly choose some points
        cur_sweep_points = points[:cur_sweep_points_num, :] # [-1, 4]
        other_sweep_points = points[cur_sweep_points_num:, :] # [-1, 4]
        if len(other_sweep_points) == 0: other_sweep_points = cur_sweep_points.copy()
        np.random.shuffle(cur_sweep_points)
        np.random.shuffle(other_sweep_points)

        input_sample_points, num_points_per_voxel = self.voxel_generator.generate_nusc(cur_sweep_points, other_sweep_points, cfg.DATASET.NUSCENE.MAX_CUR_SAMPLE_POINTS_NUM) # points, [num_voxels, num_points, 5], sem_labels, [num_voxels, num_points]
        cur_sample_points = input_sample_points[:cfg.DATASET.NUSCENE.MAX_CUR_SAMPLE_POINTS_NUM]
        other_sample_points = input_sample_points[cfg.DATASET.NUSCENE.MAX_CUR_SAMPLE_POINTS_NUM:]

        biggest_label_num = max(biggest_label_num, cur_label_num)
        return biggest_label_num, input_sample_points, cur_sample_points, other_sample_points, label_boxes_3d, ry_cls_label, residual_angle, label_classes, label_attributes, label_velocity, sample_name, cur_transformation_matrix, sweeps, original_cur_sweep_points
         

    def load_batch(self, batch_size):
        perm = np.arange(self.sample_num).tolist() # a list indicates each data
        dp = DataFromList(perm, is_train=self.is_training, shuffle=self.is_training)
        dp = MultiProcessMapData(dp, self.load_samples, self.workers_num)

        use_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1]
        use_concat = [0, 0, 0, 2, 2, 2, 2, 2, 2, 0, 0, 0, 0]

        dp = BatchDataNuscenes(dp, batch_size, use_concat=use_concat, use_list=use_list)
        dp.reset_state()
        dp = dp.get_data()
        return dp

   
    # Preprocess data
    def preprocess_samples(self, cur_scene_key, sample_data_token):
        sample_dicts = []
        biggest_label_num = 0

        cur_sample_data = self.nusc.get('sample_data', sample_data_token)
        cur_sample_token = cur_sample_data['sample_token']
        cur_sample = self.nusc.get('sample', cur_sample_token)

        ego_pose = self.nusc.get('ego_pose', cur_sample_data['ego_pose_token'])
        calibrated_sensor = self.nusc.get('calibrated_sensor', cur_sample_data['calibrated_sensor_token'])

        l2e_r = calibrated_sensor['rotation']
        l2e_t = calibrated_sensor['translation']
        e2g_r = ego_pose['rotation']
        e2g_t = ego_pose['translation']
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix
        cur_timestamp = cur_sample['timestamp']

        cur_transformation_matrix = {
            'lidar2ego_translation': l2e_t,
            'lidar2ego_rotation': l2e_r,
            'ego2global_translation': e2g_t,
            'ego2global_rotation': e2g_r,
        }

        # get point cloud in former 0.5 second
        sweeps = []
        while len(sweeps) < self.max_sweeps:
            if not cur_sample_data['prev'] == '':
                # has next frame
                cur_sample_data = self.nusc.get('sample_data', cur_sample_data['prev'])
                cur_ego_pose = self.nusc.get('ego_pose', cur_sample_data['ego_pose_token'])
                cur_calibrated_sensor = self.nusc.get('calibrated_sensor', cur_sample_data['calibrated_sensor_token'])
                cur_lidar_path, cur_sweep_boxes, _ = self.nusc.get_sample_data(cur_sample_data['token'])
                sweep = {
                    "lidar_path": cur_lidar_path,
                    "sample_data_token": cur_sample_data['token'],
                    "lidar2ego_translation": cur_calibrated_sensor['translation'],
                    "lidar2ego_rotation": cur_calibrated_sensor['rotation'],
                    "ego2global_translation": cur_ego_pose['translation'],
                    "ego2global_rotation": cur_ego_pose['rotation'],
                    "timestamp": cur_sample_data["timestamp"]
                }
                l2e_r_s = sweep["lidar2ego_rotation"]
                l2e_t_s = sweep["lidar2ego_translation"]
                e2g_r_s = sweep["ego2global_rotation"]
                e2g_t_s = sweep["ego2global_translation"]
                # sweep->ego->global->ego'->lidar
                l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
                e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix

                R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
                    np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
                T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
                    np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
                T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(
                    l2e_r_mat).T) + l2e_t @ np.linalg.inv(l2e_r_mat).T

                sweep["sweep2lidar_rotation"] = R.T  # points @ R.T + T
                sweep["sweep2lidar_translation"] = T
                sweeps.append(sweep)
            else: # prev is none
                break

        # then load gt_boxes_3d
        if self.img_list in ['train', 'val'] and cfg.TEST.WITH_GT:
            cur_data_path, all_boxes, _ = self.nusc.get_sample_data(sample_data_token)

            # then first parse boxes labels
            locs = np.array([box.center for box in all_boxes]).reshape(-1, 3)
            sizes = np.array([box.wlh for box in all_boxes]).reshape(-1, 3)
            rots = np.array([box.orientation.yaw_pitch_roll[0]
                             for box in all_boxes]).reshape(-1, 1)
            all_boxes_3d = np.concatenate([locs, sizes, -rots], axis=-1)

            annos_tokens = cur_sample['anns']
            all_velocity = np.array([self.nusc.box_velocity(ann_token)[:2] for ann_token in annos_tokens]) # [-1, 2]
            for i in range(len(all_boxes)):
                velo = np.array([*all_velocity[i], 0.0])
                velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                all_velocity[i] = velo[:2] # [-1, 2]

            attribute_tokens = [self.nusc.get('sample_annotation', ann_token)['attribute_tokens'] for ann_token in annos_tokens]
            all_attribute = []
            for attribute_token in attribute_tokens:
                if len(attribute_token) == 0:
                    all_attribute.append([])
                else:
                    all_attribute.append(self.nusc.get('attribute', attribute_token[0])['name'])
            # then filter these ignore labels
            categories = np.array([box.name for box in all_boxes])
            if self.img_list == 'train':
                useful_idx = [index for index, category in enumerate(categories) if self.useful_cls_dict[category] != 'ignore']
            else:
                useful_idx = [index for index, category in enumerate(categories)]
            if len(useful_idx) == 0:
                if self.img_list == 'train':
                    return None, biggest_label_num
                else:
                    all_boxes_3d = np.ones([1, 7], dtype=np.float32)
                    all_boxes_classes = np.array(['ignore'])
                    all_attribute = np.array([-1])
                    all_velocity = np.array([[0, 0]], dtype=np.float32)
            else:
                all_boxes_3d = all_boxes_3d[useful_idx]

                categories = categories[useful_idx]
                all_boxes_classes = np.array([self.useful_cls_dict[cate] for cate in categories])
                # now calculate the mean size of each box
                for tmp_idx, all_boxes_class in enumerate(all_boxes_classes):
                    cur_mean_size = self.cls_size_dict[all_boxes_class] * self.cls_num_dict[all_boxes_class]
                    cur_cls_num = self.cls_num_dict[all_boxes_class] + 1
                    cur_total_size = cur_mean_size + all_boxes_3d[tmp_idx, [4, 5, 3]]  # [l, w, h]
                    cur_mean_size = cur_total_size / cur_cls_num
                    self.cls_size_dict[all_boxes_class] = cur_mean_size
                    self.cls_num_dict[all_boxes_class] = cur_cls_num

                all_attribute = [all_attribute[tmp_idx] for tmp_idx in useful_idx]
                tmp_attribute = []
                for attr in all_attribute:
                    if attr == []:tmp_attribute.append(-1)
                    else:
                        tmp_attribute.append(self.attribute_idx_list[attr])
                all_attribute = tmp_attribute
                all_attribute = np.array(all_attribute, dtype=np.int32)
                all_velocity = [all_velocity[tmp_idx] for tmp_idx in useful_idx]
                all_velocity = np.array(all_velocity, dtype=np.float32)
        else:
            cur_data_path = self.nusc.get_sample_data_path(sample_data_token)

        # then generate the bev_maps
        if self.img_list in ['train', 'val', 'trainval'] and cfg.TEST.WITH_GT:
            sample_dict = {
                maps_dict.KEY_LABEL_BOXES_3D: all_boxes_3d,
                maps_dict.KEY_LABEL_CLASSES: all_boxes_classes,
                maps_dict.KEY_LABEL_ATTRIBUTES: all_attribute,
                maps_dict.KEY_LABEL_VELOCITY: all_velocity,
                maps_dict.KEY_LABEL_NUM: len(all_boxes_3d),

                maps_dict.KEY_POINT_CLOUD: cur_data_path,
                maps_dict.KEY_TRANSFORMRATION_MATRIX: cur_transformation_matrix, 
                maps_dict.KEY_SAMPLE_NAME: '{}/{}/{}'.format(cur_scene_key, cur_sample_token, sample_data_token),
                maps_dict.KEY_SWEEPS: sweeps,
                maps_dict.KEY_TIMESTAMPS: cur_timestamp,
            }
            biggest_label_num = max(len(all_boxes_3d), biggest_label_num)
        else:
            # img_list is test
            sample_dict = {
                maps_dict.KEY_POINT_CLOUD: cur_data_path,
                maps_dict.KEY_SAMPLE_NAME: '{}/{}/{}'.format(cur_scene_key, cur_sample_token, sample_data_token),
                maps_dict.KEY_TRANSFORMRATION_MATRIX: cur_transformation_matrix, 
                maps_dict.KEY_SWEEPS: sweeps,
                maps_dict.KEY_TIMESTAMPS: cur_timestamp,
            }
        return sample_dict, biggest_label_num

    def preprocess_batch(self):
        # if create_gt_dataset, then also create a boxes_numpy, saving all points
        if cfg.TRAIN.AUGMENTATIONS.MIXUP.OPEN: # also save mixup database
            mixup_label_dict = dict([(cls, []) for cls in self.mixup_db_class])

        sample_dicts_list = []
        for scene_key, v in tqdm.tqdm(self.sample_data_token_list.items()):
            for sample_data_token in v:
                sample_dict, tmp_biggest_label_num = self.preprocess_samples(scene_key, sample_data_token)
                if sample_dict is None:
                    continue
                # else save the result
                sample_dicts_list.append(sample_dict)

                # create_gt_dataset
                if self.img_list in ['train', 'val', 'trainval'] and cfg.TEST.WITH_GT and cfg.TRAIN.AUGMENTATIONS.MIXUP.OPEN:
                    mixup_sample_dicts = self.generate_mixup_sample(sample_dict)
                    if mixup_sample_dicts is None: continue
                    for mixup_sample_dict in mixup_sample_dicts:
                        cur_cls = mixup_sample_dict[maps_dict.KEY_SAMPLED_GT_CLSES]
                        mixup_label_dict[cur_cls].append(mixup_sample_dict)

        # save preprocessed data
        with open(self.train_list, 'wb') as f:
            pickle.dump(sample_dicts_list, f)
        for k, v in self.cls_num_dict.items():
            print('class name: %s / class num: %d / mean size: (%f, %f, %f)' % (k, v, self.cls_size_dict[k][0], self.cls_size_dict[k][1], self.cls_size_dict[k][2])) # [l, w, h]

        if self.img_list in ['train', 'val', 'trainval'] and cfg.TEST.WITH_GT and cfg.TRAIN.AUGMENTATIONS.MIXUP.OPEN:
            print('**** Generating groundtruth database ****')
            for cur_cls_name, mixup_sample_dict in mixup_label_dict.items():
                cur_mixup_db_cls_path = self.mixup_db_cls_path[cur_cls_name]
                cur_mixup_db_trainlist_path= self.mixup_db_trainlist_path[cur_cls_name]
                print('**** Class %s ****'%cur_cls_name)
                with open(cur_mixup_db_trainlist_path, 'w') as f:
                    for tmp_idx, tmp_cur_mixup_sample_dict in tqdm.tqdm(enumerate(mixup_sample_dict)):
                        f.write('%06d.npy\n'%tmp_idx)
                        np.save(os.path.join(cur_mixup_db_cls_path, '%06d.npy'%tmp_idx), tmp_cur_mixup_sample_dict)
        print('Ending of the preprocess !!!')


    def generate_mixup_sample(self, sample_dict):
        """ This function is bound for generating mixup dataset """
        all_boxes_3d = sample_dict[maps_dict.KEY_LABEL_BOXES_3D]
        all_boxes_classes = sample_dict[maps_dict.KEY_LABEL_CLASSES]
        point_cloud_path = sample_dict[maps_dict.KEY_POINT_CLOUD]

        # then we first cast all_boxes_3d to kitti format
        all_boxes_3d = cast_box_3d_to_kitti_format(all_boxes_3d)

        # load points
        points = np.fromfile(point_cloud_path, dtype=np.float32).reshape((-1, 5))
        points = cast_points_to_kitti(points)
        points[:, 3] /= 255
        points[:, 4] = 0 # timestamp is zero
        
        points_mask = check_inside_points(points, all_boxes_3d) # [pts_num, gt_num]
        points_masks_num = np.sum(points_masks, axis=0) # [gt_num]
        valid_box_idx = np.where(points_masks_num >= cfg.DATASET.MIN_POINTS_NUM)[0]

        if len(valid_box_idx) == 0:
            return None
        
        valid_label_boxes_3d = all_boxes_3d[valid_box_idx]
        valid_label_classes = all_boxes_classes[valid_box_idx]

        sample_dicts = []
        for index, i in enumerate(valid_box_idx):
            cur_points_mask = points_mask[:, i]
            cur_points_idx = np.where(cur_points_mask)[0]
            cur_inside_points = points[cur_points_idx, :]
            sample_dict = {
                # 0 timestamp and /255 reflectance
                maps_dict.KEY_SAMPLED_GT_POINTS: cur_inside_points, # kitti format points
                maps_dict.KEY_SAMPLED_GT_LABELS_3D: valid_label_boxes_3d[index],
                maps_dict.KEY_SAMPLED_GT_CLSES: valid_label_classes[index],
            }
            sample_dicts.append(sample_dict)
        return sample_dicts

    # Evaluation
    def set_evaluation_tensor(self, model):
        # get prediction results, bs = 1
        pred_bbox_3d = tf.squeeze(model.output[maps_dict.PRED_3D_BBOX][-1], axis=0)
        pred_cls_score = tf.squeeze(model.output[maps_dict.PRED_3D_SCORE][-1], axis=0)
        pred_cls_category = tf.squeeze(model.output[maps_dict.PRED_3D_CLS_CATEGORY][-1], axis=0)
        pred_list = [pred_bbox_3d, pred_cls_score, pred_cls_category]

        if len(model.output[maps_dict.PRED_3D_ATTRIBUTE]) > 0:
            pred_attribute = tf.squeeze(model.output[maps_dict.PRED_3D_ATTRIBUTE][-1], axis=0)
            pred_velocity = tf.squeeze(model.output[maps_dict.PRED_3D_VELOCITY][-1], axis=0)
            pred_list.extend([pred_attribute, pred_velocity])
        return pred_list

    def evaluate_map(self, sess, feeddict_producer, pred_list, val_size, cls_thresh, log_dir, placeholders=None):
        submissions = {}
        submissions['meta'] = dict()
        submissions['meta']['use_camera'] = False
        submissions['meta']['use_lidar'] = True
        submissions['meta']['use_radar'] = False
        submissions['meta']['use_map'] = False
        submissions['meta']['use_external'] = False

        submissions_results = dict()
        pred_attr_velo = (len(pred_list) == 5)

        for i in tqdm.tqdm(range(val_size)):
            feed_dict = feeddict_producer.create_feed_dict()

            if pred_attr_velo:
                pred_bbox_3d_op, pred_cls_score_op, pred_cls_category_op, pred_attr_op, pred_velo_op = sess.run(pred_list, feed_dict=feed_dict) 
            else:
                pred_bbox_3d_op, pred_cls_score_op, pred_cls_category_op = sess.run(pred_list, feed_dict=feed_dict)
            pred_cls_category_op += 1 # label from 1 to n
  
            sample_name, cur_transformation_matrix, sweeps = feeddict_producer.info
            sample_name = sample_name[0]
            cur_transformation_matrix = cur_transformation_matrix[0]
            sweeps = sweeps[0] 
            cur_scene_key, cur_sample_token, cur_sample_data_token = sample_name.split('/')

            select_idx = np.where(pred_cls_score_op >= cls_thresh)[0]
            pred_cls_score_op = pred_cls_score_op[select_idx]
            pred_cls_category_op = pred_cls_category_op[select_idx]
            pred_bbox_3d_op = pred_bbox_3d_op[select_idx]
            if pred_attr_velo:
                pred_attr_op = pred_attr_op[select_idx]
                pred_velo_op = pred_velo_op[select_idx]
            else: pred_attr_op, pred_velo_op = None, None

            if len(pred_bbox_3d_op) > 500:
                arg_sort_idx = np.argsort(pred_cls_score_op)[::-1]
                arg_sort_idx = arg_sort_idx[:500]
                pred_cls_score_op = pred_cls_score_op[arg_sort_idx]
                pred_cls_category_op = pred_cls_category_op[arg_sort_idx]
                pred_bbox_3d_op = pred_bbox_3d_op[arg_sort_idx]
                if pred_attr_velo:
                    pred_attr_op = pred_attr_op[arg_sort_idx]
                    pred_velo_op = pred_velo_op[arg_sort_idx]

            # then transform pred_bbox_op to nuscenes_box
            boxes = cast_kitti_format_to_nusc_box_3d(pred_bbox_3d_op, pred_cls_score_op, pred_cls_category_op, cur_attribute=pred_attr_op, cur_velocity=pred_velo_op, classes=self.idx2cls_dict)
            for box in boxes:
                velocity = box.velocity[:2].tolist()
                if len(sweeps) == 0:
                    velocity = (np.nan, np.nan)
                box.velocity = np.array([*velocity, 0.0])
            # then cast the box from ego to global
            boxes = _lidar_nusc_box_to_global(cur_transformation_matrix, boxes, self.idx2cls_dict, eval_version='cvpr_2019')

            annos = []
            for box in boxes:
                name = self.idx2cls_dict[box.label]
                if box.name == -1:
                    attr = self.DefaultAttribute[name]
                else:
                    attr = self.AttributeIdxLabelMapping[name][box.name]
                velocity = box.velocity[:2].tolist()
                nusc_anno = {
                    "sample_token": cur_sample_token,
                    "translation": box.center.tolist(),
                    "size": box.wlh.tolist(),
                    "rotation": box.orientation.elements.tolist(),
                    "velocity": velocity,
                    "detection_name": name,
                    "detection_score": box.score,
                    "attribute_name": attr,
                }
                annos.append(nusc_anno)
            submissions_results[info['sample_token']] = annos

        submissions['results'] = submissions_results    

        res_path = os.path.join(log_dir, "results_nusc_1.json")
        with open(res_path, "w") as f:
            json.dump(submissions, f)
        eval_main_file = os.path.join(cfg.ROOT_DIR, 'lib/core/nusc_eval.py')
        root_path = self.dataset_dir
        cmd = f"python3 {str(eval_main_file)} --root_path=\"{str(root_path)}\""
        cmd += f" --version={'v1.0-trainval'} --eval_version={'cvpr_2019'}"
        cmd += f" --res_path=\"{str(res_path)}\" --eval_set={'val'}"
        cmd += f" --output_dir=\"{LOG_FOUT_DIR}\""
        # use subprocess can release all nusc memory after evaluation
        subprocess.check_output(cmd, shell=True)
        os.system('rm \"%s\"'%res_path) # remove former result file

        with open(os.path.join(log_dir, "metrics_summary.json"), "r") as f:
            metrics = json.load(f)
        return metrics 


    def evaluate_recall(self, sess, feeddict_producer, pred_list, val_size, cls_thresh, log_dir, placeholders=None):
        pass


    def logger_and_select_best_map(self, metrics, log_string):
        detail = {}
        result = f"Nusc v1.0-trainval Evaluation\n"
        final_score = []
        for name in self.cls_list:
            detail[name] = {}
            for k, v in metrics["label_aps"][name].items():
                detail[name][f"dist@{k}"] = v
            tp_errs = []
            tp_names = []
            for k, v in metrics["label_tp_errors"][name].items():
                detail[name][k] = v
                tp_errs.append(f"{v:.4f}")
                tp_names.append(k)
            threshs = ', '.join(list(metrics["label_aps"][name].keys()))
            scores = list(metrics["label_aps"][name].values())
            final_score.append(np.mean(scores))
            scores = ', '.join([f"{s * 100:.2f}" for s in scores])
            result += f"{name} Nusc dist AP@{threshs} and TP errors\n"
            result += scores
            result += "\n"
            result += "mAP: %0.2f\n" % (np.mean(list(metrics["label_aps"][name].values())) * 100)
            result += ', '.join(tp_names) + ": " + ', '.join(tp_errs)
            result += "\n"
        result += 'NDS score: %0.2f\n' % (metrics['nd_score'] * 100)
        log_string(result)

        cur_result = metrics['nd_score']
        return cur_result

    def logger_and_select_best_recall(self, metrics, log_string):
        pass


    # save prediction results
    def save_predictions(self, sess, feeddict_producer, pred_list, val_size, cls_thresh, log_dir, placeholders=None):
        pass
