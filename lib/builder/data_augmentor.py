import tensorflow as tf
import numpy as np

import dataset.maps_dict as maps_dict
import utils.kitti_aug as kitti_aug
import utils.rotation_util as rotation_util

from builder.mixup_sampler import MixupSampler
from utils.voxelnet_aug import *

class DataAugmentor:
    def __init__(self, dataset, workers_num=1):
        """
        Data Augmentor,
        here are 4 data augmentation methods, Mix-up, single object transformation, random rotation, random scale
        pc_list: train/val/trainval
        cls_list: [Car, Pedestrian, Cyclist]
        dataset: NuScenes / KITTI / Lyft...
        """
        self.dataset = dataset # NuScenes / KITTI 

        self.cls_list = cfg.DATASET.KITTI.CLS_LIST
        self.idx2cls_dict = dict([(idx+1, cls) for idx, cls in enumerate(self.cls_list)])
        self.cls2idx_dict = dict([(cls, idx+1) for idx, cls in enumerate(self.cls_list)])

        # flip augmentation
        self.aug_flip = cfg.TRAIN.AUGMENTATIONS.FLIP
        self.aug_flip_prob = 0.5

        # mixup augmentation
        self.mixup = cfg.TRAIN.AUGMENTATIONS.MIXUP.OPEN
        if self.mixup:
            self.mixup_class = cfg.TRAIN.AUGMENTATIONS.MIXUP.CLASS
            self.mixup_number = cfg.TRAIN.AUGMENTATIONS.MIXUP.NUMBER
            self.workers_num = workers_num
            self.mixup_sampler_list = []
            for i in range(self.workers_num):
                mixup_sampler = MixupSampler()
                self.mixup_sampler_list.append(mixup_sampler)

        # Single object augmentaiton
        self.aug_type = cfg.TRAIN.AUGMENTATIONS.PROB_TYPE
        self.single_aug_prob = cfg.TRAIN.AUGMENTATIONS.PROB 


    def kitti_forward(self, points, sem_labels, sem_dists, label_boxes_3d, label_classes, plane, pipename):
        """
        Forward function of data augmentor
        points: [pts_num, c]
        sem_labels: [pts_num], whether a point is within an object
        sem_dists: [pts_num]
        label_boxes_3d: [gt_num, 7]
        label_classes: [gt_num]
        calib: calib object
        pipename: the thread name utilizing this function
        """
        # first Mixup DataAugmentation
        if self.mixup:
            mixup_result = self.kitti_mixup_sampling(points, sem_labels, sem_dists, label_boxes_3d, label_classes, plane, pipename)
            points, sem_labels, sem_dists, label_boxes_3d, label_classes, cur_label_num = mixup_result

        # randomly flip
        if self.aug_flip and np.random.rand() >= self.aug_flip_prob:
            points = kitti_aug.flip_points(points)
            label_boxes_3d = kitti_aug.flip_boxes_3d(label_boxes_3d)

        # then other data augmentation
        choice = np.random.rand(3)
        points_i = points[:, 3:]
        points = points[:, :3]

        if choice[0] <= self.single_aug_prob[0]:
            gt_boxes_mask = np.ones([label_boxes_3d.shape[0]], np.bool_)
            label_boxes_3d, points = noise_per_object_v3_(label_boxes_3d, points, valid_mask=gt_boxes_mask, rotation_perturb=cfg.TRAIN.AUGMENTATIONS.SINGLE_AUG.ROTATION_PERTURB, center_noise_std=cfg.TRAIN.AUGMENTATIONS.SINGLE_AUG.CENTER_NOISE_STD, random_scale_range=cfg.TRAIN.AUGMENTATIONS.SINGLE_AUG.RANDOM_SCALE_RANGE, global_random_rot_range=[0., 0.], scale_3_dims=cfg.TRAIN.AUGMENTATIONS.SINGLE_AUG.SCALE_3_DIMS, sem_labels=sem_labels)
            label_boxes_3d = label_boxes_3d[gt_boxes_mask]
            label_classes = label_classes[gt_boxes_mask]

        if choice[1] <= self.single_aug_prob[1]:
            random_angle = (np.random.rand() * 2 - 1) * cfg.TRAIN.AUGMENTATIONS.RANDOM_ROTATION_RANGE
            rot_matrix = rotation_util.roty(random_angle)
            points_transpose = np.transpose(points) # [3, n]
            box_3d_center = label_boxes_3d[:, :3]
            box_3d_center_transpose = np.transpose(box_3d_center) # [3, n]

            points = np.matmul(rot_matrix, points_transpose)
            points = np.transpose(points) # [n, 3]
            box_3d_center = np.matmul(rot_matrix, box_3d_center_transpose)
            box_3d_center = np.transpose(box_3d_center) # [n, 3]
            label_boxes_3d[:, :3] = box_3d_center
            label_boxes_3d[:, -1] += random_angle 
            
        if choice[2] <= self.single_aug_prob[2]:
            random_scale = (np.random.rand() * 2 - 1) * cfg.TRAIN.AUGMENTATIONS.RANDOM_SCALE_RANGE + 1
            points[:, :3] = points[:, :3] * random_scale
            label_boxes_3d[:, :6] = label_boxes_3d[:, :6] * random_scale

        points = np.concatenate([points, points_i], axis=1)

        label_boxes_3d, points, sem_labels, sem_dists = filter_points_boxes_3d(label_boxes_3d, points, sem_labels, sem_dists, enlarge_range=[0.5, 2.0, 0.5])
        return points, sem_labels, sem_dists, label_boxes_3d, label_classes
         


    def kitti_mixup_sampling(self, points, sem_labels, sem_dists, label_boxes_3d, label_classes, plane, pipename):
        mixup_sampler = self.mixup_sampler_list[pipename] 
        sampled_gt_dicts = mixup_sampler.sample()

        sampled_gt_inside_points = []
        sampled_gt_label_boxes_3d = []
        sampled_gt_label_classes = []
        for sampled_gt_dict in sampled_gt_dicts:
            sampled_gt_inside_points.append(sampled_gt_dict[maps_dict.KEY_SAMPLED_GT_POINTS])
            sampled_gt_label_boxes_3d.append(sampled_gt_dict[maps_dict.KEY_SAMPLED_GT_LABELS_3D])
            sampled_gt_label_classes.append(self.cls2idx_dict[sampled_gt_dict[maps_dict.KEY_SAMPLED_GT_CLSES]]) 

        sampled_gt_label_boxes_3d = np.stack(sampled_gt_label_boxes_3d, axis=0) # [15, 7]
        label_boxes_3d, label_classes, points, sem_labels, sem_dists = box_3d_collision_test(sampled_gt_label_boxes_3d, label_boxes_3d, sampled_gt_label_classes, label_classes, sampled_gt_inside_points, points, sem_labels, sem_dists, plane)

        return points, sem_labels, sem_dists, label_boxes_3d, label_classes, len(label_boxes_3d) 



    def nuscenes_forward(self, points, label_boxes_3d, label_classes, pipename, label_attributes, label_velocity, cur_sweep_points_num):
        """
        Forward function of data augmentor
        points: [pts_num, c]
        sem_labels: [pts_num], whether a point is within an object
        sem_dists: [pts_num]
        label_boxes_3d: [gt_num, 7]
        label_classes: [gt_num]
        calib: calib object
        pipename: the thread name utilizing this function
        """
        # first Mixup DataAugmentation
        if self.mixup:
            mixup_result = self.nuscenes_mixup_sampling(points, label_boxes_3d, label_classes, pipename, label_attributes, label_velocity, cur_sweep_points_num)
            points, label_boxes_3d, label_classes, label_attributes, label_velocity, cur_sweep_points_num = mixup_result

        # then other data augmentation
        choice = np.random.rand(3)
        points_i = points[:, 3:]
        points = points[:, :3]

        if choice[0] <= self.single_aug_prob[0]:
            # randomly flip
            points = kitti_aug.flip_points(points)
            label_boxes_3d = kitti_aug.flip_boxes_3d(label_boxes_3d)
            label_velocity[:, 0] = -label_velocity[:, 0]

        if choice[1] <= self.single_aug_prob[1]:
            random_angle = (np.random.rand() * 2 - 1) * cfg.TRAIN.AUGMENTATIONS.RANDOM_ROTATION_RANGE
            rot_matrix = kitti_util.roty(random_angle)

            points_transpose = np.transpose(points) # [3, n]
            box_3d_center = label_boxes_3d[:, :3]
            box_3d_center_transpose = np.transpose(box_3d_center) # [3, n]
            tmp_velocity_labels = np.zeros([label_velocity.shape[0], 3], dtype=label_velocity.dtype)
            tmp_velocity_labels[:, [0, 2]] = label_velocity # [n, 3]
            tmp_velocity_labels_transpose = np.transpose(tmp_velocity_labels) # [3, n]

            points = np.matmul(rot_matrix, points_transpose)
            points = np.transpose(points).astype(np.float32) # [n, 3]
            box_3d_center = np.matmul(rot_matrix, box_3d_center_transpose)
            box_3d_center = np.transpose(box_3d_center) # [n, 3]
            label_boxes_3d[:, :3] = box_3d_center
            label_boxes_3d[:, -1] += random_angle
            label_velocity = np.matmul(rot_matrix, tmp_velocity_labels_transpose)
            label_velocity = np.transpose(label_velocity) # [n, 3]
            label_velocity = label_velocity[:, [0, 2]]
            
        if choice[2] <= self.single_aug_prob[2]:
            random_scale = (np.random.rand() * 2 - 1) * cfg.TRAIN.AUGMENTATIONS.RANDOM_SCALE_RANGE + 1
            points[:, :3] = points[:, :3] * random_scale
            label_boxes_3d[:, :6] = label_boxes_3d[:, :6] * random_scale
            label_velocity = label_velocity * random_scale

        points = np.concatenate([points, points_i], axis=1)
        return points, label_boxes_3d, label_classes, label_attributes, label_velocity, cur_sweep_points_num


    def nuscenes_mixup_sampling(self, points, label_boxes_3d, label_classes, pipename, label_attributes, label_velocity, cur_sweep_points_num):
        mixup_sampler = self.mixup_sampler_list[pipename]
        sampled_gt_dicts = mixup_sampler.sample()

        # then choose these non-collision boxes
        sampled_gt_inside_points = []
        sampled_gt_label_boxes_3d = []
        sampled_gt_label_classes = []
        sampled_gt_label_attributes = []
        sampled_gt_label_velocity = []
        for sampled_gt_dict in sampled_gt_dicts:
            if points.shape[1] == 4:
                sampled_gt_inside_points.append(sampled_gt_dict[maps_dict.KEY_SAMPLED_GT_POINTS][:, [0,1,2,4]])
            else: # points have 5 channels
                sampled_gt_inside_points.append(sampled_gt_dict[maps_dict.KEY_SAMPLED_GT_POINTS])
            sampled_gt_label_boxes_3d.append(sampled_gt_dict[maps_dict.KEY_SAMPLED_GT_LABELS_3D])
            sampled_gt_label_classes.append(self.cls2idx_dict[sampled_gt_dict[maps_dict.KEY_SAMPLED_GT_CLSES]])
            sampled_gt_label_attributes.append(-1)
            sampled_gt_label_velocity.append(np.array([np.nan, np.nan]))

        sampled_gt_label_boxes_3d = np.stack(sampled_gt_label_boxes_3d, axis=0) # [15, 7]
        label_boxes_3d, label_classes, points, label_attributes, label_velocity, cur_sweep_points_num = box_3d_collision_test_nusc(sampled_gt_label_boxes_3d, label_boxes_3d, sampled_gt_label_classes, label_classes, sampled_gt_inside_points, sampled_gt_label_attributes, sampled_gt_label_velocity, points, label_attributes, label_velocity, cur_sweep_points_num=cur_sweep_points_num, enlarge_range=[0.2, 0.2, 0.2])

        return points, label_boxes_3d, label_classes, label_attributes, label_velocity, cur_sweep_points_num
       
