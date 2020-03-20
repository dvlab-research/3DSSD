import tensorflow as tf
import sys, os
import numpy as np
import cv2
import itertools
import tqdm

import utils.kitti_object as kitti_object
import utils.kitti_util as kitti_util
import utils.box_3d_utils as box_3d_utils
import dataset.maps_dict as maps_dict

from core.config import cfg
from builder.data_augmentor import DataAugmentor
from utils.anchor_encoder import encode_angle2class_np
from utils.points_filter import * 
from utils.voxelnet_aug import check_inside_points
from utils.anchors_util import project_to_image_space_corners
from utils.tf_ops.evaluation.tf_evaluate import evaluate
from dataset.data_provider.data_provider import DataFromList, MultiProcessMapData, BatchDataNuscenes

class KittiDataset:
    """
    Kitti dataset loader and producer
    """
    def __init__(self, mode, split='training', img_list='trainval', is_training=True, workers_num=1):
        """
        mode: 'loading', 'preprocessing'
        """
        self.mode = mode
        self.dataset_dir = os.path.join(cfg.ROOT_DIR, cfg.DATASET.KITTI.BASE_DIR_PATH)
        self.label_dir = os.path.join(cfg.DATASET.KITTI.BASE_DIR_PATH, split, 'label_2')
        self.kitti_object = kitti_object.kitti_object(self.dataset_dir, split)
        self.is_training = is_training
        self.img_list = img_list
        self.workers_num = workers_num

        self.cls_list = cfg.DATASET.KITTI.CLS_LIST
        self.idx2cls_dict = dict([(idx+1, cls) for idx, cls in enumerate(self.cls_list)])
        self.cls2idx_dict = dict([(cls, idx+1) for idx, cls in enumerate(self.cls_list)])

        # formulate save data_dir
        base_dir = ''
        if not cfg.TEST.WITH_GT:
            base_dir += 'no_gt/'
        self.sv_npy_path = os.path.join(cfg.ROOT_DIR, cfg.DATASET.KITTI.SAVE_NUMPY_PATH, base_dir + self.img_list, '{}'.format(self.cls_list))

        self.train_list = os.path.join(self.sv_npy_path, 'train_list.txt')

        if mode == 'loading':
            # data loader
            with open(self.train_list, 'r') as f:
                self.train_npy_list = [line.strip('\n') for line in f.readlines()]
            self.train_npy_list = np.array(self.train_npy_list)
            self.sample_num = len(self.train_npy_list)
            self.data_augmentor = DataAugmentor('KITTI', workers_num=self.workers_num)

        elif mode == 'preprocessing':
            # preprocess raw data
            if img_list == 'train':
                list_path = os.path.join(cfg.ROOT_DIR, cfg.DATASET.KITTI.TRAIN_LIST)
            elif img_list == 'val':
                list_path = os.path.join(cfg.ROOT_DIR, cfg.DATASET.KITTI.VAL_LIST)
            elif img_list == 'trainval':
                list_path = os.path.join(cfg.ROOT_DIR, cfg.DATASET.KITTI.TRAINVAL_LIST)
            else:
                list_path = os.path.join(cfg.ROOT_DIR, cfg.DATASET.KITTI.TEST_LIST)
            with open(list_path, 'r') as f:
                self.idx_list = [line.strip('\n') for line in f.readlines()]
            self.sample_num = len(self.idx_list)

            self.extents = cfg.DATASET.POINT_CLOUD_RANGE
            self.extents = np.reshape(self.extents, [3, 2])
            if not os.path.exists(self.sv_npy_path): os.makedirs(self.sv_npy_path)

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
        cur_npy = self.train_npy_list[sample_idx]
       
        cur_npy_path = os.path.join(self.sv_npy_path, cur_npy)
        sample_dict = np.load(cur_npy_path).tolist()

        sem_labels = sample_dict[maps_dict.KEY_LABEL_SEMSEG]
        sem_dists = sample_dict[maps_dict.KEY_LABEL_DIST]
        points = sample_dict[maps_dict.KEY_POINT_CLOUD]
        calib = sample_dict[maps_dict.KEY_STEREO_CALIB]

        if self.is_training or cfg.TEST.WITH_GT:
            label_boxes_3d = sample_dict[maps_dict.KEY_LABEL_BOXES_3D]
            label_classes = sample_dict[maps_dict.KEY_LABEL_CLASSES]
            cur_label_num = sample_dict[maps_dict.KEY_LABEL_NUM]
            ry_cls_label, residual_angle = encode_angle2class_np(label_boxes_3d[:, -1], num_class=cfg.MODEL.ANGLE_CLS_NUM)
        else:
            label_boxes_3d = np.zeros([1, 7], np.float32)
            label_classes = np.zeros([1], np.int32)
            cur_label_num = 1
            ry_cls_label = np.zeros([1], np.int32)
            residual_angle = np.zeros([1], np.float32)

        if self.is_training: # then add data augmentation here
            # get plane first
            sample_name = sample_dict[maps_dict.KEY_SAMPLE_NAME]
            plane = self.kitti_object.get_planes(sample_name) 
            points, sem_labels, sem_dists, label_boxes_3d, label_classes = self.data_augmentor.kitti_forward(points, sem_labels, sem_dists, label_boxes_3d, label_classes, plane, pipename)
            cur_label_num = len(label_boxes_3d)
            ry_cls_label, residual_angle = encode_angle2class_np(label_boxes_3d[:, -1], num_class=cfg.MODEL.ANGLE_CLS_NUM)

        # randomly choose points
        pts_num = points.shape[0]
        pts_idx = np.arange(pts_num)
        if pts_num >= cfg.MODEL.POINTS_NUM_FOR_TRAINING:
            sampled_idx = np.random.choice(pts_idx, cfg.MODEL.POINTS_NUM_FOR_TRAINING, replace=False)
        else:
            # pts_num < model_util.points_num_for_training
            # first random choice pts_num, replace=False
            sampled_idx_1 = np.random.choice(pts_idx, pts_num, replace=False)
            sampled_idx_2 = np.random.choice(pts_idx, cfg.MODEL.POINTS_NUM_FOR_TRAINING - pts_num, replace=True)
            sampled_idx = np.concatenate([sampled_idx_1, sampled_idx_2], axis=0)

        sem_labels = sem_labels[sampled_idx]
        sem_dists = sem_dists[sampled_idx]
        points = points[sampled_idx, :]

        biggest_label_num = max(biggest_label_num, cur_label_num)

        return biggest_label_num, points, sem_labels, sem_dists, label_boxes_3d, ry_cls_label, residual_angle, label_classes, calib.P, sample_dict[maps_dict.KEY_SAMPLE_NAME]

    def load_batch(self, batch_size): 
        """
        make data with batch_size per thread
        """
        perm = np.arange(self.sample_num).tolist() # a list indicates each data
        dp = DataFromList(perm, is_train=self.is_training, shuffle=self.is_training)
        dp = MultiProcessMapData(dp, self.load_samples, self.workers_num)

        use_concat = [0, 0, 0, 2, 2, 2, 2, 0, 0]
        dp = BatchDataNuscenes(dp, batch_size, use_concat=use_concat)
        dp.reset_state()
        dp = dp.get_data()
        return dp


    # Preprocess data
    def preprocess_samples(self, indices):
        sample_dicts = []
        biggest_label_num = 0
        for sample_idx in indices:
            sample_id = int(self.idx_list[sample_idx])
            
            img = self.kitti_object.get_image(sample_id)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            image_shape = img.shape
 
            calib = self.kitti_object.get_calibration(sample_id)

            points = self.kitti_object.get_lidar(sample_id)
            points_intensity = points[:, 3:]
            points = points[:, :3]
            # filter out this, first cast it to rect
            points = calib.project_velo_to_rect(points)

            img_points_filter = get_point_filter_in_image(points, calib, image_shape[0], image_shape[1])
            voxelnet_points_filter = get_point_filter(points, self.extents)
            img_points_filter = np.logical_and(img_points_filter, voxelnet_points_filter)
            img_points_filter = np.where(img_points_filter)[0]
            points = points[img_points_filter]
            points_intensity = points_intensity[img_points_filter]

            if self.img_list in ['train', 'val', 'trainval'] and cfg.TEST.WITH_GT:
                # then we also need to preprocess groundtruth
                objs = self.kitti_object.get_label_objects(sample_id)
                filtered_obj_list = [obj for obj in objs if obj.type in self.cls_list]
                if len(filtered_obj_list) == 0:
                    # continue if no obj
                    return None, biggest_label_num

                # then is time to generate anchors
                label_boxes_3d = np.array([box_3d_utils.object_label_to_box_3d(obj) for obj in filtered_obj_list])
                label_boxes_3d = np.reshape(label_boxes_3d, [-1, 7])
                label_classes = np.array([self.cls2idx_dict[obj.type] for obj in filtered_obj_list], np.int)

                # then calculate sem_labels and sem_dists
                tmp_label_boxes_3d = label_boxes_3d.copy()
                # expand by 0.1, so as to cover context information
                tmp_label_boxes_3d[:, 3:-1] += cfg.TRAIN.AUGMENTATIONS.EXPAND_DIMS_LENGTH
                points_mask = check_inside_points(points, tmp_label_boxes_3d) # [pts_num, gt_num]
                points_cls_index = np.argmax(points_mask, axis=1) # [pts_num]
                points_cls_index = label_classes[points_cls_index] # [pts_num]
                sem_labels = np.max(points_mask, axis=1) * points_cls_index # [pts_num]
                sem_labels = sem_labels.astype(np.int)
                sem_dists = np.ones_like(sem_labels).astype(np.float32)
            else:
                sem_labels = np.ones([points.shape[0]], dtype=np.int)
                sem_dists = np.ones([points.shape[0]], dtype=np.float32)

            points = np.concatenate([points, points_intensity], axis=-1)
 
            if np.sum(sem_labels) == 0:
                return None, biggest_label_num

            # finally return the sealed result and save as npy file 
            if self.img_list in ['train', 'val', 'trainval'] and cfg.TEST.WITH_GT:
                sample_dict = {
                    maps_dict.KEY_LABEL_BOXES_3D: label_boxes_3d,
                    maps_dict.KEY_LABEL_CLASSES: label_classes,
                    maps_dict.KEY_LABEL_SEMSEG: sem_labels,
                    maps_dict.KEY_LABEL_DIST: sem_dists,

                    maps_dict.KEY_POINT_CLOUD: points,
                    maps_dict.KEY_STEREO_CALIB: calib,

                    maps_dict.KEY_SAMPLE_NAME: sample_id,
                    maps_dict.KEY_LABEL_NUM: len(label_boxes_3d)
                }
                biggest_label_num = max(len(label_boxes_3d), biggest_label_num)
            else:
                # img_list is test
                sample_dict = {
                    maps_dict.KEY_LABEL_SEMSEG: sem_labels,
                    maps_dict.KEY_LABEL_DIST: sem_dists,
                    maps_dict.KEY_POINT_CLOUD: points,
                    maps_dict.KEY_STEREO_CALIB: calib,
                    maps_dict.KEY_SAMPLE_NAME: sample_id
                }
            sample_dicts.append(sample_dict)
        return sample_dicts, biggest_label_num


    def generate_mixup_sample(self, sample_dict):
        label_boxes_3d = sample_dict[maps_dict.KEY_LABEL_BOXES_3D]
        label_classes = sample_dict[maps_dict.KEY_LABEL_CLASSES]
        points = sample_dict[maps_dict.KEY_POINT_CLOUD]
        label_class_names = np.array([self.idx2cls_dict[label] for label in label_classes])

        tmp_label_boxes_3d = label_boxes_3d.copy()
        # expand by 0.1, so as to cover context information
        tmp_label_boxes_3d[:, 3:-1] += cfg.TRAIN.AUGMENTATIONS.EXPAND_DIMS_LENGTH
        points_mask = check_inside_points(points, tmp_label_boxes_3d) # [pts_num, gt_num]

        pts_num_inside_box = np.sum(points_mask, axis=0) # gt_num
        valid_box_idx = np.where(pts_num_inside_box >= cfg.DATASET.MIN_POINTS_NUM)[0]
        if len(valid_box_idx) == 0: return None

        valid_label_boxes_3d = label_boxes_3d[valid_box_idx, :]
        valid_label_classes = label_class_names[valid_box_idx]

        sample_dicts = []
        for index, i in enumerate(valid_box_idx):
            cur_points_mask = points_mask[:, i]
            cur_points_idx = np.where(cur_points_mask)[0]
            cur_inside_points = points[cur_points_idx, :]
            sample_dict = {
                maps_dict.KEY_SAMPLED_GT_POINTS: cur_inside_points,
                maps_dict.KEY_SAMPLED_GT_LABELS_3D: valid_label_boxes_3d[index],
                maps_dict.KEY_SAMPLED_GT_CLSES: valid_label_classes[index],
            }
            sample_dicts.append(sample_dict)
        return sample_dicts
                

    def preprocess_batch(self):
        # if create_gt_dataset, then also create a boxes_numpy, saving all points
        if cfg.TRAIN.AUGMENTATIONS.MIXUP.OPEN: # also save mixup database
            mixup_label_dict = dict([(cls, []) for cls in self.mixup_db_class])
        with open(self.train_list, 'w') as f:
            for i in tqdm.tqdm(range(0, self.sample_num)):
                sample_dicts, tmp_biggest_label_num = self.preprocess_samples([i])
                if sample_dicts is None:
                    # print('%s has no ground truth or ground truth points'%self.idx_list[i].name)
                    continue
                # else save the result
                f.write('%06d.npy\n'%i)
                np.save(os.path.join(self.sv_npy_path, '%06d.npy'%i), sample_dicts[0])

                # create_gt_dataset
                if self.img_list in ['train', 'val', 'trainval'] and cfg.TEST.WITH_GT and cfg.TRAIN.AUGMENTATIONS.MIXUP.OPEN:
                    # then also parse the sample_dicts so as to generate mixup database
                    mixup_sample_dicts = self.generate_mixup_sample(sample_dicts[0])
                    if mixup_sample_dicts is None: continue
                    for mixup_sample_dict in mixup_sample_dicts:
                        cur_cls = mixup_sample_dict[maps_dict.KEY_SAMPLED_GT_CLSES]
                        mixup_label_dict[cur_cls].append(mixup_sample_dict)

        if self.img_list in ['train', 'val', 'trainval'] and cfg.TEST.WITH_GT and cfg.TRAIN.AUGMENTATIONS.MIXUP.OPEN:
            for cur_cls_name, mixup_sample_dict in mixup_label_dict.items():
                cur_mixup_db_cls_path = self.mixup_db_cls_path[cur_cls_name]
                cur_mixup_db_trainlist_path= self.mixup_db_trainlist_path[cur_cls_name]
                with open(cur_mixup_db_trainlist_path, 'w') as f:
                    for tmp_idx, tmp_cur_mixup_sample_dict in enumerate(mixup_sample_dict):
                        f.write('%06d.npy\n'%tmp_idx)
                        np.save(os.path.join(cur_mixup_db_cls_path, '%06d.npy'%tmp_idx), tmp_cur_mixup_sample_dict)
        print('Ending of the preprocess !!!')


    # Evaluation
    def set_evaluation_tensor(self, model):
        # get prediction results, bs = 1
        pred_bbox_3d = tf.squeeze(model.output[maps_dict.PRED_3D_BBOX][-1], axis=0)
        pred_cls_score = tf.squeeze(model.output[maps_dict.PRED_3D_SCORE][-1], axis=0)
        pred_cls_category = tf.squeeze(model.output[maps_dict.PRED_3D_CLS_CATEGORY][-1], axis=0)
        pred_list = [pred_bbox_3d, pred_cls_score, pred_cls_category]

        return pred_list

    def evaluation(self, sess, feeddict_producer, pred_list, val_size, cls_thresh, log_dir):
        obj_detection_list = []
        obj_detection_num = []
        obj_detection_name = []
        for i in tqdm.tqdm(range(val_size)):
            feed_dict = feeddict_producer.create_feed_dict()

            pred_bbox_3d_op, pred_cls_score_op, pred_cls_category_op = sess.run(pred_list, feed_dict=feed_dict) 

            calib_P, sample_name = feeddict_producer.info
            sample_name = int(sample_name[0])
            calib_P = calib_P[0]

            select_idx = np.where(pred_cls_score_op >= cls_thresh)[0]
            pred_cls_score_op = pred_cls_score_op[select_idx]
            pred_cls_category_op = pred_cls_category_op[select_idx]
            pred_bbox_3d_op = pred_bbox_3d_op[select_idx]
            pred_bbox_corners_op = box_3d_utils.get_box3d_corners_helper_np(pred_bbox_3d_op[:, :3], pred_bbox_3d_op[:, -1], pred_bbox_3d_op[:, 3:-1]) 
            pred_bbox_2d = project_to_image_space_corners(pred_bbox_corners_op, calib_P)

            obj_num = len(pred_bbox_3d_op)

            obj_detection = np.zeros([obj_num, 14], np.float32)
            if 'Car' not in self.cls_list:
                pred_cls_category_op += 1
            obj_detection[:, 0] = pred_cls_category_op 
            obj_detection[:, 1:5] = pred_bbox_2d
            obj_detection[:, 6:9] = pred_bbox_3d_op[:, :3]
            obj_detection[:, 9] = pred_bbox_3d_op[:, 4] # h
            obj_detection[:, 10] = pred_bbox_3d_op[:, 5] # w
            obj_detection[:, 11] = pred_bbox_3d_op[:, 3] # l
            obj_detection[:, 12] = pred_bbox_3d_op[:, 6] # ry
            obj_detection[:, 13] = pred_cls_score_op 

            obj_detection_list.append(obj_detection)
            obj_detection_name.append(os.path.join(self.label_dir, '%06d.txt'%sample_name))
            obj_detection_num.append(obj_num)
 
        obj_detection_list = np.concatenate(obj_detection_list, axis=0)
        obj_detection_name = np.array(obj_detection_name, dtype=np.string_)
        obj_detection_num = np.array(obj_detection_num, dtype=np.int)          
        
        precision_img, aos_img, precision_ground, aos_ground, precision_3d, aos_3d = evaluate(obj_detection_list, obj_detection_name, obj_detection_num)
        precision_img_op, aos_img_op, precision_ground_op, aos_ground_op, precision_3d_op, aos_3d_op = sess.run([precision_img, aos_img, precision_ground, aos_ground, precision_3d, aos_3d])

        result_list = [precision_img_op, aos_img_op, precision_ground_op, aos_ground_op, precision_3d_op, aos_3d_op] 
        return result_list 

    def logger_and_select_best(self, result_list, log_string):
        """
            log_string: a function to print final result
        """
        precision_img_op, aos_img_op, precision_ground_op, aos_ground_op, precision_3d_op, aos_3d_op = result_list

        log_string('precision_image:')
        # [NUM_CLASS, E/M/H], NUM_CLASS: Car, Pedestrian, Cyclist
        precision_img_res = precision_img_op[:, :, 1:]
        precision_img_res = np.sum(precision_img_res, axis=-1) / 40.
        # precision_img_res = precision_img_op[:, :, ::4]
        # precision_img_res = np.sum(precision_img_res, axis=-1) / 11.
        log_string(str(precision_img_res))

        log_string('precision_ground:')
        precision_ground_res = precision_ground_op[:, :, 1:]
        precision_ground_res = np.sum(precision_ground_res, axis=-1) / 40.
        # precision_ground_res = precision_ground_op[:, :, ::4]
        # precision_ground_res = np.sum(precision_ground_res, axis=-1) / 11.
        log_string(str(precision_ground_res))

        log_string('precision_3d:')
        precision_3d_res = precision_3d_op[:, :, 1:]
        precision_3d_res = np.sum(precision_3d_res, axis=-1) / 40.
        # precision_3d_res = precision_3d_op[:, :, ::4]
        # precision_3d_res = np.sum(precision_3d_res, axis=-1) / 11.
        log_string(str(precision_3d_res))

        if 'Car' in self.cls_list:
            cur_result = precision_3d_res[0, 1]
        else: # Pedestrian and Cyclist
            cur_result = (precision_3d_res[1, 1] + precision_3d_res[2, 1]) / 2.

        return cur_result
