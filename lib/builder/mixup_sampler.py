import tensorflow as tf
import sys, os
import numpy as np
import cv2
import itertools

from core.config import cfg

class MixupSampler:
    def __init__(self, shuffle=True):
        """
        pc_list: train / val / trainval 
        cls_list: ['Car', 'Pedestrian', 'Cyclist']
        """
        self.base_dir = cfg.TRAIN.AUGMENTATIONS.MIXUP.SAVE_NUMPY_PATH
        self.pc_list = cfg.TRAIN.AUGMENTATIONS.MIXUP.PC_LIST
        self.cls_list = cfg.TRAIN.AUGMENTATIONS.MIXUP.CLASS
        self.num_list = cfg.TRAIN.AUGMENTATIONS.MIXUP.NUMBER

        self.sv_npy_cls_path = dict()
        self.sv_npy_cls_trainlist_path = dict()
        self.sv_npy_list = dict()
        self.sample_num_list = dict()
        self.sv_perm = dict()
        self.sv_idx_list = dict()

        for cls in self.cls_list:
            sv_npy_path = os.path.join(cfg.ROOT_DIR, cfg.DATASET.KITTI.SAVE_NUMPY_PATH, self.base_dir, self.pc_list, '{}'.format(cls))
            self.sv_npy_cls_path[cls] = sv_npy_path
            self.sv_npy_cls_trainlist_path[cls] = os.path.join(sv_npy_path, 'train_list.txt')
            with open(self.sv_npy_cls_trainlist_path[cls], 'r') as f:
                self.sv_npy_list[cls] = np.array([line.strip('\n') for line in f.readlines()])
            self.sample_num_list[cls] = len(self.sv_npy_list[cls])
            self.sv_perm[cls] = np.arange(self.sample_num_list[cls])
            self.sv_idx_list[cls] = 0

        if shuffle:
            for k, v in self.sv_perm.items():
                np.random.shuffle(v)
        self._shuffle = shuffle

    def _sample(self, num, cls):
        if self.sv_idx_list[cls] + num >= self.sample_num_list[cls]:
            ret = self.sv_perm[cls][self.sv_idx_list[cls]:].copy()
            self._reset(cls)
        else:
            ret = self.sv_perm[cls][self.sv_idx_list[cls]:self.sv_idx_list[cls] + num]
            self.sv_idx_list[cls] += num
        return ret

    def _reset(self, cls):
        if self._shuffle:
            np.random.shuffle(self.sv_perm[cls])
        self.sv_idx_list[cls] = 0

    def sample(self):
        return_dicts = []
        cls_list = self.cls_list
        num_list = self.num_list
        for cls, num in zip(cls_list, num_list):
            indices = self._sample(num, cls)
            for idx in indices:
                cur_npy_file = self.sv_npy_list[cls][idx]
                cur_npy_file = os.path.join(self.sv_npy_cls_path[cls], cur_npy_file)
                mat = np.load(cur_npy_file).tolist()
                return_dicts.append(mat)
        return return_dicts
