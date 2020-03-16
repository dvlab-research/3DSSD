import os, sys
import tqdm
import tensorflow as tf
import numpy as np

from core.config import cfg
from utils.tf_ops.evaluation.tf_evaluate import calc_iou 

import utils.kitti_object as kitti_object
import utils.box_3d_utils as box_3d_utils

sess = tf.Session()

ROOT_DIR = cfg.ROOT_DIR
val_list_path = os.path.join(ROOT_DIR, cfg.DATASET.KITTI.VAL_LIST)
dataset_dir = os.path.join(cfg.ROOT_DIR, cfg.DATASET.KITTI.BASE_DIR_PATH)

dataset = kitti_object.kitti_object(dataset_dir, 'training')
label_dir = os.path.join(dataset_dir, 'training', 'label_2')

with open(val_list_path, 'r') as f:
    val_list = [int(line.strip('\n')) for line in f.readlines()]

sess = tf.Session()

for val_idx in tqdm.tqdm(val_list):
    label_name = os.path.join(label_dir, '%06d.txt'%val_idx)

    label_objects = dataset.get_label_objects(val_idx)
    # then cast these objects to box_3d
    label_3d = [box_3d_utils.object_label_to_box_3d(obj) for obj in label_objects]
    label_3d = np.array(label_3d)
    label_3d = np.expand_dims(label_3d, axis=0)

    iou_bev, iou_3d = calc_iou(label_3d, label_3d)        
    print(sess.run(iou_3d))
