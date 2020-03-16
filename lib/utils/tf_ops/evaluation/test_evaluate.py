import os, sys
import tqdm
import tensorflow as tf
import numpy as np

from core.config import cfg
from utils.tf_ops.evaluation.tf_evaluate import evaluate
import utils.kitti_object as kitti_object

sess = tf.Session()

ROOT_DIR = cfg.ROOT_DIR
val_list_path = os.path.join(ROOT_DIR, cfg.DATASET.KITTI.VAL_LIST)
dataset_dir = os.path.join(cfg.ROOT_DIR, cfg.DATASET.KITTI.BASE_DIR_PATH)

dataset = kitti_object.kitti_object(dataset_dir, 'training')
label_dir = os.path.join(dataset_dir, 'training', 'label_2')

with open(val_list_path, 'r') as f:
    val_list = [int(line.strip('\n')) for line in f.readlines()]

label_dict = dict([('Car', 0), ('Pedestrian', 1), ('Cyclist', 2)])

obj_detection_list = []
obj_detection_num = []
obj_detection_name = []

i = 0

for val_idx in tqdm.tqdm(val_list):
    label_name = os.path.join(label_dir, '%06d.txt'%val_idx)
    obj_detection_name.append(label_name)

    label_objects = dataset.get_label_objects(val_idx)
    objects_num = len(label_objects)
    detect_num = 0
    obj_detections = []
    for i in range(objects_num):
        label_object = label_objects[i]
        if label_object.type not in label_dict.keys():
            continue
        obj_detection = np.zeros([14], np.float32)
        obj_detection[0] = label_dict[label_object.type]
        obj_detection[1:5] = [label_object.xmin, label_object.ymin, label_object.xmax, label_object.ymax]
        obj_detection[5] = 0
        obj_detection[6:9] = label_object.t
        obj_detection[9] = label_object.h
        obj_detection[10] = label_object.w
        obj_detection[11] = label_object.l
        obj_detection[12] = label_object.ry
        obj_detection[13] = 1
        obj_detections.append(obj_detection)
        detect_num += 1
    obj_detections = np.stack(obj_detections, axis=0)
    obj_detections = np.zeros([0, 14], np.float32)
    detect_num = 1
    obj_detection_num.append(detect_num)
    obj_detection_list.append(obj_detections)
    i += 1

obj_detection_list = np.concatenate(obj_detection_list, axis=0)
obj_detection_name = np.array(obj_detection_name, dtype=np.string_)
obj_detection_num = np.array(obj_detection_num, dtype=np.int)
        
precision_img, aos_img, precision_ground, aos_ground, precision_3d, aos_3d = evaluate(obj_detection_list, obj_detection_name, obj_detection_num)      
precision_img_op, aos_img_op, precision_ground_op, aos_ground_op, precision_3d_op, aos_3d_op = sess.run([precision_img, aos_img, precision_ground, aos_ground, precision_3d, aos_3d])

precision_img_res = precision_img_op[:, :, ::4]
print(precision_img_op)
precision_img_res = np.sum(precision_img_res, axis=-1) / 11.

precision_ground_res = precision_ground_op[:, :, ::4]
precision_ground_res = np.sum(precision_ground_res, axis=-1) / 11.

precision_3d_res = precision_3d_op[:, :, ::4]
precision_3d_res = np.sum(precision_3d_res, axis=-1) / 11.
    
print(precision_img_res)
print(precision_ground_res)
print(precision_3d_res)
