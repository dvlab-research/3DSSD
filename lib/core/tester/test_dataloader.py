import sys, os
import tensorflow as tf
import numpy as np
import argparse

from core.config import cfg, cfg_from_file, cfg_from_list, assert_and_infer_cfg
from dataset.dataloader import choose_dataset
from dataset.feeddict_builder import FeedDictCreater
from modeling import choose_model


def parse_args():
    parser = argparse.ArgumentParser(description='Trainer')
    parser.add_argument('--cfg', required=True, help='Config file for training')
    parser.add_argument('--img_list', default='train', help='Train/Val/Trainval list')
    parser.add_argument('--split', default='training', help='Dataset split')
    parser.add_argument('--restore_model_path', default=None, help='Restore model path e.g. log/model.ckpt [default: None]')
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    args = parse_args()
    cfg_from_file(args.cfg)

    batch_size = cfg.TRAIN.CONFIG.BATCH_SIZE
    gpu_num = cfg.TRAIN.CONFIG.GPU_NUM

    # first choose dataset
    dataset_func = choose_dataset()
    dataset = dataset_func('loading', split=args.split, img_list=args.img_list, is_training=True, workers_num=cfg.DATA_LOADER.NUM_THREADS)
    dataset_iter = dataset.load_batch(batch_size * gpu_num)

    # load items
    sample = next(dataset_iter, None)
    points, sem_labels, sem_dists, label_boxes_3d, ry_cls_label, residual_angle, label_classes, calib_P, sample_name = sample
    dbg_dict = dict()
    dbg_dict['pc'] = points 
    dbg_dict['sem_labels'] = sem_labels
    dbg_dict['label_boxes_3d'] = label_boxes_3d
    dbg_dict['label_classes'] = label_classes
    dbg_dict['angle_cls'] = ry_cls_label
    dbg_dict['angle_res'] = residual_angle
    np.save('dbg_input.npy', dbg_dict)
   
    
    
