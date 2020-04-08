import sys, os
import tensorflow as tf
import numpy as np
import argparse

from utils.tf_ops.grouping.tf_grouping import *
from utils.voxelnet_aug import check_inside_points
from utils.generate_anchors import generate_3d_anchors_by_point_tf

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
    points = points[:, :, :3]

    anchors = generate_3d_anchors_by_point_tf(tf.constant(points, dtype=tf.float32), [[3.8, 1.6, 1.5]])
    anchors = tf.reshape(anchors, [batch_size, -1, 7])

    sess = tf.Session()
    batch_size = len(points)

    # # test point_mask calculation
    # mask = query_boxes_3d_mask(points, label_boxes_3d)
    # # then calculate gt_result
    # np_mask_list = [] 
    # mask_op = sess.run(mask)
    # for i in range(batch_size):
    #     cur_mask = check_inside_points(points[i], label_boxes_3d[i])
    #     for j in range(len(label_boxes_3d[0])):
    #         a = len(np.where(mask_op[i, j] != 0)[0])
    #         b = len(np.where(cur_mask[:, j] != 0)[0])
    #         if a!= b: 
    #             print(np.where(mask_op[i, j] != 0)[0])
    #             print(np.where(cur_mask[:, j] != 0)[0])
    #             print(1)
    # exit()

    # test query_boxes
    # idx, pts_cnt = query_boxes_3d_points(64, points, label_boxes_3d)
    # queried_point = group_point(points, idx) # bs, gt_num, 64, 3
    # print(queried_point)
    # # bs, proposal_num, nsample, 3
    # queried_point, pts_cnt_op = sess.run([queried_point, pts_cnt]) 
    # not_in_sum = 0
    # for i in range(batch_size):
    #     for j in range(len(label_boxes_3d[0])):
    #         cur_gt_boxes = label_boxes_3d[i, j] 
    #         cur_pts_cnt = pts_cnt_op[i, j]
    #         # pts_num, 1
    #         cur_mask = np.squeeze(check_inside_points(queried_point[i, j], cur_gt_boxes[np.newaxis, :]), axis=-1)
    #         not_in = len(np.where(cur_mask == 0)[0])
    #         not_in_sum += not_in * (cur_pts_cnt > 0) 
    #         print(cur_pts_cnt, cur_pts_cnt > 0, not_in)
    # print(not_in_sum)
    # exit()

    # finally test point iou
    gt_num = label_boxes_3d.shape[1]
    anchors_num = anchors.get_shape().as_list()[1]
    iou_matrix = np.ones([batch_size, anchors_num, gt_num], dtype=label_boxes_3d.dtype)
    iou_points = query_points_iou(points, anchors, label_boxes_3d, iou_matrix)
    iou_points, anchors = sess.run([iou_points, anchors])
    iou_points_np = np.zeros([batch_size, anchors_num, gt_num], dtype=np.float32)
    for i in range(batch_size):
        cur_anchors = anchors[i]
        cur_gt_boxes = label_boxes_3d[i]
        cur_anchors_mask = check_inside_points(points[i], cur_anchors) # pts_num, anchors_num
        cur_gt_mask = check_inside_points(points[i], cur_gt_boxes) # pts_num, gt_num
        intersect = cur_anchors_mask[:, :, np.newaxis] * cur_gt_mask[:, np.newaxis, :]
        union = np.logical_or(cur_anchors_mask[:, :, np.newaxis], cur_gt_mask[:, np.newaxis, :]).astype(np.float32)
        cur_iou = np.sum(intersect, axis=0) / np.maximum(np.sum(union, axis=0), 1.)
        iou_points_np[i] = cur_iou
    print(np.where((iou_points - iou_points_np) != 0))
    
