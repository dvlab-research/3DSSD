import numpy as np
import tensorflow as tf

from core.config import cfg
from utils.tf_ops.grouping.tf_grouping import group_point, query_points_iou
from utils.tf_ops.evaluation.tf_evaluate import calc_iou
import np_functions.gt_sampler as gt_sampler

class TargetAssigner:
    def __init__(self, stage):
        """
        stage: TargetAssigner of stage1 or stage2
        """
        if stage == 0:
            cur_cfg_file = cfg.MODEL.FIRST_STAGE 
        elif stage == 1:
            cur_cfg_file = cfg.MODEL.SECOND_STAGE 
        else:
            raise Exception('Not Implementation Error')

        self.assign_method = cur_cfg_file.ASSIGN_METHOD
        self.iou_sample_type = cur_cfg_file.IOU_SAMPLE_TYPE

        # some parameters
        self.minibatch_size = cur_cfg_file.MINIBATCH_NUM
        self.positive_ratio = cur_cfg_file.MINIBATCH_RATIO 
        self.pos_iou = cur_cfg_file.CLASSIFICATION_POS_IOU
        self.neg_iou = cur_cfg_file.CLASSIFICATION_NEG_IOU
        self.effective_sample_range = cur_cfg_file.CLASSIFICATION_LOSS.SOFTMAX_SAMPLE_RANGE


        if self.assign_method == 'IoU':
            self.assign_targets_anchors = self.iou_assign_targets_anchors
        elif self.assign_method == 'Mask':
            self.assign_targets_anchors = self.mask_assign_targets_anchors

    def assign(self, points, anchors_3d, gt_boxes_3d, gt_labels, gt_angle_cls, gt_angle_res, gt_velocity, gt_attribute, valid_mask=None):
        """
        points: [bs, points_num, 3]
        anchors_3d: [bs, points_num, cls_num, 7]
        gt_boxes_3d: [bs, gt_num, 7]
        gt_labels: [bs, gt_num]
        gt_angle_cls: [bs, gt_num]
        gt_angle_res: [bs, gt_num]
        gt_velocity: [bs, gt_num, 2]
        gt_attribute: [bs, gt_num]

        return: [bs, points_num, cls_num]
        """
        bs, points_num, cls_num, _ = anchors_3d.get_shape().as_list()

        if valid_mask is None:
            valid_mask = tf.ones([bs, points_num, cls_num], dtype=points.dtype)

        assigned_idx, assigned_pmask, assigned_nmask = self.assign_targets_anchors(points, anchors_3d, gt_boxes_3d, gt_labels, valid_mask) # [bs, points_num, cls_num] 

        assigned_gt_labels = self.gather_class(gt_labels, assigned_idx) # [bs, points_num, cls_num]
        assigned_gt_labels = assigned_gt_labels * tf.cast(assigned_pmask, tf.int32)
        assigned_gt_labels = tf.reduce_sum(assigned_gt_labels, axis=-1)

        assigned_gt_boxes_3d = group_point(gt_boxes_3d, assigned_idx) 
        assigned_gt_angle_cls = self.gather_class(gt_angle_cls, assigned_idx)
        assigned_gt_angle_res = self.gather_class(gt_angle_res, assigned_idx)

        if gt_velocity is not None:
            # bs, npoint, cls_num, 2 
            assigned_gt_velocity = group_point(gt_velocity, assigned_idx)
        else: assigned_gt_velocity = None

        if gt_attribute is not None:
            # bs, npoint, cls_num
            assigned_gt_attribute = self.gather_class(gt_attribute, assigned_idx)
        else: assigned_gt_attribute = None

        returned_list = [assigned_idx, assigned_pmask, assigned_nmask, assigned_gt_boxes_3d, assigned_gt_labels, assigned_gt_angle_cls, assigned_gt_angle_res, assigned_gt_velocity, assigned_gt_attribute]

        return returned_list



    def gather_class(self, gt_labels, assigned_idx):
        # [bs, gt_num] -> [bs, points_num, cls_num]
        gt_labels_dtype = gt_labels.dtype
        gt_labels_f = tf.expand_dims(tf.cast(gt_labels, tf.float32), axis=-1)
        assigned_gt_labels = group_point(gt_labels_f, assigned_idx)
        assigned_gt_labels = tf.squeeze(tf.cast(assigned_gt_labels, gt_labels_dtype), axis=-1)
        return assigned_gt_labels
        

    def iou_assign_targets_anchors(self, points, anchors_3d, gt_boxes_3d, gt_labels, valid_mask):
        """
        Assign targets for each anchor
        points: [bs, points_num, 3]
        anchors_3d: [bs, points_num, cls_num, 7]
        gt_boxes_3d: [bs, gt_boxes_3d, 7]
        gt_labels: [bs, gt_boxes_3d]
        valid_mask: [bs, points_num, cls_num]

        Return:
        assigned_idx: [bs, points_num, cls_num], int32, the index of groundtruth
        assigned_pmask: [bs, points_num, cls_num]
        assigned_nmask: [bs, points_num, cls_num]
        """
        # first calculate IoU
        bs, points_num, cls_num, _ = anchors_3d.get_shape().as_list()
        gt_num = tf.shape(gt_boxes_3d)[1]
        anchors_3d_reshape = tf.reshape(anchors_3d, [bs, points_num * cls_num, 7])

        # bs, pts_num * cls_num, gt_num
        iou_bev, iou_3d = calc_iou(anchors_3d_reshape, gt_boxes_3d)
        if self.iou_sample_type == 'BEV':
            iou_matrix = iou_bev
        elif self.iou_sample_type == '3D':
            iou_matrix = iou_3d 
        elif self.iou_sample_type == 'Point': # point_iou
            iou_matrix = query_points_iou(points, anchors_3d_reshape, gt_boxes_3d, iou_3d)
        iou_matrix = tf.reshape(iou_matrix, [bs, points_num, cls_num, gt_num])
        
        assigned_idx, assigned_pmask, assigned_nmask = tf.py_func(gt_sampler.iou_assign_targets_anchors_np, 
            [iou_matrix, points, anchors_3d, gt_boxes_3d, gt_labels, self.minibatch_size, self.positive_ratio, self.pos_iou, self.neg_iou, self.effective_sample_range, valid_mask], 
            [tf.int32, tf.float32, tf.float32])
 
        assigned_idx = tf.reshape(assigned_idx, [bs, points_num, cls_num])
        assigned_pmask = tf.reshape(assigned_pmask, [bs, points_num, cls_num])
        assigned_nmask = tf.reshape(assigned_nmask, [bs, points_num, cls_num])
        return assigned_idx, assigned_pmask, assigned_nmask


    def mask_assign_targets_anchors(self, points, anchors_3d, gt_boxes_3d, gt_labels, valid_mask):
        """
        Assign targets for each anchor
        points: [bs, points_num, 3]
        anchors_3d: [bs, points_num, cls_num, 3] centers of anchors
        gt_boxes_3d: [bs, gt_boxes_3d, 7]
        gt_labels: [bs, gt_boxes_3d]
        valid_mask: [bs, points_num, cls_num]

        Return:
        assigned_idx: [bs, points_num, cls_num], int32, the index of groundtruth
        assigned_pmask: [bs, points_num, cls_num]
        assigned_nmask: [bs, points_num, cls_num]
        """
        bs, points_num, cls_num, _ = anchors_3d.get_shape().as_list()
        gt_num = tf.shape(gt_boxes_3d)[1]

        # then let's calculate whether a point is within a gt_boxes_3d
        assigned_idx, assigned_pmask, assigned_nmask = tf.py_func(gt_sampler.mask_assign_targets_anchors_np, 
            [points, anchors_3d, gt_boxes_3d, gt_labels, self.minibatch_size, self.positive_ratio, self.pos_iou, self.neg_iou, self.effective_sample_range, valid_mask], 
            [tf.int32, tf.float32, tf.float32])

        assigned_idx = tf.reshape(assigned_idx, [bs, points_num, cls_num])
        assigned_pmask = tf.reshape(assigned_pmask, [bs, points_num, cls_num])
        assigned_nmask = tf.reshape(assigned_nmask, [bs, points_num, cls_num])
        return assigned_idx, assigned_pmask, assigned_nmask
