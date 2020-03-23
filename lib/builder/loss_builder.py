import tensorflow as tf
import numpy as np

from core.config import cfg
from utils.box_3d_utils import transfer_box3d_to_corners
from utils.tf_ops.grouping.tf_grouping import query_corners_point
from utils.tf_ops.sampling.tf_sampling import gather_point
from utils.rotation_util import rotate_points
from np_functions.gt_sampler import vote_targets_np

import utils.model_util as model_util
import dataset.maps_dict as maps_dict


class LossBuilder:
    def __init__(self, stage):
        if stage == 0:
            self.loss_cfg = cfg.MODEL.FIRST_STAGE
        elif stage == 1:
            self.loss_cfg = cfg.MODEL.SECOND_STAGE

        self.stage = stage
 
        self.cls_loss_type = self.loss_cfg.CLASSIFICATION_LOSS.TYPE
        self.ctr_ness_range = self.loss_cfg.CLASSIFICATION_LOSS.CENTER_NESS_LABEL_RANGE
        self.cls_activation = self.loss_cfg.CLS_ACTIVATION

        if self.cls_loss_type == 'Center-ness':
            assert self.cls_activation == 'Sigmoid'

        self.cls_list = cfg.DATASET.KITTI.CLS_LIST


    def forward(self, index, label_dict, pred_dict, placeholders, vote_loss=False, attr_velo_loss=False):
        self.cls_loss(index, label_dict, pred_dict)
        self.corner_loss(index, label_dict, pred_dict)
        self.offset_loss(index, label_dict, pred_dict)
        self.angle_loss(index, label_dict, pred_dict)

        if vote_loss:
            self.vote_loss(label_dict, pred_dict, placeholders)
        if attr_velo_loss:
            self.velo_attr_loss(index, label_dict, pred_dict)


    def cls_loss(self, index, label_dict, pred_dict):
        pmask = label_dict[maps_dict.GT_PMASK][index] # bs, pts_num, cls_num
        nmask = label_dict[maps_dict.GT_NMASK][index]
        gt_cls = label_dict[maps_dict.GT_CLS][index] # bs, pts_num

        cls_mask = pmask + nmask 
        cls_mask = tf.reduce_max(cls_mask, axis=-1) # [bs, pts_num]
 
        pred_cls = pred_dict[maps_dict.PRED_CLS][index] # bs, pts_num, c

        norm_param = tf.maximum(1., tf.reduce_sum(cls_mask))

        if self.cls_activation == 'Sigmoid':
            gt_cls = tf.cast(tf.one_hot(gt_cls - 1, depth=len(self.cls_list), on_value=1, off_value=0, axis=-1), tf.float32)

        if self.cls_loss_type == 'Is-Not': # Is or Not
            if self.cls_activation == 'Softmax':
                cls_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=gt_cls, logits=pred_cls)
            else: 
                cls_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=gt_cls, logits=pred_cls)
                cls_loss = tf.reduce_mean(cls_loss, axis=-1)
                
        elif self.cls_loss_type == 'Center-ness': # Center-ness label
            base_xyz = pred_dict[maps_dict.KEY_OUTPUT_XYZ][index]
            assigned_boxes_3d = label_dict[maps_dict.GT_BOXES_ANCHORS_3D][index]
            ctr_ness = self._generate_centerness_label(base_xyz, assigned_boxes_3d, pmask)
            gt_cls = gt_cls * tf.expand_dims(ctr_ness, axis=-1)
            cls_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=gt_cls, logits=pred_cls)
            cls_loss = tf.reduce_mean(cls_loss, axis=-1)

        cls_loss = tf.reduce_sum(cls_loss * cls_mask) / norm_param 
        cls_loss = tf.identity(cls_loss, 'cls_loss%d'%index)
        tf.summary.scalar('cls_loss%d'%index, cls_loss)
        tf.add_to_collection(tf.GraphKeys.LOSSES, cls_loss)


    def _generate_centerness_label(self, base_xyz, assigned_boxes_3d, pmask, epsilon=1e-6):
        """
        base_xyz: [bs, pts_num, 3]
        assigned_boxes_3d: [bs, pts_num, cls_num, 7]
        pmask: [bs, pts_num, cls_num]

        return: [bs, pts_num]
        """
        bs, pts_num, _ = base_xyz.get_shape().as_list()

        # [bs, pts_num, 7]
        assigned_boxes_3d = tf.reduce_sum(assigned_boxes_3d * tf.expand_dims(pmask, axis=-1), axis=2)
        pmask = tf.reduce_max(pmask, axis=2) # [bs, pts_num]

        canonical_xyz = base_xyz - assigned_boxes_3d[:, :, :3]
        canonical_xyz = tf.reshape(canonical_xyz, [bs * pts_num, 1, 3])
        rys = tf.reshape(assigned_boxes_3d[:, :, -1], [bs * pts_num])
        canonical_xyz = rotate_points(canonical_xyz, -rys)
        canonical_xyz = tf.reshape(canonical_xyz, [bs, pts_num, 3])

        distance_front = assigned_boxes_3d[:, :, 3] / 2. - canonical_xyz[:, :, 0]
        distance_back = canonical_xyz[:, :, 0] + assigned_boxes_3d[:, :, 3] / 2.
        distance_bottom = 0 - canonical_xyz[:, :, 1]
        distance_top = canonical_xyz[:, :, 1] + assigned_boxes_3d[:, :, 4]
        distance_left = assigned_boxes_3d[:, :, 5] / 2. - canonical_xyz[:, :, 2]
        distance_right = canonical_xyz[:, :, 2] + assigned_boxes_3d[:, :, 5] / 2.

        ctr_ness_l = tf.minimum(distance_front, distance_back) / tf.maximum(distance_front, distance_back) * pmask 
        ctr_ness_w = tf.minimum(distance_left, distance_right) / tf.maximum(distance_left, distance_right) * pmask 
        ctr_ness_h = tf.minimum(distance_bottom, distance_top) / tf.maximum(distance_bottom, distance_top) * pmask 
        ctr_ness = tf.maximum(ctr_ness_l * ctr_ness_h * ctr_ness_w, epsilon)
        ctr_ness = tf.pow(ctr_ness, 1/3.) # [bs, points_num]

        min_ctr_ness, max_ctr_ness = self.ctr_ness_range
        ctr_ness_range = max_ctr_ness - min_ctr_ness 
        ctr_ness *= ctr_ness_range
        ctr_ness += min_ctr_ness

        return ctr_ness
        

    def vote_loss(self, label_dict, pred_dict, placeholders):
        vote_times = len(pred_dict[maps_dict.PRED_VOTE_OFFSET])
        for index in range(vote_times):
            vote_offset = pred_dict[maps_dict.PRED_VOTE_OFFSET][index]
            vote_base = pred_dict[maps_dict.PRED_VOTE_BASE][index]    
            bs, pts_num, _ = vote_offset.get_shape().as_list()
            gt_boxes_3d = placeholders[maps_dict.PL_LABEL_BOXES_3D]
            vote_mask, vote_target = tf.py_func(vote_targets_np, [vote_base, gt_boxes_3d], [tf.float32, tf.float32]) 
            vote_mask = tf.reshape(vote_mask, [bs, pts_num])
            vote_target = tf.reshape(vote_target, [bs, pts_num, 3])


            vote_loss = tf.reduce_sum(model_util.huber_loss(vote_target - vote_offset, delta=1.), axis=-1) * vote_mask
            vote_loss = tf.reduce_sum(vote_loss) / tf.maximum(1., tf.reduce_sum(vote_mask))
            vote_loss = tf.identity(vote_loss, 'vote_loss%d'%index)
            tf.summary.scalar('vote_loss%d'%index, vote_loss)
            tf.add_to_collection(tf.GraphKeys.LOSSES, vote_loss)


    def velo_attr_loss(self, index, label_dict, pred_dict):
        pmask = label_dict[maps_dict.GT_PMASK][index]
        nmask = label_dict[maps_dict.GT_NMASK][index]
        gt_attribute = label_dict[maps_dict.GT_ATTRIBUTE][index] # bs, pts_num, cls_num
        gt_velocity = label_dict[maps_dict.GT_VELOCITY][index] # bs,pts_num,cls_num,2

        pred_attribute = pred_dict[maps_dict.PRED_ATTRIBUTE][index]
        pred_velocity = pred_dict[maps_dict.PRED_VELOCITY][index]

        attr_mask = tf.cast(tf.greater_equal(gt_attribute, 0), tf.float32)
        attr_mask = attr_mask * pmask
        gt_attribute_onehot = tf.cast(tf.one_hot(gt_attribute, depth=8, on_value=1, off_value=0, axis=-1), tf.float32) # [bs, pts_num, cls_num, 8]
        attr_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=gt_attribute_onehot, logits=pred_attribute) 
        attr_loss = attr_loss * tf.expand_dims(attr_mask, axis=-1)
        attr_loss = tf.reduce_sum(attr_loss) / (tf.maximum(1., tf.reduce_sum(attr_mask)) * 8.)
        attr_loss = tf.identity(attr_loss, 'attribute_loss_%d'%index)
        tf.summary.scalar('attribute_loss_%d'%index, attr_loss)
        tf.add_to_collection(tf.GraphKeys.LOSSES, attr_loss)

        velo_mask = tf.cast(tf.logical_not(tf.is_nan(tf.reduce_sum(gt_velocity, axis=-1))), tf.float32)
        velo_mask = velo_mask * pmask
        zero_velocity = tf.zeros_like(gt_velocity)
        gt_velocity = tf.where(tf.is_nan(gt_velocity), zero_velocity, gt_velocity)
        velo_loss = model_util.huber_loss(pred_velocity - gt_velocity, delta=1.)
        velo_loss = tf.reduce_sum(velo_loss, axis=-1) * velo_mask 
        velo_loss = tf.identity(tf.reduce_sum(velo_loss) / tf.maximum(1., tf.reduce_sum(velo_mask)), 'velocity_loss_%d'%index)
        tf.summary.scalar('velocity_loss_%d'%index, velo_loss)
        tf.add_to_collection(tf.GraphKeys.LOSSES, velo_loss)


    def corner_loss(self, index, label_dict, pred_dict):
        pmask = label_dict[maps_dict.GT_PMASK][index]
        nmask = label_dict[maps_dict.GT_NMASK][index]
        gt_corners = label_dict[maps_dict.CORNER_LOSS_GT_BOXES_CORNERS][index]

        pred_corners = pred_dict[maps_dict.CORNER_LOSS_PRED_BOXES_CORNERS][index]

        norm_param = tf.maximum(1., tf.reduce_sum(pmask))

        corner_loss = model_util.huber_loss((pred_corners - gt_corners), delta=1.)
        corner_loss = tf.reduce_sum(corner_loss, axis=[-2, -1]) * pmask 
        corner_loss = tf.identity(tf.reduce_sum(corner_loss) / norm_param, 'corner_loss%d'%index)
        tf.summary.scalar('corner_loss%d'%index, corner_loss)
        tf.add_to_collection(tf.GraphKeys.LOSSES, corner_loss)


    def offset_loss(self, index, label_dict, pred_dict):
        pmask = label_dict[maps_dict.GT_PMASK][index]
        nmask = label_dict[maps_dict.GT_NMASK][index]
        gt_offset = label_dict[maps_dict.GT_OFFSET][index] 
  
        pred_offset = pred_dict[maps_dict.PRED_OFFSET][index]

        norm_param = tf.maximum(1., tf.reduce_sum(pmask))

        offset_loss = model_util.huber_loss((pred_offset - gt_offset), delta=1.)
        offset_loss = tf.reduce_sum(offset_loss, axis=-1) * pmask 
        offset_loss = tf.identity(tf.reduce_sum(offset_loss) / norm_param, 'offset_loss%d'%index)
        tf.summary.scalar('offset_loss%d'%index, offset_loss)
        tf.add_to_collection(tf.GraphKeys.LOSSES, offset_loss)


    def angle_loss(self, index, label_dict, pred_dict):
        gt_angle_cls = label_dict[maps_dict.GT_ANGLE_CLS][index] # [bs, points_num, cls_num]
        gt_angle_res = label_dict[maps_dict.GT_ANGLE_RES][index]
        pmask = label_dict[maps_dict.GT_PMASK][index]
        nmask = label_dict[maps_dict.GT_NMASK][index]

        # [bs, points_num, cls_num, cfg.MODEL.ANGLE_CLS_NUM]
        pred_angle_cls = pred_dict[maps_dict.PRED_ANGLE_CLS][index]
        pred_angle_res = pred_dict[maps_dict.PRED_ANGLE_RES][index]

        norm_param = tf.maximum(1., tf.reduce_sum(pmask))
       
        angle_cls_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred_angle_cls, labels=gt_angle_cls) * pmask # [bs, points_num, cls_num]
        angle_cls_loss = tf.identity(tf.reduce_sum(angle_cls_loss) / norm_param, 'angle_cls_loss%d'%index)
        tf.summary.scalar('angle_cls_loss%d'%index, angle_cls_loss)
        tf.add_to_collection(tf.GraphKeys.LOSSES, angle_cls_loss)

        angle_cls_onehot = tf.cast(tf.one_hot(gt_angle_cls, depth=cfg.MODEL.ANGLE_CLS_NUM, on_value=1, off_value=0, axis=-1), tf.float32)
        gt_angle_res_norm = gt_angle_res / (2 * np.pi / cfg.MODEL.ANGLE_CLS_NUM)
        pred_angle_res_norm = tf.reduce_sum(pred_angle_res * angle_cls_onehot, axis=-1)
        angle_res_loss = model_util.huber_loss((pred_angle_res_norm - gt_angle_res_norm) * pmask, delta=1.)
        angle_res_loss = tf.identity(tf.reduce_sum(angle_res_loss) / norm_param, 'angle_res_loss%d'%index)
        tf.summary.scalar('angle_res_loss%d'%index, angle_res_loss)
        tf.add_to_collection(tf.GraphKeys.LOSSES, angle_res_loss)

