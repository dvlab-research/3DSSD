import numpy as np
import tensorflow as tf

from core.config import cfg
from utils.anchors_util import project_to_bev
from utils.box_3d_utils import box_3d_to_anchor

import dataset.maps_dict as maps_dict

class PostProcessor:
    def __init__(self, stage, cls_num):
        if stage == 0:
            self.postprocessor_cfg = cfg.MODEL.FIRST_STAGE
        elif stage == 1:
            self.postprocessor_cfg = cfg.MODEL.SECOND_STAGE
        else: raise Exception('Not Implementation Error')

        self.max_output_size = self.postprocessor_cfg.MAX_OUTPUT_NUM
        self.nms_threshold = self.postprocessor_cfg.NMS_THRESH

        self.cls_num = cls_num
   
    
    def class_unaware_format(self, pred_anchors_3d, pred_score):
        """ (for rpn propose)
        Change prediction format from class-aware-format to class-ignorance-format
        pred_anchors_3d: [bs, points_num, 1/cls_num, 7]
        pred_score: [bs, points_num, cls_num]

        return: pred_anchors_3d: [bs, points_num, 1, 7]
                pred_score: [bs, points_num, 1]
        """ 
        unaware_pred_score = tf.reduce_max(pred_score, axis=-1, keepdims=True)
        cls_num = pred_anchors_3d.get_shape().as_list()[2]
        if cls_num == 1:
            return pred_anchors_3d, unaware_pred_score

        # class-aware in boundingbox prediction
        pred_cls = tf.argmax(pred_score, axis=-1)
        pred_cls_onehot = tf.cast(tf.one_hot(pred_cls, depth=cls_num, on_value=1, off_value=0, axis=-1), tf.float32)
        # bs, pts_num, cls_num, 7
        unaware_pred_anchors_3d = pred_anchors_3d * tf.expand_dims(pred_cls_onehot, axis=-1)
        unaware_pred_anchors_3d = tf.reduce_sum(unaware_pred_anchors_3d, axis=2, keepdims=True)
        return unaware_pred_anchors_3d, unaware_pred_score

    


    def forward(self, pred_anchors_3d, pred_score, output_dict, pred_attribute=None, pred_velocity=None):
        """
        pred_anchors_3d: [bs, points_num, 1/cls_num, 7]
        pred_score: [bs, points_num, cls_num]
        pred_attribute: [bs, points_num, 1/cls_num, 8]
        pred_velocity: [bs, points_num, 1/cls_num, 2]
        """
        cls_num = pred_score.get_shape().as_list()[-1] 
        if cls_num != self.cls_num: # format predictions to class-unaware predictions
            assert pred_attribute == None and pred_velocity == None, 'Not support the predictions of attribute and velocity in RPN phase'
            pred_anchors_3d, pred_score = self.class_unaware_format(pred_anchors_3d, pred_score)

        pred_anchors_3d_list = tf.unstack(pred_anchors_3d, axis=0)
        pred_scores_list = tf.unstack(pred_score, axis=0)

        pred_3d_bbox_list = []
        pred_3d_cls_score_list = []
        pred_3d_cls_cat_list = []
        pred_attribute_list = []
        pred_velocity_list = []
        for batch_idx, pred_anchors_3d, pred_scores in zip(range(len(pred_anchors_3d_list)), pred_anchors_3d_list, pred_scores_list):
            cur_pred_3d_bbox_list = []
            cur_pred_3d_cls_score_list = []
            cur_pred_3d_cls_cat_list = []
            cur_pred_attribute_list = []
            cur_pred_velocity_list = []

            for i in range(self.cls_num):
                reg_i = min(i, pred_anchors_3d.get_shape().as_list()[1] - 1)
                cur_pred_anchors_3d = pred_anchors_3d[:, reg_i, :] 

                cur_pred_anchors = box_3d_to_anchor(cur_pred_anchors_3d) 
                cur_pred_anchors_bev = project_to_bev(cur_pred_anchors) # [-1, 4]

                cur_cls_score = pred_scores[:, i]
                nms_index = tf.image.non_max_suppression(cur_pred_anchors_bev, cur_cls_score, max_output_size=self.max_output_size, iou_threshold=self.nms_threshold)
               
                cur_pred_3d_bbox_list.append(tf.gather(cur_pred_anchors_3d, nms_index)) 
                cur_pred_3d_cls_score_list.append(tf.gather(cur_cls_score, nms_index))
                cur_pred_3d_cls_cat_list.append(tf.cast(tf.ones_like(nms_index), tf.int32) * i)

                if pred_attribute is not None:
                    cur_pred_attribute_list.append(tf.gather(pred_attribute[batch_idx, :, reg_i, :], nms_index))
                if pred_velocity is not None:
                    cur_pred_velocity_list.append(tf.gather(pred_velocity[batch_idx, :, reg_i, :], nms_index))

            cur_pred_3d_bbox_list = tf.concat(cur_pred_3d_bbox_list, axis=0)
            cur_pred_3d_cls_score_list = tf.concat(cur_pred_3d_cls_score_list, axis=0)
            cur_pred_3d_cls_cat_list = tf.concat(cur_pred_3d_cls_cat_list, axis=0)

            pred_3d_bbox_list.append(cur_pred_3d_bbox_list)
            pred_3d_cls_score_list.append(cur_pred_3d_cls_score_list)
            pred_3d_cls_cat_list.append(cur_pred_3d_cls_cat_list)

            if pred_attribute is not None:
                cur_pred_attribute_list = tf.concat(cur_pred_attribute_list, axis=0)
                pred_attribute_list.append(cur_pred_attribute_list)

            if pred_velocity is not None:
                cur_pred_velocity_list = tf.concat(cur_pred_velocity_list, axis=0)
                pred_velocity_list.append(cur_pred_velocity_list)

        pred_3d_bbox_list = tf.stack(pred_3d_bbox_list, axis=0)
        pred_3d_cls_score_list = tf.stack(pred_3d_cls_score_list, axis=0)
        pred_3d_cls_cat_list = tf.stack(pred_3d_cls_cat_list, axis=0)
            
        output_dict[maps_dict.PRED_3D_BBOX].append(pred_3d_bbox_list)
        output_dict[maps_dict.PRED_3D_SCORE].append(pred_3d_cls_score_list)
        output_dict[maps_dict.PRED_3D_CLS_CATEGORY].append(pred_3d_cls_cat_list)
        if pred_attribute is not None:
            output_dict[maps_dict.PRED_3D_ATTRIBUTE].append(tf.stack(pred_attribute_list, axis=0))
        if pred_velocity is not None:
            output_dict[maps_dict.PRED_3D_VELOCITY].append(tf.stack(pred_velocity_list, axis=0))

        return output_dict
