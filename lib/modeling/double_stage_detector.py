import numpy as np
import tensorflow as tf

from core.config import cfg
from builder.anchor_builder import Anchors
from builder.target_assigner import TargetAssigner
from builder.layer_builder import LayerBuilder
from dataset.placeholders import PlaceHolders
from modeling.head_builder import HeadBuilder 
from builder.encoder_builder import EncoderDecoder
from builder.postprocessor import PostProcessor
from builder.loss_builder import LossBuilder
from builder.points_pooler import PointsPooler
from builder.sampler import Sampler

from utils.box_3d_utils import transfer_box3d_to_corners 
from utils.model_util import *

import dataset.maps_dict as maps_dict

class DoubleStageDetector:
    def __init__(self, batch_size, is_training):
        self.batch_size = batch_size
        self.is_training = is_training
        self.only_first_stage = cfg.MODEL.ONLY_FIRST_STAGE

        # placeholders
        self.placeholders_builder = PlaceHolders(self.batch_size) 
        self.placeholders_builder.get_placeholders()
        self.placeholders = self.placeholders_builder.placeholders

        self.cls_list = cfg.DATASET.KITTI.CLS_LIST
        self.cls2idx = dict([(cls, i + 1) for i, cls in enumerate(self.cls_list)])
        self.idx2cls = dict([(i + 1, cls) for i, cls in enumerate(self.cls_list)])

        # anchor_builder
        self.anchor_builder = Anchors(0, self.cls_list)

        # encoder_decoder
        self.encoder_decoder_list = [EncoderDecoder(0), EncoderDecoder(1)]

        # postprocessor
        self.postprocessor_list = [PostProcessor(0, 1), PostProcessor(1, len(self.cls_list))]

        # loss builder
        self.loss_builder_list = [LossBuilder(0), LossBuilder(1)]

        # target assigner
        self.target_assigner_list = [TargetAssigner(0), TargetAssigner(1)]

        # sampler
        self.sampler = Sampler(1)

        # points pooler
        pool_cfg = cfg.MODEL.NETWORK.FIRST_STAGE.POINTS_POOLER
        self.pool_mask_thresh = cfg.MODEL.NETWORK.FIRST_STAGE.POOLER_MASK_THRESHOLD
        self.points_pooler = PointsPooler(pool_cfg)

        ############### RPN head/network definition ##############
        ### head
        self.rpn_iou_loss = False
        self.rpn_heads = []
        head_cfg = cfg.MODEL.NETWORK.FIRST_STAGE.HEAD
        for i in range(len(head_cfg)):
            self.rpn_heads.append(HeadBuilder(self.batch_size, 
                self.anchor_builder.anchors_num, 0, head_cfg[i], is_training))
            if self.rpn_heads[-1].layer_type == 'IoU': self.rpn_iou_loss = True
        ### network
        self.rpn_vote_loss = False
        self.rpn_layers = []
        layer_cfg = cfg.MODEL.NETWORK.FIRST_STAGE.ARCHITECTURE
        for i in range(len(layer_cfg)):
            self.rpn_layers.append(LayerBuilder(i, self.is_training, layer_cfg)) 
            if self.rpn_layers[-1].layer_type == 'Vote_Layer': self.rpn_vote_loss = True

        ############### RCNN-stage head/network definition ##############
        ### head
        self.rcnn_iou_loss = False
        self.rcnn_heads = []
        head_cfg = cfg.MODEL.NETWORK.SECOND_STAGE.HEAD
        for i in range(len(head_cfg)):
            self.rcnn_heads.append(HeadBuilder(self.batch_size, 
                1, 1, head_cfg[i], is_training))
            if self.rcnn_heads[-1].layer_type == 'IoU': self.rcnn_iou_loss = True
        ### network
        self.rcnn_vote_loss = False
        self.rcnn_layers = []
        layer_cfg = cfg.MODEL.NETWORK.SECOND_STAGE.ARCHITECTURE
        for i in range(len(layer_cfg)):
            self.rcnn_layers.append(LayerBuilder(i, self.is_training, layer_cfg)) 
            if self.rcnn_layers[-1].layer_type == 'Vote_Layer': self.rcnn_vote_loss = True

        self.heads = [self.rpn_heads, self.rcnn_heads]
        self.layers = [self.rpn_layers, self.rcnn_layers]
        self.corner_loss = [cfg.MODEL.FIRST_STAGE.CORNER_LOSS, cfg.MODEL.SECOND_STAGE.CORNER_LOSS]
        self.vote_loss = [self.rpn_vote_loss, self.rcnn_vote_loss]
        self.iou_loss = [self.rpn_iou_loss, self.rcnn_iou_loss]
        self.attr_velo_loss = [False, cfg.MODEL.SECOND_STAGE.PREDICT_ATTRIBUTE_AND_VELOCITY]

        self.__init_dict()

    def __init_dict(self):
        self.output = dict()
        # sampled xyz/feature
        self.output[maps_dict.KEY_OUTPUT_XYZ] = []
        self.output[maps_dict.KEY_OUTPUT_FEATURE] = []
        # generated anchors
        self.output[maps_dict.KEY_ANCHORS_3D] = [] # generated anchors
        # vote output
        self.output[maps_dict.PRED_VOTE_OFFSET] = []
        self.output[maps_dict.PRED_VOTE_BASE] = []
        # det output
        self.output[maps_dict.PRED_CLS] = []
        self.output[maps_dict.PRED_OFFSET] = []
        self.output[maps_dict.PRED_ANGLE_CLS] = []
        self.output[maps_dict.PRED_ANGLE_RES] = []
        self.output[maps_dict.CORNER_LOSS_PRED_BOXES_CORNERS] = []
        self.output[maps_dict.PRED_ATTRIBUTE] = []
        self.output[maps_dict.PRED_VELOCITY] = []
        # iou output
        self.output[maps_dict.PRED_IOU_3D_VALUE] = []
        # final result
        self.output[maps_dict.PRED_3D_BBOX] = []
        self.output[maps_dict.PRED_3D_SCORE] = []
        self.output[maps_dict.PRED_3D_CLS_CATEGORY] = []
        self.output[maps_dict.PRED_3D_ATTRIBUTE] = []
        self.output[maps_dict.PRED_3D_VELOCITY] = []

        self.prediction_keys = self.output.keys()

        self.labels = dict()
        self.labels[maps_dict.GT_CLS] = []
        self.labels[maps_dict.GT_OFFSET] = []
        self.labels[maps_dict.GT_ANGLE_CLS] = []
        self.labels[maps_dict.GT_ANGLE_RES] = []
        self.labels[maps_dict.GT_ATTRIBUTE] = []
        self.labels[maps_dict.GT_VELOCITY] = []
        self.labels[maps_dict.GT_BOXES_ANCHORS_3D] = []
        self.labels[maps_dict.GT_IOU_3D_VALUE] = []

        self.labels[maps_dict.GT_PMASK] = []
        self.labels[maps_dict.GT_NMASK] = []
        self.labels[maps_dict.CORNER_LOSS_GT_BOXES_CORNERS] = []

    def network_forward(self, point_cloud, index, bn_decay,
        xyz_list, feature_list, fps_idx_list):

        l0_xyz = tf.slice(point_cloud, [0,0,0], [-1,-1,3])
        l0_points = tf.slice(point_cloud, [0,0,3], [-1,-1,-1])
      
        xyz_list.append(l0_xyz)
        feature_list.append(l0_points)
        fps_idx_list.append(None)

        layers, heads = self.layers[index], self.heads[index] 

        for layer in layers:
            xyz_list, feature_list, fps_idx_list = layer.build_layer(xyz_list, feature_list, fps_idx_list, bn_decay, self.output)

        cur_head_start_idx = len(self.output[maps_dict.KEY_OUTPUT_XYZ])
        for head in heads:
            head.build_layer(xyz_list, feature_list, bn_decay, self.output)
        merge_head_prediction(cur_head_start_idx, self.output, self.prediction_keys)


    def model_forward(self, bn_decay=None):
        points_input_det = self.placeholders[maps_dict.PL_POINTS_INPUT]

        # forward the point cloud
        xyz_list, feature_list, fps_idx_list = [], [], []
        self.network_forward(points_input_det, 0, bn_decay,
                             xyz_list, feature_list, fps_idx_list)

        # generate anchors
        base_xyz = self.output[maps_dict.KEY_OUTPUT_XYZ][-1]
        anchors = self.anchor_builder.generate(base_xyz) # [bs, pts_num, 1/cls_num, 7]
        self.output[maps_dict.KEY_ANCHORS_3D].append(anchors)

        if self.is_training: # training mode
            self.target_assign(-1, 0, base_xyz, anchors)
            self.train_forward(-1, 0, anchors) 

        # decode proposals
        self.test_forward(-1, 0, cfg.MODEL.FIRST_STAGE, anchors)

        if self.only_first_stage: return

        # [bs, proposal_num, 7]
        proposals = self.output[maps_dict.PRED_3D_BBOX][-1]
        proposals = tf.reshape(proposals, [self.batch_size, cfg.MODEL.FIRST_STAGE.MAX_OUTPUT_NUM, 7])
        expand_proposals = tf.expand_dims(proposals, axis=2)
        ctr_proposals = cast_bottom_to_center(proposals)

        if self.is_training:
            valid_mask = self.points_pooler.get_valid_mask(base_xyz, proposals)
            expand_proposals = self.target_assign(-1, 1, ctr_proposals[:, :, :3], expand_proposals, valid_mask)
            proposals = tf.squeeze(expand_proposals, axis=2)
            ctr_proposals = cast_bottom_to_center(proposals)
        # [bs, proposal_num, 1, 7]
        self.output[maps_dict.KEY_ANCHORS_3D].append(expand_proposals)

        # pool
        base_feature = self.output[maps_dict.KEY_OUTPUT_FEATURE][-1]
        base_mask = self.output[maps_dict.PRED_3D_SCORE][-1]
        base_mask = tf.cast(tf.greater_equal(base_mask, self.pool_mask_thresh), tf.float32)
        base_mask = tf.expand_dims(base_mask, axis=-1) # [bs, proposal_num, 1]
        pool_feature, pool_mask = self.points_pooler.pool(base_xyz, base_feature, base_mask, proposals, self.is_training, bn_decay) # [bs * proposal_num, sample_num, 3+c]
 
        # initialize the list of stage-2 with proposal center
        xyz_list, feature_list, fps_idx_list = [ctr_proposals[:, :, :3]], [None], [None]

        # second-stage forward
        self.network_forward(pool_feature, 1, bn_decay,
                             xyz_list, feature_list, fps_idx_list)

        if self.is_training: # training mode
            self.train_forward(-1, 1, expand_proposals) 
        else:
            self.test_forward(-1, 1, cfg.MODEL.SECOND_STAGE, expand_proposals, valid_mask=pool_mask)


    def target_assign(self, index, stage_index, base_xyz, anchors, valid_mask=None):
        """
        Assign target labels for each anchor/proposal
        If stage_index >= 1: also gather assigned proposals out
        """
        encoder_decoder = self.encoder_decoder_list[stage_index]
        target_assigner = self.target_assigner_list[stage_index]

        gt_boxes_3d = self.placeholders[maps_dict.PL_LABEL_BOXES_3D]
        gt_classes = self.placeholders[maps_dict.PL_LABEL_CLASSES]
        gt_angle_cls = self.placeholders[maps_dict.PL_ANGLE_CLS]
        gt_angle_res = self.placeholders[maps_dict.PL_ANGLE_RESIDUAL]

        if maps_dict.PL_LABEL_ATTRIBUTES in self.placeholders.keys():
            gt_attributes = self.placeholders[maps_dict.PL_LABEL_ATTRIBUTES]
        else: gt_attributes = None

        if maps_dict.PL_LABEL_VELOCITY in self.placeholders.keys():
            gt_velocity = self.placeholders[maps_dict.PL_LABEL_VELOCITY]
        else: gt_velocity = None

        returned_list = target_assigner.assign(base_xyz, anchors, gt_boxes_3d, gt_classes, gt_angle_cls, gt_angle_res, gt_velocity, gt_attributes, valid_mask)

        assigned_idx, assigned_pmask, assigned_nmask, assigned_gt_boxes_3d, assigned_gt_labels, assigned_gt_angle_cls, assigned_gt_angle_res, assigned_gt_velocity, assigned_gt_attribute = returned_list

        # encode offset
        assigned_gt_offset, assigned_gt_angle_cls, assigned_gt_angle_res = encoder_decoder.encode(base_xyz, assigned_gt_boxes_3d, anchors)

        if stage_index >= 1: # gather assigned proposal out for reducing memory cost
            assigned_mask = assigned_pmask + assigned_nmask

            anchors, assigned_pmask, assigned_nmask, assigned_gt_boxes_3d, assigned_gt_labels, \
            assigned_gt_offset, assigned_gt_angle_cls, assigned_gt_angle_res, \
            assigned_gt_velocity, assigned_gt_attribute = self.sampler.gather_list(\
                assigned_mask, [anchors, assigned_pmask, assigned_nmask, assigned_gt_boxes_3d, assigned_gt_labels,\
                assigned_gt_offset, assigned_gt_angle_cls, assigned_gt_angle_res, \
                assigned_gt_velocity, assigned_gt_attribute])
            

        self.labels[maps_dict.GT_CLS].append(assigned_gt_labels)
        self.labels[maps_dict.GT_BOXES_ANCHORS_3D].append(assigned_gt_boxes_3d)
        self.labels[maps_dict.GT_OFFSET].append(assigned_gt_offset)
        self.labels[maps_dict.GT_ANGLE_CLS].append(assigned_gt_angle_cls)
        self.labels[maps_dict.GT_ANGLE_RES].append(assigned_gt_angle_res)
        self.labels[maps_dict.GT_ATTRIBUTE].append(assigned_gt_attribute)
        self.labels[maps_dict.GT_VELOCITY].append(assigned_gt_velocity)
        self.labels[maps_dict.GT_PMASK].append(assigned_pmask)
        self.labels[maps_dict.GT_NMASK].append(assigned_nmask)
        return anchors


    def train_forward(self, index, stage_index, anchors, valid_mask=None):
        """
        Calculating loss
        """
        loss_builder = self.loss_builder_list[stage_index]
        encoder_decoder = self.encoder_decoder_list[stage_index]

        base_xyz = self.output[maps_dict.KEY_OUTPUT_XYZ][index]
        pred_offset = self.output[maps_dict.PRED_OFFSET][index]
        pred_angle_cls = self.output[maps_dict.PRED_ANGLE_CLS][index]
        pred_angle_res = self.output[maps_dict.PRED_ANGLE_RES][index]

        # corner_loss
        assigned_gt_angle_cls = self.labels[maps_dict.GT_ANGLE_CLS][index]
        assigned_gt_boxes_3d = self.labels[maps_dict.GT_BOXES_ANCHORS_3D][index] 
        corner_loss_angle_cls = tf.cast(tf.one_hot(assigned_gt_angle_cls, depth=cfg.MODEL.ANGLE_CLS_NUM, on_value=1, off_value=0, axis=-1), tf.float32) # bs, pts_num, cls_num, -1
        pred_anchors_3d = encoder_decoder.decode(base_xyz, pred_offset, corner_loss_angle_cls, pred_angle_res, self.is_training, anchors) # [bs, points_num, cls_num, 7]
        pred_corners = transfer_box3d_to_corners(pred_anchors_3d) # [bs, points_num, cls_num, 8, 3] 
        gt_corners = transfer_box3d_to_corners(assigned_gt_boxes_3d) # [bs, points_num, cls_num,8,3]
        self.output[maps_dict.CORNER_LOSS_PRED_BOXES_CORNERS].append(pred_corners)
        self.labels[maps_dict.CORNER_LOSS_GT_BOXES_CORNERS].append(gt_corners)

        loss_builder.forward(index, self.labels, self.output, self.placeholders, self.corner_loss[stage_index], self.vote_loss[stage_index], self.attr_velo_loss[stage_index], self.iou_loss[stage_index])


    def test_forward(self, index, stage_index, stage_cfg, anchors, valid_mask=None):
        encoder_decoder = self.encoder_decoder_list[stage_index]
        postprocessor = self.postprocessor_list[stage_index] 

        base_xyz = self.output[maps_dict.KEY_OUTPUT_XYZ][index]

        pred_cls = self.output[maps_dict.PRED_CLS][index] # [bs, points_num, cls_num + 1/0]
        pred_offset = self.output[maps_dict.PRED_OFFSET][index]
        pred_angle_cls = self.output[maps_dict.PRED_ANGLE_CLS][index]
        pred_angle_res = self.output[maps_dict.PRED_ANGLE_RES][index]

        # decode predictions
        pred_anchors_3d = encoder_decoder.decode(base_xyz, pred_offset, pred_angle_cls, pred_angle_res, self.is_training, anchors) # [bs, points_num, cls_num, 7]
        
        # decode classification
        if stage_cfg.CLS_ACTIVATION == 'Softmax':
            # softmax 
            pred_score = tf.nn.softmax(pred_cls)
            pred_score = tf.slice(pred_score, [0, 0, 1], [-1, -1, -1])
        else: # sigmoid
            pred_score = tf.nn.sigmoid(pred_cls)

        # using IoU branch proposed by sparse-to-dense
        if self.iou_loss[stage_index]:
            pred_iou = self.output[maps_dict.PRED_IOU_3D_VALUE][index]
            pred_score = pred_score * pred_iou

        if valid_mask is not None:
            valid_mask = tf.cast(valid_mask, tf.float32)
            pred_score = pred_score * valid_mask

        if len(self.output[maps_dict.PRED_ATTRIBUTE]) <= 0:
            pred_attribute = None
        else: pred_attribute = self.output[maps_dict.PRED_ATTRIBUTE][index]

        if len(self.output[maps_dict.PRED_VELOCITY]) <= 0:
            pred_velocity = None
        else: pred_velocity = self.output[maps_dict.PRED_VELOCITY][index]

        postprocessor.forward(pred_anchors_3d, pred_score, self.output, pred_attribute, pred_velocity)

