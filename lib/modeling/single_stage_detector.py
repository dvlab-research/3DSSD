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

from utils.box_3d_utils import transfer_box3d_to_corners 

import dataset.maps_dict as maps_dict

class SingleStageDetector:
    def __init__(self, batch_size, is_training):
        self.batch_size = batch_size
        self.is_training = is_training

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
        self.encoder_decoder = EncoderDecoder(0)

        # postprocessor
        self.postprocessor = PostProcessor(0)

        # loss builder
        self.loss_builder = LossBuilder(0)

        # head builder
        self.heads = []
        head_cfg = cfg.MODEL.NETWORK.FIRST_STAGE.HEAD
        self.heads.append(HeadBuilder(self.anchor_builder.anchors_num, 0, head_cfg, is_training))

        # target assigner
        self.target_assigner = TargetAssigner(0) # first stage

        self.vote_loss = False
        # layer builder
        layer_cfg = cfg.MODEL.NETWORK.FIRST_STAGE.ARCHITECTURE
        layers = []
        for i in range(len(layer_cfg)):
            layers.append(LayerBuilder(i, self.is_training, layer_cfg)) 
            if layers[-1].layer_type == 'Vote_Layer': self.vote_loss = True
        self.layers = layers

        self.attr_velo_loss = cfg.MODEL.FIRST_STAGE.PREDICT_ATTRIBUTE_AND_VELOCITY 

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
        # final result
        self.output[maps_dict.PRED_3D_BBOX] = []
        self.output[maps_dict.PRED_3D_SCORE] = []
        self.output[maps_dict.PRED_3D_CLS_CATEGORY] = []
        self.output[maps_dict.PRED_3D_ATTRIBUTE] = []
        self.output[maps_dict.PRED_3D_VELOCITY] = []
        

        self.labels = dict()
        self.labels[maps_dict.GT_CLS] = []
        self.labels[maps_dict.GT_OFFSET] = []
        self.labels[maps_dict.GT_ANGLE_CLS] = []
        self.labels[maps_dict.GT_ANGLE_RES] = []
        self.labels[maps_dict.GT_ATTRIBUTE] = []
        self.labels[maps_dict.GT_VELOCITY] = []
        self.labels[maps_dict.GT_BOXES_ANCHORS_3D] = []

        self.labels[maps_dict.GT_PMASK] = []
        self.labels[maps_dict.GT_NMASK] = []
        self.labels[maps_dict.CORNER_LOSS_GT_BOXES_CORNERS] = []

    def network_forward(self, point_cloud, bn_decay):
        l0_xyz = tf.slice(point_cloud, [0,0,0], [-1,-1,3])
        l0_points = tf.slice(point_cloud, [0,0,3], [-1,-1,-1])
        xyz_list, feature_list, fps_idx_list = [l0_xyz], [l0_points], [None]
        for layer in self.layers:
            xyz_list, feature_list, fps_idx_list = layer.build_layer(xyz_list, feature_list, fps_idx_list, bn_decay, self.output)

        pred_cls, pred_reg, base_xyz, base_feature = self.heads[0].build_layer(xyz_list, feature_list, bn_decay, self.output)
        return pred_cls, pred_reg, base_xyz, base_feature

    def model_forward(self, bn_decay=None):
        points_input_det = self.placeholders[maps_dict.PL_POINTS_INPUT]

        # forward the point cloud
        pred_cls, pred_reg, base_xyz, base_feature = self.network_forward(points_input_det, bn_decay)

        # generate anchors
        anchors = self.anchor_builder.generate(base_xyz) # [bs, pts_num, 1/cls_num, 7]
        self.output[maps_dict.KEY_ANCHORS_3D].append(anchors)

        if self.is_training: # training mode
            self.train_forward(0, anchors) 
        else: # testing mode
            self.test_forward(0, anchors)


    def train_forward(self, index, anchors):
        """
        Calculating loss
        """
        base_xyz = self.output[maps_dict.KEY_OUTPUT_XYZ][index]
        pred_offset = self.output[maps_dict.PRED_OFFSET][index]
        pred_angle_cls = self.output[maps_dict.PRED_ANGLE_CLS][index]
        pred_angle_res = self.output[maps_dict.PRED_ANGLE_RES][index]

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

        returned_list = self.target_assigner.assign(base_xyz, anchors, gt_boxes_3d, gt_classes, gt_angle_cls, gt_angle_res, gt_velocity, gt_attributes)

        assigned_idx, assigned_pmask, assigned_nmask, assigned_gt_boxes_3d, assigned_gt_labels, assigned_gt_angle_cls, assigned_gt_angle_res, assigned_gt_velocity, assigned_gt_attribute = returned_list

        # encode offset
        assigned_gt_offset = self.encoder_decoder.encode(base_xyz, assigned_gt_boxes_3d, anchors)

        # corner_loss
        corner_loss_angle_cls = tf.cast(tf.one_hot(assigned_gt_angle_cls, depth=cfg.MODEL.ANGLE_CLS_NUM, on_value=1, off_value=0, axis=-1), tf.float32) # bs, pts_num, cls_num, -1
        pred_anchors_3d = self.encoder_decoder.decode(base_xyz, pred_offset, corner_loss_angle_cls, pred_angle_res, self.is_training, anchors) # [bs, points_num, cls_num, 7]
        pred_corners = transfer_box3d_to_corners(pred_anchors_3d) # [bs, points_num, cls_num, 8, 3] 
        gt_corners = transfer_box3d_to_corners(assigned_gt_boxes_3d) # [bs, points_num, cls_num,8,3]
        self.output[maps_dict.CORNER_LOSS_PRED_BOXES_CORNERS].append(pred_corners)
        self.labels[maps_dict.CORNER_LOSS_GT_BOXES_CORNERS].append(gt_corners)
        

        self.labels[maps_dict.GT_CLS].append(assigned_gt_labels)
        self.labels[maps_dict.GT_BOXES_ANCHORS_3D].append(assigned_gt_boxes_3d)
        self.labels[maps_dict.GT_OFFSET].append(assigned_gt_offset)
        self.labels[maps_dict.GT_ANGLE_CLS].append(assigned_gt_angle_cls)
        self.labels[maps_dict.GT_ANGLE_RES].append(assigned_gt_angle_res)
        self.labels[maps_dict.GT_ATTRIBUTE].append(assigned_gt_attribute)
        self.labels[maps_dict.GT_VELOCITY].append(assigned_gt_velocity)
        self.labels[maps_dict.GT_PMASK].append(assigned_pmask)
        self.labels[maps_dict.GT_NMASK].append(assigned_nmask)

        self.loss_builder.forward(index, self.labels, self.output, self.placeholders, self.vote_loss, self.attr_velo_loss)


    def test_forward(self, index, anchors):
        base_xyz = self.output[maps_dict.KEY_OUTPUT_XYZ][index]

        pred_cls = self.output[maps_dict.PRED_CLS][index] # [bs, points_num, cls_num + 1/0]
        pred_offset = self.output[maps_dict.PRED_OFFSET][index]
        pred_angle_cls = self.output[maps_dict.PRED_ANGLE_CLS][index]
        pred_angle_res = self.output[maps_dict.PRED_ANGLE_RES][index]

        # decode predictions
        pred_anchors_3d = self.encoder_decoder.decode(base_xyz, pred_offset, pred_angle_cls, pred_angle_res, self.is_training, anchors) # [bs, points_num, cls_num, 7]
        
        # decode classification
        if cfg.MODEL.FIRST_STAGE.CLS_ACTIVATION == 'Softmax':
            # softmax 
            pred_score = tf.nn.softmax(pred_cls)
            pred_score = tf.slice(pred_score, [0, 0, 1], [-1, -1, -1])
        else: # sigmoid
            pred_score = tf.nn.sigmoid(pred_cls)

        if len(self.output[maps_dict.PRED_ATTRIBUTE]) <= index:
            pred_attribute = None
        else: pred_attribute = self.output[maps_dict.PRED_ATTRIBUTE][index]

        if len(self.output[maps_dict.PRED_VELOCITY]) <= index:
            pred_velocity = None
        else: pred_velocity = self.output[maps_dict.PRED_VELOCITY][index]

        self.postprocessor.forward(pred_anchors_3d, pred_score, self.output, pred_attribute, pred_velocity)

