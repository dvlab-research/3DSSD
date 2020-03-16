######################################################################
# dataset loader keys
######################################################################
KEY_LABEL_BOXES_3D = 'label_boxes_3d'
KEY_LABEL_ANCHORS = 'label_anchors'
KEY_LABEL_CLASSES = 'label_classes'
KEY_LABEL_Y_ANGLE = 'label_y_angle'
KEY_LABEL_RES_ANGLE = 'label_res_angle'
KEY_LABEL_SEMSEG = 'label_sem_seg'
KEY_LABEL_DIST = 'label_dist'
KEY_GROUNDTURTH_SEMSEG = 'label_ground_truth_sem_seg'

############## NuScenes Labels #########
KEY_LABEL_ATTRIBUTES = 'label_attributes'
KEY_LABEL_VELOCITY = 'label_velocity'
KEY_SWEEPS = 'sweeps'
KEY_TRANSFORMRATION_MATRIX = 'transformation_matrix'
KEY_TIMESTAMPS = 'time_stamps'
############## NuScenes Labels #########

KEY_OUTPUT_XYZ = 'final_xyz'
KEY_OUTPUT_FEATURE = 'final_feature'
KEY_ANCHORS_3D = 'anchors_3d'

KEY_IMAGE_INPUT = 'image_input'
KEY_BEV_INPUT = 'bev_input'

KEY_LABEL_NUM = 'label_num'

KEY_SAMPLE_IDX = 'sample_idx'
KEY_SAMPLE_NAME = 'sample_name'
KEY_SAMPLE_AUGS = 'sample_augs'

KEY_POINT_CLOUD = 'point_cloud'
KEY_GROUND_PLANE = 'ground_plane'
KEY_STEREO_CALIB = 'stereo_calib'
KEY_VOLUMES = 'volumes'


######################################################################
# SECOND data augmentations
######################################################################
KEY_SAMPLED_GT_POINTS = 'sampled_gt_points'
KEY_SAMPLED_GT_LABELS_3D = 'sampled_gt_labels_3d'
KEY_SAMPLED_GT_SEM_LABELS = 'sampled_gt_sem_labels'
KEY_SAMPLED_GT_SEM_SCORES = 'sampled_gt_sem_scores'
KEY_SAMPLED_GT_CLSES = 'sampled_gt_classes'
KEY_SAMPLED_GT_ATTRIBUTES = 'sampled_gt_attributes'
KEY_SAMPLED_GT_VELOCITY = 'sampled_gt_velocity'

######################################################################
# point cloud object detection keys
######################################################################
PL_POINTS_INPUT = 'point_cloud_pl'
PL_ANGLE_CLS = 'angle_cls_pl'
PL_ANGLE_RESIDUAL = 'angle_res_pl'

PL_CUR_SWEEP_POINTS_INPUT = 'cur_sweep_points_input_pl'
PL_OTHER_SWEEP_POINTS_INPUT = 'other_sweep_points_input_pl'
PL_POINTS_NUM_PER_VOXEL = 'points_num_per_voxel_pl'

PL_POINTS_LOCATION = 'points_location_pl'
PL_POINTS_FEATURE = 'points_feature_pl'
PL_POINTS_SEMSEG = 'points_semseg'
PL_POINTS_AFTER_SEMSEG = 'points_after_semseg'
PL_CLS_PRED_FEATURE = 'cls_pred_feature_pl'
PL_REG_PRED_FEATURE = 'reg_pred_feature_pl'

PL_PREDICTED_BOXES3D = 'predicted_box3d_pl'
PL_PREDICTED_BOXES_ANCHORS = 'predicted_boxes_anchors'

PL_CALIB_P2 = 'calib_p2_pl'


######################################################################
# rpn keys
######################################################################
##############################
# Keys for Placeholders
##############################
PL_BEV_INPUT = 'bev_input_pl'
PL_IMG_INPUT = 'img_input_pl'
PL_ANCHORS = 'anchors_pl'

# anchors project to bev
PL_BEV_ANCHORS = 'bev_anchors_pl'
PL_BEV_ANCHORS_NORM = 'bev_anchors_norm_pl'
# anchors project to img
PL_IMG_ANCHORS = 'img_anchors_pl'
PL_IMG_ANCHORS_NORM = 'img_anchors_norm_pl'
PL_LABEL_ANCHORS = 'label_anchors_pl'
PL_LABEL_BOXES_3D = 'label_boxes_3d_pl'
PL_LABEL_CLASSES = 'label_classes_pl'
PL_LABEL_SEMSEGS = 'label_semantic_segmentation_pl'
PL_LABEL_DIST = 'dist_between_points_to_ctr'
PL_LABEL_ATTRIBUTES = 'label_attributes_pl'
PL_LABEL_VELOCITY = 'label_velocity_pl'

PL_ANCHOR_IOUS = 'anchor_ious_pl'
PL_ANCHOR_OFFSETS = 'anchor_offsets_pl'
PL_ANCHOR_CLASSES = 'anchor_classes_pl'

# Sample info, including keys for projection to image space
# (e.g. camera matrix, image index, etc.)
PL_CALIB_P2 = 'frame_calib_p2'
PL_IMG_IDX = 'current_img_idx'
PL_GROUND_PLANE = 'ground_plane'
PL_IMG = 'current_img'

##############################
# Keys for Predictions
##############################
PRED_ANCHORS = 'generate_anchors'
PRED_VOTE_OFFSET = 'pred_vote_offset'
PRED_VOTE_BASE = 'pred_vote_base'
PRED_CLS = 'pred_cls'
PRED_OFFSET = 'pred_offset'
PRED_CORNER_RES = 'pred_corner_residual'
PRED_FORCED_6_DISTANCE = 'pred_forced_6_distance'
PRED_CTR = 'pred_ctr'
PRED_ANGLE_CLS = 'pred_angle_cls'
PRED_ANGLE_RES = 'pred_angle_residual'
PRED_VELOCITY = 'pred_velocity'
PRED_ATTRIBUTE = 'pred_attribute'

# origin fpointnet network
PRED_STAGE_1_CTR = 'pred_stage_1_ctr'

# Fpointnet Regression offset
PRED_OFFSET_CLS = 'pred_offset_cls'
PRED_OFFSET_REG = 'pred_offset_reg'

# center regression network
PRED_CTR_REGRESSION = 'pred_ctr_regression'

# detectron reg loss
MASK_XYZ_MEAN = 'mask_xyz_mean'

# keys for decoded result
PRED_3D_BBOX_BEV = 'pred_3d_bbox_bev'
PRED_3D_BBOX = 'pred_3d_bbox'
PRED_3D_BBOX_RY = 'pred_3d_bbox_ry'
PRED_3D_BBOX_CORNERS = 'pred_3d_bbox_corners'
PRED_3D_CLS_CATEGORY = 'pred_3d_class_Category'
PRED_3D_SCORE = 'pred_3d_score'
PRED_3D_VELOCITY = 'pred_3d_velocity'
PRED_3D_ATTRIBUTE = 'pred_3d_attribute'

PRED_3D_BBOX_BEV_2 = 'pred_3d_bbox_bev_2'
PRED_3D_BBOX_2 = 'pred_3d_bbox_2'
PRED_3D_BBOX_RY_2 = 'pred_3d_bbox_ry_2'
pred_3D_BBOX_CORNERS_2 = 'pred_3d_bbox_corners_2'
pred_3D_NMS_INDICES_2 = 'pred_3d_nms_indices_2'

############################
# keys for point_rcnn
############################
# stage_1
PRED_XBIN_CLS = 'pred_xbin_cls'
PRED_XBIN_RES = 'pred_xbin_res'
PRED_ZBIN_CLS = 'pred_zbin_cls'
PRED_ZBIN_RES = 'pred_zbin_res'
PRED_Y_RES = 'pred_y_res'

# stage_2
PRED_XBIN_CLS_STAGE2 = 'pred_xbin_cls_2'
PRED_XBIN_RES_STAGE2 = 'pred_xbin_res_2'
PRED_ZBIN_CLS_STAGE2 = 'pred_zbin_cls_2'
PRED_ZBIN_RES_STAGE2 = 'pred_zbin_res_2'
PRED_Y_RES_STAGE2 = 'pred_y_res_2'
PRED_OFFSET_STAGE2 = 'pred_offset_2'
PRED_ANCHORS_STAGE2 = 'pred_anchors_2'
PRED_ANGLE_CLS_STAGE2 = 'pred_angle_cls_2'
PRED_ANGLE_RES_STAGE2 = 'pred_angle_res_2'

# f-pointnet
PRED_CTR_STAGE2 = 'pred_ctr_2'

#############################
# keys for training
#############################
GT_BOXES_ANCHORS_3D = 'gt_boxes_anchors_3d'
GT_BOXES_ANCHORS = 'gt_boxes_anchors'
GT_PMASK = 'gt_pmask'
GT_NMASK = 'gt_nmask'
GT_CLS = 'gt_cls'
GT_OFFSET = 'gt_offset'
GT_CTR = 'gt_ctr'
GT_CORNER_RES = 'gt_corner_residual'
GT_FORCED_6_DISTANCE = 'gt_forced_6_distance'
GT_ANGLE_CLS = 'gt_angle_cls'
GT_ANGLE_RES = 'gt_angle_res'
GT_OFFSET_CLS = 'gt_offset_cls'
GT_OFFSET_RES = 'gt_offset_res'
LOSS = 'loss'
GT_ATTRIBUTE = 'gt_3d_attribute'
GT_VELOCITY = 'gt_3d_velocity'

############################
# keys for point_rcnn
############################
GT_XBIN_CLS = 'gt_xbin_cls'
GT_XBIN_RES = 'gt_xbin_res'
GT_ZBIN_CLS = 'gt_zbin_cls'
GT_ZBIN_RES = 'gt_zbin_res'
GT_Y_RES = 'gt_y_res'

# stage2
# original
GT_PMASK_STAGE2 = 'gt_pmask2'
GT_NMASK_STAGE2 = 'gt_nmask2'
GT_XBIN_CLS_STAGE2 = 'gt_xbin_cls2'
GT_XBIN_RES_STAGE2 = 'gt_xbin_res2'
GT_ZBIN_CLS_STAGE2 = 'gt_zbin_cls2'
GT_ZBIN_RES_STAGE2 = 'gt_zbin_res2'
GT_Y_RES_STAGE2 = 'gt_y_res2'
GT_OFFSET_STAGE2 = 'gt_offset2'
GT_ANGLE_CLS_STAGE2 = 'gt_angle_cls2'
GT_ANGLE_RES_STAGE2 = 'gt_angle_res2'
GT_BOXES_ANCHORS_STAGE2 = 'gt_boxes_anchors2'

# F-PointNet
GT_CTR_STAGE2 = 'gt_center2'


############################
# key for evaluations
############################
EVAL_GT_BOXES_CORNERS = 'eval_gt_boxes_corners'
EVAL_ANCHORS_BOXES_CORNERS = 'eval_anchors_boxes_corners'
EVAL_GT_BOXES_CORNERS_2 = 'eval_gt_boxes_corners_2'
EVAL_ANCHORS_BOXES_CORNERS_2 = 'eval_anchors_boxes_corners_2'

############################
# Key for Corner Loss
############################
# labels
CORNER_LOSS_GT_BOXES_CORNERS = 'corner_loss_gt_boxes_corners'
CORNER_LOSS_GT_BOXES_CORNERS_FLIP = 'corner_loss_gt_boxes_corners_flip'
# output
CORNER_LOSS_PRED_BOXES_CORNERS = 'corner_loss_pred_boxes_corners'

#############################
# Key for IoU loss
#############################
GT_IOU_3D_VALUE = 'gt_iou_3d_matrix'
GT_IOU_BEV_VALUE = 'gt_iou_bev_matrix'
# output
PRED_IOU_3D_VALUE = 'pred_iou_3d_value'
PRED_IOU_BEV_VALUE = 'pred_iou_bev_value'
