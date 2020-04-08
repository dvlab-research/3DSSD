import numpy as np

from core.config import cfg

import utils.tf_ops.nms.cython_nms as cython_nms
from utils.voxelnet_aug import check_inside_points

def iou_guided_nms(iou_matrix, pred_boxes_3d, pred_scores, pred_iou_3d, iou_thresh):
    """ Calculate iou guided nms, using IoU 2d and IoU 3d to guide NMS

    Args:
        pred_boxes_bev: [-1, 4], xmin, ymin, xmax,ymax
        pred_boxes_3d: [-1, 7], xyzlhw, theta
        pred_scores: [-1]
        pred_iou_3d: [-1]
        pred_iou_bev: [-1]
        iou_thresh: float
    """    
    keep_idx, pred_boxes_3d, pred_scores = cython_nms.matrix_iou_guided_nms(iou_matrix, pred_boxes_3d, pred_scores, pred_iou_3d, iou_thresh)
    pred_boxes_3d = pred_boxes_3d[keep_idx]
    pred_scores = pred_scores[keep_idx]
    keep_idx = keep_idx.astype(np.int32)

    return keep_idx, pred_boxes_3d, pred_scores



def vote_targets_np(vote_base, gt_boxes_3d):
    """ Generating vote_targets for each vote_base point
    vote_base: [bs, points_num, 3]
    gt_boxes_3d: [bs, gt_num, 7]

    Return:
        vote_mask: [bs, points_num]
        vote_target: [bs, points_num, 3]
    """
    bs, points_num, _ = vote_base.shape
    vote_mask = np.zeros([bs, points_num], dtype=np.float32)
    vote_target = np.zeros([bs, points_num, 3], dtype=np.float32)

    for i in range(bs):
        cur_vote_base = vote_base[i]
        cur_gt_boxes_3d = gt_boxes_3d[i]

        filter_idx = np.where(np.any(np.not_equal(cur_gt_boxes_3d, 0), axis=-1))[0]
        cur_gt_boxes_3d = cur_gt_boxes_3d[filter_idx]

        cur_expand_boxes_3d = cur_gt_boxes_3d.copy()
        cur_expand_boxes_3d[:, 3:-1] += cfg.TRAIN.AUGMENTATIONS.EXPAND_DIMS_LENGTH
        cur_points_mask = check_inside_points(cur_vote_base, cur_expand_boxes_3d) # [pts_num, gt_num]

        cur_vote_mask = np.max(cur_points_mask, axis=1).astype(np.float32)
        vote_mask[i] = cur_vote_mask

        cur_vote_target_idx = np.argmax(cur_points_mask, axis=1) # [pts_num]
        cur_vote_target = cur_gt_boxes_3d[cur_vote_target_idx]
        cur_vote_target[:, 1] = cur_vote_target[:, 1] - cur_vote_target[:, 4] / 2.
        cur_vote_target = cur_vote_target[:, :3] - cur_vote_base
        vote_target[i] = cur_vote_target

    return vote_mask, vote_target



def iou_assign_targets_anchors_np(batch_iou_matrix, batch_points, batch_anchors_3d, batch_gt_boxes_3d, batch_gt_labels, minibatch_size, positive_rate, pos_iou, neg_iou, effective_sample_range, valid_mask):
    """ IoU assign targets function
    batch_iou_matrix: [bs, points_num, cls_num, gt_num]
    batch_points: [bs, points_num, 3]
    batch_anchors_3d: [bs, points_num, cls_num, 7]
    batch_gt_boxes_3d: [bs, gt_num, 7]
    batch_gt_labels: [bs, gt_num]
    valid_mask: [bs, points_num, cls_num]

    return:
        assigned_idx: [bs, points_num, cls_num], int32, the index of groundtruth
        assigned_pmask: [bs, points_num, cls_num], float32
        assigned_nmask: [bs, points_num, cls_num], float32
    """
    bs, pts_num, cls_num, gt_num = batch_iou_matrix.shape

    positive_size = int(minibatch_size * positive_rate)
    
    batch_assigned_idx = np.zeros([bs, pts_num, cls_num], np.int32)
    batch_assigned_pmask = np.zeros([bs, pts_num, cls_num], np.float32)
    batch_assigned_nmask = np.zeros([bs, pts_num, cls_num], np.float32)

    for i in range(bs):
        # first calc the 3d iou matrix or 2d iou
        # pts_num, cls_num, 7
        cur_points = batch_points[i]
        cur_anchors_3d = batch_anchors_3d[i] # [pts_num, cls_num, 7]
        cur_valid_mask = valid_mask[i]

        # gt_num
        cur_gt_labels = batch_gt_labels[i] # [gt_num]
        cur_gt_boxes_3d = batch_gt_boxes_3d[i] # [gt_num, 7]
        iou_matrix = batch_iou_matrix[i] # [pts_num, cls_num, gt_num]

        # first filter gt_boxes
        filter_idx = np.where(np.any(np.not_equal(cur_gt_boxes_3d, 0), axis=-1))[0]
        cur_gt_labels = cur_gt_labels[filter_idx]
        cur_gt_boxes_3d = cur_gt_boxes_3d[filter_idx]
        iou_matrix = iou_matrix[:, :, filter_idx]

        # first we check whether a point is within a box 
        points_mask = check_inside_points(cur_points, cur_gt_boxes_3d) # [pts_num, gt_num]
        sampled_gt_idx = np.argmax(points_mask, axis=-1) # [pts_num]
        # used for generating label_mask
        assigned_gt_label = cur_gt_labels[sampled_gt_idx] # [pts_num]
        assigned_gt_label = assigned_gt_label - 1 # 1... -> 0...
        # used for generating dist_mask
        assigned_gt_boxes = cur_gt_boxes_3d[sampled_gt_idx] # [pts_num, 7]
        # then calc the distance between anchors and assigned_boxes
        dist = np.linalg.norm(cur_anchors_3d[:, :, :3] - assigned_gt_boxes[:, np.newaxis, :3], axis=-1) # [pts_num, cls_num] 
    
        # then we get assigned_idx by whether a point is within an object
        filtered_assigned_idx = filter_idx[sampled_gt_idx] # [pts_num]
        filtered_assigned_idx = np.tile(np.reshape(filtered_assigned_idx, [pts_num, 1]), [1, cls_num])
        batch_assigned_idx[i] = filtered_assigned_idx 
        
        # then we generate pos/neg mask 
        assigned_idx = np.tile(np.reshape(sampled_gt_idx, [pts_num, 1, 1]), [1, cls_num, 1])
        iou_mask = np.tile(np.reshape(np.arange(len(filter_idx)), [1, 1, len(filter_idx)]), [pts_num, cls_num, 1]) # [pts_num, cls_num, len(filter_idx)]
        iou_matrix = np.sum(np.equal(iou_mask, assigned_idx).astype(np.float32) * iou_matrix, axis=-1) # [pts_num, cls_num]
        if cls_num > 1:
            label_mask = np.tile(np.reshape(np.arange(cls_num), [1, cls_num]), [pts_num, 1])
            label_mask = np.equal(label_mask, assigned_gt_label[:, np.newaxis]).astype(np.float32)
        else: 
            label_mask = np.ones([pts_num, cls_num], dtype=np.float32)
        iou_matrix = iou_matrix * label_mask + (1-label_mask) * np.ones_like(iou_matrix) * -1 # count and ignored

        pmask = np.greater_equal(iou_matrix, pos_iou) # [pts_num, gt_num]
        dist_mask = np.less_equal(dist, effective_sample_range)
        pmask = np.logical_and(pmask, dist_mask).astype(np.float32)
        nmask = np.logical_and(np.less(iou_matrix, neg_iou), np.greater_equal(iou_matrix, 0.05)).astype(np.float32)
        pmask = pmask * cur_valid_mask
        nmask = nmask * cur_valid_mask

        # finally let's randomly choice some points
        if minibatch_size != -1: 
            pts_pmask = np.any(pmask, axis=1) # [pts_num]
            pts_nmask = np.any(nmask, axis=1) # [pts_num]

            positive_inds = np.where(pts_pmask)[0]
            cur_positive_num = np.minimum(len(positive_inds), positive_size)
            if cur_positive_num > 0:
                positive_inds = np.random.choice(positive_inds, cur_positive_num, replace=False)
            pts_pmask = np.zeros_like(pts_pmask)
            pts_pmask[positive_inds] = 1
 
            cur_negative_num = minibatch_size - cur_positive_num
            negative_inds = np.where(pts_nmask)[0]       
            cur_negative_num = np.minimum(len(negative_inds), cur_negative_num)
            if cur_negative_num > 0:
                negative_inds = np.random.choice(negative_inds, cur_negative_num, replace=False) 
            pts_nmask = np.zeros_like(pts_nmask)
            pts_nmask[negative_inds] = 1

            pmask = pmask * pts_pmask[:, np.newaxis]
            nmask = nmask * pts_nmask[:, np.newaxis]

        batch_assigned_pmask[i] = pmask
        batch_assigned_nmask[i] = nmask

    return batch_assigned_idx, batch_assigned_pmask, batch_assigned_nmask


def mask_assign_targets_anchors_np(batch_points, batch_anchors_3d, batch_gt_boxes_3d, batch_gt_labels, minibatch_size, positive_rate, pos_iou, neg_iou, effective_sample_range, valid_mask):
    """ Mask assign targets function
    batch_points: [bs, points_num, 3]
    batch_anchors_3d: [bs, points_num, cls_num, 7]
    batch_gt_boxes_3d: [bs, gt_num, 7]
    batch_gt_labels: [bs, gt_num]
    valid_mask: [bs, points_num, cls_num]

    return:
        assigned_idx: [bs, points_num, cls_num], int32, the index of groundtruth
        assigned_pmask: [bs, points_num, cls_num], float32
        assigned_nmask: [bs, points_num, cls_num], float32
    """
    bs, pts_num, cls_num, _ = batch_anchors_3d.shape

    positive_size = int(minibatch_size * positive_rate)
    
    batch_assigned_idx = np.zeros([bs, pts_num, cls_num], np.int32)
    batch_assigned_pmask = np.zeros([bs, pts_num, cls_num], np.float32)
    batch_assigned_nmask = np.zeros([bs, pts_num, cls_num], np.float32)

    for i in range(bs):
        cur_points = batch_points[i]
        cur_anchors_3d = batch_anchors_3d[i] # [pts_num, cls_num, 3/7]
        cur_valid_mask = valid_mask[i] # [pts_num, cls_num]

        # gt_num
        cur_gt_labels = batch_gt_labels[i] # [gt_num]
        cur_gt_boxes_3d = batch_gt_boxes_3d[i] # [gt_num, 7]

        # first filter gt_boxes
        filter_idx = np.where(np.any(np.not_equal(cur_gt_boxes_3d, 0), axis=-1))[0]
        cur_gt_labels = cur_gt_labels[filter_idx]
        cur_gt_boxes_3d = cur_gt_boxes_3d[filter_idx]

        points_mask = check_inside_points(cur_points, cur_gt_boxes_3d) # [pts_num, gt_num]
        sampled_gt_idx = np.argmax(points_mask, axis=-1) # [pts_num]
        # used for label_mask
        assigned_gt_label = cur_gt_labels[sampled_gt_idx] # [pts_num]
        assigned_gt_label = assigned_gt_label - 1 # 1... -> 0...
        # used for dist_mask
        assigned_gt_boxes = cur_gt_boxes_3d[sampled_gt_idx] # [pts_num, 7]
        # then calc the distance between anchors and assigned_boxes
        dist = np.linalg.norm(cur_anchors_3d[:, :, :3] - assigned_gt_boxes[:, np.newaxis, :3], axis=-1) # [pts_num, cls_num]
 
        filtered_assigned_idx = filter_idx[sampled_gt_idx] # [pts_num]
        filtered_assigned_idx = np.tile(np.reshape(filtered_assigned_idx, [pts_num, 1]), [1, cls_num])
        batch_assigned_idx[i] = filtered_assigned_idx

        if cls_num == 1: # anchor_free
            label_mask = np.ones([pts_num, cls_num], dtype=np.float32)
        else: # multiple anchors
            label_mask = np.tile(np.reshape(np.arange(cls_num), [1, cls_num]), [pts_num, 1])
            label_mask = np.equal(label_mask, assigned_gt_label[:, np.newaxis]).astype(np.float32)

        pmask = np.max(points_mask, axis=1) > 0
        dist_mask = np.less_equal(dist, effective_sample_range) # pts_num, cls_num
        pmask = np.logical_and(pmask[:, np.newaxis], dist_mask).astype(np.float32)
        pmask = pmask * label_mask
        pmask = pmask * cur_valid_mask

        nmask = np.max(points_mask, axis=1) == 0
        nmask = np.tile(np.reshape(nmask, [pts_num, 1]), [1, cls_num])
        nmask = nmask * label_mask
        nmask = nmask * cur_valid_mask

        # then randomly sample
        if minibatch_size != -1:
            pts_pmask = np.any(pmask, axis=1) # pts_num
            pts_nmask = np.any(nmask, axis=1) # [pts_num]

            positive_inds = np.where(pts_pmask)[0]
            cur_positive_num = np.minimum(len(positive_inds), positive_size)
            if cur_positive_num > 0:
                positive_inds = np.random.choice(positive_inds, cur_positive_num, replace=False)
            pts_pmask = np.zeros_like(pts_pmask)
            pts_pmask[positive_inds] = 1

            cur_negative_num = minibatch_size - cur_positive_num
            negative_inds = np.where(pts_nmask)[0]
            cur_negative_num = np.minimum(len(negative_inds), cur_negative_num)
            if cur_negative_num > 0:
                negative_inds = np.random.choice(negative_inds, cur_negative_num, replace=False)
            pts_nmask = np.zeros_like(pts_nmask)
            pts_nmask[negative_inds] = 1

            pmask = pmask * pts_pmask[:, np.newaxis]
            nmask = nmask * pts_nmask[:, np.newaxis]

        batch_assigned_pmask[i] = pmask
        batch_assigned_nmask[i] = nmask
    return batch_assigned_idx, batch_assigned_pmask, batch_assigned_nmask




