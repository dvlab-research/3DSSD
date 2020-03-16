import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
evaluate_module=tf.load_op_library(os.path.join(BASE_DIR, 'tf_evaluate_so.so'))

def evaluate(detections, names, numlist):
    '''
    Input:
        detections: (n, 12)
        names: (m,)
        numlist: (m,)
    Output:
        precision_image: (NUM_CLASS, 3, 41)
        aos_image: (NUM_CLASS, 3, 41)
        precision_ground: (NUM_CLASS, 3, 41)
        aos_ground: (NUM_CLASS, 3, 41)
        precision_3d: (NUM_CLASS, 3, 41)
        aos_3d: (NUM_CLASS, 3, 41)
    '''
    return evaluate_module.evaluate(detections, names, numlist)
ops.NoGradient('Evaluate')

def calc_iou(detections, groundtruths):
    '''
    detections: [bs, dets_num, 7]
    groundtruths: [bs, gt_num, 7]
    '''
    iou_bev, iou_3d = evaluate_module.calc_iou(detections, groundtruths)
    return iou_bev, iou_3d
ops.NoGradient('CalcIou')

def calc_iou_match(detections, groundtruths):
    '''
    detections: [bs, dets_num, 7]
    groundtruths: [bs, dets_num, 7]
    '''
    iou_bev, iou_3d = evaluate_module.calc_matching_iou(detections, groundtruths)
    return iou_bev, iou_3d
ops.NoGradient('CalcMatchingIou')
