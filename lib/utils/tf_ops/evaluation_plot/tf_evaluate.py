import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os
BASE_DIR = os.path.dirname(__file__)
sys.path.append(BASE_DIR)
evaluate_module=tf.load_op_library(os.path.join(BASE_DIR, 'tf_evaluate_so.so'))

def evaluate(detections, names, numlist, model_name):
    '''
    Input:
        detections: (n, 12)
        names: (m,)
        numlist: (m,)
        model_name: (1,)
    Output:
        precision_image: (NUM_CLASS, 3, 41)
        aos_image: (NUM_CLASS, 3, 41)
        precision_ground: (NUM_CLASS, 3, 41)
        aos_ground: (NUM_CLASS, 3, 41)
        precision_3d: (NUM_CLASS, 3, 41)
        aos_3d: (NUM_CLASS, 3, 41)
    '''
    print(os.path.join(BASE_DIR, 'tf_evaluate_so.so'))
    return evaluate_module.evaluate_plot(detections, names, numlist, model_name)
ops.NoGradient('EvaluatePlot')
