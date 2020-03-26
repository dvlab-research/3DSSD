import tensorflow as tf
import numpy as np
import utils.tf_util as tf_util

def align_channel_network(info, mlp_list, bn, is_training, bn_decay, scope):
    with tf.variable_scope(scope) as sc:
        for i, num_out_channel in enumerate(mlp_list):
            info = tf_util.conv2d(info, 
                   num_out_channel, 
                   [1, 1], 
                   padding='VALID', 
                   bn=bn, 
                   is_training=is_training, 
                   scope='conv%d'%i, 
                   bn_decay=bn_decay)
    return info
