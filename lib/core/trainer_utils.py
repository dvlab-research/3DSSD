import tensorflow as tf
import numpy as np

from tensorflow.python import pywrap_tensorflow
from core.config import cfg

def get_learning_rate(batch):
    steps = cfg.SOLVER.STEPS
    values = [cfg.SOLVER.BASE_LR] + [cfg.SOLVER.BASE_LR * cfg.SOLVER.GAMMA ** (index + 1) for index, step in enumerate(steps)]
    learning_rate = tf.train.piecewise_constant(batch, steps, values)

    learing_rate = tf.maximum(learning_rate, 0.00001) # CLIP THE LEARNING RATE!
    return learning_rate

def get_bn_decay(batch):
    steps = cfg.SOLVER.STEPS
    values = [cfg.SOLVER.BN_INIT_DECAY] + [cfg.SOLVER.BN_INIT_DECAY * cfg.SOLVER.BN_DECAY_DECAY_RATE ** (index + 1) for index, step in enumerate(steps)]
    bn_momentum = tf.train.piecewise_constant(batch, steps, values)

    bn_decay = tf.minimum(cfg.SOLVER.BN_DECAY_CLIP, 1 - bn_momentum)
    return bn_decay

def average_gradients(tower_grads):
    # ref:
    # https://github.com/tensorflow/models/blob/6db9f0282e2ab12795628de6200670892a8ad6ba/tutorials/image/cifar10/cifar10_multi_gpu_train.py#L103
    # but only contains grads, no vars
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for g in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        grad_and_var = grad
        average_grads.append(grad_and_var)
    return average_grads

def get_variables_in_checkpoint_file(file_name):
    try:
        reader = pywrap_tensorflow.NewCheckpointReader(file_name)
        var_to_shape_map = reader.get_variable_to_shape_map()
        return var_to_shape_map
    except Exception as e:  # pylint: disable=broad-except
        print(str(e))

def get_trainable_parameter(prefix_list):
    """
    Given the prefix list of trainable params and return them
    """
    if len(prefix_list) == 0:
        return tf.trainable_variables()
    else:
        param_list = []
        for prefix in prefix_list:
            param = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, prefix)
            param_list.extend(param)
        return param_list
  
def get_trainable_loss(prefix_list, scope):
    """
    Given the prefix list of trainable loss and return them
    """
    if len(prefix_list) == 0:
        return tf.get_collection('losses', scope) 
    else:
        loss_list = []
        for prefix in prefix_list:
            loss = tf.get_collection('losses',
                '%s%s'%(scope, prefix))
            loss_list.extend(loss)
        return loss_list
