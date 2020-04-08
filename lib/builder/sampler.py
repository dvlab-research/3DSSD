import tensorflow as tf
import numpy as np

from core.config import cfg
from utils.tf_ops.sampling.tf_sampling import gather_by_mask

class Sampler:
    """
    Sample assigned proposals out according to mask tensor
    """
    def __init__(self, stage_index):
        if stage_index == 0:
            self.sampler_cfg = cfg.MODEL.FIRST_STAGE
        elif stage_index == 1:
            self.sampler_cfg = cfg.MODEL.SECOND_STAGE
  
        self.proposal_num = self.sampler_cfg.MINIBATCH_NUM
         

    def gather_list(self, mask, tensor_list):
        """
        Gather according to mask
        mask: [bs, pts_num, 1]
        tensor_list: tensors with different shapes
                     ---> {
                         [bs, pts_num],
                         [bs, pts_num, 1, -1],
                         [bs, pts_num, 1] 
                     }
        return: [bs, proposal_num, ...]
        """
        mask = tf.squeeze(mask, axis=-1)
        return_list = []
        for tensor in tensor_list:
            if tensor is not None:
                tensor = self.gather_tensor(mask, tensor)
            return_list.append(tensor)
        return return_list
            
            
    def gather_tensor(self, mask, tensor):
        tensor_shape = tensor.get_shape().as_list()
        tensor_dtype = tensor.dtype
        tensor = tf.cast(tensor, tf.float32)
        if len(tensor_shape) == 2:
            tensor = tf.expand_dims(tensor, axis=-1) 
            tensor = gather_by_mask(self.proposal_num, tensor, mask)
            tensor = tf.squeeze(tensor, axis=-1)
        elif len(tensor_shape) == 3:
            tensor = gather_by_mask(self.proposal_num, tensor, mask)
        elif len(tensor_shape) == 4: # bs, pts_num, 1, -1
            tensor = tf.squeeze(tensor, axis=2)
            tensor = gather_by_mask(self.proposal_num, tensor, mask)
            tensor = tf.expand_dims(tensor, axis=2)
        else: raise Exception('Not support for more than 4 dimensions gather')
        
        tensor = tf.cast(tensor, tensor_dtype)
        return tensor
