import tensorflow as tf
from tensorflow.python.framework import ops
import sys
import os
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sampling_module=tf.load_op_library(os.path.join(BASE_DIR, 'tf_sampling_so.so'))
def prob_sample(inp,inpr):
    '''
input:
    batch_size * ncategory float32
    batch_size * npoints   float32
returns:
    batch_size * npoints   int32
    '''
    return sampling_module.prob_sample(inp,inpr)
ops.NoGradient('ProbSample')
# TF1.0 API requires set shape in C++
#@tf.RegisterShape('ProbSample')
#def _prob_sample_shape(op):
#    shape1=op.inputs[0].get_shape().with_rank(2)
#    shape2=op.inputs[1].get_shape().with_rank(2)
#    return [tf.TensorShape([shape2.dims[0],shape2.dims[1]])]
def gather_point(inp,idx):
    '''
input:
    batch_size * ndataset * c   float32
    batch_size * npoints        int32
returns:
    batch_size * npoints * c    float32
    '''
    return sampling_module.gather_point(inp,idx)
#@tf.RegisterShape('GatherPoint')
#def _gather_point_shape(op):
#    shape1=op.inputs[0].get_shape().with_rank(3)
#    shape2=op.inputs[1].get_shape().with_rank(2)
#    return [tf.TensorShape([shape1.dims[0],shape2.dims[1],shape1.dims[2]])]
@tf.RegisterGradient('GatherPoint')
def _gather_point_grad(op,out_g):
    inp=op.inputs[0]
    idx=op.inputs[1]
    return [sampling_module.gather_point_grad(inp,idx,out_g),None]
def farthest_point_sample(npoint,inp):
    '''
input:
    int32
    batch_size * ndataset * c   float32
returns:
    batch_size * npoint         int32
    '''
    return sampling_module.farthest_point_sample(inp, npoint)
ops.NoGradient('FarthestPointSample')

def farthest_point_sample_with_distance(npoint, dist):
    """
    Args:
        int32
        batch_size * ndataset * ndataset float32, distance matrics
    Return:
        batch_size * npoint, int32
    """
    return sampling_module.farthest_point_sample_with_distance(dist, npoint)
ops.NoGradient('FarthestPointSampleWithDistance')
    
def farthest_point_sample_with_preidx(npoint, inp, preidx):
    """
    Args:
        int32
        batch_size * ndataset * ndataset float32, distance matrics
    Return:
        batch_size * npoint, int32
    """
    return sampling_module.farthest_point_sample_with_preidx(inp, preidx, npoint)
ops.NoGradient('FarthestPointSampleWithPreidx')

def gather_by_mask(proposal_num, inp, mask):
    """
    proposal_num: training proposal number for stage2
    inp: [bs, pts_num, -1]
    mask: [bs, pts_num]

    out: [bs, proposal_num, -1]
    """
    return sampling_module.gather_by_mask(inp, mask, proposal_num)
ops.NoGradient('GatherByMask')

