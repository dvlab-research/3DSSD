from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import six
import os
import os.path as osp
import copy
from ast import literal_eval

import numpy as np
import yaml

from utils.collections import AttrDict

__C = AttrDict()
# Consumers can get config by:
#   from fast_rcnn_config import cfg
cfg = __C

__C.ROOT_DIR = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))

# Random note: avoid using '.ON' as a config key since yaml converts it to True;
# prefer 'ENABLED' instead
# ---------------------------------------------------------------------------- #
# DATASET Options 
# ---------------------------------------------------------------------------- #
__C.DATASET = AttrDict()

__C.DATASET.SELF_SPLIT_DATASET = False 

__C.DATASET.POINT_CLOUD_RANGE = (-40, 40, -5, 3, 0, 70)

__C.DATASET.VOXEL_SIZE = (0.2, 0.2, 0.2)

__C.DATASET.MAX_NUMBER_OF_POINT_PER_VOXEL = 100


# mix-up data augmentation
__C.DATASET.MIN_POINTS_NUM = 5
__C.DATASET.TYPE = 'KITTI' # KITTI, NuScenes, Lyft


__C.DATASET.KITTI = AttrDict()

__C.DATASET.KITTI.PREPROCESS_IMG_SIZE = (360, 1200)

__C.DATASET.KITTI.PREPROCESS_IMG_MEAN = [123.68, 116.779, 103.939]

__C.DATASET.KITTI.CLS_LIST = ('Car', 'Pedestrian', 'Cyclist')

__C.DATASET.KITTI.BASE_DIR_PATH = 'dataset/KITTI/object'

__C.DATASET.KITTI.TRAINVAL_LIST = 'dataset/KITTI/object/trainval.txt'

__C.DATASET.KITTI.TRAIN_LIST = 'dataset/KITTI/object/train.txt'

__C.DATASET.KITTI.VAL_LIST = 'dataset/KITTI/object/val.txt'

__C.DATASET.KITTI.TEST_LIST = 'dataset/KITTI/object/test.txt'

__C.DATASET.KITTI.SAVE_NUMPY_PATH = 'data/KITTI'

__C.DATASET.NUSCENES = AttrDict()

__C.DATASET.NUSCENES.MAX_NUMBER_OF_VOXELS = 32768

__C.DATASET.NUSCENES.MAX_CUR_SAMPLE_POINTS_NUM = 16384

__C.DATASET.NUSCENES.NSWEEPS = 10

__C.DATASET.NUSCENES.INPUT_FEATURE_CHANNEL = 4


# ---------------------------------------------------------------------------- #
# Training options
# ---------------------------------------------------------------------------- #
__C.TRAIN = AttrDict()

# Augmentations
__C.TRAIN.AUGMENTATIONS = AttrDict() 
__C.TRAIN.AUGMENTATIONS.OPEN = False
__C.TRAIN.AUGMENTATIONS.EXPAND_DIMS_LENGTH = 0.1

# translate, rotate, scale
__C.TRAIN.AUGMENTATIONS.PROB_TYPE = 'Simultaneously' # Simultaneously or Seperately
__C.TRAIN.AUGMENTATIONS.PROB = [0.5, 0.5, 0.5]
__C.TRAIN.AUGMENTATIONS.RANDOM_ROTATION_RANGE = 45 / 180 * np.pi 
__C.TRAIN.AUGMENTATIONS.RANDOM_SCALE_RANGE = 0.1

# random flip or not
__C.TRAIN.AUGMENTATIONS.FLIP = False

# mix up data augmentation
__C.TRAIN.AUGMENTATIONS.MIXUP = AttrDict()
__C.TRAIN.AUGMENTATIONS.MIXUP.OPEN = False
__C.TRAIN.AUGMENTATIONS.MIXUP.SAVE_NUMPY_PATH = 'mixup_database'
__C.TRAIN.AUGMENTATIONS.MIXUP.PC_LIST = 'train'
__C.TRAIN.AUGMENTATIONS.MIXUP.CLASS = ('Car', )
__C.TRAIN.AUGMENTATIONS.MIXUP.NUMBER = (15, )

# single object data augmentation
__C.TRAIN.AUGMENTATIONS.SINGLE_AUG = AttrDict()
__C.TRAIN.AUGMENTATIONS.SINGLE_AUG.ROTATION_PERTURB = [-np.pi / 3, np.pi / 3]
__C.TRAIN.AUGMENTATIONS.SINGLE_AUG.CENTER_NOISE_STD = [1.0, 1.0, 0.] 
__C.TRAIN.AUGMENTATIONS.SINGLE_AUG.RANDOM_SCALE_RANGE = [1.0, 1.0] 
__C.TRAIN.AUGMENTATIONS.SINGLE_AUG.SCALE_3_DIMS = False
__C.TRAIN.AUGMENTATIONS.SINGLE_AUG.FIX_LENGTH = False
# Augmentations

# training configuration
__C.TRAIN.CONFIG = AttrDict()

__C.TRAIN.CONFIG.BATCH_SIZE = 1

__C.TRAIN.CONFIG.GPU_NUM = 1

__C.TRAIN.CONFIG.MAX_ITERATIONS = 500

__C.TRAIN.CONFIG.CHECKPOINT_INTERVAL = 50

__C.TRAIN.CONFIG.MAX_CHECKPOINTS_TO_KEEP = 10

__C.TRAIN.CONFIG.SUMMARY_INTERVAL = 10

__C.TRAIN.CONFIG.SUMMARY_HISTOGRAMS = True

__C.TRAIN.CONFIG.SUMMARY_IMG_IMAGES = True

__C.TRAIN.CONFIG.SUMMARY_BEV_IMAGES = True

# STD training
__C.TRAIN.CONFIG.ONLY_IOU_BRANCH = False

__C.TRAIN.CONFIG.ONLY_TRAIN_BEV_IOU_BRANCH = False

__C.TRAIN.CONFIG.FIX_BN_LAYER = False

__C.TRAIN.CONFIG.FIX_IMG_FEATURE_EXTRACTOR = False


# ---------------------------------------------------------------------------- #
# Data loader options
# ---------------------------------------------------------------------------- #
__C.DATA_LOADER = AttrDict()

# Number of Python threads to use for the data loader (warning: using too many
# threads can cause GIL-based interference with Python Ops leading to *slower*
# training; 4 seems to be the sweet spot in our experience)
__C.DATA_LOADER.NUM_THREADS = 4


# ---------------------------------------------------------------------------- #
# Inference ('test') options
# ---------------------------------------------------------------------------- #
__C.TEST = AttrDict()

# load or not load gt
__C.TEST.WITH_GT = True

# ---------------------------------------------------------------------------- #
# Model options
# ---------------------------------------------------------------------------- #
__C.MODEL = AttrDict()

__C.MODEL.POINTS_NUM_FOR_TRAINING = 16384 


__C.MODEL.USING_ORIGIN_PLANE = False

# SingleStage or DoubleStage
__C.MODEL.TYPE = 'SingleStage'

# Paths config
__C.MODEL.PATH = AttrDict()

__C.MODEL.PATH.CHECKPOINT_DIR = 'log' 

__C.MODEL.PATH.EVALUATION_DIR= 'result'

__C.MODEL.BBOX_REG_WEIGHT = (1., 1., 1., 1., 1., 1.)

__C.MODEL.ENLARGE_ANCHORS_LENGTH = 0.1 

# Network
__C.MODEL.NETWORK = AttrDict()

__C.MODEL.NETWORK.USE_BN = True
__C.MODEL.NETWORK.SYNC_BN = False
__C.MODEL.NETWORK.USE_GN = False

__C.MODEL.ANGLE_CLS_NUM = 12

# backbone
__C.MODEL.NETWORK.AGGREGATION_SA_FEATURE = False

# VoteNet
__C.MODEL.NETWORK.ONLY_POS_DEFORMABLE_LOSS = False
__C.MODEL.MAX_TRANSLATE_RANGE = [-3.0, -2.0, -3.0]

# model architecture
__C.MODEL.NETWORK.FIRST_STAGE = AttrDict()
################################################# 
# for each layer
# 0: use xyz from which layer, 1: use feature from which layer
# 2: radius_list, 3: nsample_list, 4: mlp_list 5: bn, 
# 6: fps_sample_range_list 7: fps_method_list (F-FPS, D-FPS, FS) 8: fps_npoint_list 
# 9: former_fps_idx: (fps_idx from which layer)
# 10: use_attention: whether using attention in grouping points
# 11: layer type: ['SA_Layer', 'Vote_Layer', 'FP_Layer', 'SA_Layer_SSG_Last']
# 12: scope, 
# 13: dilated_group
# 14: deformable-center, /e.g: votenet center
# 15: aggregation channel
################################################# 
__C.MODEL.NETWORK.FIRST_STAGE.ARCHITECTURE = [
    [[0], [0], [0.2,0.4,0.8], [32,64,128], [[32,32,64], [64,64,128], [64,96,128]], True,
     [-1], ['D-FPS'], [4096], 
     -1, False, 'SA_Layer', 'layer1', False, -1, 128], # layer1
    [[1], [1], [0.4,0.8,1.6], [64,64,64], [[64,64,128], [128,128,256], [128,128,256]], True, 
     [-1], ['FS'], [1024], 
     -1, True, 'SA_Layer', 'layer2', False, -1, 256], # layer2
    [[2], [2], [1.6,4.8], [64, 128], [[128,128,256], [128,256,256]], True,
     [1024, -1], ['F-FPS', 'D-FPS'], [256, 0], 
     -1, True, 'SA_Layer', 'layer3', False, -1, 256], # layer3
    [[3], [3], -1, -1, [256], True,
     [-1], [-1], [-1], 
     -1, -1, 'Vote_Layer', 'layer3_vote', False, -1, -1], # layer3-vote
    [[2], [2], [1.6,3.2,4.8], [64,64,128], [[128,128,256], [128,128,256], [128,256,256]], True, 
     [1024, -1], ['F-FPS', 'D-FPS'], [0, 256], 
     3, True, 'SA_Layer', 'layer3_frf', False, -1, 256], # layer3_frf
    [[5], [5], [4.8, 6.4], [32, 32], [[256,256,512], [256,512,1024]], True, 
     [-1], ['D-FPS'], [256], 
     -1, False, 'SA_Layer', 'layer4', False, 4, 512], # layer4
]

################################################# 
# for each layer
# 0: use xyz from which layer, 1: use feature from which layer
# 2: composed by which op: 'conv1d', 'conv2d', 'fc', etc
# 3: mlp_list
# 4: bn
# 5: layer type: 'Det' or 'IoU' 
# 6: scope
################################################# 
__C.MODEL.NETWORK.FIRST_STAGE.HEAD = [[[6], [6], 'conv1d', [128,], True, 'Det', 'detection_head']]

#################################################
# PointsPooler
# 0: layer type: 'RegionPool'(PointRCNN), 'PointsPool'(STD)
# 1: additional information keys: ['r', 'mask', 'dist']
# 2: align_mlp_list
# 3: sample_pts_num
# 4: pool_size, [l,h,w, sample_num] for PointsPool only
# 5: vfe_channel_list, for PointsPool only 
# 5: bn
# 6: scope
#################################################
__C.MODEL.NETWORK.FIRST_STAGE.POINTS_POOLER = ['RegionPool', ['mask', 'dist'], [128], 512, [6, 6, 6, 10], [128], True, 'roi_pool']

__C.MODEL.NETWORK.FIRST_STAGE.POOLER_MASK_THRESHOLD = 0.5


# model architecture for second stage
__C.MODEL.NETWORK.SECOND_STAGE = AttrDict()
################################################# 
# for each layer
# 0: use xyz from which layer, 1: use feature from which layer
# 2: radius_list, 3: nsample_list, 4: mlp_list 5: bn, 
# 6: fps_end_idx_list 7: fps_method_list (F-FPS, D-FPS, FS) 8: fps_npoint_list 
# 9: former_fps_idx: (fps_idx from which layer)
# 10: use_attention: whether using attention in grouping points
# 11: layer type: [SA layer, VoteLayer]
# 12: scope, 
# 13: dilated_group
# 14: deformable-center, /e.g: votenet center
# 15: aggregation channel
################################################# 
__C.MODEL.NETWORK.SECOND_STAGE.ARCHITECTURE = [
]

################################################# 
# for each layer
# 0: use xyz from which layer, 1: use feature from which layer
# 2: mlp_list
# 3: bn
# 4: scope
################################################# 
__C.MODEL.NETWORK.SECOND_STAGE.HEAD = [[6], [6], [128,], True, 'detection_head']


# RPN
__C.MODEL.FIRST_STAGE = AttrDict()
__C.MODEL.FIRST_STAGE.TYPE = 'PointRCNN' # PointRCNN or STD or 3DSSD
__C.MODEL.FIRST_STAGE.MAX_OUTPUT_NUM = 300
__C.MODEL.FIRST_STAGE.NMS_THRESH = 0.7
__C.MODEL.FIRST_STAGE.NUM_OBJECT_POINT = 128 # for std only
__C.MODEL.FIRST_STAGE.MINIBATCH_NUM = 64
__C.MODEL.FIRST_STAGE.MINIBATCH_RATIO = 0.25
# whether using points iou to sample anchors
__C.MODEL.FIRST_STAGE.POINTS_SAMPLE_IOU = False
# Log-Anchor, Dist-Anchor, Dist-Anchor-free
__C.MODEL.FIRST_STAGE.REGRESSION_METHOD = 'Dist-Anchor' 
# for nuscenes
__C.MODEL.FIRST_STAGE.REGRESSION_MULTI_HEAD = False
__C.MODEL.FIRST_STAGE.MULTI_HEAD_DISTRUBUTE = [['car'], ['construction_vehicle', 'truck'], ['bus', 'trailer'], ['barrier'], ['motorcycle', 'bicycle'], ['pedestrian', 'traffic_cone']]
# Sigmoid or Softmax
__C.MODEL.FIRST_STAGE.CLS_ACTIVATION = 'Sigmoid' 
# assign P/N anchors
__C.MODEL.FIRST_STAGE.ASSIGN_METHOD = 'IoU' # IoU or Mask
__C.MODEL.FIRST_STAGE.IOU_SAMPLE_TYPE = '3D' # 3D or BEV or Point
__C.MODEL.FIRST_STAGE.CLASSIFICATION_POS_IOU = 0.7
__C.MODEL.FIRST_STAGE.CLASSIFICATION_NEG_IOU = 0.55
# FCOS center_ness label
__C.MODEL.FIRST_STAGE.CLASSIFICATION_LOSS = AttrDict()
__C.MODEL.FIRST_STAGE.CLASSIFICATION_LOSS.TYPE = 'Center-ness' # Center-ness or Is-Not
__C.MODEL.FIRST_STAGE.CLASSIFICATION_LOSS.CENTER_NESS_LABEL_RANGE = (0.0, 1.0)
# using softmax-cross-entropy loss
__C.MODEL.FIRST_STAGE.CLASSIFICATION_LOSS.SOFTMAX_SAMPLE_RANGE = 10.0
######### Whether Add Corner loss here #########
__C.MODEL.FIRST_STAGE.CORNER_LOSS = False
__C.MODEL.FIRST_STAGE.PREDICT_ATTRIBUTE_AND_VELOCITY = False



# second stage
__C.MODEL.SECOND_STAGE = AttrDict()
__C.MODEL.SECOND_STAGE.NUM_OBJECT_POINT = 512 
__C.MODEL.SECOND_STAGE.NMS_THRESH = 0.7
__C.MODEL.SECOND_STAGE.MAX_OUTPUT_NUM = 100
__C.MODEL.SECOND_STAGE.MINIBATCH_NUM = 64
__C.MODEL.SECOND_STAGE.MINIBATCH_RATIO = 0.25
# Log-Anchor, Dist-Anchor, Dist-Anchor-free
__C.MODEL.SECOND_STAGE.REGRESSION_METHOD = 'Dist-Anchor' 
# for nuscenes
__C.MODEL.SECOND_STAGE.REGRESSION_MULTI_HEAD = False
__C.MODEL.SECOND_STAGE.MULTI_HEAD_DISTRUBUTE = [['car'], ['construction_vehicle', 'truck'], ['bus', 'trailer'], ['barrier'], ['motorcycle', 'bicycle'], ['pedestrian', 'traffic_cone']]
# Sigmoid or Softmax
__C.MODEL.SECOND_STAGE.CLS_ACTIVATION = 'Sigmoid' 
# assign P/N anchors
__C.MODEL.SECOND_STAGE.ASSIGN_METHOD = 'IoU' # IoU or Mask
__C.MODEL.SECOND_STAGE.IOU_SAMPLE_TYPE = 'BEV' # 3D or BEV
__C.MODEL.SECOND_STAGE.CLASSIFICATION_POS_IOU = 0.7
__C.MODEL.SECOND_STAGE.CLASSIFICATION_NEG_IOU = 0.55
# FCOS center_ness label
__C.MODEL.SECOND_STAGE.CLASSIFICATION_LOSS = AttrDict()
__C.MODEL.SECOND_STAGE.CLASSIFICATION_LOSS.TYPE = 'Center-ness' # Center-ness or Is-Not 
__C.MODEL.SECOND_STAGE.CLASSIFICATION_LOSS.CENTER_NESS_LABEL_RANGE = (0.0, 1.0)
# using softmax-cross-entropy loss
__C.MODEL.SECOND_STAGE.CLASSIFICATION_LOSS.SOFTMAX_SAMPLE_RANGE = 10.0
# Corner loss
__C.MODEL.SECOND_STAGE.CORNER_LOSS = False
__C.MODEL.SECOND_STAGE.PREDICT_ATTRIBUTE_AND_VELOCITY = False

# ---------------------------------------------------------------------------- #
# Solver options
# Note: all solver options are used exactly as specified; the implication is
# that if you switch from training on 1 GPU to N GPUs, you MUST adjust the
# solver configuration accordingly. We suggest using gradual warmup and the
# linear learning rate scaling rule as described in
# "Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour" Goyal et al.
# https://arxiv.org/abs/1706.02677
# ---------------------------------------------------------------------------- #
__C.SOLVER = AttrDict()

# e.g 'SGD', 'Adam'
__C.SOLVER.TYPE = 'SGD'

# Base learning rate for the specified schedule
__C.SOLVER.BASE_LR = 0.001

# BN_DECAY
__C.SOLVER.BN_INIT_DECAY = 0.5
__C.SOLVER.BN_DECAY_DECAY_RATE = 0.5
__C.SOLVER.BN_DECAY_CLIP = 0.99

# Schedule type (see functions in utils.lr_policy for options)
# E.g., 'step', 'steps_with_decay', ...
__C.SOLVER.LR_POLICY = 'step'

# Some LR Policies (by example):
# 'step'
#   lr = SOLVER.BASE_LR * SOLVER.GAMMA ** (cur_iter // SOLVER.STEP_SIZE)
# 'steps_with_decay'
#   SOLVER.STEPS = [0, 60000, 80000]
#   SOLVER.GAMMA = 0.1
#   lr = SOLVER.BASE_LR * SOLVER.GAMMA ** current_step
#   iters [0, 59999] are in current_step = 0, iters [60000, 79999] are in
#   current_step = 1, and so on
# 'steps_with_lrs'
#   SOLVER.STEPS = [0, 60000, 80000]
#   SOLVER.LRS = [0.02, 0.002, 0.0002]
#   lr = LRS[current_step]

# Hyperparameter used by the specified policy
# For 'step', the current LR is multiplied by SOLVER.GAMMA at each step
__C.SOLVER.GAMMA = 0.1

# Uniform step size for 'steps' policy
__C.SOLVER.STEP_SIZE = 30000

# Non-uniform step iterations for 'steps_with_decay' or 'steps_with_lrs'
# policies
__C.SOLVER.STEPS = []

# Learning rates to use with 'steps_with_lrs' policy
__C.SOLVER.LRS = []

# Maximum number of SGD iterations
__C.SOLVER.MAX_ITER = 40000

# Momentum to use with SGD
__C.SOLVER.MOMENTUM = 0.9

# L2 regularization hyperparameter
__C.SOLVER.WEIGHT_DECAY = 0.0005

# Whether to double the learning rate for bias
__C.SOLVER.BIAS_DOUBLE_LR = True

# Whether to have weight decay on bias as well
__C.SOLVER.BIAS_WEIGHT_DECAY = False

# Warm up to SOLVER.BASE_LR over this number of SGD iterations
__C.SOLVER.WARM_UP_ITERS = 500

# Start the warm up from SOLVER.BASE_LR * SOLVER.WARM_UP_FACTOR
__C.SOLVER.WARM_UP_FACTOR = 1.0 / 3.0

# WARM_UP_METHOD can be either 'constant' or 'linear' (i.e., gradual)
__C.SOLVER.WARM_UP_METHOD = 'linear'

# Scale the momentum update history by new_lr / old_lr when updating the
# learning rate (this is correct given MomentumSGDUpdateOp)
__C.SOLVER.SCALE_MOMENTUM = True
# Only apply the correction if the relative LR change exceeds this threshold
# (prevents ever change in linear warm up from scaling the momentum by a tiny
# amount; momentum scaling is only important if the LR change is large)
__C.SOLVER.SCALE_MOMENTUM_THRESHOLD = 1.1

# Suppress logging of changes to LR unless the relative change exceeds this
# threshold (prevents linear warm up from spamming the training log)
__C.SOLVER.LOG_LR_CHANGE_THRESHOLD = 1.1


# ------------------------------
# Data directory
__C.DATA_DIR = osp.abspath(osp.join(__C.ROOT_DIR, 'data'))

# [Deprecate]
__C.POOLING_MODE = 'crop'

# [Deprecate] Size of the pooled region after RoI pooling
__C.POOLING_SIZE = 7

__C.CROP_RESIZE_WITH_MAX_POOL = True

# [Infered value]
__C.CUDA = False

__C.DEBUG = False


# ---------------------------------------------------------------------------- #
# mask heads or keypoint heads that share res5 stage weights and
# training forward computation with box head.
# ---------------------------------------------------------------------------- #
_SHARE_RES5_HEADS = set(
    [
        'mask_rcnn_heads.mask_rcnn_fcn_head_v0upshare',
    ]
)


def assert_and_infer_cfg(make_immutable=True):
    """Call this function in your script after you have finished setting all cfg
    values that are necessary (e.g., merging a config from a file, merging
    command line config options, etc.). By default, this function will also
    mark the global cfg as immutable to prevent changing the global cfg settings
    during script execution (which can lead to hard to debug errors or code
    that's harder to understand than is necessary).
    """
    if __C.MODEL.RPN_ONLY or __C.MODEL.FASTER_RCNN:
        __C.RPN.RPN_ON = True
    if __C.RPN.RPN_ON or __C.RETINANET.RETINANET_ON:
        __C.TEST.PRECOMPUTED_PROPOSALS = False
    if __C.MODEL.LOAD_IMAGENET_PRETRAINED_WEIGHTS:
        assert __C.RESNETS.IMAGENET_PRETRAINED_WEIGHTS, \
            "Path to the weight file must not be empty to load imagenet pertrained resnets."
    if set([__C.MRCNN.ROI_MASK_HEAD, __C.KRCNN.ROI_KEYPOINTS_HEAD]) & _SHARE_RES5_HEADS:
        __C.MODEL.SHARE_RES5 = True
    if make_immutable:
        cfg.immutable(True)


def merge_cfg_from_file(cfg_filename):
    """Load a yaml config file and merge it into the global config."""
    with open(cfg_filename, 'r') as f:
        yaml_cfg = AttrDict(yaml.load(f))
    _merge_a_into_b(yaml_cfg, __C)

cfg_from_file = merge_cfg_from_file


def merge_cfg_from_cfg(cfg_other):
    """Merge `cfg_other` into the global config."""
    _merge_a_into_b(cfg_other, __C)


def merge_cfg_from_list(cfg_list):
    """Merge config keys, values in a list (e.g., from command line) into the
    global config. For example, `cfg_list = ['TEST.NMS', 0.5]`.
    """
    assert len(cfg_list) % 2 == 0
    for full_key, v in zip(cfg_list[0::2], cfg_list[1::2]):
        # if _key_is_deprecated(full_key):
        #     continue
        # if _key_is_renamed(full_key):
        #     _raise_key_rename_error(full_key)
        key_list = full_key.split('.')
        d = __C
        for subkey in key_list[:-1]:
            assert subkey in d, 'Non-existent key: {}'.format(full_key)
            d = d[subkey]
        subkey = key_list[-1]
        assert subkey in d, 'Non-existent key: {}'.format(full_key)
        value = _decode_cfg_value(v)
        value = _check_and_coerce_cfg_value_type(
            value, d[subkey], subkey, full_key
        )
        d[subkey] = value

cfg_from_list = merge_cfg_from_list


def _merge_a_into_b(a, b, stack=None):
    """Merge config dictionary a into config dictionary b, clobbering the
    options in b whenever they are also specified in a.
    """
    assert isinstance(a, AttrDict), 'Argument `a` must be an AttrDict'
    assert isinstance(b, AttrDict), 'Argument `b` must be an AttrDict'

    for k, v_ in a.items():
        full_key = '.'.join(stack) + '.' + k if stack is not None else k
        # a must specify keys that are in b
        if k not in b:
            # if _key_is_deprecated(full_key):
            #     continue
            # elif _key_is_renamed(full_key):
            #     _raise_key_rename_error(full_key)
            # else:
            raise KeyError('Non-existent config key: {}'.format(full_key))

        v = copy.deepcopy(v_)
        v = _decode_cfg_value(v)
        v = _check_and_coerce_cfg_value_type(v, b[k], k, full_key)

        # Recursively merge dicts
        if isinstance(v, AttrDict):
            try:
                stack_push = [k] if stack is None else stack + [k]
                _merge_a_into_b(v, b[k], stack=stack_push)
            except BaseException:
                raise
        else:
            b[k] = v


def _decode_cfg_value(v):
    """Decodes a raw config value (e.g., from a yaml config files or command
    line argument) into a Python object.
    """
    # Configs parsed from raw yaml will contain dictionary keys that need to be
    # converted to AttrDict objects
    if isinstance(v, dict):
        return AttrDict(v)
    # All remaining processing is only applied to strings
    if not isinstance(v, six.string_types):
        return v
    # Try to interpret `v` as a:
    #   string, number, tuple, list, dict, boolean, or None
    try:
        v = literal_eval(v)
    # The following two excepts allow v to pass through when it represents a
    # string.
    #
    # Longer explanation:
    # The type of v is always a string (before calling literal_eval), but
    # sometimes it *represents* a string and other times a data structure, like
    # a list. In the case that v represents a string, what we got back from the
    # yaml parser is 'foo' *without quotes* (so, not '"foo"'). literal_eval is
    # ok with '"foo"', but will raise a ValueError if given 'foo'. In other
    # cases, like paths (v = 'foo/bar' and not v = '"foo/bar"'), literal_eval
    # will raise a SyntaxError.
    except ValueError:
        pass
    except SyntaxError:
        pass
    return v


def _check_and_coerce_cfg_value_type(value_a, value_b, key, full_key):
    """Checks that `value_a`, which is intended to replace `value_b` is of the
    right type. The type is correct if it matches exactly or is one of a few
    cases in which the type can be easily coerced.
    """
    # The types must match (with some exceptions)
    type_b = type(value_b)
    type_a = type(value_a)
    if type_a is type_b:
        return value_a

    # Exceptions: numpy arrays, strings, tuple<->list
    if isinstance(value_b, np.ndarray):
        value_a = np.array(value_a, dtype=value_b.dtype)
    elif isinstance(value_b, six.string_types):
        value_a = str(value_a)
    elif isinstance(value_a, tuple) and isinstance(value_b, list):
        value_a = list(value_a)
    elif isinstance(value_a, list) and isinstance(value_b, tuple):
        value_a = tuple(value_a)
    else:
        raise ValueError(
            'Type mismatch ({} vs. {}) with values ({} vs. {}) for config '
            'key: {}'.format(type_b, type_a, value_b, value_a, full_key)
        )
    return value_a
