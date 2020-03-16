#!/usr/bin/env python
# -*- coding: utf-8 -*-
### modified from https://github.com/ppwwyyxx/tensorpack

import os
import sys
import msgpack
import msgpack_numpy
msgpack_numpy.patch()
import json
import numpy as np

from .logger import *

# https://github.com/apache/arrow/pull/1223#issuecomment-359895666
old_mod = sys.modules.get('torch', None)
sys.modules['torch'] = None
try:
    import pyarrow as pa
except ImportError:
    pa = None
if old_mod is not None:
    sys.modules['torch'] = old_mod
else:
    del sys.modules['torch']

import pickle

__all__ = ['loads', 'dumps', 'dump_pkl', 'load_pkl']


def dumps_msgpack(obj):
    """
    Serialize an object.
    Returns:
        Implementation-dependent bytes-like object
    """
    return msgpack.dumps(obj, use_bin_type=True)


def loads_msgpack(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return msgpack.loads(buf, raw=False)


def dumps_pyarrow(obj):
    """
    Serialize an object.

    Returns:
        Implementation-dependent bytes-like object
    """
    return pa.serialize(obj).to_buffer()


def loads_pyarrow(buf):
    """
    Args:
        buf: the output of `dumps`.
    """
    return pa.deserialize(buf)


def dump_pkl(name, obj):
    with open('{}'.format(name), 'wb') as f:
        pickle.dump( obj, f, pickle.HIGHEST_PROTOCOL )

def load_pkl(name):
    with open('{}'.format(name), 'rb') as f:
        ret = pickle.load( f )
    return ret

if pa is None:
    loads = loads_msgpack
    dumps = dumps_msgpack
else:
    loads = loads_pyarrow
    dumps = dumps_pyarrow

def dump_dict(obj):
    dic = dict()
    for i, attr in enumerate(dir(obj)):
        if attr.startswith('_'): continue
        k = obj.__getattribute__(attr)
        if callable(k): continue
        dic[attr] = k
    return dic

def comp_dict(dic1, dic2, cfg1='cfg1', cfg2='cfg2'):
    diff_count = 0
    extra1_count = 0
    extra2_count = 0

    set1 = set(dic1.keys())
    set2 = set(dic2.keys())
    
    ign_cond = lambda i: 'dir' in i.lower() or 'path' in i.lower()

    def ign_word_last(list):
        ign_list = [i for i in list if ign_cond(i)]
        else_list = [i for i in list if not ign_cond(i)]
        return else_list + ign_list

    for k in ign_word_last(sorted(set1 & set2)):
        same = True
        if (isinstance(dic1[k], list) or isinstance(dic1[k], tuple) or isinstance(dic1[k], np.ndarray)):
            try:
                if len(dic1[k]) == len(dic2[k]) == 0: continue
                if np.asarray(dic1[k]).shape != np.asarray(dic2[k]).shape or \
                    np.any(np.asarray(dic1[k]) != np.asarray(dic2[k])):
                    diff_count += 1
                    same = False
            except Exception as e:
                print_red(e)
                diff_count += 1
                same = False
        else:
            if dic1[k] != dic2[k]:
                diff_count += 1
                same = False
        if not same:
            if diff_count == 1:
                print_red('### Difference between {} and {}:'.format(cfg1, cfg2))
            print('{:>20s} {:>10s}: {} <---> {}'.format(red('Diff'), k, dic1[k], dic2[k]))

    for k in ign_word_last(sorted(set1 - set2)):
        if extra1_count == 0:
            print_red('### Extra items in {}: '.format(cfg1))
        print('{:>20s} {:>10s}: {} <---> [None]'.format(green('New'), k, dic1[k]))
        diff_count += 1
        extra1_count += 1

    for k in ign_word_last(sorted(set2 - set1)):
        if extra2_count == 0:
            print_red('### Extra items in {}: '.format(cfg2))
        print('{:>20s} {:>10s}: [None] <---> {}'.format(red('Lack'), k, dic2[k]))
        diff_count += 1
        extra2_count += 1

    return (diff_count == 0) # issame

def dump_class(obj):
    def json_encode(x):
        if type(x) is np.ndarray:
            return x.tolist()
        else:
            return x
    dic = dump_dict(obj)
    return json.dumps(dic, indent=4, skipkeys=True, default=json_encode, sort_keys=True)
