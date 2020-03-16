#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import os
import os.path as osp
import sys
import numpy as np
from datetime import datetime
from glob import glob
from itertools import chain
import gc

from .logger import red, green, yellow

THIS_PROCESS_RNG = None

check_flag = False
def check_once(cond=True, msg=None, err_msg="Check Failed"):
    global check_flag
    if not check_flag:
        if cond:
            print(yellow(msg))
        else:
            print(yellow(err_msg))
            sys.exit(0)
        check_flag = True

def check_none(cond=True, err_msg="Check Failed"):
    if cond:
        print(red(err_msg))
        sys.exit(0)

def cmd(command):
    import subprocess
    output = subprocess.check_output(command, shell=True)
    output = output.decode()
    return output

def mem_info():
    import subprocess
    dev = subprocess.check_output(
        "nvidia-smi | grep MiB | awk -F '|' '{print $3}' | awk -F '/' '{print $1}' | grep -Eo '[0-9]{1,10}'",
        shell=True)
    dev = dev.decode()
    dev_mem = list(map(lambda x: int(x), dev.split('\n')[:-1]))
    return dev_mem

def get_file_dir(file):
    return osp.dirname(osp.abspath(file))

def add_pypath(path):
    if path not in sys.path:
        sys.path.insert(0, path)

def make_link(dest_path, link_path):
    if os.path.islink(link_path):
        os.system('rm {}'.format(link_path))
    os.system('ln -s {} {}'.format(dest_path, link_path))

def make_dir(path):
    if os.path.exists(path) or os.path.islink(path):
        return 
    os.makedirs(path)

def del_file(path, msg='{} deleted.'):
    if os.path.exists(path):
        os.remove(path)
        print(msg.format(path))
    else:
        print("{} doesn't exist.".format(path))

def approx_equal(a, b, eps=1e-9):
    return np.fabs(a-b) < eps

def random_int(obj=None):
    return (id(obj) + os.getpid() + int(datetime.now().strftime("%Y%m%d%H%M%S%f"))) % 4294967295

def get_rng(obj=None):
    """
    Get a good RNG seeded with time, pid and the object.

    Args:
        obj: some object to use to generate random seed.
    Returns:
        np.random.RandomState: the RNG.
    """
    seed = random_int(obj)
    return np.random.RandomState(seed)

def set_np_seed(obj=None):
    global THIS_PROCESS_RNG
    if THIS_PROCESS_RNG is None:
        THIS_PROCESS_RNG = random_int()
    print('numpy applys seed {}'.format(THIS_PROCESS_RNG))
    np.random.seed(THIS_PROCESS_RNG)

import cv2

def GetVideoInfo(video_path):
    assert os.path.isfile(video_path), "Path {} is not a video file.".format(video_path)
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened(): 
        raise ValueError("could not open {}".format(video_path))

    info = dict(
        cap = cap,
        length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        fps    = cap.get(cv2.CAP_PROP_FPS)
    )

    return info

def Video2Frame(video_path, downsample_rate, output_path=None):
    info = GetVideoInfo(video_path)

    if output_path is None:
        output_path = video_path
    make_dir(output_path)

    cap = info['cap']
    count = 0
    save_count = 0
    success = True
    while success:
        success, image = cap.read()
        if (count % downsample_rate == 0):
            cv2.imwrite(osp.join(output_path, 'video_{:06d}'.format(count) + '.jpg'), image)
            save_count += 1
        count += 1
    print('written {}/{} frames in {}'.format( save_count, count, output_path))
    print('--------------')

from contextlib import contextmanager
@contextmanager
def VideoWriter(video_path, frame_shape, fps=20):
    # frame_shape (height, width)
    #hack(only support mp4 media type)
    vwriter = cv2.VideoWriter(video_path, cv2.VideoWriter_fourcc(*'h264'), fps, tuple([frame_shape[1], frame_shape[0]]))
    yield vwriter
    print('Wrote video {}.'.format(video_path))
    vwriter.release()

def Frame2Video(frame_path_regex, video_path, fps=20):
    frame_list = glob(frame_path_regex)
    with VideoWriter(video_path, cv2.imread(frame_list[0]).shape, fps=fps) as vw:
        for f in frame_list:
            img = cv2.imread(f)
            vw.write(img)

def aggregate_batch(data_holder):
    if isinstance(data_holder[0], dict):
        keys = data_holder[0]
        results = dict( [ [k, [i[k] for i in data_holder]] for k in data_holder[0]] )
        for k in results:
            if isinstance(results[k][0], list):
                results[k] = list(chain(*results[k]))
            elif isinstance(results[k][0], np.ndarray):
                results[k] = np.concatenate(results[k], axis=0)
            else:
                from IPython import embed; embed()
                raise TypeError('Unsupported type when aggregating batch.')
    elif isinstance(data_holder[0], list):
        results = list(chain(data_holder))
    elif isinstance(data_holder[0], np.ndarray):
        results = np.concatenate(data_holder)
    else:
        raise TypeError('Unsupported type when aggregating batch.')
    return results

def del_list(x):
    del x[:]
    del x

def clear_memory():
    gc.collect()
