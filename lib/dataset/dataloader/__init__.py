import tensorflow as tf
import numpy as np

from core.config import cfg
from .kitti_dataloader import KittiDataset
from .nuscenes_dataloader import NuScenesDataset

def choose_dataset():
    dataset_dict = {
        'KITTI': KittiDataset,
        'NuScenes': NuScenesDataset,
    }
    return dataset_dict[cfg.DATASET.TYPE]
