import numpy as np

from core.config import cfg
from builder.voxel_generator.point_cloud_ops import points_to_voxel, points_to_voxel_nusc


class VoxelGenerator:
    def __init__(self, max_voxels=None):
        point_cloud_range = cfg.DATASET.POINT_CLOUD_RANGE # (-50, 50, -4, 2, -50, 50)
        point_cloud_range = np.array(point_cloud_range, dtype=np.float32)
        point_cloud_range = np.reshape(point_cloud_range, [3, 2])
        point_cloud_range = np.transpose(point_cloud_range, [2, 3])
        point_cloud_range = np.reshape(point_cloud_range, [-1]) # (-50, -4, -50, 50, 2, 50)

        voxel_size = cfg.DATASET.VOXEL_SIZE # [0.5, 1, 0.5]
        voxel_size = np.array(voxel_size, dtype=np.float32)
        grid_size = (
            point_cloud_range[3:] - point_cloud_range[:3]) / voxel_size
        # [500, 500, 30]
        grid_size = np.round(grid_size).astype(np.int64)

        max_num_points = int(cfg.DATASET.MAX_NUMBER_OF_POINT_PER_VOXEL)
        max_voxels = int(cfg.DATASET.NUSCENE.MAX_NUMBER_OF_VOXELS) 

        self._voxel_size = voxel_size
        self._point_cloud_range = point_cloud_range
        self._max_num_points = max_num_points
        self._max_voxels = max_voxels
        self._grid_size = grid_size

    def generate(self, points):
        return points_to_voxel(
            points, self._voxel_size, self._point_cloud_range,
            self._max_num_points, self._max_voxels)

    def generate_nusc(self, cur_sweep_points, other_sweep_points, max_cur_sample_num):
        return points_to_voxel_nusc(
            cur_sweep_points, other_sweep_points, self._voxel_size, 
            self._point_cloud_range,
            self._max_num_points, self._max_voxels, max_cur_sample_num
        ) 

    @property
    def voxel_size(self):
        return self._voxel_size

    @property
    def max_num_points_per_voxel(self):
        return self._max_num_points


    @property
    def point_cloud_range(self):
        return self._point_cloud_range

    @property
    def grid_size(self):
        return self._grid_size
