import time

import numba
import numpy as np
from points2voxel import points_to_voxel_3d_np, nusc_points_to_voxel_3d_np
import time

@numba.jit(nopython=True)
def _points_to_voxel_kernel(points, # [-1, 4/5]
                            sem_labels, # [-1]
                            voxel_size, # [0.2, 0.2, 0.2]
                            coors_range, # [-50, -50, -4, 50, 50, 2]
                            num_points_per_voxel, # (max_voxels,)
                            coor_to_voxelidx, # (500, 500, 30)
                            voxels, # (max_voxels, max_points, 4/5)
                            voxel_sem_labels, # (max_voxels, max_points)
                            coors, # (max_voxels, 3)
                            max_points=35,
                            max_voxels=20000):
    # need mutex if write in cuda, but numba.cuda don't support mutex.
    # in addition, pytorch don't support cuda in dataloader(tensorflow support this).
    # put all computations to one loop.
    # we shouldn't create large array in main jit code, otherwise
    # decrease performance
    N = points.shape[0]
    # ndim = points.shape[1] - 1
    ndim = 3
    grid_size = (coors_range[3:] - coors_range[:3]) / voxel_size
    # grid_size = np.round(grid_size).astype(np.int64)(np.int32)
    grid_size = np.round(grid_size, 0, grid_size).astype(np.int32)

    lower_bound = coors_range[:3]
    upper_bound = coors_range[3:]
    coor = np.zeros(shape=(3, ), dtype=np.int32)
    voxel_num = 0
    cur_sample_num = 0
    failed = False
    for i in range(N):
        failed = False
        for j in range(ndim):
            c = np.floor((points[i, j] - coors_range[j]) / voxel_size[j])
            if c < 0 or c >= grid_size[j]:
                failed = True
                break
            coor[j] = c
        if failed:
            continue
        voxelidx = coor_to_voxelidx[coor[0], coor[1], coor[2]]
        if voxelidx == -1: # no current voxel
            voxelidx = voxel_num
            if voxel_num >= max_voxels:
                break
            voxel_num += 1
            if sem_labels[i] == 1: cur_sample_num += 1
            coor_to_voxelidx[coor[0], coor[1], coor[2]] = voxelidx
            coors[voxelidx] = coor
        num = num_points_per_voxel[voxelidx]
        if num < max_points:
            voxels[voxelidx, num] = points[i]
            voxel_sem_labels[voxelidx, num] = sem_labels[i]
            num_points_per_voxel[voxelidx] += 1
    return voxel_num, cur_sample_num


def points_to_voxel(points,
                    voxel_size,
                    coors_range,
                    max_points=35,
                    max_voxels=20000):
    """convert kitti points(N, >=3) to voxels. This version calculate
    everything in one loop. now it takes only 4.2ms(complete point cloud) 
    with jit and 3.2ghz cpu.(don't calculate other features)
    Note: this function in ubuntu seems faster than windows 10.

    Args:
        points: [N, ndim] float tensor. points[:, :3] contain xyz points and
            points[:, 3:] contain other information such as reflectivity.
        sem_labels: [N, ] int tensor. whether a point is labeled positive
        voxel_size: [3] list/tuple or array, float. xyz, indicate voxel size
        coors_range: [6] list/tuple or array, float. indicate voxel range.
            format: xyzxyz, minmax
        max_points: int. indicate maximum points contained in a voxel.
        reverse_index: boolean. indicate whether return reversed coordinates.
            if points has xyz format and reverse_index is True, output 
            coordinates will be zyx format, but points in features always
            xyz format.
        max_voxels: int. indicate maximum voxels this function create.
            for second, 20000 is a good choice. you should shuffle points
            before call this function because max_voxels may drop some points.

    Returns:
        voxels: [M, max_points, ndim] float tensor. only contain points.
        coordinates: [M, 3] int32 tensor.
        num_points_per_voxel: [M] int32 tensor.
    """
    if not isinstance(voxel_size, np.ndarray):
        voxel_size = np.array(voxel_size, dtype=points.dtype)
    if not isinstance(coors_range, np.ndarray):
        coors_range = np.array(coors_range, dtype=points.dtype)
    voxelmap_shape = (coors_range[3:] - coors_range[:3]) / voxel_size
    # [lx, ly, lz], [500, 30, 500, 30]
    voxelmap_shape = tuple(np.round(voxelmap_shape).astype(np.int32).tolist())
    # don't create large array in jit(nopython=True) code.
    num_points_per_voxel = np.zeros(shape=(max_voxels, ), dtype=np.int32)
    coor_to_voxelidx = -np.ones(shape=voxelmap_shape, dtype=np.int32)
    voxels = np.zeros(
        shape=(max_voxels, max_points, points.shape[-1]), dtype=points.dtype)
    coors = np.zeros(shape=(max_voxels, 3), dtype=np.int32)

    voxel_num = points_to_voxel_3d_np(
                points, voxels,
                coors,
                num_points_per_voxel, coor_to_voxelidx, voxel_size.tolist(),
                coors_range.tolist(), max_points, max_voxels)

    coors = coors[:voxel_num]
    voxels = voxels[:voxel_num]
    num_points_per_voxel = num_points_per_voxel[:voxel_num]
    
    return voxels, num_points_per_voxel


def points_to_voxel_nusc(cur_sweep_points,
                         other_sweep_points, # indicate whether a point within cur_sample
                         voxel_size,
                         coors_range,
                         max_points=35,
                         max_voxels=20000,
                         cur_sample_num=16384):
    """convert kitti points(N, >=3) to voxels. This version calculate
    everything in one loop. now it takes only 4.2ms(complete point cloud) 
    with jit and 3.2ghz cpu.(don't calculate other features)
    Note: this function in ubuntu seems faster than windows 10.

    Args:
        cur_sweep_points: [N, ndim] float tensor. points[:, :3] contain xyz points and
                          points[:, 3:] contain other information such as reflectivity.
        other_sweep_points: [M, ndim] float tensor.
        voxel_size: [3] list/tuple or array, float. xyz, indicate voxel size
        coors_range: [6] list/tuple or array, float. indicate voxel range.
            format: xyzxyz, minmax
        max_points: int. indicate maximum points contained in a voxel.
        reverse_index: boolean. indicate whether return reversed coordinates.
            if points has xyz format and reverse_index is True, output 
            coordinates will be zyx format, but points in features always
            xyz format.
        max_voxels: int. indicate maximum voxels this function create.
            for second, 20000 is a good choice. you should shuffle points
            before call this function because max_voxels may drop some points.

    Returns:
        voxels: [M, max_points, ndim] float tensor. only contain points.
        coordinates: [M, 3] int32 tensor.
        num_points_per_voxel: [M] int32 tensor.
    """
    if not isinstance(voxel_size, np.ndarray):
        voxel_size = np.array(voxel_size, dtype=points.dtype)
    if not isinstance(coors_range, np.ndarray):
        coors_range = np.array(coors_range, dtype=points.dtype)
    voxelmap_shape = (coors_range[3:] - coors_range[:3]) / voxel_size
    # [lx, ly, lz], [500, 30, 500]
    voxelmap_shape = tuple(np.round(voxelmap_shape).astype(np.int32).tolist())
    # don't create large array in jit(nopython=True) code.
    num_points_per_voxel = np.zeros(shape=(max_voxels, ), dtype=np.int32)
    coor_to_voxelidx = -np.ones(shape=voxelmap_shape, dtype=np.int32)
    voxels = np.zeros(
        shape=(max_voxels, max_points, cur_sweep_points.shape[-1]), dtype=cur_sweep_points.dtype)
    coors = np.zeros(shape=(max_voxels, 3), dtype=np.int32)

    voxel_num = nusc_points_to_voxel_3d_np(
                cur_sweep_points, other_sweep_points, voxels, 
                coors,
                num_points_per_voxel, coor_to_voxelidx, voxel_size.tolist(),
                coors_range.tolist(), max_points, max_voxels, cur_sample_num)

    # coors = coors[:voxel_num]
    # voxels = voxels[:voxel_num]
    # num_points_per_voxel = num_points_per_voxel[:voxel_num]
    # voxel_sem_labels = voxel_sem_labels[:voxel_num]
    # voxels[:, :, -3:] = voxels[:, :, :3] - \
    #     voxels[:, :, :3].sum(axis=1, keepdims=True)/num_points_per_voxel.reshape(-1, 1, 1)

    # finally, get the sem_labels for each voxel
    # first calculate the center point if each voxels (voxel_num, point_num, 3)
    # if one point inside this voxel is positive, then this voxel is labeled positive
    # voxel_sem_labels = voxel_sem_labels.max(axis=1) # (voxel_num)
    
    return voxels, num_points_per_voxel


@numba.jit(nopython=True)
def bound_points_jit(points, upper_bound, lower_bound):
    # to use nopython=True, np.bool is not supported. so you need
    # convert result to np.bool after this function.
    N = points.shape[0]
    ndim = points.shape[1]
    keep_indices = np.zeros((N, ), dtype=np.int32)
    success = 0
    for i in range(N):
        success = 1
        for j in range(ndim):
            if points[i, j] < lower_bound[j] or points[i, j] >= upper_bound[j]:
                success = 0
                break
        keep_indices[i] = success
    return keep_indices
