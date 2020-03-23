import numpy as np
import numba

from core.config import cfg

from utils.box_3d_utils import get_box3d_corners_helper_np
from utils.rotation_util import rotate_points, symmetric_rotate_points, roty

def noise_per_object_v3_(gt_boxes,
                         points=None,
                         valid_mask=None,
                         rotation_perturb=np.pi / 4,
                         center_noise_std=1.0,
                         global_random_rot_range=[0., 0.],
                         random_scale_range=[0.9, 1.1],
                         scale_3_dims=False,
                         sem_labels=None,
                         num_try=100):
    """random rotate or remove each groundtrutn independently.
    use kitti viewer to test this function points_transform_
    Args:
        gt_boxes: [N, 7], gt box in lidar.points_transform_
        points: [M, 4], point cloud in lidar.
    """
    num_boxes = gt_boxes.shape[0]
    if not isinstance(rotation_perturb, (list, tuple, np.ndarray)):
        rotation_perturb = [-rotation_perturb, rotation_perturb]
    if not isinstance(global_random_rot_range, (list, tuple, np.ndarray)):
        global_random_rot_range = [
            -global_random_rot_range, global_random_rot_range
        ]
    if not isinstance(random_scale_range, (list, tuple, np.ndarray)):
        random_scale_range = [1 - random_scale_range,
                              1 + random_scale_range]
    enable_grot = np.abs(global_random_rot_range[0] -
                         global_random_rot_range[1]) >= 1e-3
    if not isinstance(center_noise_std, (list, tuple, np.ndarray)):
        center_noise_std = [
            center_noise_std, center_noise_std, center_noise_std
        ]
    if valid_mask is None:
        valid_mask = np.ones((num_boxes, ), dtype=np.bool_)
    center_noise_std = np.array(center_noise_std, dtype=gt_boxes.dtype)
    loc_noises = np.random.normal(
        scale=center_noise_std, size=[num_boxes, num_try, 3])
    # loc_noises = np.random.uniform(
    #     -center_noise_std, center_noise_std, size=[num_boxes, num_try, 3])
    rot_noises = np.random.uniform(
        rotation_perturb[0], rotation_perturb[1], size=[num_boxes, num_try])
    if scale_3_dims: # random on each dimensions
        scale_noises = np.random.uniform(
            random_scale_range[0], random_scale_range[1], size=[num_boxes, num_try, 3])
        if cfg.TRAIN.AUGMENTATIONS.SINGLE_AUG.FIX_LENGTH:
            scale_noises[:, :, 0] = 1.
    else:
        scale_noises = np.random.uniform(
            random_scale_range[0], random_scale_range[1], size=[num_boxes, num_try, 1])

    # then reshape the gt_boxes to [x, z, y, l, w, h, angle]
    gt_boxes = gt_boxes[:, [0, 2, 1, 3, 5, 4, 6]]
    gt_boxes_expand = gt_boxes.copy()
    gt_boxes_expand[:, 3:6] += float(cfg.TRAIN.AUGMENTATIONS.EXPAND_DIMS_LENGTH)
    points = points[:, [0, 2, 1]]

    assert sem_labels is not None, 'only move these points within each groundtruth, not change background'
    pos_index = np.where(sem_labels)[0]
    pos_points = points[pos_index]

    origin = [0.5, 0.5, 1.0]
    gt_box_corners = center_to_corner_box3d(
        gt_boxes_expand[:, :3],
        gt_boxes_expand[:, 3:6],
        gt_boxes_expand[:, 6],
        origin=origin,
        axis=2)

    if not enable_grot:
        selected_noise = noise_per_box(gt_boxes_expand[:, [0, 1, 3, 4, 6]],
                                       valid_mask, loc_noises, rot_noises, scale_noises, scale_3_dims)
    else:
        raise Exception('Not Implementation Fault ----')

    loc_transforms = _select_transform(loc_noises, selected_noise)
    rot_transforms = _select_transform(rot_noises, selected_noise)
    scale_transforms = _select_transform(scale_noises, selected_noise)
    if not scale_3_dims:
        scale_transforms = np.squeeze(scale_transforms, axis=-1)
    surfaces = corner_to_surfaces_3d_jit(gt_box_corners)
    if points is not None:
        point_masks = points_in_convex_polygon_3d_jit(pos_points[:, :3], surfaces)
        dbg_point_masks = np.where(np.sum(point_masks, axis=1) > 0)[0]
        points_transform_(pos_points, gt_boxes[:, :3], point_masks, loc_transforms,
                          rot_transforms, scale_transforms, valid_mask, 
                          scale_3_dims, gt_boxes[:, -1])

    box3d_transform_(gt_boxes, loc_transforms, rot_transforms, scale_transforms, valid_mask)
    gt_boxes = gt_boxes[:, [0, 2, 1, 3, 5, 4, 6]]
    points[pos_index] = pos_points
    points = points[:, [0, 2, 1]]
    return gt_boxes, points

def add_symmetric_points_to_gt(boxes, points, sem_labels, sem_dists):
    """
    Add symmetric points to ground truth
    boxes: [n, 7]
    points:[m, -1]
    sem_labels: [m]
    sem_dists: [m]
    """
    pts_num = points.shape[0]
    # new_pts_list = [points]
    # new_sem_labels_list = [sem_labels]
    # new_sem_dists_list = [sem_dists]
    enlarge_boxes = boxes.copy()
    enlarge_boxes[:, 3:-1] += cfg.TRAIN.AUGMENTATIONS.EXPAND_DIMS_LENGTH
    total_inside_pts_mask = check_inside_points(points, enlarge_boxes) # [pts_num, gt_num]
    # first add all labels that not for any boxes in
    bg_pts_mask = np.where(np.max(total_inside_pts_mask, axis=-1) == 0)[0]
    new_pts_list = [points[bg_pts_mask]]
    new_sem_labels_list = [sem_labels[bg_pts_mask]]
    new_sem_dists_list = [sem_dists[bg_pts_mask]]
    for index, box in enumerate(boxes):
        # first gather points inside this box
        inside_pts_mask = total_inside_pts_mask[:, index]
        inside_pts_idx = np.where(inside_pts_mask > 0)[0]
        inside_pts = points[inside_pts_idx].copy() # [-1, 4]
        # then we flip these points according to z-axis
        inside_pts_i = inside_pts[:, 3:]
        inside_pts_xyz = inside_pts[:, :3]

        # flip
        inside_pts_xyz = inside_pts_xyz - box[np.newaxis, :3]
        inside_pts_xyz = symmetric_rotate_points_np(inside_pts_xyz[np.newaxis, :, :], box[np.newaxis, -1])
        inside_pts_xyz = np.squeeze(inside_pts_xyz, axis=0)
        inside_pts_xyz = inside_pts_xyz + box[np.newaxis, :3]
        inside_pts_flip = np.concatenate([inside_pts_xyz, inside_pts_i], axis=-1)

        new_pts_list.append(inside_pts_flip)
        new_sem_labels_list.append(np.ones([inside_pts_flip.shape[0]], dtype=sem_labels.dtype))
        new_sem_dists_list.append(np.ones([inside_pts_flip.shape[0]], dtype=sem_dists.dtype))
    new_pts_list = np.concatenate(new_pts_list, axis=0)
    new_sem_labels_list = np.concatenate(new_sem_labels_list, axis=0)
    new_sem_dists_list = np.concatenate(new_sem_dists_list, axis=0)
    return new_pts_list, new_sem_labels_list, new_sem_dists_list


def add_symmetric_points_to_gt_original_idx(boxes, points, sem_labels=None, sem_dists=None):
    """
    Add symmetric points to ground truth
    difference from former method is that, it add same points into same position
    boxes: [n, 7]
    points:[m, -1]
    sem_labels: [m]
    sem_dists: [m]
    """
    pts_num = points.shape[0]
    # new_pts_list = [points]
    # new_sem_labels_list = [sem_labels]
    # new_sem_dists_list = [sem_dists]
    points = points.copy()
    if sem_labels is not None and sem_dists is not None:
        sem_labels = sem_labels.copy()
        sem_dists = sem_dists.copy()
    enlarge_boxes = boxes.copy()
    # first of all, let's filter these empty gt
    useful_gt_mask = np.logical_not(np.all(np.equal(enlarge_boxes, 0), axis=-1))
    useful_gt_idx = np.where(useful_gt_mask)[0]
    enlarge_boxes = enlarge_boxes[useful_gt_idx]
    enlarge_boxes[:, 3:-1] += cfg.TRAIN.AUGMENTATIONS.EXPAND_DIMS_LENGTH
    total_inside_pts_mask = check_inside_points(points, enlarge_boxes) # [pts_num, gt_num]
    for index, box in enumerate(enlarge_boxes):
        # first gather points inside this box
        inside_pts_mask = total_inside_pts_mask[:, index]
        inside_pts_idx = np.where(inside_pts_mask > 0)[0]
        inside_pts = points[inside_pts_idx].copy() # [-1, 4]
        # then we flip these points according to z-axis
        inside_pts_i = inside_pts[:, 3:]
        inside_pts_xyz = inside_pts[:, :3]

        # flip
        inside_pts_xyz = inside_pts_xyz - box[np.newaxis, :3]
        inside_pts_xyz = symmetric_rotate_points_np(inside_pts_xyz[np.newaxis, :, :], box[np.newaxis, -1])
        inside_pts_xyz = np.squeeze(inside_pts_xyz, axis=0)
        inside_pts_xyz = inside_pts_xyz + box[np.newaxis, :3]
        inside_pts_flip = np.concatenate([inside_pts_xyz, inside_pts_i], axis=-1)

        points[inside_pts_idx] = inside_pts_flip
        if sem_labels is not None and sem_dists is not None:
            sem_labels[inside_pts_idx] = np.ones([inside_pts_flip.shape[0]], dtype=sem_labels.dtype)
            sem_dists[inside_pts_idx] = np.ones([inside_pts_flip.shape[0]], dtype=sem_dists.dtype)
    if sem_labels is not None and sem_dists is not None:
        return points, sem_labels, sem_dists
    else:
        return points


def box_3d_collision_test(boxes, q_boxes, classes, q_classes, boxes_points, points, sem_labels, sem_dists, plane, enlarge_range=[0.5, 2.0, 0.5]):
    """ Calculate whether two boxes are collided
        Args:
            boxes: [n, 7], x, y, z, l, h, w, ry, generate boxes
            q_boxes: [m, 7], x, y, z, l, h, w, ry, original boxes
            classes: [n], generate_boxes labels
            q_classes: [m], original_boxes labels
            boxes_points: [[], [], ...]
            points: [points_num, 3]
        Return:
            collision_matrix: [n, m] whether collision or not 
    """
    a, b, c, d = plane
    # first cast these boxes to the function used format
    enlarge_range = np.array(enlarge_range)

    avoid_boxes = boxes.copy()
    avoid_boxes[:, 3:-1] += enlarge_range
    boxes_bev = avoid_boxes[:, [0, 2, 3, 5, 6]]
    boxes_bev_corners = box2d_to_corner_jit(boxes_bev)   
    num_boxes = boxes_bev.shape[0]    

    for i in range(num_boxes):
        q_boxes_bev = q_boxes[:, [0, 2, 3, 5, 6]]
        q_boxes_bev_corners = box2d_to_corner_jit(q_boxes_bev)

        cur_boxes = boxes[i, :]
        cur_classes = classes[i]
        cur_boxes_bev_corners = boxes_bev_corners[i, :, :] 

        # 1: collision, 0: non-collision 
        coll_mat = box_collision_test(cur_boxes_bev_corners[np.newaxis, :, :], q_boxes_bev_corners)

        if not np.any(coll_mat):
            cur_boxes_points = boxes_points[i]
            cur_sem_labels = np.ones([len(cur_boxes_points)], dtype=np.int32) * cur_classes
            cur_sem_dists = np.ones([len(cur_boxes_points)], dtype=np.float32)

            cur_height = (-d - a * cur_boxes[0] - c * cur_boxes[2]) / b
            move_height = cur_boxes[1] - cur_height
            cur_boxes_points[:, 1] -= move_height
            cur_boxes[1] -= move_height

            points = np.concatenate([points, cur_boxes_points], axis=0)
            sem_labels = np.concatenate([sem_labels, cur_sem_labels], axis=0)
            sem_dists = np.concatenate([sem_dists, cur_sem_dists], axis=0)

            # finally, update the q_boxes
            q_boxes = np.concatenate([q_boxes, cur_boxes[np.newaxis, :]], axis=0)
            q_classes = np.concatenate([q_classes, [cur_classes]], axis=0)

    return q_boxes, q_classes, points, sem_labels, sem_dists 


def box_3d_collision_test_nusc(boxes, q_boxes, classes, q_classes, boxes_points, boxes_attributes, boxes_velocity, points, attributes=None, velocity=None, cur_sweep_points_num=None, enlarge_range=[0.5, 2.0, 0.5]):
    """ Calculate whether two boxes are collided
        Args:
            boxes: [n, 7], x, y, z, l, h, w, ry, generate boxes
            q_boxes: [m, 7], x, y, z, l, h, w, ry, original boxes
            classes: [n], generate_boxes labels
            q_classes: [m], original_boxes labels
            boxes_points: [[], [], ...]
            points: [points_num, 3]
        Return:
            collision_matrix: [n, m] whether collision or not 
    """
    # first cast these boxes to the function used format
    enlarge_range = np.array(enlarge_range)

    avoid_boxes = boxes.copy()
    avoid_boxes[:, 3:-1] += enlarge_range
    boxes_bev = avoid_boxes[:, [0, 2, 3, 5, 6]]
    boxes_bev_corners = box2d_to_corner_jit(boxes_bev)   
    num_boxes = boxes_bev.shape[0]    
    
    sampled_points = []
    sampled_boxes = []
    for i in range(num_boxes):
        q_boxes_bev = q_boxes[:, [0, 2, 3, 5, 6]]
        q_boxes_bev_corners = box2d_to_corner_jit(q_boxes_bev)

        cur_boxes = boxes[i, :]
        cur_classes = classes[i]
        cur_attributes = boxes_attributes[i]
        cur_velocity = boxes_velocity[i]
        cur_boxes_bev_corners = boxes_bev_corners[i, :, :] 

        # 1: collision, 0: non-collision 
        coll_mat = box_collision_test(cur_boxes_bev_corners[np.newaxis, :, :], q_boxes_bev_corners)

        if not np.any(coll_mat):
            # now, add current points in
            cur_boxes_points = boxes_points[i]

            sampled_points.append(cur_boxes_points)
            sampled_boxes.append(cur_boxes)

            # finally, update the q_boxes
            q_boxes = np.concatenate([q_boxes, cur_boxes[np.newaxis, :]], axis=0)
            q_classes = np.concatenate([q_classes, [cur_classes]], axis=0)
            if attributes is not None:
                attributes = np.concatenate([attributes, [cur_attributes]], axis=0)
            if velocity is not None:
                velocity = np.concatenate([velocity, cur_velocity[np.newaxis, :]], axis=0)

    # finally remove points within boxes
    if len(sampled_points) != 0:
        sampled_points = np.concatenate(sampled_points, axis=0)
        sampled_boxes = np.stack(sampled_boxes, axis=0) # [-1, 7]
        point_masks = check_inside_points(points, sampled_boxes)  # [num_points, gt_boxes]
        point_masks = np.max(point_masks, axis=1) # num_points
        keep_points = np.where(point_masks == 0)[0]

        cur_sample_num = np.sum(np.less(keep_points, cur_sweep_points_num).astype(np.int)) + len(sampled_points)

        points = np.concatenate([sampled_points, points[keep_points]], axis=0)
    else:
        cur_sample_num = cur_sweep_points_num

    return q_boxes, q_classes, points, attributes, velocity, cur_sample_num 

def check_inside_points(points, cur_boxes):
    """
    points: [num, n]
    cur_boxes: [m 7]
    return: [num, m]
    """
    # first cast points to second format
    points = points.copy()
    cur_boxes = cur_boxes.copy()
    points_xyz = points[:, :3]
    points_xyz = points_xyz[:, [0, 2, 1]]

    cur_boxes = cur_boxes[:, [0, 2, 1, 3, 5, 4, 6]]

    origin = [0.5, 0.5, 1.0]
    cur_box_corners = center_to_corner_box3d(
        cur_boxes[:, :3],
        cur_boxes[:, 3:6],
        cur_boxes[:, 6],
        origin=origin,
        axis=2)
    surfaces = corner_to_surfaces_3d_jit(cur_box_corners)
    point_masks = points_in_convex_polygon_3d_jit(points_xyz, surfaces) # [num_points, gt_boxes]

    return point_masks


def filter_points_boxes_3d(label_boxes_3d, points, sem_labels, dist_labels, enlarge_range=[0.5, 2.0, 0.5]):
    """
    Filter points inside each label_boxes_3d but not in sem_labels
    label_boxes_3d: [gt_num, 7]
    points: [points_num, 4]
    sem_labels: [points_num]
    dist_labels: [points_num]
    """
    # first transpose points
    label_boxes_3d[:, 3:-1] += np.array(enlarge_range)

    pos_index = np.where(sem_labels >= 1)[0]
    neg_index = np.where(sem_labels == 0)[0]
    neg_points = points[neg_index]

    point_masks = check_inside_points(neg_points, label_boxes_3d) # [pts_num, gt_num]

    point_masks = np.equal(np.max(point_masks, axis=1), 0) # [points_num]
    stored_index = np.where(point_masks)[0] # neg points not in label_boxes_3d

    stored_index = neg_index[stored_index]
    stored_index = np.concatenate([pos_index, stored_index])
    points = points[stored_index]
    sem_labels = sem_labels[stored_index]
    dist_labels = dist_labels[stored_index]

    label_boxes_3d[:, 3:-1] -= np.array(enlarge_range)
    return label_boxes_3d, points, sem_labels, dist_labels

def put_boxes_on_planes(label_boxes_3d, points, sem_labels, plane, expand_dims_length):
    """
    label_boxes_3d: [gt_num, 7]
    plane: 4 params, a/b/c/d
    """
    a,b,c,d = plane
    gt_num = label_boxes_3d.shape[0]

    cp_label_boxes_3d = label_boxes_3d.copy()
    cp_label_boxes_3d[:, 3:-1] += expand_dims_length

    pos_index = np.where(sem_labels >= 1)[0]
    pos_points = points[pos_index]
    pos_points_mask = check_inside_points(pos_points, cp_label_boxes_3d) # [pts_num, gt_num]
    assigned_gt = np.argmax(pos_points_mask, axis=1) # [pts_num]

    # gt_num
    y_plane = (-d - a * label_boxes_3d[:, 0] - c * label_boxes_3d[:, 2]) / b
    mv_vector_box = label_boxes_3d[:, 1] - y_plane
    mv_vector_pts = mv_vector_box[assigned_gt]

    pos_points[:, 1] -= mv_vector_pts
    label_boxes_3d[:, 1] -= mv_vector_box

    points[pos_index] = pos_points
    return points, label_boxes_3d


@numba.njit
def noise_per_box(boxes, valid_mask, loc_noises, rot_noises, scale_noises, scale_3_dims):
    # boxes: [N, 5]
    # valid_mask: [N]
    # loc_noises: [N, M, 3]
    # rot_noises: [N, M]
    # scale_noises: [N, M], if scale_3_dims ---> [N, M, 3]
    # scale_3_dims: whether scale on each dims
    num_boxes = boxes.shape[0]
    num_tests = loc_noises.shape[1]
    box_corners = box2d_to_corner_jit(boxes)
    current_corners = np.zeros((4, 2), dtype=boxes.dtype)
    rot_mat_T = np.zeros((2, 2), dtype=boxes.dtype)
    success_mask = -np.ones((num_boxes, ), dtype=np.int64)
    # print(valid_mask)
    for i in range(num_boxes):
        if valid_mask[i]:
            for j in range(num_tests):
                current_corners[:] = box_corners[i]
                current_corners -= boxes[i, :2]
                _rotation_box2d_jit_(current_corners, rot_noises[i, j],
                                     rot_mat_T)
                if scale_3_dims:
                    current_corners[:, 0] *= scale_noises[i, j, 0]
                    current_corners[:, 2] *= scale_noises[i, j, 2]
                else:
                    current_corners *= scale_noises[i, j, 0]
                current_corners += boxes[i, :2] + loc_noises[i, j, :2]
                coll_mat = box_collision_test(
                    current_corners.reshape(1, 4, 2), box_corners)
                coll_mat[0, i] = False
                # print(coll_mat)
                if not coll_mat.any():
                    success_mask[i] = j
                    box_corners[i] = current_corners
                    break
    return success_mask

@numba.njit
def noise_per_box_v2_(boxes, valid_mask, loc_noises, rot_noises,
                      global_rot_noises):
    # boxes: [N, 5]
    # valid_mask: [N]
    # loc_noises: [N, M, 3]
    # rot_noises: [N, M]
    num_boxes = boxes.shape[0]
    num_tests = loc_noises.shape[1]
    box_corners = box2d_to_corner_jit(boxes)
    current_corners = np.zeros((4, 2), dtype=boxes.dtype)
    current_box = np.zeros((1, 5), dtype=boxes.dtype)
    rot_mat_T = np.zeros((2, 2), dtype=boxes.dtype)
    dst_pos = np.zeros((2, ), dtype=boxes.dtype)
    success_mask = -np.ones((num_boxes, ), dtype=np.int64)
    corners_norm = np.zeros((4, 2), dtype=boxes.dtype)
    corners_norm[1, 1] = 1.0
    corners_norm[2] = 1.0
    corners_norm[3, 0] = 1.0
    corners_norm -= np.array([0.5, 0.5], dtype=boxes.dtype)
    corners_norm = corners_norm.reshape(4, 2)
    for i in range(num_boxes):
        if valid_mask[i]:
            for j in range(num_tests):
                current_box[0, :] = boxes[i]
                current_radius = np.sqrt(boxes[i, 0]**2 + boxes[i, 1]**2)
                current_grot = np.arctan2(boxes[i, 0], boxes[i, 1])
                dst_grot = current_grot + global_rot_noises[i, j]
                dst_pos[0] = current_radius * np.sin(dst_grot)
                dst_pos[1] = current_radius * np.cos(dst_grot)
                current_box[0, :2] = dst_pos
                current_box[0, -1] += (dst_grot - current_grot)

                rot_sin = np.sin(current_box[0, -1])
                rot_cos = np.cos(current_box[0, -1])
                rot_mat_T[0, 0] = rot_cos
                rot_mat_T[0, 1] = -rot_sin
                rot_mat_T[1, 0] = rot_sin
                rot_mat_T[1, 1] = rot_cos
                current_corners[:] = current_box[0, 2:
                                                 4] * corners_norm @ rot_mat_T + current_box[0, :
                                                                                             2]
                current_corners -= current_box[0, :2]
                _rotation_box2d_jit_(current_corners, rot_noises[i, j],
                                     rot_mat_T)
                current_corners += current_box[0, :2] + loc_noises[i, j, :2]
                coll_mat = box_collision_test(
                    current_corners.reshape(1, 4, 2), box_corners)
                coll_mat[0, i] = False
                if not coll_mat.any():
                    success_mask[i] = j
                    box_corners[i] = current_corners
                    loc_noises[i, j, :2] += (dst_pos - boxes[i, :2])
                    rot_noises[i, j] += (dst_grot - current_grot)
                    break
    return success_mask

@numba.jit(nopython=False)
def surface_equ_3d_jit(polygon_surfaces):
    # return [a, b, c], d in ax+by+cz+d=0
    # polygon_surfaces: [num_polygon, num_surfaces, num_points_of_polygon, 3]
    surface_vec = polygon_surfaces[:, :, :2, :] - polygon_surfaces[:, :, 1:3, :]
    # normal_vec: [..., 3]
    normal_vec = np.cross(surface_vec[:, :, 0, :], surface_vec[:, :, 1, :])
    # print(normal_vec.shape, points[..., 0, :].shape)
    # d = -np.inner(normal_vec, points[..., 0, :])
    d = np.einsum('aij, aij->ai', normal_vec, polygon_surfaces[:, :, 0, :])
    return normal_vec, -d

@numba.jit(nopython=True)
def box2d_to_corner_jit(boxes):
    num_box = boxes.shape[0]
    corners_norm = np.zeros((4, 2), dtype=boxes.dtype)
    corners_norm[1, 1] = 1.0
    corners_norm[2] = 1.0
    corners_norm[3, 0] = 1.0
    corners_norm -= np.array([0.5, 0.5], dtype=boxes.dtype)
    corners = boxes.reshape(num_box, 1, 5)[:, :, 2:4] * corners_norm.reshape(
        1, 4, 2)
    rot_mat_T = np.zeros((2, 2), dtype=boxes.dtype)
    box_corners = np.zeros((num_box, 4, 2), dtype=boxes.dtype)
    for i in range(num_box):
        rot_sin = np.sin(boxes[i, -1])
        rot_cos = np.cos(boxes[i, -1])
        rot_mat_T[0, 0] = rot_cos
        rot_mat_T[0, 1] = -rot_sin
        rot_mat_T[1, 0] = rot_sin
        rot_mat_T[1, 1] = rot_cos
        box_corners[i] = corners[i] @ rot_mat_T + boxes[i, :2]
    return box_corners

@numba.jit(nopython=False)
def points_in_convex_polygon_3d_jit(points,
                                    polygon_surfaces,
                                    ):
    """check points is in 3d convex polygons.
    Args:
        points: [num_points, 3] array.
        polygon_surfaces: [num_polygon, max_num_surfaces, 
            max_num_points_of_surface, 3] 
            array. all surfaces' normal vector must direct to internal.
            max_num_points_of_surface must at least 3.
        num_surfaces: [num_polygon] array. indicate how many surfaces 
            a polygon contain
    Returns:
        [num_points, num_polygon] bool array.
    """
    max_num_surfaces, max_num_points_of_surface = polygon_surfaces.shape[1:3]
    num_points = points.shape[0]
    num_polygons = polygon_surfaces.shape[0]
    num_surfaces = np.full((num_polygons,), 9999999, dtype=np.int64)
    normal_vec, d = surface_equ_3d_jit(polygon_surfaces[:, :, :3, :])
    # normal_vec: [num_polygon, max_num_surfaces, 3]
    # d: [num_polygon, max_num_surfaces]
    ret = np.ones((num_points, num_polygons), dtype=np.bool_)
    sign = 0.0
    for i in range(num_points):
        for j in range(num_polygons):
            for k in range(max_num_surfaces):
                if k > num_surfaces[j]:
                    break
                sign = points[i, 0] * normal_vec[j, k, 0] \
                     + points[i, 1] * normal_vec[j, k, 1] \
                     + points[i, 2] * normal_vec[j, k, 2] + d[j, k]
                if sign >= 0:
                    ret[i, j] = False
                    break
    return ret

@numba.jit(nopython=True)
def corner_to_surfaces_3d_jit(corners):
    """convert 3d box corners from corner function above
    to surfaces that normal vectors all direct to internal.
    Args:
        corners (float array, [N, 8, 3]): 3d box corners. 
    Returns:
        surfaces (float array, [N, 6, 4, 3]): 
    """
    # box_corners: [N, 8, 3], must from corner functions in this module
    num_boxes = corners.shape[0]
    surfaces = np.zeros((num_boxes, 6, 4, 3), dtype=corners.dtype)
    corner_idxes = np.array([
        0, 1, 2, 3, 7, 6, 5, 4, 0, 3, 7, 4, 1, 5, 6, 2, 0, 4, 5, 1, 3, 2, 6, 7
    ]).reshape(6, 4)
    for i in range(num_boxes):
        for j in range(6):
            for k in range(4):
                surfaces[i, j, k] = corners[i, corner_idxes[j, k]]
    return surfaces

@numba.njit
def points_transform_(points, centers, point_masks, loc_transform,
                      rot_transform, scale_transform, valid_mask,
                      scale_3_dims, angles):
    num_box = centers.shape[0]
    num_points = points.shape[0]
    rot_mat_T = np.zeros((num_box, 3, 3), dtype=points.dtype)
    if scale_3_dims:
        rot_mat_original_T = np.zeros((num_box, 3, 3), dtype=points.dtype)
        for i in range(num_box):
            _rotation_matrix_3d_(rot_mat_original_T[i], -angles[i], 2)
            _rotation_matrix_3d_(rot_mat_T[i], rot_transform[i] + angles[i], 2)
    else:
        for i in range(num_box):
            _rotation_matrix_3d_(rot_mat_T[i], rot_transform[i], 2)
    for i in range(num_points):
        for j in range(num_box):
            if valid_mask[j]:
                if point_masks[i, j] == 1:
                    points[i, :3] -= centers[j, :3]
                    if scale_3_dims:
                        points[i:i+1, :3] = points[i:i+1, :3] @ rot_mat_original_T[j]
                        points[i, :3] *= scale_transform[j]
                    else:
                        points[i, :3] *= scale_transform[j]
                    points[i:i + 1, :3] = points[i:i + 1, :3] @ rot_mat_T[j]
                    points[i, :3] += centers[j, :3]
                    points[i, :3] += loc_transform[j]
                    break  # only apply first box's transform

@numba.njit
def corner_to_standup_nd_jit(boxes_corner):
    num_boxes = boxes_corner.shape[0]
    ndim = boxes_corner.shape[-1]
    result = np.zeros((num_boxes, ndim * 2), dtype=boxes_corner.dtype)
    for i in range(num_boxes):
        for j in range(ndim):
            result[i, j] = np.min(boxes_corner[i, :, j])
        for j in range(ndim):
            result[i, j + ndim] = np.max(boxes_corner[i, :, j])
    return result

@numba.jit(nopython=True)
def box_collision_test(boxes, qboxes, clockwise=True):
    N = boxes.shape[0]
    K = qboxes.shape[0]
    ret = np.zeros((N, K), dtype=np.bool_)
    slices = np.array([1, 2, 3, 0])
    lines_boxes = np.stack(
        (boxes, boxes[:, slices, :]), axis=2)  # [N, 4, 2(line), 2(xy)]
    lines_qboxes = np.stack((qboxes, qboxes[:, slices, :]), axis=2)
    # vec = np.zeros((2,), dtype=boxes.dtype)
    boxes_standup = corner_to_standup_nd_jit(boxes)
    qboxes_standup = corner_to_standup_nd_jit(qboxes)
    for i in range(N):
        for j in range(K):
            # calculate standup first
            iw = (min(boxes_standup[i, 2], qboxes_standup[j, 2]) - max(
                boxes_standup[i, 0], qboxes_standup[j, 0]))
            if iw > 0:
                ih = (min(boxes_standup[i, 3], qboxes_standup[j, 3]) - max(
                    boxes_standup[i, 1], qboxes_standup[j, 1]))
                if ih > 0:
                    for k in range(4):
                        for l in range(4):
                            A = lines_boxes[i, k, 0]
                            B = lines_boxes[i, k, 1]
                            C = lines_qboxes[j, l, 0]
                            D = lines_qboxes[j, l, 1]
                            acd = (D[1] - A[1]) * (C[0] - A[0]) > (
                                C[1] - A[1]) * (D[0] - A[0])
                            bcd = (D[1] - B[1]) * (C[0] - B[0]) > (
                                C[1] - B[1]) * (D[0] - B[0])
                            if acd != bcd:
                                abc = (C[1] - A[1]) * (B[0] - A[0]) > (
                                    B[1] - A[1]) * (C[0] - A[0])
                                abd = (D[1] - A[1]) * (B[0] - A[0]) > (
                                    B[1] - A[1]) * (D[0] - A[0])
                                if abc != abd:
                                    ret[i, j] = True  # collision.
                                    break
                        if ret[i, j] is True:
                            break
                    if ret[i, j] is False:
                        # now check complete overlap.
                        # box overlap qbox:
                        box_overlap_qbox = True
                        for l in range(4):  # point l in qboxes
                            for k in range(4):  # corner k in boxes
                                vec = boxes[i, k] - boxes[i, (k + 1) % 4]
                                if clockwise:
                                    vec = -vec
                                cross = vec[1] * (
                                    boxes[i, k, 0] - qboxes[j, l, 0])
                                cross -= vec[0] * (
                                    boxes[i, k, 1] - qboxes[j, l, 1])
                                if cross >= 0:
                                    box_overlap_qbox = False
                                    break
                            if box_overlap_qbox is False:
                                break

                        if box_overlap_qbox is False:
                            qbox_overlap_box = True
                            for l in range(4):  # point l in boxes
                                for k in range(4):  # corner k in qboxes
                                    vec = qboxes[j, k] - qboxes[j, (k + 1) % 4]
                                    if clockwise:
                                        vec = -vec
                                    cross = vec[1] * (
                                        qboxes[j, k, 0] - boxes[i, l, 0])
                                    cross -= vec[0] * (
                                        qboxes[j, k, 1] - boxes[i, l, 1])
                                    if cross >= 0:  #
                                        qbox_overlap_box = False
                                        break
                                if qbox_overlap_box is False:
                                    break
                            if qbox_overlap_box:
                                ret[i, j] = True  # collision.
                        else:
                            ret[i, j] = True  # collision.
    return ret

@numba.njit
def _rotation_matrix_3d_(rot_mat_T, angle, axis):
    rot_sin = np.sin(angle)
    rot_cos = np.cos(angle)
    rot_mat_T[:] = np.eye(3)
    if axis == 1:
        rot_mat_T[0, 0] = rot_cos
        rot_mat_T[0, 2] = -rot_sin
        rot_mat_T[2, 0] = rot_sin
        rot_mat_T[2, 2] = rot_cos
    elif axis == 2 or axis == -1:
        rot_mat_T[0, 0] = rot_cos
        rot_mat_T[0, 1] = -rot_sin
        rot_mat_T[1, 0] = rot_sin
        rot_mat_T[1, 1] = rot_cos
    elif axis == 0:
        rot_mat_T[1, 1] = rot_cos
        rot_mat_T[1, 2] = -rot_sin
        rot_mat_T[2, 1] = rot_sin
        rot_mat_T[2, 2] = rot_cos

@numba.njit
def box3d_transform_(boxes, loc_transform, rot_transform, scale_transforms, valid_mask):
    num_box = boxes.shape[0]
    for i in range(num_box):
        if valid_mask[i]:
            boxes[i, :3] += loc_transform[i]
            boxes[i, 3:6] *= scale_transforms[i]
            boxes[i, 6] += rot_transform[i]

@numba.njit
def _rotation_box2d_jit_(corners, angle, rot_mat_T):
    rot_sin = np.sin(angle)
    rot_cos = np.cos(angle)
    rot_mat_T[0, 0] = rot_cos
    rot_mat_T[0, 1] = -rot_sin
    rot_mat_T[1, 0] = rot_sin
    rot_mat_T[1, 1] = rot_cos
    corners[:] = corners @ rot_mat_T

# Dont need numba
def center_to_corner_box3d(centers,
                           dims,
                           angles=None,
                           origin=[0.5, 1.0, 0.5],
                           axis=1):
    """convert kitti locations, dimensions and angles to corners
    
    Args:
        centers (float array, shape=[N, 3]): locations in kitti label file.
        dims (float array, shape=[N, 3]): dimensions in kitti label file.
        angles (float array, shape=[N]): rotation_y in kitti label file.
        origin (list or array or float): origin point relate to smallest point.
            use [0.5, 1.0, 0.5] in camera and [0.5, 0.5, 0] in lidar.
        axis (int): rotation axis. 1 for camera and 2 for lidar.
    Returns:
        [type]: [description]
    """
    # 'length' in kitti format is in x axis.
    # yzx(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(wlh)(lidar)
    # center in kitti format is [0.5, 1.0, 0.5] in xyz.
    corners = corners_nd(dims, origin=origin)
    # corners: [N, 8, 3]
    if angles is not None:
        corners = rotation_3d_in_axis(corners, angles, axis=axis)
    corners += centers.reshape([-1, 1, 3])
    return corners

def corners_nd(dims, origin=0.5):
    """generate relative box corners based on length per dim and
    origin point. 
    
    Args:
        dims (float array, shape=[N, ndim]): array of length per dim
        origin (list or array or float): origin point relate to smallest point.
    
    Returns:
        float array, shape=[N, 2 ** ndim, ndim]: returned corners. 
        point layout example: (2d) x0y0, x0y1, x1y0, x1y1;
            (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
            where x0 < x1, y0 < y1, z0 < z1
    """
    ndim = int(dims.shape[1])
    corners_norm = np.stack(
        np.unravel_index(np.arange(2**ndim), [2] * ndim), axis=1).astype(
            dims.dtype)
    # now corners_norm has format: (2d) x0y0, x0y1, x1y0, x1y1
    # (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
    # so need to convert to a format which is convenient to do other computing.
    # for 2d boxes, format is clockwise start with minimum point
    # for 3d boxes, please draw lines by your hand.
    if ndim == 2:
        # generate clockwise box corners
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array(origin, dtype=dims.dtype)
    corners = dims.reshape([-1, 1, ndim]) * corners_norm.reshape(
        [1, 2**ndim, ndim])
    return corners

def rotation_3d_in_axis(points, angles, axis=0):
    # points: [N, point_size, 3]
    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    ones = np.ones_like(rot_cos)
    zeros = np.zeros_like(rot_cos)
    if axis == 1:
        rot_mat_T = np.stack([[rot_cos, zeros, -rot_sin], [zeros, ones, zeros],
                              [rot_sin, zeros, rot_cos]])
    elif axis == 2 or axis == -1:
        rot_mat_T = np.stack([[rot_cos, -rot_sin, zeros],
                              [rot_sin, rot_cos, zeros], [zeros, zeros, ones]])
    elif axis == 0:
        rot_mat_T = np.stack([[zeros, rot_cos, -rot_sin],
                              [zeros, rot_sin, rot_cos], [ones, zeros, zeros]])
    else:
        raise ValueError("axis should in range")

    return np.einsum('aij,jka->aik', points, rot_mat_T)

def _select_transform(transform, indices):
    result = np.zeros(
        (transform.shape[0], *transform.shape[2:]), dtype=transform.dtype)
    for i in range(transform.shape[0]):
        if indices[i] != -1:
            result[i] = transform[i, indices[i]]
    return result

############## Get image feature of each points ###################
@numba.jit(nopython=True)
def get_coeff(x, y):
    x = np.abs(x)
    y = np.abs(y)
    return (1 - x) * (1 - y)

@numba.jit(nopython=True)
def get_data(img_feature, x, y, width, height):
    overflow = (x < 0) or (y < 0) or (x >= width) or (y >= height)
    ret_val = np.zeros((img_feature.shape[-1]), dtype=img_feature.dtype)
    if not overflow:
        ret_val = img_feature[y, x]
    return ret_val

@numba.jit(nopython=True)
def get_point_image_feature(img_feature, image_v):
    """ Get Image info
        img_feature: [h, w, c]
        image_v: [n, 2], x, y location for each
    """
    points_num = image_v.shape[0]
    channels = img_feature.shape[-1]
    return_feature = np.zeros((points_num, channels), dtype=img_feature.dtype)
    height = img_feature.shape[0]
    width = img_feature.shape[1]
    for i in range(points_num):
        cur_x, cur_y = image_v[i]

        # then get current value
        x1 = np.int(np.floor(cur_x))
        y1 = np.int(np.floor(cur_y))
        ret_val = get_data(img_feature, x1, y1, width, height) * get_coeff(cur_x - x1, cur_y - y1)

        x1 = np.int(np.floor(cur_x) + 1)
        y1 = np.int(np.floor(cur_y))
        ret_val += get_data(img_feature, x1, y1, width, height) * get_coeff(cur_x - x1, cur_y - y1)

        x1 = np.int(np.floor(cur_x))
        y1 = np.int(np.floor(cur_y) + 1)
        ret_val += get_data(img_feature, x1, y1, width, height) * get_coeff(cur_x - x1, cur_y - y1)

        x1 = np.int(np.floor(cur_x) + 1)
        y1 = np.int(np.floor(cur_y) + 1)
        ret_val += get_data(img_feature, x1, y1, width, height) * get_coeff(cur_x - x1, cur_y - y1)

        return_feature[i] = ret_val
    return return_feature
