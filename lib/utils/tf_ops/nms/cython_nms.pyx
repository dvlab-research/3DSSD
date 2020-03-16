cimport cython
import numpy as np
cimport numpy as np

cdef inline np.float32_t max(np.float32_t a, np.float32_t b) nogil:
    return a if a >= b else b

cdef inline np.float32_t min(np.float32_t a, np.float32_t b) nogil:
    return a if a <= b else b

@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def nms(np.ndarray[np.float32_t, ndim=2] dets, np.float32_t thresh):
    cdef np.ndarray[np.float32_t, ndim=1] x1 = dets[:, 0]
    cdef np.ndarray[np.float32_t, ndim=1] y1 = dets[:, 1]
    cdef np.ndarray[np.float32_t, ndim=1] x2 = dets[:, 2]
    cdef np.ndarray[np.float32_t, ndim=1] y2 = dets[:, 3]
    cdef np.ndarray[np.float32_t, ndim=1] scores = dets[:, 4]

    cdef np.ndarray[np.float32_t, ndim=1] areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    cdef np.ndarray[np.int_t, ndim=1] order = scores.argsort()[::-1]

    cdef int ndets = dets.shape[0]
    cdef np.ndarray[np.int_t, ndim=1] suppressed = \
            np.zeros((ndets), dtype=np.int)

    # nominal indices
    cdef int _i, _j
    # sorted indices
    cdef int i, j
    # temp variables for box i's (the box currently under consideration)
    cdef np.float32_t ix1, iy1, ix2, iy2, iarea
    # variables for computing overlap with box j (lower scoring box)
    cdef np.float32_t xx1, yy1, xx2, yy2
    cdef np.float32_t w, h
    cdef np.float32_t inter, ovr

    with nogil:
      for _i in range(ndets):
          i = order[_i]
          if suppressed[i] == 1:
              continue
          ix1 = x1[i]
          iy1 = y1[i]
          ix2 = x2[i]
          iy2 = y2[i]
          iarea = areas[i]
          for _j in range(_i + 1, ndets):
              j = order[_j]
              if suppressed[j] == 1:
                  continue
              xx1 = max(ix1, x1[j])
              yy1 = max(iy1, y1[j])
              xx2 = min(ix2, x2[j])
              yy2 = min(iy2, y2[j])
              w = max(0.0, xx2 - xx1 + 1)
              h = max(0.0, yy2 - yy1 + 1)
              inter = w * h
              ovr = inter / (iarea + areas[j] - inter)
              if ovr >= thresh:
                  suppressed[j] = 1

    return np.where(suppressed == 0)[0]


# ----------------------------------------------------------
# Soft-NMS: Improving Object Detection With One Line of Code
# Copyright (c) University of Maryland, College Park
# Licensed under The MIT License [see LICENSE for details]
# Written by Navaneeth Bodla and Bharat Singh
# ----------------------------------------------------------
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def soft_nms(
    np.ndarray[float, ndim=2] boxes_in,
    np.ndarray[float, ndim=2] boxes_3d_in,
    float sigma=0.5,
    float Nt=0.3,
    float threshold=0.001,
    unsigned int method=0
):
    boxes = boxes_in.copy()
    boxes_3d = boxes_3d_in.copy()
    cdef unsigned int N = boxes.shape[0]
    cdef float iw, ih, box_area
    cdef float ua
    cdef int pos = 0
    cdef float maxscore = 0
    cdef int maxpos = 0
    cdef float x1, x2, y1, y2, tx1, tx2, ty1, ty2, ts, area, weight, ov
    cdef float x, y, z, l, w, h, theta
    inds = np.arange(N).astype(np.int32)

    for i in range(N):
        maxscore = boxes[i, 4]
        maxpos = i

        tx1 = boxes[i,0]
        ty1 = boxes[i,1]
        tx2 = boxes[i,2]
        ty2 = boxes[i,3]
        ts = boxes[i,4]
        ti = inds[i]

        x = boxes_3d_in[i, 0]
        y = boxes_3d_in[i, 1]
        z = boxes_3d_in[i, 2]
        l = boxes_3d_in[i, 3]
        w = boxes_3d_in[i, 4]
        h = boxes_3d_in[i, 5]
        theta = boxes_3d_in[i, 6]

        pos = i + 1
        # get max box
        while pos < N:
            if maxscore < boxes[pos, 4]:
                maxscore = boxes[pos, 4]
                maxpos = pos
            pos = pos + 1

        # add max box as a detection
        boxes[i,0] = boxes[maxpos,0]
        boxes[i,1] = boxes[maxpos,1]
        boxes[i,2] = boxes[maxpos,2]
        boxes[i,3] = boxes[maxpos,3]
        boxes[i,4] = boxes[maxpos,4]
        inds[i] = inds[maxpos]

        boxes_3d[i, 0] = boxes_3d[maxpos, 0]
        boxes_3d[i, 1] = boxes_3d[maxpos, 1]
        boxes_3d[i, 2] = boxes_3d[maxpos, 2]
        boxes_3d[i, 3] = boxes_3d[maxpos, 3]
        boxes_3d[i, 4] = boxes_3d[maxpos, 4]
        boxes_3d[i, 5] = boxes_3d[maxpos, 5]
        boxes_3d[i, 6] = boxes_3d[maxpos, 6]

        boxes_3d[maxpos, 0] = x
        boxes_3d[maxpos, 1] = y
        boxes_3d[maxpos, 2] = z
        boxes_3d[maxpos, 3] = l
        boxes_3d[maxpos, 4] = w
        boxes_3d[maxpos, 5] = h
        boxes_3d[maxpos, 6] = theta

        # swap ith box with position of max box
        boxes[maxpos,0] = tx1
        boxes[maxpos,1] = ty1
        boxes[maxpos,2] = tx2
        boxes[maxpos,3] = ty2
        boxes[maxpos,4] = ts
        inds[maxpos] = ti

        tx1 = boxes[i,0]
        ty1 = boxes[i,1]
        tx2 = boxes[i,2]
        ty2 = boxes[i,3]
        ts = boxes[i,4]

        pos = i + 1
        # NMS iterations, note that N changes if detection boxes fall below
        # threshold
        while pos < N:
            x1 = boxes[pos, 0]
            y1 = boxes[pos, 1]
            x2 = boxes[pos, 2]
            y2 = boxes[pos, 3]
            s = boxes[pos, 4]

            area = (x2 - x1 + 1) * (y2 - y1 + 1)
            iw = (min(tx2, x2) - max(tx1, x1) + 1)
            if iw > 0:
                ih = (min(ty2, y2) - max(ty1, y1) + 1)
                if ih > 0:
                    ua = float((tx2 - tx1 + 1) * (ty2 - ty1 + 1) + area - iw * ih)
                    ov = iw * ih / ua #iou between max box and detection box

                    if method == 1: # linear
                        if ov > Nt:
                            weight = 1 - ov
                        else:
                            weight = 1
                    elif method == 2: # gaussian
                        weight = np.exp(-(ov * ov)/sigma)
                    else: # original NMS
                        if ov > Nt:
                            weight = 0
                        else:
                            weight = 1

                    boxes[pos, 4] = weight*boxes[pos, 4]

                    # if box score falls below threshold, discard the box by
                    # swapping with last box update N
                    if boxes[pos, 4] < threshold:
                        boxes[pos,0] = boxes[N-1, 0]
                        boxes[pos,1] = boxes[N-1, 1]
                        boxes[pos,2] = boxes[N-1, 2]
                        boxes[pos,3] = boxes[N-1, 3]
                        boxes[pos,4] = boxes[N-1, 4]

                        boxes_3d[pos, 0] = boxes_3d[N-1, 0]
                        boxes_3d[pos, 1] = boxes_3d[N-1, 1]
                        boxes_3d[pos, 2] = boxes_3d[N-1, 2]
                        boxes_3d[pos, 3] = boxes_3d[N-1, 3]
                        boxes_3d[pos, 4] = boxes_3d[N-1, 4]
                        boxes_3d[pos, 5] = boxes_3d[N-1, 5]
                        boxes_3d[pos, 6] = boxes_3d[N-1, 6]

                        inds[pos] = inds[N-1]
                        N = N - 1
                        pos = pos - 1

            pos = pos + 1

    return boxes[:N], inds[:N], boxes_3d[:N]


# ----------------------------------------------------------
# IoU guided NMS methods with calculated IoU matrix
# ----------------------------------------------------------
@cython.boundscheck(False)
@cython.cdivision(True)
@cython.wraparound(False)
def matrix_iou_guided_nms(
    np.ndarray[np.float32_t, ndim=2] iou_matrix, # [-1, -1] num_proposals x num_proposals
    np.ndarray[np.float32_t, ndim=2] boxes_3d_in, # [-1, 7]
    np.ndarray[np.float32_t, ndim=1] scores_in, # [-1]
    np.ndarray[np.float32_t, ndim=1] iou_3d_in, # [-1]
    float iou_thresh
):
    # sort by 3D IoU * score
    cdef np.ndarray[np.float32_t, ndim=1] ensemble_score = scores_in * iou_3d_in
    cdef np.ndarray[np.int_t, ndim=1] order = ensemble_score.argsort()[::-1]

    cdef int proposals_num = boxes_3d_in.shape[0]
    cdef np.ndarray[np.int_t, ndim=1] suppressed = \
            np.zeros((proposals_num), dtype=np.int)

    # nominal indices
    cdef np.ndarray[np.float32_t, ndim=1] ibox = np.zeros((7), dtype=np.float32)
    cdef np.ndarray[np.float32_t, ndim=1] jbox = np.zeros((7), dtype=np.float32)
    cdef int _i, _j
    # sorted indices
    cdef int i, j
    # temp variables for box i's (the box currently under consideration)
    cdef np.float32_t iscore
    cdef np.float32_t ovr

    for _i in range(proposals_num):
        i = order[_i]
        if suppressed[i] == 1: continue
        iscore = ensemble_score[i]
        ibox = boxes_3d_in[i]
        for _j in range(_i + 1, proposals_num):
            j = order[_j]
            if suppressed[j] == 1: continue
            ovr = iou_matrix[i, j]
            if ovr >= iou_thresh:
                suppressed[j] = 1
        # finally, add ibox back to boxes_3d_in
        boxes_3d_in[i] = ibox
        scores_in[i] = iscore
    return np.where(suppressed == 0)[0], boxes_3d_in, scores_in


