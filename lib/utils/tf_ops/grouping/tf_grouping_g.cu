#include <stdio.h>
#include <iostream>
#include <vector>
#include <time.h>
#include <math.h>

#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      std::cout << cudaGetErrorString(error) << std::endl; \
    } \
  } while (0)

#define CUDA_1D_KERNEL_LOOP(i, n)                            \
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; \
            i += blockDim.x * gridDim.x)

# define M_PI           3.14159265358979323846

const int block_num = 512;
#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))
const int threadsPerBlock = sizeof(unsigned long long) * 8;


__device__ static int point_inside_box_3d(float x, float y, float z, float cx, float by, float cz, float l, float h, float w, float ry, float max_distance){
    float cos_ry, sin_ry;
    float canonical_x, canonical_z;
    int inside;
    if ((fabsf(x - cx) > max_distance) || (y > by) || ((by - y) > h) || (fabsf(z - cz) > max_distance)){
        return 0;
    }
    cos_ry = cos(ry); sin_ry = sin(ry);
    canonical_x = (x - cx) * cos_ry - (z - cz) * sin_ry;
    canonical_z = (x - cx) * sin_ry + (z - cz) * cos_ry;

    inside = (canonical_x >= -l / 2.0) & (canonical_x <= l / 2.0) & (canonical_z >= -w / 2.0) & (canonical_z <= w / 2.0);
    return inside;
}


/* query boxes 3d points */
// input: nsample (1), xyz (b,n,3), proposals (b,m,7)
// output: idx (b,m,nsample), pts_cnt (b,m)
__global__ void query_boxes_3d_points_gpu(int b, int n, int m, int nsample, const float *xyz, const float *proposals, int *idx, int *pts_cnt) {
    int total_idx = b * m;
    CUDA_1D_KERNEL_LOOP(point_inds, total_idx){
        int batch_index = point_inds / m;

        const float* cur_xyz;
        const float* cur_proposal;
        cur_xyz = xyz + n*3*batch_index;
        cur_proposal = proposals + point_inds * 7;

        int* cur_idx;
        int* cur_pts_cnt;
        cur_idx = idx + nsample*point_inds;
        cur_pts_cnt = pts_cnt + point_inds; // counting how many unique points selected in local region

        float cx= cur_proposal[0];
        float by= cur_proposal[1];
        float cz= cur_proposal[2];
        float l = cur_proposal[3];
        float h = cur_proposal[4];
        float w = cur_proposal[5];
        float ry= cur_proposal[6];
        float max_distance = max(sqrtf((l / 2.) * (l / 2.)+(w / 2.)*(w / 2.)),1e-20f);

        float x, y, z;
        int inside;

        int cnt = 0;
        for (int k=0;k<n;++k) {
            if (cnt == nsample)
                break; // only pick the FIRST nsample points in the ball
            x=cur_xyz[k*3+0];
            y=cur_xyz[k*3+1];
            z=cur_xyz[k*3+2];

            inside = point_inside_box_3d(x, y, z, cx, by, cz, l, h, w, ry, max_distance); 

            if (inside) {
                if (cnt==0) {
                    for (int l=0;l<nsample;++l)
                        cur_idx[l] = k;
                }
                cur_idx[cnt] = k;
                cnt+=1;
            }
        }
        cur_pts_cnt[0] = cnt;
    }
}


/* query boxes 3d mask */
// input: xyz (b,n,3), boxes_3d (b,m,7)
// output: mask (b,m,n)
__global__ void query_boxes_3d_mask_gpu(int b, int n, int m, const float *xyz, const float *boxes_3d, int *mask){
    int total_idx = b * m * n;
    CUDA_1D_KERNEL_LOOP(point_inds, total_idx){
        int batch_index = point_inds / (m * n);
        int box_index = point_inds / n;
        int point_index = point_inds % n;

        const float* cur_xyz;
        const float* cur_boxes_3d;
        cur_xyz = xyz + batch_index * n * 3 + point_index * 3;
        cur_boxes_3d = boxes_3d + box_index * 7;

        int* cur_mask;
        cur_mask = mask + point_inds;

        float cx= cur_boxes_3d[0];
        float by= cur_boxes_3d[1];
        float cz= cur_boxes_3d[2];
        float l = cur_boxes_3d[3];
        float h = cur_boxes_3d[4];
        float w = cur_boxes_3d[5];
        float ry= cur_boxes_3d[6];
        float max_distance = max(sqrtf((l / 2.) * (l / 2.)+(w / 2.)*(w / 2.)),1e-20f);

        float x = cur_xyz[0];
        float y = cur_xyz[1];
        float z = cur_xyz[2];
        int inside;

        inside = point_inside_box_3d(x, y, z, cx, by, cz, l, h, w, ry, max_distance); 
        cur_mask[0] = inside;
    }
}


/* query points iou */
// input: xyz (b,n,3), anchors_3d (b,anchors_num,7), gt_boxes_3d (b, gt_num, 7)
// input: iou_matrix (b, anchors_num, gt_num)
// output: iou_points(b, anchors_num, gt_num)
__global__ void query_points_iou_gpu(int b, int n, int anchors_num, int gt_num,
    const float* xyz, const float* anchors_3d, const float* gt_boxes_3d,
    const float* iou_matrix, float* iou_points){

    int total_idx = b * anchors_num * gt_num;
    CUDA_1D_KERNEL_LOOP(point_inds, total_idx){
        float iou_value = iou_matrix[point_inds]; 
        if (iou_value < 1e-3f){
            // if no overlaps around two boxes_3d, then directly return 0
            iou_points[point_inds] = 0.;
            continue;
        }
        // has overlaps, then calculate PointIoU
        int batch_index = point_inds / (anchors_num * gt_num);
        int anchor_index = point_inds / gt_num;
        int gt_index = point_inds % gt_num;

        const float* cur_xyz;
        const float* cur_anchors_3d;
        const float* cur_gt_boxes_3d;
        cur_xyz = xyz + batch_index * n * 3;
        cur_anchors_3d = anchors_3d + anchor_index * 7;
        cur_gt_boxes_3d = gt_boxes_3d + batch_index * gt_num * 7 + gt_index * 7;

        float* cur_iou_points;
        cur_iou_points = iou_points + point_inds;
        int in = 0, un = 0;

        float gt_boxes_cx= cur_gt_boxes_3d[0];
        float gt_boxes_by= cur_gt_boxes_3d[1];
        float gt_boxes_cz= cur_gt_boxes_3d[2];
        float gt_boxes_l = cur_gt_boxes_3d[3];
        float gt_boxes_h = cur_gt_boxes_3d[4];
        float gt_boxes_w = cur_gt_boxes_3d[5];
        float gt_boxes_ry= cur_gt_boxes_3d[6];
        float gt_boxes_max_distance = max(sqrtf((gt_boxes_l / 2.) * (gt_boxes_l / 2.)
                                    + (gt_boxes_w  / 2.) * (gt_boxes_w / 2.)),1e-20f);

        float anchors_cx= cur_anchors_3d[0];
        float anchors_by= cur_anchors_3d[1];
        float anchors_cz= cur_anchors_3d[2];
        float anchors_l = cur_anchors_3d[3];
        float anchors_h = cur_anchors_3d[4];
        float anchors_w = cur_anchors_3d[5];
        float anchors_ry= cur_anchors_3d[6];
        float anchors_max_distance = max(sqrtf((anchors_l / 2.) * (anchors_l / 2.)
                                    + (anchors_w / 2.) * (anchors_w / 2.)),1e-20f);

        float x, y, z;
        int inside_anchors, inside_gt;

        for (int k=0;k<n;++k) {
            x=cur_xyz[k*3+0];
            y=cur_xyz[k*3+1];
            z=cur_xyz[k*3+2];

            inside_anchors = point_inside_box_3d(x, y, z, 
                anchors_cx, anchors_by, anchors_cz, anchors_l, 
                anchors_h, anchors_w, anchors_ry, anchors_max_distance);

            inside_gt = point_inside_box_3d(x, y, z,
                gt_boxes_cx, gt_boxes_by, gt_boxes_cz, gt_boxes_l,
                gt_boxes_h, gt_boxes_w, gt_boxes_ry, gt_boxes_max_distance); 

            un += (inside_gt | inside_anchors);
            in += (inside_gt & inside_anchors);
        }
        un = max(un, 1);
        cur_iou_points[0] = float(in) / float(un);
    }
}



// input: radius (1), nsample (1), xyz1 (b,n,3), xyz2 (b,m,3)
// output: idx (b,m,nsample), pts_cnt (b,m)
__global__ void query_ball_point_gpu(int b, int n, int m, float radius, int nsample, const float *xyz1, const float *xyz2, int *idx, int *pts_cnt) {
    int total_idx = b * m;
    CUDA_1D_KERNEL_LOOP(point_inds, total_idx){
        int batch_index = point_inds / m;

        const float* cur_xyz1;
        const float* cur_xyz2;
        cur_xyz1 = xyz1 + n*3*batch_index;
        cur_xyz2 = xyz2 + point_inds * 3;

        int* cur_idx;
        int* cur_pts_cnt;
        cur_idx = idx + nsample*point_inds;
        cur_pts_cnt = pts_cnt + point_inds; // counting how many unique points selected in local region

        float x2=cur_xyz2[0];
        float y2=cur_xyz2[1];
        float z2=cur_xyz2[2];

        float x1, y1, z1, d;

        int cnt = 0;
        for (int k=0;k<n;++k) {
            if (cnt == nsample)
                break; // only pick the FIRST nsample points in the ball
            x1=cur_xyz1[k*3+0];
            y1=cur_xyz1[k*3+1];
            z1=cur_xyz1[k*3+2];
            d=max(sqrtf((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1)),1e-20f);
            if (d<radius) {
                if (cnt==0) { // set ALL indices to k, s.t. if there are less points in ball than nsample, we still have valid (repeating) indices
                    for (int l=0;l<nsample;++l)
                        cur_idx[l] = k;
                }
                cur_idx[cnt] = k;
                cnt+=1;
            }
        }
        cur_pts_cnt[0] = cnt;
    }
}


// input: radius (1), nsample (1), xyz1 (b,n,3), xyz2 (b,m,3), sort_idx (b, m, n)
// output: idx (b,m,nsample), pts_cnt (b,m)
__global__ void query_ball_point_withidx_gpu(int b, int n, int m, float radius, int nsample, const float *xyz1, const float *xyz2, const int* sort_idx, int *idx, int *pts_cnt) {
    int total_idx = b * m;
    CUDA_1D_KERNEL_LOOP(point_inds, total_idx){
        int batch_index = point_inds / m;

        const float* cur_xyz1;
        const float* cur_xyz2;
        const int* cur_sort_idx;
        cur_xyz1 = xyz1 + n*3*batch_index;
        cur_xyz2 = xyz2 + point_inds * 3;
        cur_sort_idx = sort_idx + point_inds * n;

        int* cur_idx;
        int* cur_pts_cnt;
        cur_idx = idx + nsample*point_inds;
        cur_pts_cnt = pts_cnt + point_inds; // counting how many unique points selected in local region

        float x2=cur_xyz2[0];
        float y2=cur_xyz2[1];
        float z2=cur_xyz2[2];

        float x1, y1, z1, d;

        int cnt = 0;
        int k;
        for (int i=0;i<n;++i) {
            if (cnt == nsample)
                break; // only pick the FIRST nsample points in the ball
            k = cur_sort_idx[i];
            x1=cur_xyz1[k*3+0];
            y1=cur_xyz1[k*3+1];
            z1=cur_xyz1[k*3+2];
            d=max(sqrtf((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1)),1e-20f);
            if (d<radius) {
                if (cnt==0) { // set ALL indices to k, s.t. if there are less points in ball than nsample, we still have valid (repeating) indices
                    for (int l=0;l<nsample;++l)
                        cur_idx[l] = k;
                }
                cur_idx[cnt] = k;
                cnt+=1;
            }
        }
        cur_pts_cnt[0] = cnt;
    }
}

// input: min_radius (1), max_radius (1), nsample (1), xyz1 (b,n,3), xyz2 (b,m,3)
// output: idx (b,m,nsample), pts_cnt (b,m)
__global__ void query_ball_point_dilated_gpu(int b, int n, int m, float min_radius, float max_radius, int nsample, const float *xyz1, const float *xyz2, int *idx, int *pts_cnt) {
    int total_idx = b * m;
    CUDA_1D_KERNEL_LOOP(point_inds, total_idx){
        int batch_index = point_inds / m;

        const float* cur_xyz1;
        const float* cur_xyz2;
        cur_xyz1 = xyz1 + n*3*batch_index;
        cur_xyz2 = xyz2 + point_inds * 3;

        int* cur_idx;
        int* cur_pts_cnt;
        cur_idx = idx + nsample*point_inds;
        cur_pts_cnt = pts_cnt + point_inds; // counting how many unique points selected in local region

        float x2=cur_xyz2[0];
        float y2=cur_xyz2[1];
        float z2=cur_xyz2[2];

        float x1, y1, z1, d;

        int cnt = 0;
        for (int k=0;k<n;++k) {
            if (cnt == nsample)
                break; // only pick the FIRST nsample points in the ball
            x1=cur_xyz1[k*3+0];
            y1=cur_xyz1[k*3+1];
            z1=cur_xyz1[k*3+2];
            d=sqrtf((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1));
            if (d == 0){
                // x2, y2, z2: set all indices to k
                if (cnt == 0){
                    for (int l=0;l<nsample;++l)
                        cur_idx[l] = k;
                }
                cur_idx[cnt] = k;
                cnt += 1;
            }
            else if (d >= min_radius && d < max_radius) {
                if (cnt==0) { // set ALL indices to k, s.t. if there are less points in ball than nsample, we still have valid (repeating) indices
                    for (int l=0;l<nsample;++l)
                        cur_idx[l] = k;
                }
                cur_idx[cnt] = k;
                cnt+=1;
            }
        }
        cur_pts_cnt[0] = cnt;
    }
}


// input: points (b,n,c), idx (b,m,nsample)
// output: out (b,m,nsample,c)
__global__ void group_point_gpu(int b, int n, int c, int m, int nsample, const float *points, const int *idx, float *out) {
    int total_idx = b * m * nsample * c;
    CUDA_1D_KERNEL_LOOP(point_inds, total_idx){
        int batch_inds = point_inds / (m * nsample * c);
        int idx_inds = point_inds / c;
        int cur_channel = point_inds % c;

        const float* cur_points = points + batch_inds * n * c;
        int cur_idx = idx[idx_inds];
        float *cur_out = out + point_inds;

        if (cur_idx == -1){
            cur_out[0] = float(0);
        } else{
            cur_out[0] = cur_points[cur_idx * c + cur_channel];
        }
    }
}

// input: grad_out (b,m,nsample,c), idx (b,m,nsample),
// output: grad_points (b,n,c)
__global__ void group_point_grad_gpu(int b, int n, int c, int m, int nsample, const float *grad_out, const int *idx, float *grad_points) {
    int total_idx = b * m * nsample * c;
    CUDA_1D_KERNEL_LOOP(point_inds, total_idx){
        int batch_index = point_inds / (m * nsample * c);
        int idx_inds = point_inds / c;
        int cur_channel = point_inds % c;

        const float* cur_grad_out = grad_out + point_inds;
        int cur_idx = idx[idx_inds];
        float* cur_grad_points = grad_points + batch_index * n * c;

        if (cur_idx != -1){
            atomicAdd(&cur_grad_points[cur_idx * c + cur_channel], cur_grad_out[0]);
        }
    }
}

// input: k (1), distance matrix dist (b,m,n)
// output: idx (b,m,n), dist_out (b,m,n)
// only the top k results within n are useful
__global__ void selection_sort_gpu(int b, int n, int m, int k, const float *dist, int *outi, float *out) {
    int batch_index = blockIdx.x;
    dist+=m*n*batch_index;
    outi+=m*n*batch_index;
    out+=m*n*batch_index;

    int index = threadIdx.x;
    int stride = blockDim.x;

    // copy from dist to dist_out
    for (int j=index;j<m;j+=stride) {
        for (int s=0;s<n;++s) {
            out[j*n+s] = dist[j*n+s];
            outi[j*n+s] = s;
        }
    }

    float *p_dist;
    for (int j=index;j<m;j+=stride) {
        p_dist = out+j*n;
        // selection sort for the first k elements
        for (int s=0;s<k;++s) {
            int min=s; 
            // find the min
            for (int t=s+1;t<n;++t) {
                if (p_dist[t]<p_dist[min]) {
                    min = t;
                }
            }
            // swap min-th and i-th element
            if (min!=s) {
                float tmp = p_dist[min];
                p_dist[min] = p_dist[s];
                p_dist[s] = tmp;
                int tmpi = outi[j*n+min];
                outi[j*n+min] = outi[j*n+s];
                outi[j*n+s] = tmpi;
            }
        }
    }
}



void queryBoxes3dPointsLauncher(int b, int n, int m, int nsample, const float *xyz, const float *proposals, int *idx, int *pts_cnt){
    query_boxes_3d_points_gpu<<<block_num, threadsPerBlock>>>(b,n,m,nsample,xyz,proposals,idx,pts_cnt);
}

void queryBoxes3dMaskLauncher(int b, int n, int m, const float *xyz, const float *boxes_3d, int *mask){
    query_boxes_3d_mask_gpu<<<block_num, threadsPerBlock>>>(b,n,m,xyz,boxes_3d,mask);
}

void queryPointsIouLauncher(int b, int n, int anchors_num, int gt_num, const float* xyz, const float* anchors_3d, const float* gt_boxes_3d, const float* iou_matrix, float* iou_points){
    query_points_iou_gpu<<<block_num, threadsPerBlock>>>(b,n,anchors_num,gt_num,
                                                         xyz, anchors_3d, gt_boxes_3d,
                                                         iou_matrix, iou_points);
}

void queryBallPointLauncher(int b, int n, int m, float radius, int nsample, const float *xyz1, const float *xyz2, int *idx, int *pts_cnt) {
    query_ball_point_gpu<<<block_num,threadsPerBlock>>>(b,n,m,radius,nsample,xyz1,xyz2,idx,pts_cnt);
    //cudaDeviceSynchronize();
}
void queryBallPointDilatedLauncher(int b, int n, int m, float min_radius, float max_radius, int nsample, const float *xyz1, const float *xyz2, int *idx, int *pts_cnt) {
    query_ball_point_dilated_gpu<<<block_num,threadsPerBlock>>>(b,n,m,min_radius,max_radius,nsample,xyz1,xyz2,idx,pts_cnt);
    //cudaDeviceSynchronize();
}
void queryBallPointWithidxLauncher(int b, int n, int m, float radius, int nsample, const float *xyz1, const float *xyz2, const int* sort_idx, int *idx, int *pts_cnt){
    query_ball_point_withidx_gpu<<<block_num,threadsPerBlock>>>(b,n,m,radius,nsample,xyz1,xyz2,sort_idx,idx,pts_cnt);
}
void selectionSortLauncher(int b, int n, int m, int k, const float *dist, int *outi, float *out) {
    selection_sort_gpu<<<b,256>>>(b,n,m,k,dist,outi,out); 
    //cudaDeviceSynchronize();
}
void groupPointLauncher(int b, int n, int c, int m, int nsample, const float *points, const int *idx, float *out){
    group_point_gpu<<<block_num,threadsPerBlock>>>(b,n,c,m,nsample,points,idx,out);
    //cudaDeviceSynchronize();
}
void groupPointGradLauncher(int b, int n, int c, int m, int nsample, const float *grad_out, const int *idx, float *grad_points){
    group_point_grad_gpu<<<block_num,threadsPerBlock>>>(b,n,c,m,nsample,grad_out,idx,grad_points);
    //group_point_grad_gpu<<<1,1>>>(b,n,c,m,nsample,grad_out,idx,grad_points);
    //cudaDeviceSynchronize();
}
