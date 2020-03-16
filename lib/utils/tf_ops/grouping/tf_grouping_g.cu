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


// __global__ void calculate_points_iou_gpu(int b, int n, int anchors_num, int gt_num, const float* batch_points, const float* batch_anchors_corners, const float* batch_label_corners, float *out){
//     // batch_points [b, n, 3], batch_anchors_corners[b, anchors_num, 8, 3]
//     // batch_label_corners [b, gt_num, 8, 3]
//     // out [b, anchors_num, gt_num], whether a point inside a proposal
//     int loop_times = b * anchors_num * gt_num;
//     CUDA_1D_KERNEL_LOOP(index, loop_times){
//         // loop_kernel
//         int cur_batch_index = index / (anchors_num * gt_num);
//         int cur_anchor_index = (index / gt_num) % anchors_num;
//         int cur_gt_index = index % gt_num;
//        
//         const float *cur_batch_anchors_corners = batch_anchors_corners + cur_batch_index * (anchors_num * 8 * 3) + cur_anchor_index * 8 * 3; 
//         const float* cur_batch_label_corners = batch_label_corners + cur_batch_index * (gt_num * 8 * 3)  + cur_gt_index * 8 * 3;
//         const float *cur_batch_points = batch_points + cur_batch_index * (n * 3) 
//         float *cur_out = out + cur_batch_index * anchors_num * gt_num + cur_anchor_index * gt_num + cur_gt_index;
// 
//         const float* anchors_P1 = cur_batch_anchors_corners;
//         const float* anchors_P2 = cur_batch_anchors_corners + 3;
//         const float* anchors_P4 = cur_batch_anchors_corners + 9; 
//         const float* anchors_P5 = cur_batch_anchors_corners + 12;
//         // u
//         float anchors_u_0 = anchors_P2[0] - anchors_P1[0];
//         float anchors_u_1 = anchors_P2[1] - anchors_P1[1];
//         float anchors_u_2 = anchors_P2[2] - anchors_P1[2];
//         // v
//         float anchors_v_0 = anchors_P4[0] - anchors_P1[0];
//         float anchors_v_1 = anchors_P4[1] - anchors_P1[1];
//         float anchors_v_2 = anchors_P4[2] - anchors_P1[2];
//         // w
//         float anchors_w_0 = anchors_P5[0] - anchors_P1[0];
//         float anchors_w_1 = anchors_P5[1] - anchors_P1[1];
//         float anchors_w_2 = anchors_P5[2] - anchors_P1[2];
// 
//         const float* label_P1 = cur_batch_label_corners; 
//         const float* label_P2 = cur_batch_label_corners + 3; 
//         const float* label_P4 = cur_batch_label_corners + 9; 
//         const float* label_P5 = cur_batch_label_corners + 12; 
//         float label_u_0 = label_P2[0] - label_P1[0];
//         float label_u_1 = label_P2[1] - label_P1[1];
//         float label_u_2 = label_P2[2] - label_P1[2];
//         float label_v_0 = label_P4[0] - label_P1[0];
//         float label_v_1 = label_P4[1] - label_P1[1];
//         float label_v_2 = label_P4[2] - label_P1[2];
//         float label_w_0 = label_P5[0] - label_P1[0];
//         float label_w_1 = label_P5[1] - label_P1[1];
//         float label_w_2 = label_P5[2] - label_P1[2];
// 
//         float union_points_num = 0;
//         float inter_points_num = 0;
//         for (int i = 0; i < n; i++){
//             // anchors
//             float anchors_u_dot_x =  anchors_u_0 * cur_batch_points[0] + anchors_u_1 * cur_batch_points[1] + anchors_u_2 * cur_batch_points[2];
//             float anchors_u_dot_p1 = anchors_u_0 * anchors_P1[0] + anchors_u_1 * anchors_P1[1] + anchors_u_2 * anchors_P1[2];
//             float anchors_u_dot_p2 = anchors_u_0 * anchors_P2[0] + anchors_u_1 * anchors_P2[1] + anchors_u_2 * anchors_P2[2];
//   
//             float anchors_v_dot_x =  anchors_v_0 * cur_batch_points[0] + anchors_v_1 * cur_batch_points[1] + anchors_v_2 * cur_batch_points[2];
//             float anchors_v_dot_p1 = anchors_v_0 * anchors_P1[0] + anchors_v_1 * anchors_P1[1] + anchors_v_2 * anchors_P1[2];
//             float anchors_v_dot_p4 = anchors_v_0 * anchors_P4[0] + anchors_v_1 * anchors_P4[1] + anchors_v_2 * anchors_P4[2];
// 
//             float anchors_w_dot_x =  anchors_w_0 * cur_batch_points[0] + anchors_w_1 * cur_batch_points[1] + anchors_w_2 * cur_batch_points[2];
//             float anchors_w_dot_p1 = anchors_w_0 * anchors_P1[0] + anchors_w_1 * anchors_P1[1] + anchors_w_2 * anchors_P2[2];
//             float anchors_w_dot_p5 = anchors_w_0 * anchors_P5[0] + anchors_w_1 * anchors_P5[1] + anchors_w_2 * anchors_P5[2];
// 
//             // then determine whether in or out
//             int anchors_cur_point_mask = (anchors_u_dot_p1 < anchors_u_dot_x) & (anchors_u_dot_x < anchors_u_dot_p2) &
//                                          (anchors_v_dot_p1 < anchors_v_dot_x) & (anchors_v_dot_x < anchors_v_dot_p4) &
//                                          (anchors_w_dot_p1 < anchors_w_dot_x) & (anchors_w_dot_x < anchors_w_dot_p5);
// 
//             // labels 
//             float labels_u_dot_x =  labels_u_0 * cur_batch_points[0] + labels_u_1 * cur_batch_points[1] + labels_u_2 * cur_batch_points[2];
//             float labels_u_dot_p1 = labels_u_0 * labels_P1[0] + labels_u_1 * labels_P1[1] + labels_u_2 * labels_P1[2];
//             float labels_u_dot_p2 = labels_u_0 * labels_P2[0] + labels_u_1 * labels_P2[1] + labels_u_2 * labels_P2[2];
//   
//             float labels_v_dot_x =  labels_v_0 * cur_batch_points[0] + labels_v_1 * cur_batch_points[1] + labels_v_2 * cur_batch_points[2];
//             float labels_v_dot_p1 = labels_v_0 * labels_P1[0] + labels_v_1 * labels_P1[1] + labels_v_2 * labels_P1[2];
//             float labels_v_dot_p4 = labels_v_0 * labels_P4[0] + labels_v_1 * labels_P4[1] + labels_v_2 * labels_P4[2];
// 
//             float labels_w_dot_x =  labels_w_0 * cur_batch_points[0] + labels_w_1 * cur_batch_points[1] + labels_w_2 * cur_batch_points[2];
//             float labels_w_dot_p1 = labels_w_0 * labels_P1[0] + labels_w_1 * labels_P1[1] + labels_w_2 * labels_P2[2];
//             float labels_w_dot_p5 = labels_w_0 * labels_P5[0] + labels_w_1 * labels_P5[1] + labels_w_2 * labels_P5[2];
// 
//             // then determine whether in or out
//             int labels_cur_point_mask = (labels_u_dot_p1 < labels_u_dot_x) & (labels_u_dot_x < labels_u_dot_p2) &
//                                         (labels_v_dot_p1 < labels_v_dot_x) & (labels_v_dot_x < labels_v_dot_p4) &
//                                         (labels_w_dot_p1 < labels_w_dot_x) & (labels_w_dot_x < labels_w_dot_p5);
// 
//             union_points_num += (labels_cur_point_mask | anchors_cur_point_mask);
//             inter_points_num += (labels_cur_point_mask & anchors_cur_point_mask);
//         }
//         cur_out[0] = inter_points_num / union_points_num;
//     }
// }

__global__ void query_corners_point_gpu(int b, int n, int proposals_num, const float* batch_points, const float* batch_anchors_corners, int *out){
    // batch_points [b, n, 3], batch_anchors_corners[b, proposals_num, 8, 3]
    // out [b, num_proposals, n], whether a point inside a proposal
    int loop_times = b * proposals_num * n;
    CUDA_1D_KERNEL_LOOP(index, loop_times){
        // loop_kernel
        int cur_batch_index = index / (n * proposals_num);
        int cur_proposals_index = (index / n) % proposals_num;
        int cur_point_index = index % n;
       
        const float *cur_batch_anchors_corners = batch_anchors_corners + cur_batch_index * (proposals_num * 8 * 3) + cur_proposals_index * 8 * 3; 
        const float *cur_batch_points = batch_points + cur_batch_index * (n * 3) + cur_point_index * 3;
        int *cur_out = out + cur_batch_index * proposals_num * n + cur_proposals_index * n + cur_point_index;

        const float* P1 = cur_batch_anchors_corners;
        const float* P2 = cur_batch_anchors_corners + 3;
        const float* P4 = cur_batch_anchors_corners + 9; 
        const float* P5 = cur_batch_anchors_corners + 12;

        // u
        float u_0 = P2[0] - P1[0];
        float u_1 = P2[1] - P1[1];
        float u_2 = P2[2] - P1[2];
        // v
        float v_0 = P4[0] - P1[0];
        float v_1 = P4[1] - P1[1];
        float v_2 = P4[2] - P1[2];
        // w
        float w_0 = P5[0] - P1[0];
        float w_1 = P5[1] - P1[1];
        float w_2 = P5[2] - P1[2];

        float u_dot_x = u_0 * cur_batch_points[0] + u_1 * cur_batch_points[1] + u_2 * cur_batch_points[2];
        float u_dot_p1 = u_0 * P1[0] + u_1 * P1[1] + u_2 * P1[2];
        float u_dot_p2 = u_0 * P2[0] + u_1 * P2[1] + u_2 * P2[2];
  
        float v_dot_x = v_0 * cur_batch_points[0] + v_1 * cur_batch_points[1] + v_2 * cur_batch_points[2];
        float v_dot_p1 = v_0 * P1[0] + v_1 * P1[1] + v_2 * P1[2];
        float v_dot_p4 = v_0 * P4[0] + v_1 * P4[1] + v_2 * P4[2];

        float w_dot_x = w_0 * cur_batch_points[0] + w_1 * cur_batch_points[1] + w_2 * cur_batch_points[2];
        float w_dot_p1 = w_0 * P1[0] + w_1 * P1[1] + w_2 * P2[2];
        float w_dot_p5 = w_0 * P5[0] + w_1 * P5[1] + w_2 * P5[2];

        // then determine whether in or out
        int cur_point_mask = (u_dot_p1 < u_dot_x) & (u_dot_x < u_dot_p2) &
                             (v_dot_p1 < v_dot_x) & (v_dot_x < v_dot_p4) &
                             (w_dot_p1 < w_dot_x) & (w_dot_x < w_dot_p5);
        cur_out[0] = cur_point_mask; 
    }
}

// input: radius (1), nsample (1), subcube_num(1), total_subcube_num(1), xyz1 (b,n,3), xyz2 (b,m,3)
// output: idx (b,m,nsample), pts_cnt (b,m,total_subcube_num), subcube_location (b,m,total_subcube_num,3)
__global__ void query_cube_point_gpu(int b, int n, int m, float radius, int nsample, int subcube_num, int total_subcube_num, const float *xyz1, const float *xyz2, int *idx, int *pts_cnt, float* subcube_location) {
    int total_idx = b * m;
    CUDA_1D_KERNEL_LOOP(point_inds, total_idx){
        int batch_index = point_inds / m;

        const float* cur_xyz1;
        const float* cur_xyz2;
        cur_xyz1 = xyz1 + n*3*batch_index;
        cur_xyz2 = xyz2 + point_inds * 3;

        int* cur_idx;
        int* cur_pts_cnt;
        float* cur_subcube_location;
        cur_idx = idx + nsample*point_inds;
        cur_pts_cnt = pts_cnt + point_inds * total_subcube_num; // pts_num per subcube 
        cur_subcube_location = subcube_location + point_inds * total_subcube_num * 3;

        // current center xyz location
        float x2=cur_xyz2[0];
        float y2=cur_xyz2[1];
        float z2=cur_xyz2[2];

        float radius_per_subcube = radius * 2 / float(subcube_num); 
        float ctr_x = x2 - radius;
        float ctr_y = y2 - radius;
        float ctr_z = z2 - radius;
        // now calculate cur_subcube_location
        int cur_subcube_location_idx = 0;
        for (int i=0; i<subcube_num; i++){
            for (int j=0; j<subcube_num; j++){
                for (int k=0; k<subcube_num; k++){
                    cur_subcube_location_idx = i * subcube_num * subcube_num + j * subcube_num + k;
                    cur_pts_cnt[cur_subcube_location_idx] = 0;
                    cur_subcube_location[cur_subcube_location_idx * 3 + 0] = ctr_x + radius_per_subcube * (0.5 + i); 
                    cur_subcube_location[cur_subcube_location_idx * 3 + 1] = ctr_y + radius_per_subcube * (0.5 + j); 
                    cur_subcube_location[cur_subcube_location_idx * 3 + 2] = ctr_z + radius_per_subcube * (0.5 + k); 
                }
            }
        }

        float x1, y1, z1;
        int cnt = 0;
        for (int k=0;k<n;++k) {
            if (cnt == nsample)
                break; // only pick the FIRST nsample points in the ball
            x1=cur_xyz1[k*3+0];
            y1=cur_xyz1[k*3+1];
            z1=cur_xyz1[k*3+2];
            if (abs(x1 - x2) < radius && abs(y1 - y2) < radius && abs(z1 - z2) < radius){
                cur_subcube_location_idx = floor((x1 - ctr_x) / radius_per_subcube) * subcube_num * subcube_num \
                                         + floor((y1 - ctr_y) / radius_per_subcube) * subcube_num \
                                         + floor((z1 - ctr_z) / radius_per_subcube);
                if (cnt==0) { // set ALL indices to k, s.t. if there are less points in ball than nsample, we still have valid (repeating) indices
                    for (int l=0;l<nsample;++l)
                        cur_idx[l] = k;
                }
                cur_idx[cnt] = k;
                cur_pts_cnt[cur_subcube_location_idx] += 1;
                cnt+=1;
            }
        }
    }
}


// input: nsample (1), xyz1 (b,n,3), xyz2 (b,m,3), radius (b, m, split_bin_num)
// output: idx (b,m,nsample), pts_cnt (b,m), radius_idx (b,m,nsample,2), radius_rate (b,m,nsample,2)
__global__ void query_ball_point_dynamic_shape_gpu(int b, int n, int m, int split_bin_num, int nsample, const float *xyz1, const float *xyz2, const float* radius, int *idx, int *pts_cnt, int* radius_idx, float* radius_rate) {
    int total_idx = b * m;
    CUDA_1D_KERNEL_LOOP(point_inds, total_idx){
        int batch_index = point_inds / m;

        const float* cur_xyz1;
        const float* cur_xyz2;
        cur_xyz1 = xyz1 + n*3*batch_index;
        cur_xyz2 = xyz2 + point_inds * 3;

        const float* cur_radius;
        cur_radius = radius + point_inds * split_bin_num; // split_bin_num [0, 2*pi]

        int* cur_idx;
        int* cur_pts_cnt;
        cur_idx = idx + nsample*point_inds;
        cur_pts_cnt = pts_cnt + point_inds; // counting how many unique points selected in local region
 
        int* cur_radius_idx;
        float* cur_radius_rate;
        cur_radius_idx = radius_idx + point_inds * nsample * 2;
        cur_radius_rate = radius_rate + point_inds * nsample * 2;

        float x2=cur_xyz2[0];
        float y2=cur_xyz2[1];
        float z2=cur_xyz2[2];

        float x1, y1, z1, d;
        float d_theta;
        float cur_theta;
        float interval = 2 * M_PI / float(split_bin_num); // 2pi / split_bin
        int low_idx, high_idx;
        float low_rate, high_rate;
        float judge_radius;

        int cnt = 0;
        for (int k=0;k<n;++k) {
            if (cnt == nsample)
                break; // only pick the FIRST nsample points in the ball
            x1=cur_xyz1[k*3+0];
            y1=cur_xyz1[k*3+1];
            z1=cur_xyz1[k*3+2];
            d=max(sqrtf((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1)+(z2-z1)*(z2-z1)),1e-20f);
            d_theta=max(sqrtf((x2-x1)*(x2-x1)+(z2-z1)*(z2-z1)),1e-20f);
            cur_theta = acos((x1-x2)/d_theta); // [0, pi]
            // first of all, according to current x,y,z determine the two radius to use
            if (z1 - z2 >= 0) { // [0, pi]
                low_idx = floorf(cur_theta / interval); 
                high_idx = low_idx + 1;
            } 
            else{ // z1 - z2 < 0 [pi, 2pi]
                cur_theta = 2*M_PI - cur_theta;
                low_idx = floorf(cur_theta / interval); 
                high_idx = low_idx + 1;
            }
            low_rate = (cur_theta - float(low_idx) * interval) / interval;
            high_rate = (float(high_idx) * interval - cur_theta) / interval;
            high_idx = high_idx % split_bin_num;
            low_idx = low_idx % split_bin_num;
            judge_radius = high_rate * cur_radius[low_idx] + low_rate * cur_radius[high_idx]; 
            // printf("current theta is: (%f, %f, %f), %f, %f, %f, %f, %f, %d, %d", x1,y1,z1,cur_theta,low_rate,high_rate, d, judge_radius, low_idx, high_idx);
            if (d<judge_radius) {
                if (cnt==0) { // set ALL indices to k, s.t. if there are less points in ball than nsample, we still have valid (repeating) indices
                    for (int l=0;l<nsample;++l){
                        cur_idx[l] = k;
                        cur_radius_idx[2*l] = low_idx;
                        cur_radius_idx[2*l+1] = high_idx;
                        cur_radius_rate[2*l] = high_rate;
                        cur_radius_rate[2*l+1] = low_rate;
                    }
                }
                cur_idx[cnt] = k;
                cur_radius_idx[2*cnt] = low_idx;
                cur_radius_idx[2*cnt+1] = high_idx;
                cur_radius_rate[2*cnt] = high_rate;
                cur_radius_rate[2*cnt+1] = low_rate;
                cnt+=1;
            }
        }
        cur_pts_cnt[0] = cnt;
    }
}


// input: xyz1 (b, n, nsample, 3) radius (b, n, split_bin_num)
// output: radius_idx (b,n,nsample,2), radius_rate (b,n,nsample,2)
__global__ void query_dynamic_radius_for_points_gpu(int b, int n, int nsample, int split_bin_num, const float *xyz1, const float* radius, int* radius_idx, float* radius_rate) {
    int total_idx = b * n * nsample;
    CUDA_1D_KERNEL_LOOP(point_inds, total_idx){
        const float* cur_xyz1;
        cur_xyz1 = xyz1 + point_inds * 3;

        int* cur_radius_idx;
        float* cur_radius_rate;
        cur_radius_idx = radius_idx + point_inds * 2;
        cur_radius_rate = radius_rate + point_inds * 2;

        float x1=cur_xyz1[0];
        float z1=cur_xyz1[2];

        float d_theta;
        float cur_theta;
        float interval = 2 * M_PI / float(split_bin_num); // 2pi / split_bin
        int low_idx, high_idx;
        float low_rate, high_rate;

        d_theta=max(sqrtf(x1*x1+z1*z1),1e-20f);
        cur_theta = acos(x1/d_theta); // [0, pi]
        // first of all, according to current x,y,z determine the two radius to use
        if (z1 >= 0) { // [0, pi]
            low_idx = floorf(cur_theta / interval); 
            high_idx = low_idx + 1;
        } 
        else{ // z1 < 0 [pi, 2pi]
            cur_theta = 2*M_PI - cur_theta;
            low_idx = floorf(cur_theta / interval); 
            high_idx = low_idx + 1;
        }
        low_rate = (cur_theta - float(low_idx) * interval) / interval;
        high_rate = (float(high_idx) * interval - cur_theta) / interval;
        high_idx = high_idx % split_bin_num;
        low_idx = low_idx % split_bin_num;
        cur_radius_idx[0] = low_idx;
        cur_radius_idx[1] = high_idx;
        cur_radius_rate[0] = high_rate;
        cur_radius_rate[1] = low_rate;
    }
}

// input: xyz1 (b, n, 3) gt_boxes_3d (b, n, 7)
// output: target_dist (b, n, split_bin_num)
__global__ void query_target_distance_for_points_gpu(int b, int n, int split_bin_num, const float* xyz1, const float* gt_boxes_3d, float* target_dist) {
    int total_idx = b * n * split_bin_num;
    CUDA_1D_KERNEL_LOOP(point_inds, total_idx){
        int pts_index = point_inds / split_bin_num;
        int cur_split_bin = point_inds % split_bin_num;

        const float* cur_xyz1;
        cur_xyz1 = xyz1 + pts_index * 3;

        const float* cur_gt_boxes_3d;
        cur_gt_boxes_3d = gt_boxes_3d + pts_index * 7;

        float* cur_target_dist;
        cur_target_dist = target_dist + point_inds;

        float x1=cur_xyz1[0];
        float z1=cur_xyz1[2];
        float l = cur_gt_boxes_3d[3] / 2.;
        float w = cur_gt_boxes_3d[5] / 2.;
        float cur_box_angle = cur_gt_boxes_3d[6];

        float interval = 2 * M_PI / float(split_bin_num); // 2pi / split_bin

        float cur_theta = float(cur_split_bin) * interval + cur_box_angle + float(2 * M_PI); 
        cur_theta = fmod(cur_theta, float(2 * M_PI));
        float cur_x, cur_y, cur_distance;

        if (cur_theta < M_PI / 2.){
            cur_y = abs(w - z1);
            cur_x = abs(cur_y / tan(cur_theta)); 
            if (cur_x > abs(l - x1)) {
                cur_x = abs(l - x1);
                cur_y = abs(cur_x * tan(cur_theta));
            }
            cur_distance = sqrtf(cur_x * cur_x + cur_y * cur_y);
        } else if (cur_theta >= M_PI / 2. && cur_theta < M_PI) {
            cur_y = abs(w - z1);
            cur_x = abs(cur_y / tan(cur_theta)); 
            if (cur_x > abs(-l - x1)) {
                cur_x = abs(-l - x1);
                cur_y = abs(cur_x * tan(cur_theta));
            }
            cur_distance = sqrtf(cur_x * cur_x + cur_y * cur_y);
        } else if (cur_theta >= M_PI && cur_theta < M_PI * 3. / 2.) {
            cur_y = abs(-w - z1);
            cur_x = abs(cur_y / tan(cur_theta)); 
            if (cur_x > abs(-l - x1)) {
                cur_x = abs(-l - x1);
                cur_y = abs(cur_x * tan(cur_theta));
            }
            cur_distance = sqrtf(cur_x * cur_x + cur_y * cur_y);
        } else if (cur_theta >= M_PI * 3. / 2.) {
            cur_y = abs(-w - z1);
            cur_x = abs(cur_y / tan(cur_theta)); 
            if (cur_x > abs(l-x1)) {
                cur_x = abs(l-x1);
                cur_y = abs(cur_x * tan(cur_theta));
            }
            cur_distance = sqrtf(cur_x * cur_x + cur_y * cur_y);
        } 
        cur_target_dist[0] = cur_distance;
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


// input: nsample (1), xyz1 (b,n,3), xyz2 (b,m,3), radius (b, m)
// output: idx (b,m,nsample), pts_cnt (b,m)
__global__ void query_ball_point_dynamic_radius_gpu(int b, int n, int m, int nsample, const float *xyz1, const float *xyz2, const float* radius, int *idx, int *pts_cnt) {
    int total_idx = b * m;
    CUDA_1D_KERNEL_LOOP(point_inds, total_idx){
        int batch_index = point_inds / m;
        float cur_radius = radius[point_inds];

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
            if (d<cur_radius) {
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


// void calculatePointsIouLauncher(int b, int n, int anchors_num, int gt_num, const float* batch_points, const float* batch_anchors_corners, const float* batch_label_corners, float *out){
//     calculate_points_iou_gpu<<<block_num, threadsPerBlock>>>(b, n, anchors_num, gt_num, batch_points, batch_anchors_corners, batch_label_corners, out);
// }

void queryCornersPointLauncher(int b, int n, int proposals_num, const float* batch_points, const float* batch_anchors_corners, int *out){
    query_corners_point_gpu<<<block_num, threadsPerBlock>>>(b, n, proposals_num, batch_points, batch_anchors_corners, out);
}

void queryCubePointLauncher(int b, int n, int m, float radius, int nsample, int subcube_num, int total_subcube_num, const float *xyz1, const float *xyz2, int *idx, int *pts_cnt, float* subcube_location) {
    query_cube_point_gpu<<<block_num,threadsPerBlock>>>(b,n,m,radius,nsample,subcube_num,total_subcube_num,xyz1,xyz2,idx,pts_cnt,subcube_location);
    //cudaDeviceSynchronize();
}

void queryBallPointDynamicShapeLauncher(int b, int n, int m, int split_bin_num, int nsample, const float *xyz1, const float *xyz2, const float* radius, int *idx, int *pts_cnt, int* radius_idx, float* radius_rate){
    query_ball_point_dynamic_shape_gpu<<<block_num,threadsPerBlock>>>(b,n,m,split_bin_num,nsample,xyz1,xyz2,radius,idx,pts_cnt,radius_idx,radius_rate);
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
void queryBallPointDynamicRadiusLauncher(int b, int n, int m, int nsample, const float *xyz1, const float *xyz2, const float* radius, int *idx, int *pts_cnt){
    query_ball_point_dynamic_radius_gpu<<<block_num,threadsPerBlock>>>(b,n,m,nsample,xyz1,xyz2,radius,idx,pts_cnt);
}
void queryDynamicRadiusForPointsLauncher(int b, int n, int nsample, int split_bin_num, const float *xyz1, const float* radius, int* radius_idx, float* radius_rate){
    query_dynamic_radius_for_points_gpu<<<block_num,threadsPerBlock>>>(b, n, nsample, split_bin_num, xyz1, radius, radius_idx, radius_rate);
}
void queryTargetDistanceForPointsLauncher(int b, int n, int split_bin_num, const float* xyz1, const float* gt_boxes_3d, float* target_dist){
    query_target_distance_for_points_gpu<<<block_num,threadsPerBlock>>>(b, n, split_bin_num, xyz1, gt_boxes_3d, target_dist);
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
