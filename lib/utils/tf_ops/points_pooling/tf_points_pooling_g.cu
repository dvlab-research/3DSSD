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

const int block_num = 512;
#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))
const int threadsPerBlock = sizeof(unsigned long long) * 8;

/*
    PointsPooling module for sparse-to-dense
    Args:
        pc: [bs, proposal_num, point_num, channel_num]
        box_3d: [bs, proposal_num, 6] x, y, z, l, h, w 
        pc_loc: [bs, proposal_num, point_num, 3]
    Return:
        out_features: [bs, proposal_num, l, h, w, sample_num, c]
        out_idx: [bs, proposal_num, l, h, w, sample_num]
        sampled_num_lists: [bs, proposal_num, l, h, w]
        pillars: [bs, proposal_num, l, h, w, 3]
*/
__global__ void points_pooling_gpu(const int bs, const int proposal_num, const int point_num, const int channel_num, 
    const int l, const int h, const int w, const int sample_num, 
    const float* pc, const float* box_3d, const float* pc_loc, 
    float* out_features, int* out_idx, int *sampled_num_lists, float* pillars){

    int loop_times = bs * proposal_num;
    CUDA_1D_KERNEL_LOOP(batch_inds, loop_times){
        // recurrent the proposals
        const float* cur_pc = pc + batch_inds * point_num * channel_num;
        const float* cur_box_3d = box_3d + batch_inds * 6;
        const float* cur_pc_loc = pc_loc + batch_inds * point_num * 3;
 
        float box_cx = cur_box_3d[0];
        float box_by = cur_box_3d[1];
        float box_cz = cur_box_3d[2];
        float box_l  = cur_box_3d[3];
        float box_h  = cur_box_3d[4];
        float box_w  = cur_box_3d[5];

        float interval_l = box_l / float(l);
        float interval_h = box_h / float(h);
        float interval_w = box_w / float(w);
       
        float xmin = box_cx - box_l / 2.;
        float ymin = box_by - box_h;
        float zmin = box_cz - box_w / 2.;

        float* cur_out_features = out_features + batch_inds * l * h * w * sample_num * channel_num;
        int* cur_out_idx = out_idx + batch_inds * l * h * w * sample_num;
        int* cur_sampled_num_list = sampled_num_lists + batch_inds * l * h * w; 
        float* cur_pillars = pillars + batch_inds * l * h * w;

        float tmp_x, tmp_y, tmp_z;
        int tmp_idx;
        for (int i=0; i < l; i++){
            for (int j=0; j<h; j++){
                for (int k = 0; k < w; k++){
                    tmp_x = xmin + (i + 0.5) * interval_l;
                    tmp_y = ymin + (j + 0.5) * interval_h;
                    tmp_z = zmin + (k + 0.5) * interval_w; 
                    tmp_idx = (i * h * w + j * w + k) * 3; 
                    cur_pillars[tmp_idx] = tmp_x;
                    cur_pillars[tmp_idx + 1] = tmp_y;
                    cur_pillars[tmp_idx + 2] = tmp_z;
                }
            }
        }

        float cur_pc_x, cur_pc_y, cur_pc_z;
        for (int i = 0; i < point_num; i++){
            // calculate the result
            cur_pc_x = cur_pc_loc[i * 3 + 0];
            cur_pc_y = cur_pc_loc[i * 3 + 1];
            cur_pc_z = cur_pc_loc[i * 3 + 2]; 

            int x_inds = min(max(int(floor((cur_pc_x - xmin) / interval_l)), 0), l - 1);
            int y_inds = min(max(int(floor((cur_pc_y - ymin) / interval_h)), 0), h - 1);
            int z_inds = min(max(int(floor((cur_pc_z - zmin) / interval_w)), 0), w - 1);

            int grid_inds = x_inds * h * w + y_inds * w + z_inds;
            if (cur_sampled_num_list[grid_inds] >= sample_num)
                continue;
            int cur_pc_inds = cur_sampled_num_list[grid_inds];
            int out_grid_inds = grid_inds * sample_num + cur_pc_inds;;
            
            cur_out_idx[out_grid_inds] = i;
            for (int j = 0; j < channel_num; j ++){
                cur_out_features[out_grid_inds * channel_num + j] = cur_pc[i * channel_num + j];
            }
            cur_sampled_num_list[grid_inds] += 1;
        }
    }
}


/*  Calculate gradients of PointsPool operation in sparse to dense
    Args:
        pc: [bs, proposal_num, point_num, channel_num]
        out_idx: [bs, proposal_num, l, h, w, sample_num]    
        sampled_num_lists: [bs, proposal_num, l, h, w]
        features_grad: [bs, proposal_num, l, h, w, sample_num, channel_num]
    Return:
        pc_grad: [bs, proposal_num, point_num, channel_num]
*/
__global__ void points_pooling_grad_gpu(const int bs, const int proposal_num, const int point_num, const int channel_num,
    const int l, const int h, const int w, const int sample_num,
    const float* pc, const int* out_idx, const int *sampled_num_lists, const float* features_grad,
    float *pc_grad){

    int loop_times = bs * proposal_num * l * h * w * sample_num * channel_num;
    CUDA_1D_KERNEL_LOOP(point_inds, loop_times){
        int proposal_idx = point_inds / (l * h * w * sample_num * channel_num);
        int sample_num_lists_idx = point_inds / (sample_num * channel_num);

        int out_idx_idx = point_inds / channel_num;
        int channel_idx = point_inds % channel_num;

        int cur_sample_idx = out_idx_idx % sample_num;
        if (cur_sample_idx >= sampled_num_lists[sample_num_lists_idx])
            continue;

        const int* cur_out_idx = out_idx + out_idx_idx;
        float* cur_pc_grad = pc_grad + proposal_idx * point_num * channel_num + 
                             cur_out_idx[0] * channel_num + channel_idx; 
        atomicAdd(&cur_pc_grad[0], features_grad[point_inds]);
    }
}


void pointsPoolingLauncher(const int bs, const int proposal_num, const int point_num, const int channel_num, 
    const int l, const int h, const int w, const int sample_num, 
    const float* pc, const float* box_3d, const float* pc_loc, 
    float* out_features, int* out_idx, int *sampled_num_lists, float* pillars){

    points_pooling_gpu<<<block_num, threadsPerBlock>>>(
        bs, proposal_num, point_num, channel_num,
        l, h, w, sample_num,
        pc, box_3d, pc_loc,
        out_features, out_idx, sampled_num_lists, pillars
    );
}

void pointsPoolingGradLauncher(const int bs, const int proposal_num, const int point_num, const int channel_num, 
    const int l, const int h, const int w, const int sample_num, 
    const float* pc, const int* out_idx, const int *sampled_num_lists, const float* features_grad, 
    float *pc_grad){

    points_pooling_grad_gpu<<<block_num, threadsPerBlock>>>(
        bs, proposal_num, point_num, channel_num, 
        l, h, w, sample_num, 
        pc, out_idx, sampled_num_lists, features_grad, 
        pc_grad);
}
