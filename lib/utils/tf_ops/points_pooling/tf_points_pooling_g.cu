// input: iou_matrix, [n, n], points_sampling, [n, npoint], merge function, 0:union, 1: intersection
// min_keep_num
// output: keep_inds [n, n], 0/1
// nmsed_points_sample: [n, npoint], 0/1
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

__global__ void points_pooling_gpu(const int proposal_num, const int points_num, const int channels, const int l, const int h, const int w, const int sample_num, const float* pc, const float* proposals, const float* pc_location, int *sampled_num_list, float* out_features, int *out_idx, float* anchors_pillars){
    // pc: [b, 512, c], proposals: [b, 6], pc_locations: [b, 512, 3]
    // sampled_num_list [b, l, h, w]: how many sampled points num
    // out_features, [b, l, h, w, sample_num, c]
    // anchors_pillars, [b, l, h, w, 3], centers for each pillars
    CUDA_1D_KERNEL_LOOP(batch_inds, proposal_num){
        // recurrent the proposals
        const float* cur_pc = pc + batch_inds * points_num * channels;
        const float* cur_proposals = proposals + batch_inds * 6;
        const float* cur_pc_locations = pc_location + batch_inds * points_num * 3;
 
        float cur_xctr = cur_proposals[0];
        float cur_yctr = cur_proposals[1];
        float cur_zctr = cur_proposals[2];
        float cur_length = cur_proposals[3];
        float cur_height = cur_proposals[4];
        float cur_width = cur_proposals[5];

        float length_gap = cur_length / l;
        float height_gap = cur_height / h;
        float width_gap = cur_width / w;
       
        float xmin = cur_xctr - cur_length / 2.;
        float ymin = cur_yctr - cur_height;
        float zmin = cur_zctr - cur_width / 2.;

        int* cur_sampled_num_list = sampled_num_list + batch_inds * l * h * w; 
        float* cur_out_features = out_features + batch_inds * l * h * w * sample_num * channels;
        int* cur_out_idx = out_idx + batch_inds * l * h * w * sample_num;
        float* cur_anchors_pillars = anchors_pillars + batch_inds * l * h * w;

        float tmp_x, tmp_y, tmp_z;
        int tmp_idx;
        for (int i=0; i < l; i ++){
            for (int j=0; j<h; j++){
                for (int k = 0; k < w; k++){
                    tmp_x = xmin + (i + 0.5) * length_gap;
                    tmp_y = ymin + (j + 0.5) * height_gap;
                    tmp_z = zmin + (k + 0.5) * width_gap; 
                    tmp_idx = (i * h * w + j * w + k) * 3; 
                    cur_anchors_pillars[tmp_idx] = tmp_x;
                    cur_anchors_pillars[tmp_idx + 1] = tmp_y;
                    cur_anchors_pillars[tmp_idx + 2] = tmp_z;
                }
            }
        }

        //int max_align_grid_inds = l * h * w; 

        for (int i = 0; i < points_num; i ++){
            // calculate the result
            float cur_pc_x = cur_pc_locations[i * 3 + 0];
            float cur_pc_y = cur_pc_locations[i * 3 + 1];
            float cur_pc_z = cur_pc_locations[i * 3 + 2]; 

            int align_x_inds = min(max(int(floor((cur_pc_x - xmin) / length_gap)), 0), l - 1);
            int align_y_inds = min(max(int(floor((cur_pc_y - ymin) / height_gap)), 0), h - 1);
            int align_z_inds = min(max(int(floor((cur_pc_z - zmin) / width_gap)), 0), w - 1);

            int align_grid_inds = align_x_inds * h * w + align_y_inds * w + align_z_inds;
            //printf ("%d, %d, %d\n", l, h, w);
            //printf ("%d, %d, %d, %d, %f\n", align_x_inds, align_y_inds, align_z_inds, align_grid_inds, length_gap);
            // align_grid_inds = min(align_grid_inds, max_align_grid_inds - 1);
            if (cur_sampled_num_list[align_grid_inds] >= sample_num)
                continue;
            int cur_pooled_pts_num = cur_sampled_num_list[align_grid_inds];
            int out_align_grid_inds = align_grid_inds * sample_num + cur_pooled_pts_num;
            
            cur_out_idx[out_align_grid_inds] = i;
            for (int j = 0; j < channels; j ++){
                cur_out_features[out_align_grid_inds * channels + j] = cur_pc[i * channels + j];
            }
            //cudaMemcpy(&cur_out_features[out_align_grid_inds * channels], &cur_pc[i * channels], sizeof(float) * channels, cudaMemcpyDeviceToDevice);
            cur_sampled_num_list[align_grid_inds] += 1;
        }
    }
}

__global__ void points_pooling_grad_gpu(const int proposal_num, const int points_num, const int channels, const int l, const int h, const int w, const int sample_num, const float* pc, const int* out_idx, const int *sampled_num_list, const float* features_grad, float *pc_grad){
    // pc: [b, points_num, c], out_idx[b, l, h, w, sample_num], features_grad: [b, l, h, w, sample_num, c]
    // sampled_num_list: [b, l, h, w]
    // pc_grad, [b, points_num. c]
    for (int batch_inds = blockIdx.x; batch_inds < proposal_num; batch_inds += gridDim.x){
        const int *cur_out_idx = out_idx + batch_inds * l * h * w * sample_num;
        const int *cur_sampled_num_list = sampled_num_list + batch_inds * l * h * w; 
        const float *cur_features_grad = features_grad + batch_inds * l * h * w * sample_num * channels;
        float *cur_pc_grad = pc_grad + batch_inds * points_num * channels;

        int x_index = threadIdx.x;
        int x_stride = blockDim.x;
        for (int i = x_index; i < l * h * w; i += x_stride){
            // look through each x_index
            int cur_sampled_num = cur_sampled_num_list[i];
            for (int j=0; j < cur_sampled_num; j++){
                int cur_align_out_idx = cur_out_idx[i * sample_num + j];
                for (int k = 0; k < channels; k++){
                    atomicAdd(&cur_pc_grad[cur_align_out_idx * channels + k], cur_features_grad[i * sample_num * channels + j * channels + k]);
                }
            }
        }
    }
}


void pointsPoolingLauncher(const int proposal_num, const int points_num, const int channels, const int l, const int h, const int w, const int sample_num, const float* pc, const float* proposals, const float* pc_location, int *sampled_num_list, float* out_features, int *out_idx, float* anchors_pillars){
    //std::cout << "beginning forwarding" << std::endl;
    points_pooling_gpu<<<block_num, threadsPerBlock>>>(proposal_num, points_num, channels, l, h, w, sample_num, pc, proposals, pc_location, sampled_num_list, out_features, out_idx, anchors_pillars);
    //std::cout << "Finishing forwarding" << std::endl;
}

void pointsPoolingGradLauncher(const int proposal_num, const int points_num, const int channels, const int l, const int h, const int w, const int sample_num, const float* pc, const int* out_idx, const int *sampled_num_list, const float* features_grad, float *pc_grad){
    //std::cout << "Beginning grad" << std::endl;
    points_pooling_grad_gpu<<<block_num, threadsPerBlock>>>(proposal_num, points_num, channels, l, h, w, sample_num, pc, out_idx, sampled_num_list, features_grad, pc_grad);
    //std::cout << "Finishing grad" << std::endl;
}
