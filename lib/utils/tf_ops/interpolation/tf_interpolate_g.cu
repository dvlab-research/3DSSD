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

__global__ void three_nn_gpu(const int b, const int n, const int m, const float* xyz1, const float* xyz2, float* dist, int* idx){
    // Find three nearest neighbors with square distance, from xyz1 to xyz2
    // input: xyz1: (b, n, 3), xyz2: (b, m, 3)
    // output: dist: (b, n, 3), idx: (b, n, 3)
    int total_idx = b * n;
    CUDA_1D_KERNEL_LOOP(point_inds, total_idx){
        int cur_batch_idx = point_inds / n;

        const float* cur_xyz1 = xyz1 + point_inds * 3;
        const float cur_xyz1_x = cur_xyz1[0];
        const float cur_xyz1_y = cur_xyz1[1];
        const float cur_xyz1_z = cur_xyz1[2]; 

        float cur_xyz2_x, cur_xyz2_y, cur_xyz2_z;
        const float* cur_xyz2 = xyz2 + cur_batch_idx * m * 3;

        float* cur_dist = dist + point_inds * 3;
        int* cur_idx = idx + point_inds * 3;

        double best1 = 1e40;
        double best2 = 1e40;
        double best3 = 1e40;
        double d;
        int besti1 = 0;
        int besti2 = 0;
        int besti3 = 0;
        for (int i = 0; i < m; i++){
            // compare the distance to each xyz2 points
            cur_xyz2_x = cur_xyz2[i * 3 + 0];
            cur_xyz2_y = cur_xyz2[i * 3 + 1];
            cur_xyz2_z = cur_xyz2[i * 3 + 2]; 

            d = (cur_xyz2_x - cur_xyz1_x) * (cur_xyz2_x - cur_xyz1_x) + (cur_xyz2_y - cur_xyz1_y) * (cur_xyz2_y - cur_xyz1_y) + (cur_xyz2_z - cur_xyz1_z) * (cur_xyz2_z - cur_xyz1_z);
            if (d < best1){
                best3=best2;
                besti3=besti2;
                best2=best1;
                besti2=besti1;
                best1=d;
                besti1=i;
            }
            else if (d < best2){
                best3=best2;
                besti3=besti2;
                best2=d;
                besti2=i;
            }
            else if (d < best3){
                best3=d;
                besti3=i;
            }
        }
        cur_dist[0] = best1; 
        cur_dist[1] = best2;
        cur_dist[2] = best3;

        cur_idx[0] = besti1;
        cur_idx[1] = besti2;
        cur_idx[2] = besti3;
    }
}


__global__ void three_interpolate_gpu(const int b, const int m, const int c, const int n, const float* points, const int* idx, const float* weight, float* out){
    // input: points: (b, m, c), idx: (b, n, 3), weight: (b, n, 3)
    // out: (b, n, c) 
    int total_idx = b * n * c;
    CUDA_1D_KERNEL_LOOP(point_inds, total_idx){
        int cur_batch_inds = point_inds / (n * c);
        int cur_point_inds = point_inds / c;
        int cur_channel_inds = point_inds % c;

        const float* cur_points = points + cur_batch_inds * m * c;
        const int* cur_idx = idx + cur_point_inds * 3; 
        const float* cur_weight = weight + cur_point_inds * 3;

        float w1 = cur_weight[0];
        float w2 = cur_weight[1];
        float w3 = cur_weight[2];
        int i1 = cur_idx[0];
        int i2 = cur_idx[1];
        int i3 = cur_idx[2];

        float c1 = cur_points[i1 * c + cur_channel_inds];
        float c2 = cur_points[i2 * c + cur_channel_inds];
        float c3 = cur_points[i3 * c + cur_channel_inds];

        out[point_inds] = c1 * w1 + c2 * w2 + c3 * w3;
    }
}

__global__ void three_interpolate_grad_gpu(const int b, const int n, const int c, const int m, const float* grad_out, const int* idx, const float* weight, float* grad_points){
    // input: grad_out: [b, n, c] idx [b, n, 3], weight [b, n, 3]
    // output: grad_points [b, m, c]
    int total_idx = b * n * c;
    CUDA_1D_KERNEL_LOOP(points_inds, total_idx){
        int cur_batch_inds = points_inds / (n * c);
        int cur_points_inds = points_inds / c;
        int cur_channel_inds = points_inds % c;

        float* cur_grad_points = grad_points + cur_batch_inds * m * c;
        const float* cur_grad_out = grad_out + points_inds;
        const int* cur_idx = idx + cur_points_inds * 3;
        const float* cur_weight = weight + cur_points_inds * 3;
   
        float w1 = cur_weight[0];
        float w2 = cur_weight[1];
        float w3 = cur_weight[2];
        int i1 = cur_idx[0];
        int i2 = cur_idx[1];
        int i3 = cur_idx[2];

        atomicAdd(&cur_grad_points[i1 * c + cur_channel_inds], cur_grad_out[0] * w1);
        atomicAdd(&cur_grad_points[i2 * c + cur_channel_inds], cur_grad_out[0] * w2);
        atomicAdd(&cur_grad_points[i3 * c + cur_channel_inds], cur_grad_out[0] * w3);
    }
}

__global__ void k_interpolate_gpu(const int b, const int m, const int c, const int n, const int k, const float* points, const int* idx, const float* weight, float* out){
    // input: points: (b, m, c), idx: (b, n, k), weight: (b, n, k)
    // out: (b, n, c) 
    int total_idx = b * n * c;
    CUDA_1D_KERNEL_LOOP(point_inds, total_idx){
        int cur_batch_inds = point_inds / (n * c);
        int cur_point_inds = point_inds / c;
        int cur_channel_inds = point_inds % c;

        const float* cur_points = points + cur_batch_inds * m * c;
        const int* cur_idx = idx + cur_point_inds * k; 
        const float* cur_weight = weight + cur_point_inds * k;

        float w, ci;
        int index;
        out[point_inds] = 0;
        for (int i=0; i < k; i++){
            index = cur_idx[i]; 
            w = cur_weight[i];
            ci = cur_points[index * c + cur_channel_inds];
            out[point_inds] += w * ci;
        }
    }
}

__global__ void k_interpolate_grad_gpu(const int b, const int n, const int c, const int m, const int k, const float* grad_out, const int* idx, const float* weight, float* grad_points){
    // input: grad_out: [b, n, c] idx [b, n, k], weight [b, n, k]
    // output: grad_points [b, m, c]
    int total_idx = b * n * c;
    CUDA_1D_KERNEL_LOOP(points_inds, total_idx){
        int cur_batch_inds = points_inds / (n * c);
        int cur_points_inds = points_inds / c;
        int cur_channel_inds = points_inds % c;

        float* cur_grad_points = grad_points + cur_batch_inds * m * c;
        const float* cur_grad_out = grad_out + points_inds;
        const int* cur_idx = idx + cur_points_inds * k;
        const float* cur_weight = weight + cur_points_inds * k;
   
        float w;
        int index;
        for (int i=0; i<k; i++){
            w = cur_weight[i];
            index = cur_idx[i];
            atomicAdd(&cur_grad_points[index * c + cur_channel_inds], cur_grad_out[0] * w);
        }
    }
}

void ThreeNNLauncher(const int b, const int n, const int m, const float* xyz1, const float* xyz2, float* dist, int* idx){
    //std::cout << "beginning forwarding" << std::endl;
    three_nn_gpu<<<block_num, threadsPerBlock>>>(b, n, m, xyz1, xyz2, dist, idx);
    //std::cout << "Finishing forwarding" << std::endl;
}


void ThreeInterpolateLauncher(const int b, const int m, const int c, const int n, const float* points, const int* idx, const float* weight, float* out){
    three_interpolate_gpu<<<block_num, threadsPerBlock>>>(b, m, c, n, points, idx, weight, out);
}

void ThreeInterpolateGradLauncher(const int b, const int n, const int c, const int m, const float* grad_out, const int* idx, const float* weight, float* grad_points){
    // grad_out: [b, n, c]
    // idx: [b, n, 3], weight: [b. n, 3], grad_points: [b, m, c]
    three_interpolate_grad_gpu<<<block_num, threadsPerBlock>>>(b, n, c, m, grad_out, idx, weight, grad_points);
}

void KInterpolateLauncher(const int b, const int m, const int c, const int n, const int k, const float* points, const int* idx, const float* weight, float* out){
    k_interpolate_gpu<<<block_num, threadsPerBlock>>>(b, m, c, n, k, points, idx, weight, out);
}

void KInterpolateGradLauncher(const int b, const int n, const int c, const int m, const int k, const float* grad_out, const int* idx, const float* weight, float* grad_points){
    // grad_out: [b, n, c]
    // idx: [b, n, 3], weight: [b. n, 3], grad_points: [b, m, c]
    k_interpolate_grad_gpu<<<block_num, threadsPerBlock>>>(b, n, c, m, k, grad_out, idx, weight, grad_points);
}
