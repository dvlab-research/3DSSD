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

__device__ static float points_fuse_get_data(const float* data, const int h, const int w, const int c, const int height, const int width, const int channels){
    // h, w: current index 
    // height, width: bound of the image
    bool overflow = (h < 0) || (w < 0) || (h >= height) || (w >= width);
    float retVal = overflow ? 0.0f : data[h * width * channels + w * channels + c];
    return retVal;
}

__device__ static float points_fuse_get_coeff(float dh, float dw){
    dw = dw > 0 ? dw : -dw;
    dh = dh > 0 ? dh : -dh;
    return (1.0f - dh) * (1.0f - dw);
}

__device__ static float points_fuse_interpolation(const float* data, const float h, const float w, const int c, const int height, const int width, const int channels){
    float retVal = 0.0f;
    int h1 = floorf(h); 
    int w1 = floorf(w);
    //printf("cur get_data result is: %f, %d, %d, %d, %d\n", points_fuse_get_data(data, h1, w1, c, height, width, channels), h1, w1, height, width);
    retVal += points_fuse_get_data(data, h1, w1, c, height, width, channels) * points_fuse_get_coeff(h - float(h1), w - float(w1));

    h1 = floorf(h) + 1;
    w1 = floorf(w);
    retVal += points_fuse_get_data(data, h1, w1, c, height, width, channels) * points_fuse_get_coeff(h - float(h1), w - float(w1));

    h1 = floorf(h);
    w1 = floorf(w) + 1;
    retVal += points_fuse_get_data(data, h1, w1, c, height, width, channels) * points_fuse_get_coeff(h - float(h1), w - float(w1));

    h1 = floorf(h) + 1;
    w1 = floorf(w) + 1;
    retVal += points_fuse_get_data(data, h1, w1, c, height, width, channels) * points_fuse_get_coeff(h - float(h1), w - float(w1));
    return retVal;
}

__device__ static void points_fuse_add_grad_back(float *img_grad, const float pc_grad, const int h, const int w, const int c, const int height, const int width, const int channels, const float coeff){
    // add the grad * coeff back to img_grad 
    bool overflow = (h < 0) || (w < 0) || (h >= height) || (w >= width);
    if (!overflow){
        float *cur_img_grad = img_grad + h * width * channels + w * channels + c; 
        atomicAdd(cur_img_grad, pc_grad * coeff);
    }
}

__device__ static void points_fuse_grad_interpolation(float* img_grad, const float pc_grad, const float h, const float w, const int c, const int height, const int width, const int channels){
    // add the gradient to the img_feature
    // img_grad: [b, h, w, c]
    int h1 = floorf(h);
    int w1 = floorf(w);
    points_fuse_add_grad_back(img_grad, pc_grad, h1, w1, c, height, width, channels, points_fuse_get_coeff(h - float(h1), w - float(w1)));

    h1 = floorf(h) + 1;
    w1 = floorf(w);
    points_fuse_add_grad_back(img_grad, pc_grad, h1, w1, c, height, width, channels, points_fuse_get_coeff(h - float(h1), w - float(w1)));

    h1 = floorf(h);
    w1 = floorf(w) + 1;
    points_fuse_add_grad_back(img_grad, pc_grad, h1, w1, c, height, width, channels, points_fuse_get_coeff(h - float(h1), w - float(w1)));

    h1 = floorf(h) + 1;
    w1 = floorf(w) + 1;
    points_fuse_add_grad_back(img_grad, pc_grad, h1, w1, c, height, width, channels, points_fuse_get_coeff(h - float(h1), w - float(w1)));
}

// CUDA code for points fusion method
__global__ void points_fuse_gpu(const int b, const int n, const int h, const int w, const int c, const float down_sample_rate, const float* points, const float* img_feature, const float *calib, float* out_2d_locations, float* out_2d_features){
    // points, [batch_size, n, 3], img_features, [b,h, w, c], calib, [b, 3, 4]
    // out_2d_locations, [batch_size, n, 2], out_2d_features, [batch_size, n, c]
    int loop_times = b * n * c;
    CUDA_1D_KERNEL_LOOP(index, loop_times){
        int cur_batch_index = index / (n * c);
        int cur_point_index = (index / c) % n;
        int cur_channel = index % c;

        float *cur_out_2d_locations = out_2d_locations + cur_batch_index * n * 2 + cur_point_index * 2;
        float *cur_out_2d_features = out_2d_features + cur_batch_index * n * c + cur_point_index * c;

        const float* cur_point = points + cur_batch_index * n * 3 + cur_point_index * 3; 
        const float* cur_img_feature = img_feature + cur_batch_index * h * w * c;
        const float* cur_calib = calib + cur_batch_index * 3 * 4;                
   
        float location_2d_x = cur_calib[0] * cur_point[0] + cur_calib[1] * cur_point[1] + cur_calib[2] * cur_point[2] + cur_calib[3] * 1;
        cur_calib += 4;
        float location_2d_y = cur_calib[0] * cur_point[0] + cur_calib[1] * cur_point[1] + cur_calib[2] * cur_point[2] + cur_calib[3] * 1;
        cur_calib += 4;
        float location_2d_z = cur_calib[0] * cur_point[0] + cur_calib[1] * cur_point[1] + cur_calib[2] * cur_point[2] + cur_calib[3] * 1;

        // now we get the true 2d locations
        location_2d_x = location_2d_x / location_2d_z;
        location_2d_y = location_2d_y / location_2d_z;
        
        // get true location in cur_feature_map
        location_2d_x = location_2d_x / down_sample_rate;
        location_2d_y = location_2d_y / down_sample_rate;

        const float cur_point_feature = points_fuse_interpolation(cur_img_feature, location_2d_y, location_2d_x, cur_channel, h, w, c);
        
        cur_out_2d_locations[0] = location_2d_x;
        cur_out_2d_locations[1] = location_2d_y;

        cur_out_2d_features[cur_channel] = cur_point_feature;
    } 
}

// CUDA code for points fusion gradient method
__global__ void points_fuse_grad_gpu(const int b, const int n, const int h, const int w, const int c, const float down_sample_rate, const float* img_feature, const float* img_pc, const float *feature_grad, float* img_grad){
    // img_feature: [b, h, w, c], img_pc: [b, n, 2], feature_grad: [b, n, c]
    // img_grad: [b, h, w, c]
    int loop_times = b * n * c;
    CUDA_1D_KERNEL_LOOP(index, loop_times){
        int cur_batch_index = index / (n * c);
        int cur_point_index = (index / c) % n;
        int cur_channel = index % c;

        const float* cur_img_pc = img_pc + cur_batch_index * n * 2 + cur_point_index * 2;
        const float* cur_feature_grad = feature_grad + index;
        float *cur_img_grad = img_grad + h * w * c * cur_batch_index;

        const float location_2d_x = cur_img_pc[0];
        const float location_2d_y = cur_img_pc[1];

        points_fuse_grad_interpolation(cur_img_grad, cur_feature_grad[0], location_2d_y, location_2d_x, cur_channel, h, w, c); 
    }
}

void pointsFuseLauncher(const int b, const int n, const int h, const int w, const int c, const float down_sample_rate, const float* points, const float* img_feature, const float *calib, float* out_2d_locations, float* out_2d_features){
    points_fuse_gpu<<<block_num, threadsPerBlock>>>(b, n, h, w, c, down_sample_rate, points, img_feature, calib, out_2d_locations, out_2d_features);
}

void pointsFuseGradLauncher(const int b, const int n, const int h, const int w, const int c, const float down_sample_rate, const float* img_feature, const float* img_pc, const float* feature_grad, float* img_grad){
    points_fuse_grad_gpu<<<block_num, threadsPerBlock>>>(b, n, h, w, c, down_sample_rate, img_feature, img_pc, feature_grad, img_grad);
}
