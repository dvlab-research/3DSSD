// input: iou_matrix, [n, n], points_sampling, [n, npoint], merge function, 0:union, 1: intersection
// min_keep_num
// output: keep_inds [n, n], 0/1 
// nmsed_points_sample: [n, npoint], 0/1
#include <stdio.h> 
#include <iostream>
#include <vector>
#include <time.h>

#define CUDA_CHECK(condition) \
  /* Code block avoids redefinition of cudaError_t error */ \
  do { \
    cudaError_t error = condition; \
    if (error != cudaSuccess) { \
      std::cout << cudaGetErrorString(error) << std::endl; \
    } \
  } while (0)

const int block_num = 512;
#define DIVUP(m,n) ((m) / (n) + ((m) % (n) > 0))
const int threadsPerBlock = sizeof(unsigned long long) * 8;

__global__ void points_inside_boxes(const int n, const int npoint, const float *points, const float* anchors, int* points_sample_mask){
    // n: boxes_num, npoint: points_num, points: points_num x 3, anchors: boxes_num, 6
    // return: points_sample_mask: boxes_num x npoint
    for (int batch_idx=blockIdx.x; batch_idx < n; batch_idx += gridDim.x){
        // xmin, ymin, zmin, xmax, ymax, zmax
        const float* cur_anchors = anchors + batch_idx * 6;
        int *cur_points_sample_mask = points_sample_mask + batch_idx * npoint;

        int x_index = threadIdx.x;
        int x_stride = blockDim.x;

        const float cur_anchors_xmin = cur_anchors[0] - cur_anchors[3] / 2.;
        const float cur_anchors_ymin = cur_anchors[1] - cur_anchors[4];
        const float cur_anchors_zmin = cur_anchors[2] - cur_anchors[5] / 2.;
        const float cur_anchors_xmax = cur_anchors[0] + cur_anchors[3] / 2.;
        const float cur_anchors_ymax = cur_anchors[1];
        const float cur_anchors_zmax = cur_anchors[2] + cur_anchors[5] / 2.;

        for (int points_idx = x_index; points_idx < npoint; points_idx += x_stride){
            const float* cur_points = points + points_idx * 3;

            const float cur_points_x = cur_points[0];
            const float cur_points_y = cur_points[1];
            const float cur_points_z = cur_points[2];

            int _x = (cur_points_x >= cur_anchors_xmin) * (cur_points_x <= cur_anchors_xmax);
            int _y = (cur_points_y >= cur_anchors_ymin) * (cur_points_y <= cur_anchors_ymax);
            int _z = (cur_points_z >= cur_anchors_zmin) * (cur_points_z <= cur_anchors_zmax);
            cur_points_sample_mask[points_idx] = _x * _y * _z;
        }
    }
}


__global__ void points_iou_kernel(const int n, const int npoint, const int* points_sample_mask, float* iou_matrix){
  // points_sample_mask, [n, npoint], 0/1
  // iou_matrix, [n, n]
  for (int x_num_idx=blockIdx.x; x_num_idx<n; x_num_idx+=gridDim.x){
    for(int y_num_idx=blockIdx.y; y_num_idx<n; y_num_idx+=gridDim.y){
      const int* x_points_sample_mask = points_sample_mask + x_num_idx * npoint;
      const int* y_points_sample_mask = points_sample_mask + y_num_idx * npoint; 
      
      int x_index = threadIdx.x;
      int x_stride = blockDim.x;
  
      __shared__ float intersect_list[threadsPerBlock]; 
      __shared__ float union_list[threadsPerBlock];
      // first initialize intersect_list and union_list by zero
      intersect_list[x_index] = 0;
      union_list[x_index] = 0;
      __syncthreads();

      for(int i_x=x_index; i_x<npoint; i_x+= x_stride){

        intersect_list[x_index] = intersect_list[x_index] + float(x_points_sample_mask[i_x] && y_points_sample_mask[i_x]);
        union_list[x_index] = union_list[x_index] + float(x_points_sample_mask[i_x] || y_points_sample_mask[i_x]);

      }
      __syncthreads();
      // after calc the intersect
      // then get the sum
      __shared__ float intersect_sum;
      __shared__ float union_sum;
      intersect_sum = 0;
      union_sum = 0;
      __syncthreads();

      atomicAdd(&intersect_sum, intersect_list[x_index]);
      atomicAdd(&union_sum, union_list[x_index]);
      __syncthreads();

      float iou = intersect_sum / max(union_sum, 1.);
      iou_matrix[x_num_idx * n + y_num_idx] = iou;
    }
  }
}


__device__ inline float devIou(const int *a, const int *b, int npoint) {
    // a:[npoint], b[npoint], then calc the iou
    float intersect = 0;
    float union_sect = 0;

    for (int i = 0; i < npoint; i ++){
        intersect += a[i] && b[i];
        union_sect += a[i] || b[i];
    }
    return intersect / union_sect;
}

__global__ void points_nms_block_kernel(const int n, const int npoint, const int merge_function, const float iou_thresh, const int*points_sample, unsigned long long *keep_inds, int *nmsed_points_sample){
  const int row_start = blockIdx.y;
  const int col_start = blockIdx.x;

  const int row_size = min(n - row_start * threadsPerBlock, threadsPerBlock);
  const int col_size = min(n - col_start * threadsPerBlock, threadsPerBlock);
  const int* col_points_sample = points_sample + (threadsPerBlock * col_start) * npoint;

  if (threadIdx.x < row_size){
    const int cur_box_idx = threadsPerBlock * row_start + threadIdx.x;
    const int *cur_points_sample = points_sample + cur_box_idx * npoint;
    int *cur_nmsed_points_sample = nmsed_points_sample + cur_box_idx * npoint;
    int i = 0;
    unsigned long long t = 0;
    int start = 0;
    if (row_start == col_start){
      start = threadIdx.x + 1;
    }  
    for (i = start; i < col_size; i ++){
      if (devIou(cur_points_sample, col_points_sample + i * npoint, npoint) > iou_thresh) {
        // merge the nmsed_points_sample
        const int *merged_col_points_sample = col_points_sample + i * npoint;
        if (merge_function == 0){
          for (int j = 0; j < npoint; j++){
            atomicOr(&cur_nmsed_points_sample[j], merged_col_points_sample[j]);        
          }
        }
        else if (merge_function == 1){
          for (int j = 0; j < npoint; j++){
            atomicAnd(&cur_nmsed_points_sample[j], merged_col_points_sample[j]);
          }
        }
        t |= 1ULL << i;
      }
    }
    const int col_blocks = DIVUP(n, threadsPerBlock);
    // keep_inds, [col_blocks, threadsPerBlock]
    keep_inds[cur_box_idx * col_blocks + col_start] = t;
  }
}


__global__ void points_nms_kernel(const int n, const int npoint, const int merge_function, float iou_thresh, const float *iou_matrix, const int *points_sample, int *keep_inds, int *nmsed_points_sample) {
  // nmsed_points_sample [n, npoint]
  for (int x_num_idx=blockIdx.x; x_num_idx<n; x_num_idx+=gridDim.x){
    for(int y_num_idx=blockIdx.y; y_num_idx<n; y_num_idx+=gridDim.y){
      if (x_num_idx == y_num_idx)
        continue;
      // const int* x_points_sample = points_sample + x_num_idx * npoint;
      const int* y_points_sample = points_sample + y_num_idx * npoint;

      const float* x_iou_matrix = iou_matrix + x_num_idx * n;
      int *x_keep_inds = keep_inds + x_num_idx * n;
      int* x_nmsed_points_sample = nmsed_points_sample + x_num_idx * npoint; 

      int index = threadIdx.x;
      int stride = blockDim.x;

      
      float cur_iou = x_iou_matrix[y_num_idx];
      if (cur_iou > iou_thresh){
        // merge them togethor
        x_keep_inds[y_num_idx] = 1;
        for (int i=index;i<npoint;i+=stride){
          // merge the result
          if (merge_function == 0){
            // union the two vector
            atomicOr(&x_nmsed_points_sample[i], y_points_sample[i]);
          }
          else if(merge_function == 1){
            atomicAnd(&x_nmsed_points_sample[i], y_points_sample[i]);
          }
          else{
            continue;
          }
        }
      }
    }
  }
}

__global__ void points_nms_sample(const int n, const int npoint, int merge_function, int* nmsed_points_sample_media, int* nmsed_points_sample){
    for (int num_idx=blockIdx.x; num_idx<n; num_idx+=gridDim.x){
        int *batch_nmsed_points_sample_media = nmsed_points_sample_media + num_idx * n *npoint;
        int *batch_nmsed_points_sample = nmsed_points_sample + num_idx * npoint;

        int index = threadIdx.x;
        int stride = blockDim.x;

        for (int i=index; i<n; i+=stride){
            for(int j=0; j < npoint; j++){
                if (merge_function == 0 || merge_function == 2){
                    // union or keep the origin
                    atomicOr(&batch_nmsed_points_sample[j], batch_nmsed_points_sample_media[i * npoint + j]);
                    // batch_nmsed_points_sample[j] = batch_nmsed_points_sample[j] + batch_nmsed_points_sample_media[i * npoint + j];
                }
                else if (merge_function == 1){
                    atomicAnd(&batch_nmsed_points_sample[j], batch_nmsed_points_sample_media[i * npoint + j]);
                    // batch_nmsed_points_sample[j] = batch_nmsed_points_sample[j] && batch_nmsed_points_sample_media[i * npoint + j];
                }
            }
        }
    }
}

void points_iou_gpu(const int n, const int npoint, const int* points_sample_mask, float* iou_matrix){
    dim3 blocks(512, 512);
    points_iou_kernel<<<blocks, threadsPerBlock>>>(n, npoint, points_sample_mask, iou_matrix);
    // std::cout << "Iou Caluculating Done!!" << std::endl;
}

void points_inside_boxes_gpu(const int n, const int npoint, const float *points, const float* anchors, int* points_sample_mask){
    CUDA_CHECK(cudaMemset(points_sample_mask, 1, n * npoint * sizeof(int)));
    points_inside_boxes<<<512, threadsPerBlock>>>(n, npoint, points, anchors, points_sample_mask);
}


void points_nms_block_gpu(const int n, const int npoint, const int merge_function, const float iou_thresh, const int num_to_keep, const int *points_sample, int *keep_inds, int *nmsed_points_sample){
  unsigned long long* mask_dev = NULL;
  const int col_blocks = DIVUP(n, threadsPerBlock);

  CUDA_CHECK(cudaMalloc(&mask_dev, n * col_blocks * sizeof(unsigned long long))); 

  CUDA_CHECK(cudaMemcpy(nmsed_points_sample, points_sample, sizeof(int) * n * npoint, cudaMemcpyDeviceToDevice));

  time_t c_start, c_end;
  c_start = clock();
  cudaEvent_t start, stop;   // variables that holds 2 events 
  float time;                // Variable that will hold the time
  cudaEventCreate(&start);   // creating the event 1
  cudaEventCreate(&stop);    // creating the event 2
  cudaEventRecord(start, 0); // start measuring  the time

  dim3 blocks(DIVUP(n, threadsPerBlock),
              DIVUP(n, threadsPerBlock));
  dim3 threads(threadsPerBlock);
  points_nms_block_kernel<<<blocks, threads>>>(n, npoint, merge_function, iou_thresh, points_sample, mask_dev, nmsed_points_sample);

  c_end = clock();
  cudaEventRecord(stop, 0);                  // Stop time measuring
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&time, start, stop);
  std::cout << difftime(c_end,c_start) << std::endl;
  std::cout << time << std::endl;
  std::cout << "Finished main working !!!" << std::endl;

  c_start = clock();
  std::vector<unsigned long long> mask_host(n * col_blocks);
  cudaMemcpy(&mask_host[0],
                        mask_dev,
                        sizeof(unsigned long long) * n * col_blocks,
                        cudaMemcpyDeviceToHost);
  c_end = clock();
  std::cout << difftime(c_end,c_start) << std::endl;
  std::cout << "Finished copying" << std::endl;

  c_start = clock();
  std::vector<unsigned long long> remv(col_blocks);
  memset(&remv[0], 0, sizeof(unsigned long long) * col_blocks);

  std::vector<int> cpu_keep_inds(n);
  memset(&cpu_keep_inds[0], -1, sizeof(int) * num_to_keep);
  std::cout << "setting the output to -1" << std::endl;

  int keeping_num = 0;
  for (int i=0; i < n; i ++){
    int nblock = i / threadsPerBlock;
    int inblock = i % threadsPerBlock;
  
    if (!(remv[nblock] & (1ULL << inblock))){
      cpu_keep_inds[keeping_num++] = i;
      if (keeping_num >= num_to_keep)
          break;
      unsigned long long *p = &mask_host[0] + i * col_blocks; 
      for (int j = nblock; j < col_blocks; j ++){
        remv[j] |= p[j];
      }
    } 
  }
  c_end = clock();
  std::cout << difftime(c_end,c_start) << std::endl;
  CUDA_CHECK(cudaFree(mask_dev));
  CUDA_CHECK(cudaMemcpy(keep_inds, &cpu_keep_inds[0], sizeof(int) * num_to_keep, cudaMemcpyHostToDevice));
  std::cout << "Finished!!!" << std::endl;
}



void points_nms_gpu(const int n, const int npoint, const int merge_function, float iou_thresh, const float *iou_matrix, const int *points_sample, int *keep_inds, int *nmsed_points_sample) {
    // std::cout << "Beginning points nms !!!" << std::endl;
    int *remove_inds = NULL;
    CUDA_CHECK(cudaMalloc(&remove_inds, n * n * sizeof(int))); 
    CUDA_CHECK(cudaMemset(remove_inds, 0, n * n * sizeof(int)));

    std::vector<int> cpu_keep_inds(n, 1);

    // First initialize the nmsed_points_sample by the points_sample
    CUDA_CHECK(cudaMemcpy(nmsed_points_sample, points_sample, sizeof(int) * n * npoint, cudaMemcpyDeviceToDevice));

    dim3 blocks(block_num, block_num);
    points_nms_kernel<<<blocks, threadsPerBlock>>>(n, npoint, merge_function, iou_thresh, iou_matrix, points_sample, remove_inds, nmsed_points_sample);

    // Using for Debug
    // std::vector<int> debug(n * npoint);
    // CUDA_CHECK(cudaMemcpy(&debug[0], media_nmsed_points_sample, sizeof(int) * n * npoint, cudaMemcpyDeviceToHost));
    // for (int i=0; i<n; i++){
    //     for (int j=0; j< npoint; j++)
    //         std::cout << debug[i * npoint + j] << " ";
    //     std::cout << std::endl;
    // }
    // std::cout << std::endl;

    std::vector<int> cpu_remove_inds(n * n);
    CUDA_CHECK(cudaMemcpy(&cpu_remove_inds[0], remove_inds, sizeof(int) * n * n, cudaMemcpyDeviceToHost));
    // std::cout << "points nms_remove inds Done !!!" << std::endl;

    // finally get the keep_inds
    for (int i=0; i<n; i++){
        // std::cout << 1 << std::endl;
        if (cpu_keep_inds[i] == 0){
            continue;
        }
        
        for(int j=i+1; j<n; j++){
            if (cpu_remove_inds[i * n + j] == 1){
                // remove this point
                cpu_keep_inds[j] = 0;
            }
        }
    }
    // at last, make it back
    CUDA_CHECK(cudaMemcpy(keep_inds, &cpu_keep_inds[0], sizeof(int) * n, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaFree(remove_inds));
    // std::cout << "points nms Done !!!" << std::endl;
}
