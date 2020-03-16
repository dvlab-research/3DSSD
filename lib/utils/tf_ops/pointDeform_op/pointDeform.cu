#include <stdio.h>

__global__ void selection_k_radius_gpu(int b, int m, int k, float radius, const int* idx, const float* val, int* idx_out, float* val_out){
    int batch_index = blockIdx.x;
    int stride = batch_index * m * k;
    idx += stride;
    val += stride;
    idx_out += stride;
    val_out += stride;
    for(int i = threadIdx.x; i < m;i += blockDim.x) {

        for(int j = 0;j < k;j ++) {
            if(val[i * k + j] < radius) {
                idx_out[i * k + j] = idx[i * k + j];
                val_out[i * k + j] = val[i * k + j];
            } else {
                idx_out[i * k + j] = idx[i * k ];
                val_out[i * k + j] = val[i * k ];
            }
        }
    }
}

__global__ void nearest_select(int b, int n,float radius, const float* xyz, int* idx_out) {
    // b: batch_size
    // n: num_points, radius: search range in an octant
    // xyz: [b, n, 3], idx_out: [b, n, num]
    int batch_idx = blockIdx.x;
    xyz += batch_idx * n * 3;
    const int num = 8;
    idx_out += batch_idx * n * num;
    float temp_dist[num];
    float judge_dist = radius * radius;
    int insert_index = 1e8;
    for(int i = threadIdx.x; i < n;i += blockDim.x) {
        // for x in n
        // first get the current points
        float x = xyz[i * 3];
        float y = xyz[i * 3 + 1];
        float z = xyz[i * 3 + 2];
        for(int j = 0;j < num;j ++) {
            temp_dist[j] = 1e8;
            idx_out[i * num + j] = i; // if not found, just return itself..
        }
        for(int j = 0;j < n;j ++) {
            if(i == j) continue;
            float tx = xyz[j * 3];
            float ty = xyz[j * 3 + 1];
            float tz = xyz[j * 3 + 2];
            float dist = (x - tx) * (x - tx) + (y - ty) * (y - ty) + (z - tz) * (z - tz);
            if(dist > judge_dist) continue;
            // now add this dist to the nearest_select
            // from 0 to num, insert sort
            for (int k = 0; k < num; k ++){
                if (dist < temp_dist[k]){
                    // if < current dist
                    insert_index = k;
                    break;
                }
            }
            // then we insert that into the temp_dist or idx_out
            for (int k = num - 1; k > insert_index; k --){
                temp_dist[k] = temp_dist[k - 1];
                idx_out[i * num + k] = idx_out[i * num + k - 1];
            }
            if (insert_index < num)
                temp_dist[insert_index] = dist;
                idx_out[i * num + insert_index] = j;
        }
        // after n range
        // we just where index == i to the front of the list
        for(int j = 0; j < num; j ++){
            if (idx_out[i * num + j] == i){
                // if find the i, then we add it to the front
                // for (int k = num - j; k < num; k ++){
                for (int k = num - 1; k > num - j - 1; k--){
                    idx_out[i * num + k] = idx_out[i * num + k - num + j];
                }
                for (int k = 0; k < num - j; k ++){
                    idx_out[i * num + k] = i;
                }
                break;
            }
        }
    }
}



void selectionKRadiusLauncher(int b, int m, int k, float radius, const int* idx, const float* val, int* idx_out, float* val_out){
    selection_k_radius_gpu<<<b,256>>>(b, m, k, radius, idx, val, idx_out, val_out);
}
void nearestSelectLauncher(int b, int n, float radius, const float* xyz, int* idx_out) {
    // find the nearest num points in the range radius, if not enough, then just choose itself
    nearest_select<<<b, 512>>>(b, n, radius, xyz, idx_out);
}
