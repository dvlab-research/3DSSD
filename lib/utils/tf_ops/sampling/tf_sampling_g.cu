#include <stdio.h>
#include <iostream>
#include <vector>


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


__global__ void cumsumKernel(int b,int n,const float * __restrict__ inp,float * __restrict__ out){
  const int BlockSize=2048;
  const int paddingLevel=5;
  __shared__ float buffer4[BlockSize*4];
  __shared__ float buffer[BlockSize+(BlockSize>>paddingLevel)];
  for (int i=blockIdx.x;i<b;i+=gridDim.x){
    float runningsum=0,runningsum2=0;
    for (int j=0;j<n;j+=BlockSize*4){
      int n24_i=min(n-j,BlockSize*4);
      int n24=(n24_i+3)&~3;
      int n2=n24>>2;
      for (int k=threadIdx.x*4;k<n24_i;k+=blockDim.x*4){
        if (k+3<n24_i){
          float v1=inp[i*n+j+k];
          float v2=inp[i*n+j+k+1];
          v2+=v1;
          float v3=inp[i*n+j+k+2];
          float v4=inp[i*n+j+k+3];
          v4+=v3;
          v3+=v2;
          v4+=v2;
          buffer4[k]=v1;
          buffer4[k+1]=v2;
          buffer4[k+2]=v3;
          buffer4[k+3]=v4;
          buffer[(k>>2)+(k>>(2+paddingLevel))]=v4;
        }else{
          float v=0;
          for (int k2=k;k2<n24_i;k2++){
            v+=inp[i*n+j+k2];
            buffer4[k2]=v;
          }
          for (int k2=n24_i;k2<n24;k2++){
            buffer4[k2]=v;
          }
          buffer[(k>>2)+(k>>(2+paddingLevel))]=v;
        }
      }
      int u=0;
      for (;(2<<u)<=n2;u++){
        __syncthreads();
        for (int k=threadIdx.x;k<int(n2>>(u+1));k+=blockDim.x){
          int i1=(((k<<1)+2)<<u)-1;
          int i2=(((k<<1)+1)<<u)-1;
          i1+=i1>>paddingLevel;
          i2+=i2>>paddingLevel;
          buffer[i1]+=buffer[i2];
        }
      }
      u--;
      for (;u>=0;u--){
        __syncthreads();
        for (int k=threadIdx.x;k<int((n2-(1<<u))>>(u+1));k+=blockDim.x){
          int i1=(((k<<1)+3)<<u)-1;
          int i2=(((k<<1)+2)<<u)-1;
          i1+=i1>>paddingLevel;
          i2+=i2>>paddingLevel;
          buffer[i1]+=buffer[i2];
        }
      }
      __syncthreads();
      for (int k=threadIdx.x*4;k<n24;k+=blockDim.x*4){
        if (k!=0){
          int k2=((k>>2)-1)+(((k>>2)-1)>>paddingLevel);
          buffer4[k]+=buffer[k2];
          buffer4[k+1]+=buffer[k2];
          buffer4[k+2]+=buffer[k2];
          buffer4[k+3]+=buffer[k2];
        }
      }
      __syncthreads();
      for (int k=threadIdx.x;k<n24_i;k+=blockDim.x){
        out[i*n+j+k]=buffer4[k]+runningsum;
      }
      float t=buffer[(n2-1)+((n2-1)>>paddingLevel)]+runningsum2;
      float r2=runningsum+t;
      runningsum2=t-(r2-runningsum);
      runningsum=r2;
      __syncthreads();
    }
  }
}

__global__ void binarysearchKernel(int b,int n,int m,const float * __restrict__ dataset,const float * __restrict__ query, int * __restrict__ result){
  int base=1;
  while (base<n)
    base<<=1;
  for (int i=blockIdx.x;i<b;i+=gridDim.x){
    for (int j=blockIdx.y*blockDim.x+threadIdx.x;j<m;j+=blockDim.x*gridDim.y){
      float q=query[i*m+j]*dataset[i*n+n-1];
      int r=n-1;
      for (int k=base;k>=1;k>>=1)
        if (r>=k && dataset[i*n+r-k]>=q)
          r-=k;
      result[i*m+j]=r;
    }
  }
}

template <unsigned int BlockSize>
__global__ void farthestpointsamplingKernel(int b,int n,int c,int m,const float * __restrict__ dataset,float * __restrict__ temp,int * __restrict__ idxs){
  if (m<=0)
    return;
  // const int BlockSize=512;
  __shared__ float dists[BlockSize];
  __shared__ int dists_i[BlockSize];
  for (int i=blockIdx.x;i<b;i+=gridDim.x){
    int old=0;
    if (threadIdx.x==0)
      idxs[i*m+0]=old;
    //initialize temp
    for (int j=threadIdx.x;j<n;j+=blockDim.x){
      temp[blockIdx.x*n+j]=1e38;
    }
    __syncthreads();
    for (int j=1;j<m;j++){
      int besti=0;
      float best=-1;
      for (int k=threadIdx.x;k<n;k+=blockDim.x){
        float td=temp[blockIdx.x*n+k];
        float d = 0;
        float p1, p2;
        for (int l=0;l<c;l++){
          p1 = dataset[i*n*c+old*c+l];
          p2 = dataset[i*n*c+k*c+l];
          d += (p2-p1) * (p2-p1);
        }
        float d2=min(d,td);
        if (d2!=td)
          temp[blockIdx.x*n+k]=d2;
        if (d2>best){
          best=d2;
          besti=k;
        }
      }
      dists[threadIdx.x]=best;
      dists_i[threadIdx.x]=besti;
      for (int u=0;(1<<u)<blockDim.x;u++){
        __syncthreads();
        if (threadIdx.x<(blockDim.x>>(u+1))){
          int i1=(threadIdx.x*2)<<u;
          int i2=(threadIdx.x*2+1)<<u;
          if (dists[i1]<dists[i2]){
            dists[i1]=dists[i2];
            dists_i[i1]=dists_i[i2];
          }
        }
      }
      __syncthreads();
      old=dists_i[0];
      if (threadIdx.x==0)
        idxs[i*m+j]=old;
    }
  }
}

template <unsigned int BlockSize>
__global__ void farthestpointsamplingwithdistKernel(int b,int n,int m,const float * __restrict__ dataset,float * __restrict__ temp,int * __restrict__ idxs){
  if (m<=0)
    return;
  // const int BlockSize=512;
  __shared__ float dists[BlockSize];
  __shared__ int dists_i[BlockSize];
  for (int i=blockIdx.x;i<b;i+=gridDim.x){
    int old=0;
    if (threadIdx.x==0)
      idxs[i*m+0]=old;
    //initialize temp
    for (int j=threadIdx.x;j<n;j+=blockDim.x){
      temp[blockIdx.x*n+j]=1e38;
    }
    __syncthreads();
    for (int j=1;j<m;j++){
      int besti=0;
      float best=-1;
      for (int k=threadIdx.x;k<n;k+=blockDim.x){
        float td=temp[blockIdx.x*n+k];
        float d = 0;
        d = dataset[i * n * n + old * n + k];
        float d2=min(d,td);
        if (d2!=td)
          temp[blockIdx.x*n+k]=d2;
        if (d2>best){
          best=d2;
          besti=k;
        }
      }
      dists[threadIdx.x]=best;
      dists_i[threadIdx.x]=besti;
      for (int u=0;(1<<u)<blockDim.x;u++){
        __syncthreads();
        if (threadIdx.x<(blockDim.x>>(u+1))){
          int i1=(threadIdx.x*2)<<u;
          int i2=(threadIdx.x*2+1)<<u;
          if (dists[i1]<dists[i2]){
            dists[i1]=dists[i2];
            dists_i[i1]=dists_i[i2];
          }
        }
      }
      __syncthreads();
      old=dists_i[0];
      if (threadIdx.x==0)
        idxs[i*m+j]=old;
    }
  }
}


template <unsigned int BlockSize>
__global__ void farthestpointsamplingwithpreidxKernel(int b,int n,int c,int m,int m1,const float * __restrict__ dataset,const int * __restrict__ preidx,float * __restrict__ temp,int * __restrict__ idxs){
  // b: batch_size, n: ndataset, c: channel_num, m: points_num after fps, m1: preidx number
  // dataset: [b, n, c] preidx: [b, m1], temp: [b, n], idxs: [b, m] 
  if (m<=0)
    return;
  // const int BlockSize=512;
  __shared__ float dists[BlockSize];
  __shared__ int dists_i[BlockSize];
  for (int i=blockIdx.x;i<b;i+=gridDim.x){
    for (int j=threadIdx.x;j<n;j+=blockDim.x){
      temp[blockIdx.x*n+j]=1e38;
    }
    int pre_idx;
    for (int j=threadIdx.x;j<n;j+=blockDim.x){
      // update temp metrics
      float pre_best = 1e38;
      float pre_p1, pre_p2;
      for (int k=0; k<m1; k++){
        pre_idx = preidx[i * m1 + k];
        float pre_d = 0;
        for (int l=0; l < c; l++){
          pre_p1 = dataset[i * n * c + pre_idx * c + l];
          pre_p2 = dataset[i * n * c + j * c + l];
          pre_d += (pre_p2 - pre_p1) * (pre_p2 - pre_p1);
        } 
        pre_best = min(pre_best, pre_d);
      }
      temp[blockIdx.x*n+j] = pre_best;
    }
    // then find current smallest distance as current old
    __syncthreads();
    int old=0;
    float pre_best = -1; 
    for (int j=0; j<n; j++){
      if (pre_best < temp[blockIdx.x*n+j]){
        pre_best = temp[blockIdx.x*n+j];
        old = j;
      } 
    }
    if (threadIdx.x==0)
      idxs[i*m+0]=old;
    //initialize temp
    __syncthreads();
    for (int j=1;j<m;j++){
      int besti=0;
      float best=-1;
      for (int k=threadIdx.x;k<n;k+=blockDim.x){
        float td=temp[blockIdx.x*n+k];
        float d = 0;
        float p1, p2;
        for (int l=0;l<c;l++){
          p1 = dataset[i*n*c+old*c+l];
          p2 = dataset[i*n*c+k*c+l];
          d += (p2-p1) * (p2-p1);
        }
        float d2=min(d,td);
        if (d2!=td)
          temp[blockIdx.x*n+k]=d2;
        if (d2>best){
          best=d2;
          besti=k;
        }
      }
      dists[threadIdx.x]=best;
      dists_i[threadIdx.x]=besti;
      for (int u=0;(1<<u)<blockDim.x;u++){
        __syncthreads();
        if (threadIdx.x<(blockDim.x>>(u+1))){
          int i1=(threadIdx.x*2)<<u;
          int i2=(threadIdx.x*2+1)<<u;
          if (dists[i1]<dists[i2]){
            dists[i1]=dists[i2];
            dists_i[i1]=dists_i[i2];
          }
        }
      }
      __syncthreads();
      old=dists_i[0];
      if (threadIdx.x==0)
        idxs[i*m+j]=old;
    }
  }
}

// inp: [b, n, c] idx: [b, m]
// out: [b, m, c]
__global__ void gatherpointKernel(int b,int n,int m,int c,const float * __restrict__ inp,const int * __restrict__ idx,float * __restrict__ out){
  int loop_time = b * m * c;
  CUDA_1D_KERNEL_LOOP(index, loop_time){
    int cur_batch_size = index / (m * c);
    int cur_point_idx = index / c;
    int cur_channel = index % c;

    int a=idx[cur_point_idx];
    int current_idx = cur_batch_size * (n * c) + a * c + cur_channel;
    out[index] = inp[current_idx];
  }
}

// out_g: [b, m, c] idx: [b, m]
// inp_g: [b, n, c]
__global__ void scatteraddpointKernel(int b,int n,int m,int c,const float * __restrict__ out_g,const int * __restrict__ idx,float * __restrict__ inp_g){
  int loop_time = b * m * c;
  CUDA_1D_KERNEL_LOOP(index, loop_time){
    int cur_batch_size = index / (m * c);
    int cur_point_idx = index / c;
    int cur_channel = index % c;

    int a = idx[cur_point_idx];
    int current_idx = cur_batch_size * n * c + a * c + cur_channel;
    atomicAdd(&inp_g[current_idx],out_g[index]);
  }
}


// inp: [b, n, c] mask: [b, n]
// out: [b, proposal_num, c]
__global__ void GatherByMaskKernel(int b,int n,int c,int proposal_num,const float *inp,const float *mask,float *out){
  for (int cur_batch=blockIdx.x; cur_batch<b; cur_batch+=gridDim.x){
    const float *cur_inp = inp + cur_batch * n * c; 
    const float *cur_mask = mask + cur_batch * n;
    float* cur_out = out + cur_batch * proposal_num * c;

    int proposal_cnt = 0;
    int loop_time, tmp_channel_idx;
    for (int cur_pts=0; cur_pts<n; cur_pts++){
        if(int(cur_mask[cur_pts]) == 0) continue;
        if(proposal_cnt == proposal_num) break;
        // a valid proposal
        if (proposal_cnt == 0){
            loop_time = proposal_num * c;
            for (int i=threadIdx.x; i<loop_time; i+=blockDim.x){
                tmp_channel_idx = i % c;
                cur_out[i] = cur_inp[cur_pts * c + tmp_channel_idx];
            }
            __syncthreads();
        } 
        else {
            loop_time = c;
            for (int i=threadIdx.x; i<loop_time; i+=blockDim.x){
                cur_out[proposal_cnt * c + i] = cur_inp[cur_pts * c + i];
            }
            __syncthreads();
        }
        proposal_cnt += 1;
    }
  }
}

void cumsumLauncher(int b,int n,const float * inp,float * out){
  cumsumKernel<<<32,512>>>(b,n,inp,out);
}
//require b*n working space
void probsampleLauncher(int b,int n,int m,const float * inp_p,const float * inp_r,float * temp,int * out){
  cumsumKernel<<<32,512>>>(b,n,inp_p,temp);
  binarysearchKernel<<<dim3(32,8,1),512>>>(b,n,m,temp,inp_r,out);
}
//require 32*n working space
void farthestpointsamplingLauncher(int b,int n,int c,int m,const float * inp,float * temp,int * out){
  farthestpointsamplingKernel<1024><<<b,1024>>>(b,n,c,m,inp,temp,out);
}
//require 32*n working space
void farthestpointsamplingwithdistLauncher(int b,int n,int m,const float * inp,float * temp,int * out){
  farthestpointsamplingwithdistKernel<1024><<<b,1024>>>(b,n,m,inp,temp,out);
}
//require 32*n working space
void farthestpointsamplingwithpreidxLauncher(int b,int n,int c,int m,int m1,const float * inp, const int* preidx,float * temp,int * out){
  farthestpointsamplingwithpreidxKernel<1024><<<b,1024>>>(b,n,c,m,m1,inp,preidx,temp,out);
}
void gatherpointLauncher(int b,int n,int m,int c,const float * inp,const int * idx,float * out){
  gatherpointKernel<<<block_num,threadsPerBlock>>>(b,n,m,c,inp,idx,out);
  //int thread_num = 512 / b;
  // gatherpointKernel<<<dim3(256,8,1),512>>>(b,n,m,inp,idx,out);
}
void scatteraddpointLauncher(int b,int n,int m,int c,const float * out_g,const int * idx,float * inp_g){
  scatteraddpointKernel<<<block_num,threadsPerBlock>>>(b,n,m,c,out_g,idx,inp_g);
}

void GatherByMaskLauncher(int b,int n,int c,int proposal_num,const float *inp,const float *mask,float *out){
  GatherByMaskKernel<<<block_num,threadsPerBlock>>>(b,n,c,proposal_num,inp,mask,out);
}

