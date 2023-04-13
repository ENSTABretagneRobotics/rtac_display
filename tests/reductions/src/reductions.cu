#include "reductions.h"

#include <rtac_base/cuda/reductions.hcu>

namespace rtac { namespace display {


__global__ void do_reduce(float* out, const float* in, unsigned int N)
{
    extern __shared__ unsigned char sdata_[];
    float *sdata = reinterpret_cast<float*>(sdata_);

    unsigned int tid = threadIdx.x;

    unsigned int i = 256*blockIdx.x + tid;
    unsigned int gridSize = 256*gridDim.x;

    sdata[tid] = 0.0f;
    while(i < N) {
        sdata[tid] += in[i];
        i += gridSize;
    }
    __syncthreads();
    
    if(tid < 128) { 
        sdata[tid] += sdata[tid + 128];
        __syncthreads();
    }
    if(tid < 64) { 
        sdata[tid] += sdata[tid + 64];
        __syncthreads();
    }
    if(tid < 32) { 
        sdata[tid] += sdata[tid + 32];
        __syncthreads();
    }
    if(tid < 16) { 
        sdata[tid] += sdata[tid + 16];
        __syncthreads();
    }
    if(tid < 8) { 
        sdata[tid] += sdata[tid + 8];
        __syncthreads();
    }
    if(tid < 4) { 
        sdata[tid] += sdata[tid + 4];
        __syncthreads();
    }
    if(tid < 2) { 
        sdata[tid] += sdata[tid + 2];
        __syncthreads();
    }

    if(tid == 0) out[blockIdx.x] = sdata[0] + sdata[1];
}



float sum(cuda::CudaVector<float>& data)
{
    cuda::device::reduce(data.data(), data.data(), data.size());
    CUDA_CHECK_LAST();


    //unsigned int N = data.size();
    //while(N > 0) {
    //    unsigned int blockCount = N / (2*256);
    //    if(blockCount == 0)
    //        do_reduce<<<1,256,256*sizeof(float)>>>(data.data(), data.data(), N);
    //    else
    //        do_reduce<<<blockCount,256,256*sizeof(float)>>>(data.data(), data.data(), N);
    //    cudaDeviceSynchronize();
    //    N = blockCount;
    //}
    //CUDA_CHECK_LAST();
    return 0.0f;
}

}; //namespace display
}; //namespace rtac

