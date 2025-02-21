#pragma once

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <cuda_texture_types.h>

#ifdef __CUDA__
#define __twin__ __host__ __device__
#define __gpu__ __device__
#define __cpu__ __host__
#define __kernel__ __global__
#else
#define __twin__
#define __gpu__
#define __cpu__
#define __kernel__
#endif


#define checkCudaErrors(call)\
{\
  const cudaError_t error=call;\
  if(error!=cudaSuccess)\
  {\
      printf("ERROR: %s:%d,",__FILE__,__LINE__);\
      printf("code:%d,reason:%s\n",error,cudaGetErrorString(error));\
      exit(1);\
  }\
}