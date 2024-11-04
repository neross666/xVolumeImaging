﻿#ifdef __CUDA__
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


constexpr void checkCudaErrors(cudaError_t error) {
	if (error != cudaSuccess)
	{
		//spdlog::error("ERROR: {}:{},", __FILE__, __LINE__);
		//spdlog::error("code: {}, reason:{},", val, cudaGetErrorString(val));
		printf("ERROR: %s:%d,", __FILE__, __LINE__);
		printf("code:%d,reason:%s\n", error, cudaGetErrorString(error));
		exit(1);
	}
}