#pragma once
#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

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


__kernel__ void random_init(int max_x, int max_y, curandState* rand_state) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y)) return;
	int pixel_index = j * max_x + i;
	//Each thread gets same seed, a different sequence number, no offset
	curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}


__gpu__ float rnd(curandState* rand_state)
{
	return curand_uniform(rand_state);
}
