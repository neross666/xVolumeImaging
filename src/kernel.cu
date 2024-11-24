#include "raytrace.h"


__kernel__ void random_init(int max_x, int max_y, curandState* rand_state) {
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	if ((i >= max_x) || (j >= max_y)) return;
	int pixel_index = j * max_x + i;
	//Each thread gets same seed, a different sequence number, no offset
	curand_init(1984, pixel_index, 0, &rand_state[pixel_index]);
}


__kernel__ void renderKernel(const DensityGrid grid, const RenderSetting setting, float* fb, curandState* rand_state)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if ((x >= setting.width) || (y >= setting.height)) return;

	int pixel_index = y * setting.width * 3 + x * 3;

	// local randomstate
	curandState local_rand_state = rand_state[y * setting.width + x];
	Sampler sampler(&local_rand_state);

	Vec3f Color(0.0f);
	for (int i = 0; i < setting.samples; i++)
	{
		// SSAA
		const float u =
			(x + sampler.getNext1D()) / setting.width;
		const float v =
			(y + sampler.getNext1D()) / setting.height;

		Ray firstRay;
		if (setting.camera.sampleRay(Vec2f(u, v), firstRay))
		{
			//Color += NormalIntegrate(firstRay, grid, setting);
			Color += RayTraceNEE(firstRay, grid, setting, sampler);
		}
	}
	Color /= float(setting.samples);

	// Gamma Process
	Color = gamma(Color);

	// write color buffer
	fb[pixel_index + 0] = Color[0];
	fb[pixel_index + 1] = Color[1];
	fb[pixel_index + 2] = Color[2];
}


void render(const DensityGrid& grid, const RenderSetting& setting, float* h_fb)
{
	//get data from render setting
	int nx = setting.width;
	int ny = setting.height;
	int tx = 8;
	int ty = 8;
	dim3 blocks(nx / tx + 1, ny / ty + 1);
	dim3 threads(tx, ty);

	//init random state per pixel
	curandState* d_rand_state = nullptr;
	checkCudaErrors(cudaMalloc((void**)&d_rand_state, nx * ny * sizeof(curandState)));
	random_init << <blocks, threads >> > (nx, ny, d_rand_state);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());



	int num_pixels = nx * ny;
	size_t fb_size = 3 * num_pixels * sizeof(float);

	float* d_fb;
	checkCudaErrors(cudaMalloc(&d_fb, fb_size));

	//call kernel function
	renderKernel << <blocks, threads >> > (grid, setting, d_fb, d_rand_state);
	checkCudaErrors(cudaGetLastError());
	checkCudaErrors(cudaDeviceSynchronize());
	checkCudaErrors(cudaMemcpy(h_fb, d_fb, fb_size, cudaMemcpyDeviceToHost));

	//free several data
	checkCudaErrors(cudaFree(d_rand_state));
	checkCudaErrors(cudaFree(d_fb));
}