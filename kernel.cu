#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include "base.h"
#include <curand_kernel.h>


__global__ void vectorAddKernel(const float* A, const float* B, float* C, int N) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < N) {
		C[i] = A[i] + B[i];
	}
}


void vectorAdd(const float* A, const float* B, float* C, int N) {
	float* d_A, * d_B, * d_C;
	size_t size = N * sizeof(float);

	checkCudaErrors(cudaMalloc(&d_A, size));
	checkCudaErrors(cudaMalloc(&d_B, size));
	checkCudaErrors(cudaMalloc(&d_C, size));

	checkCudaErrors(cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice));
	checkCudaErrors(cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice));

	dim3 blocksPerGrid(256, 1, 1);
	dim3 threadsPerBlock((N + 256 - 1) / 256, 1, 1);

	vectorAddKernel << <blocksPerGrid, threadsPerBlock >> > (d_A, d_B, d_C, N);

	checkCudaErrors(cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost));

	checkCudaErrors(cudaFree(d_A));
	checkCudaErrors(cudaFree(d_B));
	checkCudaErrors(cudaFree(d_C));
}


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



__gpu__ Vec3f gamma(const Vec3f color)
{
	Vec3f result;
	result[0] = fmaxf(fminf(powf(color[0], 1 / 2.2), 1.0), 0.0);
	result[1] = fmaxf(fminf(powf(color[1], 1 / 2.2), 1.0), 0.0);
	result[2] = fmaxf(fminf(powf(color[2], 1 / 2.2), 1.0), 0.0);

	return result;
}

__gpu__ Vec3f NormalIntegrate(const Ray* ray_in, const DensityGrid* grid, const RenderSetting* setting)
{
	Vec3f radiance(0);
	auto bbox = grid->getBounds();
	float t_near = 0.0f;
	float t_far = 0.0f;
	if (bbox.intersect(ray_in, &t_near, &t_far))
	{
		return Vec3f(0.5f);
	}
	return Vec3f(0.0f);
}


__gpu__ float distant_sample(float sigma_t, float u)
{
	return -logf(1 - u) / sigma_t;
}
__gpu__ int sampleEvent(const Vec3f events, float u)
{
	// 0-->absorp, 1-->scatter, 2-->null scatter
	Vec3f cdf = Vec3f{ events[0],
						events[0] + events[1],
						events[0] + events[1] + events[2] }
	/ (events[0] + events[1] + events[2]);

	if (u < cdf[0])
		return 0;
	else if (u < cdf[1])
		return 1;
	else
		return 2;
}
__gpu__ float henyey_greenstein_phase(float costheta, float g)
{
	const float PI_MUL_4_INV = 0.25f * 3.14159265358979323846f;
	float denomi = 1 + g * g - 2 * g * costheta;
	return PI_MUL_4_INV * (1 - g * g) / powf(denomi, 3.0f / 2.0f);
}
__gpu__ Vec3f SunLightNEE(const Ray shadowRay,
	curandState* rand_state,
	const DensityGrid* grid,
	const float sigma_s,
	const float sigma_a)
{
	Vec3f throughput{ 1.0, 1.0, 1.0 };

	const float max_t = (sigma_s + sigma_a) * grid->getMaxDensity();
	AABB bbox = grid->getBounds();
	float t_near = 0.0, t_far = 0.0f;
	if (!bbox.intersect(&shadowRay, &t_near, &t_far))
		return throughput;

	//Ratio Tracking
	float t = t_near;
	while (true)
	{
		//distance sampling
		t += distant_sample(max_t, rnd(rand_state));

		//sampled distance is out of volume --> break
		if (t >= t_far)
			break;

		// calculate several parametor in now position
		float density = grid->getDensity(shadowRay(t));
		float absorp_weight = sigma_a * density;
		float scatter_weight = sigma_s * density;
		float null_weight = max_t - absorp_weight - scatter_weight;

		//estimate transmittance
		throughput *= null_weight / max_t;
	}
	return throughput;
}
__gpu__ void branchlessONB(const Vec3f n, Vec3f* b1, Vec3f* b2)
{
	float sign = copysignf(1.0f, n[2]);
	const float a = -1.0f / (sign + n[2]);
	const float b = n[0] * n[1] * a;
	*b1 = Vec3f(1.0f + sign * n[0] * n[0] * a, sign * b, -sign * n[0]);
	*b2 = Vec3f(b, sign + n[1] * n[1] * a, -n[1]);
}
__gpu__ Vec3f henyey_greenstein_sample(float g, float u, float v)
{
	const float PI_MUL_2 = 2.0f * 3.14159265358979323846f;

	float cosTheta;
	if (abs(g) < 1e-3)
	{
		cosTheta = 2.0f * u - 1.0f;
	}
	else
	{
		const float sqrTerm = (1 - g * g) / (1 - g + 2 * g * u);
		cosTheta = (1 + g * g - sqrTerm * sqrTerm) / (2 * g);
	}
	float sinTheta = sqrtf(abs(1 - cosTheta * cosTheta));
	float phi = PI_MUL_2 * v;

	//Note: This renderer assumes that Y-up.
	//      So This return has to be interpreted as {x, z, y} 
	return { sinTheta * cosf(phi),
			sinTheta * sinf(phi),
			cosTheta };
}
__gpu__ Vec3f local2world(const Vec3f local, const Vec3f x, const Vec3f y, const Vec3f z)
{
	return x * local[0] + y * local[1] + z * local[2];
}
__gpu__ Vec3f RayTraceNEE(const Ray* ray_in, const DensityGrid* grid, const RenderSetting* setting, curandState* rand_state)
{
	Vec3f radiance(0);
	Vec3f background(0.0f);

	Ray ray = *ray_in;
	ray.throughput = Vec3f(1.0f, 1.0f, 1.0f);

	float max_t = (setting->sigma_s + setting->sigma_a) * grid->getMaxDensity();

	auto bbox = grid->getBounds();
	uint32_t depth = 0;
	while (depth < setting->max_depth)
	{
		float t_near = 0.0f;
		float t_far = 0.0f;
		if (!bbox.intersect(&ray, &t_near, &t_far))
		{
			radiance += ray.throughput * background * (depth != 0);
			break;
		}
		//return Vec3f((t_far - t_near) / 4.0);

		// russian roulette
		if (depth > 0) {
			const float russian_roulette_prob = Xmin(
				(ray.throughput[0] + ray.throughput[1] + ray.throughput[2]) /
				3.0f,
				1.0f);
			if (rnd(rand_state) >= russian_roulette_prob) { break; }
			ray.throughput /= russian_roulette_prob;
		}

		// distance sample
		float t = t_near;
		float d_sampled = distant_sample(max_t, rnd(rand_state));
		t += d_sampled;
		//transmit
		if (t >= t_far)
		{
			// 考虑边界点的辐射量？
			// direct light
			float costheta = dot(setting->lightdir, ray.direction);
			float nee_phase = henyey_greenstein_phase(costheta, setting->g);
			Vec3f transmittance = SunLightNEE(Ray(ray(t_far), setting->lightdir), rand_state, grid, setting->sigma_s, setting->sigma_a);
			radiance += ray.throughput * nee_phase * setting->l_intensity * transmittance;

			break;
		}

		// density sample
		auto pos = ray(t);
		float density = grid->getDensity(pos);

		float absorp_weight = setting->sigma_a * density;
		float scatter_weight = setting->sigma_s * density;
		float null_weight = max_t - absorp_weight - scatter_weight;
		Vec3f events{ absorp_weight, scatter_weight, null_weight };

		//Sample Event???如果只考虑散射或者只考虑发射时可以通过调整透射率，免去事件采样
		// 0-->absorp, 1-->scatter, 2-->null scatter
		int e = sampleEvent(events, rnd(rand_state));
		if (e == 0)//absorp
		{
			//Todo: correspond to emission
			break;
		}
		else if (e == 1)//scatter
		{
			// direct light
			float costheta = dot(setting->lightdir, ray.direction);
			float nee_phase = henyey_greenstein_phase(costheta, setting->g);
			Vec3f transmittance = SunLightNEE(Ray(pos, setting->lightdir), rand_state, grid, setting->sigma_s, setting->sigma_a);
			radiance += ray.throughput * nee_phase * setting->l_intensity * transmittance;

			//make next scatter Ray
			//   localize
			Vec3f b1, b2;
			branchlessONB(ray.direction, &b1, &b2);
			//   sample scatter dir
			Vec3f local_scatterdir = henyey_greenstein_sample(setting->g, rnd(rand_state), rnd(rand_state));
			//   reset local ray to world ray
			Vec3f scatterdir = local2world(local_scatterdir, b1, b2, ray.direction);
			//   reset ray
			ray.direction = scatterdir;
			ray.origin = pos;

			depth++;
		}
		else // null scatter
		{
			// renew ray
			ray.origin = pos;
			continue;
		}


	}
	return radiance;
}

// kernel函数似乎不允许const T*类型参数？！
__kernel__ void renderKernel(const DensityGrid grid, const RenderSetting setting, float* fb, curandState* rand_state)
{
	int x = threadIdx.x + blockIdx.x * blockDim.x;
	int y = threadIdx.y + blockIdx.y * blockDim.y;

	if ((x >= setting.width) || (y >= setting.height)) return;

	int pixel_index = y * setting.width * 3 + x * 3;

	// local randomstate
	curandState local_rand_state = rand_state[y * setting.width + x];

	Vec3f Color(0.0f);
	for (int i = 0; i < setting.samples; i++)
	{
		// SSAA
		const float u =
			(x + rnd(&local_rand_state)) / setting.width;
		const float v =
			(y + rnd(&local_rand_state)) / setting.height;

		Ray firstRay;
		if (setting.camera.sampleRay(Vec2f(u, v), &firstRay))
		{
			//Color += NormalIntegrate(&firstRay, &grid, &setting);
			Color += RayTraceNEE(&firstRay, &grid, &setting, &local_rand_state);
			// Color += RayTrace(grid, setting.lightdir, setting.l_intensity, &local_rand_state,
			//                   setting.max_density, setting.max_depth, setting.sigma_s, setting.sigma_a, setting.g, firstRay);
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
