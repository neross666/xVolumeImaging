#pragma once
#include <filesystem>
#include "geometry.h"
#include "random.hpp"
//#include <spdlog/spdlog.h>






// uniform distribution sampler
class Sampler
{
private:
	int count = 0;

public:
	__gpu__ float getNext1D(curandState* rand_state) { count++; return rnd(rand_state); }
	__gpu__ Vec2f getNext2D(curandState* rand_state) { count += 2; return Vec2f(rnd(rand_state), rnd(rand_state)); }
};





/*
__gpu__ Vec3f gamma(const Vec3f& color)
{
	Vec3f result;
	result[0] = fmaxf(fminf(powf(color[0], 1 / 2.2), 1.0), 0.0);
	result[1] = fmaxf(fminf(powf(color[1], 1 / 2.2), 1.0), 0.0);
	result[2] = fmaxf(fminf(powf(color[2], 1 / 2.2), 1.0), 0.0);

	return result;
}

__gpu__ Vec3f NormalIntegrate(const Ray& ray_in, const DensityGrid& grid, const RenderSetting& setting)
{
	Vec3f radiance(0);
	auto bbox = grid.getBounds();
	float t_near = 0.0f;
	float t_far = 0.0f;
	if (bbox.intersect(ray_in, t_near, t_far))
	{
		return Vec3f(0.5);
	}
}

//render function
__gpu__ void doRender(int i, int j, float* fb, const DensityGrid& grid, const RenderSetting & setting, curandState * rand_state)
{
	if ((i >= setting.width) || (j >= setting.height)) return;

	int pixel_index = j * setting.width * 3 + i * 3;

	// local randomstate
	curandState local_rand_state = rand_state[j * setting.width + i];

	// Let's Montecarlo
	Vec3f Color{};
	for (int i = 0; i < setting.samples; i++)
	{
		// SSAA
		const float u =
			(j + rnd(rand_state)) / setting.width;
		const float v =
			(i + rnd(rand_state)) / setting.height;
		Ray firstRay;
		if (setting.camera->sampleRay(Vec2f(u, v), firstRay))
		{
			Color += NormalIntegrate(firstRay, grid, setting);
			//Color += RayTraceNEE(grid, setting.lightdir, setting.l_intensity, &local_rand_state,
			//	setting.max_density, setting.max_depth, setting.sigma_s, setting.sigma_a, setting.g, firstRay);
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
};
*/


