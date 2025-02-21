#pragma once
#include "base.h"

__gpu__ Vec3f gamma(const Vec3f& color);

__gpu__ Vec3f NormalIntegrate(const Ray& ray_in, const DensityGrid& grid, const RenderSetting& setting);

__gpu__ Vec3f transmittance(float t, const Vec3f& sigma);

__gpu__ float distant_sample(float sigma_t, float u);

__gpu__ int sampleEvent(const Vec3f& events, float u);

__gpu__ float henyey_greenstein_phase(float costheta, float g);

__gpu__ Vec3f SunLightNEE(const Ray shadowRay,
	Sampler& sampler,
	const DensityGrid& grid,
	const Vec3f sigma_s,
	const Vec3f sigma_a);

__gpu__ Vec3f henyey_greenstein_sample(float g, float u, float v);

// ratio tracking
__gpu__ bool sampleMedium(Ray& ray, float t_near, float t_far, Sampler& sampler, const DensityGrid& grid, const RenderSetting& setting);

__gpu__ Vec3f RayTraceNEE(const Ray& ray_in, const DensityGrid& grid, const RenderSetting& setting, Sampler& sampler);