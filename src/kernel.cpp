#include "raytrace.h"


Vec3f doRender(const DensityGrid& grid, const RenderSetting& setting, Sampler& sampler, int x, int y)
{
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

	return Color;
}



void renderX(const DensityGrid& grid, const RenderSetting& setting, float* h_fb)
{
	const uint32_t width = setting.width;
	const uint32_t height = setting.height;

	for (uint32_t i = 0; i < height; ++i)
	{
		for (uint32_t j = 0; j < width; ++j)
		{
			const std::unique_ptr<Sampler> sampler_per_pixel = std::make_unique<Sampler>(nullptr);
			sampler_per_pixel->setSeed(j + width * i);


			int pixel_index = i * setting.width * 3 + j * 3;

			Vec3f radiance = doRender(grid, setting, *sampler_per_pixel, j, i);


			// gamma
			radiance = gamma(radiance);

			h_fb[pixel_index] = radiance[0];
			h_fb[pixel_index + 1] = radiance[1];
			h_fb[pixel_index + 2] = radiance[2];
		}
	}


}