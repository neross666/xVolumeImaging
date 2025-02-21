#include "base.h"
#include "window.h"
#include <chrono>

void render(const DensityGrid& grid, const RenderSetting& setting, float* frameBuffer);

void convert(float* fb, unsigned char* pixel, int width, int height)
{
	for (int j = 0; j < height; j++)
	{
		for (int i = 0; i < width; i++)
		{
			size_t pixel_index = j * 3 * width + i * 3;
			float r = fb[pixel_index + 0];
			float g = fb[pixel_index + 1];
			float b = fb[pixel_index + 2];

			const uint8_t R =
				std::clamp(static_cast<uint32_t>(255.0f * r), 0u, 255u);
			const uint8_t G =
				std::clamp(static_cast<uint32_t>(255.0f * g), 0u, 255u);
			const uint8_t B =
				std::clamp(static_cast<uint32_t>(255.0f * b), 0u, 255u);

			pixel[pixel_index + 0] = B;
			pixel[pixel_index + 1] = G;
			pixel[pixel_index + 2] = R;
		}
	}
}

int main()
{
	Window window(700, 512);

	RenderSetting setting;
	setting.width = 700;
	setting.height = 512;
	setting.l_intensity = Vec3f(2.5f);
	setting.lightdir = { 0.0, 0.0, -1.0f };
	setting.sigma_a = Vec3f(0.0f);
	setting.sigma_s = Vec3f(0.86f, 0.63f, 0.48f);
	setting.samples = 1024;
	setting.g = 0.0f;
	setting.max_depth = 10;

	const float aspect_ratio = static_cast<float>(setting.width) / setting.height;
	const Matrix44f c2w(
		1.0, 0.0, 0.0, 0.0,
		0.0, 1.0, 0.0, 0.0,
		0.0, 0.0, 1.0, 5.0,
		0.0, 0.0, 0.0, 1.0);
	const float FOV = 30.0f;
	setting.camera = PinholeCamera(aspect_ratio, c2w, FOV);
	window.SetRenderSetting(&setting);

	// 
	std::string dataDir = DATA_DIR;
	DensityGrid grid(dataDir + "rest.raw");
	window.SetDensityGrid(&grid);

	// 
	auto Render = [](const DensityGrid& grid, const RenderSetting& setting, unsigned char* pixel)
	{
		auto start = std::chrono::steady_clock::now();
		std::unique_ptr<float[]> frameBuffer{ new float[setting.width * setting.height * 3] };
		render(grid, setting, frameBuffer.get());
		convert(frameBuffer.get(), pixel, setting.width, setting.height);
		auto end = std::chrono::steady_clock::now();
		return 1000.f / std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	};
	window.SetRenderCallback(Render);

	window.Render();
	window.Run();

	return 0;
}
