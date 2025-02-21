#include "base.h"
#include "window.h"
#include <chrono>

void render(const DensityGrid& grid, const RenderSetting& setting, unsigned char* frameBuffer);

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
		render(grid, setting, pixel);
		auto end = std::chrono::steady_clock::now();
		return 1000.f / std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
	};
	window.SetRenderCallback(Render);

	window.Render();
	window.Run();

	return 0;
}
