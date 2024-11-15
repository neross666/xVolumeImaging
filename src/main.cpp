#include <spdlog/spdlog.h>
#include "base.h"

void render(const DensityGrid& grid, const RenderSetting& setting, float* frameBuffer);
void renderX(const DensityGrid& grid, const RenderSetting& setting, float* frameBuffer);


#include <opencv2/opencv.hpp>
void SaveMat(float* fb, int width, int height)
{
	cv::Mat mat(height, width, CV_8UC3);
	for (int j = 0; j < height; j++)
	{
		auto prow = mat.ptr<uchar>(j);
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

			prow[3 * i] = B;
			prow[3 * i + 1] = G;
			prow[3 * i + 2] = R;
		}
	}

	cv::imshow("output", mat);
	cv::waitKey(0);
}

// optimization strategy:
// 1. Random number precomputation
// 2. First intersect with AABB boxes, then intersect in triangles
int main()
{
	RenderSetting setting;
	setting.height = 512;
	setting.width = 700;
	setting.l_intensity = Vec3f(2.5f);
	setting.lightdir = { 0.0, 0.0, -1.0f };
	setting.sigma_a = Vec3f(0.0f);
	setting.sigma_s = Vec3f(0.86f, 0.63f, 0.48f);
	setting.samples = 1024;
	setting.g = 0.0f;
	setting.max_depth = 100;

	const float aspect_ratio = static_cast<float>(setting.width) / setting.height;
	const Matrix44f c2w(
		1.0, 0.0, 0.0, 0.0,
		0.0, 1.0, 0.0, 0.0,
		0.0, 0.0, 1.0, 0.0,
		0.0, 0.0, 5.0, 1.0);
	const float FOV = 30.0f;
	setting.camera = PinholeCamera(aspect_ratio, c2w, FOV);
	
	std::string dataDir = DATA_DIR;
	DensityGrid grid(dataDir+"rest.raw");

	std::unique_ptr<float[]> frameBuffer{ new float[setting.width * setting.height * 3] };

	spdlog::info("start rendering a image");
	render(grid, setting, frameBuffer.get());
	spdlog::info("render done.");

	//show Image
	SaveMat(frameBuffer.get(), setting.width, setting.height);


	return 0;
}