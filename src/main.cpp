#include <spdlog/spdlog.h>
//#include "render.h"
//#include "random.hpp"
#include "base.h"

void vectorAdd(const float* A, const float* B, float* C, int N);
void render(const DensityGrid& grid, const RenderSetting& setting, float* frameBuffer);

void demo()
{
	const int N = 1024;
	float a[N], b[N], c[N];
	int* dev_a, * dev_b, * dev_c;
	for (int i = 0; i < N; ++i) // 为数组a、b赋值
	{
		a[i] = i;
		b[i] = i * i;
	}

	vectorAdd(a, b, c, N);

	spdlog::info("{},{},{},{}", c[0], c[1], c[2], c[1023]);
}


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

int main()
{
	RenderSetting setting;
	setting.height = 512;
	setting.width = 700;
	setting.l_intensity = Vec3f(1.0f);
	setting.lightdir = { 0.0, 0.0, 1.0 };
	setting.sigma_a = Vec3f(0.01f);
	setting.sigma_s = Vec3f(0.09f, 0.05f, 0.05f);
	setting.samples = 100;
	setting.g = -0.1;
	setting.max_depth = 100;

	const float aspect_ratio = static_cast<float>(setting.width) / setting.height;
	const Matrix44f c2w(
		1.0, 0.0, 0.0, 0.0,
		0.0, 1.0, 0.0, 0.0,
		0.0, 0.0, 1.0, 0.0,
		0.0, 0.0, 5.0, 1.0);
	const float FOV = 60.0f;
	setting.camera = PinholeCamera(aspect_ratio, c2w, FOV);

	DensityGrid grid;

	std::unique_ptr<float[]> frameBuffer{ new float[setting.width * setting.height * 3] };

	spdlog::info("start rendering a image");
	render(grid, setting, frameBuffer.get());
	spdlog::info("render done.");

	//show Image
	SaveMat(frameBuffer.get(), setting.width, setting.height);

	//demo();

	return 0;
}