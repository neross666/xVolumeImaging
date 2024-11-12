﻿#pragma once
#include <random>
#include <fstream>

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <curand_kernel.h>
#include <cuda_texture_types.h>

#include "geometry.h"


class Ray
{
public:
	Vec3f origin;
	Vec3f direction;
	Vec3f throughput;

	__twin__ Ray() {}
	__twin__ Ray(const Vec3f& origin, const Vec3f& direction)
		: origin(origin), direction(direction)
	{
	}

	__twin__ Vec3f operator()(float t) const { return origin + t * direction; }
};

class PinholeCamera
{
private:
	float FOV;
	float scale;
	float aspect_ratio;
	Matrix44f camera2world;

public:
	PinholeCamera() = default;
	PinholeCamera(float aspect_ratio_, const Matrix44f& c2w,
		float FOV = 90.0f) : aspect_ratio(aspect_ratio_), camera2world(c2w)
	{
		auto forward = multDirMatrix(Vec3f(0.0f, 0.0f, -1.0f), c2w);
		scale = std::tan(0.5f * deg2rad(FOV));

		//spdlog::info("[Camera] position: ({}, {}, {})", camera2world[3][0], camera2world[3][1],
		//	camera2world[3][2]);
		//spdlog::info("[Camera] forward: ({}, {}, {})", forward[0], forward[1],
		//	forward[2]);
		//spdlog::info("[PinholeCamera] scale: {}", scale);
	}

	void setTransform(const Matrix44f& c2w) {
		camera2world = c2w;
	}

	__twin__ bool sampleRay(const Vec2f& pixel, Ray& ray) const
	{
		const Vec3f dir((2 * pixel[0] - 1) * scale, (1 - 2 * pixel[1]) * scale / aspect_ratio, -1);

		ray.direction = normalize(multDirMatrix(dir, camera2world));
		ray.origin = Vec3f(camera2world[3][0], camera2world[3][1], camera2world[3][2]);

		return true;
	}
};

class RenderSetting
{
public:
	int width;
	int height;
	Vec3f lightdir = { 1,0,0 };
	Vec3f l_intensity;
	int max_depth;
	Vec3f sigma_s;
	Vec3f sigma_a;
	float g;
	int samples;
	PinholeCamera camera;
};

struct AABB {
	Vec3f pMin;
	Vec3f pMax;

	__twin__ bool intersect(const Ray& ray, float& t_near, float& t_far) {
		Vec3f direction_inv = 1.0 / ray.direction;
		Vec3f t_top = direction_inv * (pMax - ray.origin);
		Vec3f t_bottom = direction_inv * (pMin - ray.origin);
		Vec3f t_min = vmin(t_top, t_bottom);
		Vec3f t_max = vmax(t_top, t_bottom);

		float t_0 = Xmax(Xmax(t_min[0], t_min[1]), t_min[2]);
		float t_1 = Xmin(Xmin(t_max[0], t_max[1]), t_max[2]);

		if (t_0 > t_1 || t_1 <= 0.0f) {
			return false;
		}

		t_0 = Xmax(t_0, 0.0f); // Ensure t_0 is not negative

		t_near = t_0;
		t_far = t_1;

		return true;
	}
};


class DensityGrid
{
private:
	Vec3f m_bounds;

public:
	cudaTextureObject_t tex3DObj;

	DensityGrid() = default;
	DensityGrid(std::string filename)
	{
		//spdlog::info("[OpenVDBGrid] loading: {}", filepath.generic_string());

		// read grid data from file.

		AABB bbox = getBounds();
		//spdlog::info("[Scene] OpenVDB Volume bounding box");
		//spdlog::info("[Scene] pMin: ({}, {}, {})", bbox.pMin[0], bbox.pMin[1],
		//	bbox.pMin[2]);
		//spdlog::info("[Scene] pMax: ({}, {}, {})", bbox.pMax[0], bbox.pMax[1],
		//	bbox.pMax[2]);


		// 
		std::ifstream rawFile(filename, std::ios::binary);
		if (!rawFile) {
			std::cerr << "无法打开文件：" << filename << std::endl;
			return ;
		}

		int width = 0;
		int height = 0;
		int length = 0;
		int scantype = 0;
		rawFile.read((char*)&width, sizeof(int));
		rawFile.read((char*)&height, sizeof(int));
		rawFile.read((char*)&length, sizeof(int));
		rawFile.read((char*)&scantype, sizeof(int));

		std::unique_ptr<unsigned char[]> raw_data{ new unsigned char[width * height * length] };
		rawFile.read((char*)raw_data.get(), width * height * length * sizeof(unsigned char));

		allocateTexture3D(raw_data.get(), width, height, length);

		m_bounds = Vec3f(width, height, length);
	}

	__twin__ AABB getBounds() const
	{
		//AABB ret{ Vec3f(-1.0f), Vec3f(1.0f) };
		auto scale = getBoundsScale();
		AABB ret{ -scale, scale };
		return ret;
	}

	__twin__ Vec3f getBoundsScale() const
	{
		auto max_edge = Xmax(Xmax(m_bounds[0], m_bounds[1]), m_bounds[2]);
		return m_bounds / max_edge;
	}

	//__twin__ float getDensity(const Vec3f& pos) const
	//{
	//	return 0.9f;
	//}

	__twin__ float getMaxDensity() const
	{
		return 1.0f;
	}

private:
	__cpu__ void allocateTexture3D(unsigned char* data, int width, int height, int depth) {
		cudaArray* cuArray;
		cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<unsigned char>();

		// 申请CUDA数组
		checkCudaErrors(cudaMalloc3DArray(&cuArray, &channelDesc, make_cudaExtent(width, height, depth)));

		// 复制数据到CUDA数组
		cudaMemcpy3DParms copyParams = { 0 };
		copyParams.srcPtr = make_cudaPitchedPtr(data, width * sizeof(unsigned char), width, height);
		copyParams.dstArray = cuArray;
		copyParams.extent = make_cudaExtent(width, height, depth);
		copyParams.kind = cudaMemcpyHostToDevice;
		checkCudaErrors(cudaMemcpy3D(&copyParams));

		// 设置纹理对象的描述符
		cudaResourceDesc resDesc;
		memset(&resDesc, 0, sizeof(cudaResourceDesc));
		resDesc.resType = cudaResourceTypeArray;
		resDesc.res.array.array = cuArray;

		cudaTextureDesc texDesc;
		memset(&texDesc, 0, sizeof(cudaTextureDesc));
		texDesc.addressMode[0] = cudaAddressModeWrap; // X轴包裹模式
		texDesc.addressMode[1] = cudaAddressModeWrap; // Y轴包裹模式
		texDesc.addressMode[2] = cudaAddressModeWrap; // Z轴包裹模式
		texDesc.filterMode = cudaFilterModeLinear; // 线性插值
		texDesc.readMode = cudaReadModeNormalizedFloat; // 读取模式
		texDesc.normalizedCoords = true; // 坐标归一化

		// 创建纹理对象
		checkCudaErrors(cudaCreateTextureObject(&tex3DObj, &resDesc, &texDesc, nullptr));
	}
};

// uniform distribution sampler
#ifdef __CUDA__
class Sampler
{
public:
	__twin__ Sampler(curandState* rs) : rand_state(rs) {}
	__gpu__ float getNext1D() { count++; return curand_uniform(rand_state); }
	__gpu__ Vec2f getNext2D() { count += 2; return Vec2f(curand_uniform(rand_state), curand_uniform(rand_state)); }
	void setSeed(uint32_t seed) {  }

private:
	curandState* rand_state = nullptr;
	int count = 0;
};
#else
class Sampler
{
public:
	__twin__ Sampler() : dis(0.0f, 1.0f) {}
	float getNext1D() { count++; return dis(gen); }
	Vec2f getNext2D() { count += 2; return Vec2f(dis(gen), dis(gen)); }

	void setSeed(uint32_t seed) { gen.seed(seed); }

private:
	std::mt19937 gen;
	std::uniform_real_distribution<float> dis;
	int count = 0;
};
#endif // __CUDA__