#include "raytrace.h"

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
		return Vec3f(0.5f);
	}
	return Vec3f(0.0f);
}

__gpu__ Vec3f transmittance(float t, const Vec3f& sigma)
{
	return exp(-sigma * t);
}
__gpu__ float distant_sample(float sigma_t, float u)
{
	return -logf(1 - u) / sigma_t;
}
__gpu__ int sampleEvent(const Vec3f& events, float u)
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
__gpu__ float getDensity(Vec3f xyz, Vec3f scale, cudaTextureObject_t tex3DObj)
{
	Vec3f uvw = (xyz + scale) / scale / 2.0f;
	float sample = tex3D<float>(tex3DObj, uvw[0], uvw[1], uvw[2]);
	return sample > 0.05f ? sample : 0.0f;
}
__gpu__ Vec3f SunLightNEE(const Ray shadowRay,
	Sampler& sampler,
	const DensityGrid& grid,
	const Vec3f sigma_s,
	const Vec3f sigma_a)
{
	Vec3f throughput{ 1.0, 1.0, 1.0 };

	const Vec3f max_t = (sigma_s + sigma_a) * grid.getMaxDensity();
	float majorant = Xmax(max_t[0], Xmax(max_t[1], max_t[2]));

	AABB bbox = grid.getBounds();
	float t_near = 0.0, t_far = 0.0f;
	if (!bbox.intersect(shadowRay, t_near, t_far))
		return throughput;

	//Ratio Tracking
	float t = t_near;
	while (true)
	{
		//distance sampling
		t += distant_sample(majorant, sampler.getNext1D());

		//sampled distance is out of volume --> break
		if (t >= t_far)
			break;

		// calculate several parametor in now position
		float density = getDensity(shadowRay(t), grid.getBoundsScale(), grid.tex3DObj);
		Vec3f absorp_weight = sigma_a * density;
		Vec3f scatter_weight = sigma_s * density;
		Vec3f null_weight = Vec3f(majorant) - absorp_weight - scatter_weight;

		//estimate transmittance
		throughput *= null_weight / Vec3f(majorant);
	}
	return throughput;
}

__gpu__ Vec3f henyey_greenstein_sample(float g, float u, float v)
{
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

__gpu__ int sampleWavelength(const Vec3f& throughput, const Vec3f& albedo,
	Sampler& sampler, Vec3f& pmf)
{
	// create empirical discrete distribution
	const Vec3f throughput_albedo = throughput * albedo;
	DiscreteEmpiricalDistribution1D distribution(throughput_albedo);
	pmf = Vec3f(distribution.getPDF(0), distribution.getPDF(1),
		distribution.getPDF(2));

	// sample index of wavelength from empirical discrete distribution
	float _pdf;
	const uint32_t channel = distribution.sample(sampler.getNext1D(), _pdf);
	return channel;
}

// ratio tracking
__gpu__ bool sampleMedium(Ray& ray, float t_near, float t_far, Sampler& sampler, const DensityGrid& grid, const RenderSetting& setting)
{
	float t = t_near;
	Vec3f throughput_tracking(1, 1, 1);
	Vec3f max_t = (setting.sigma_s + setting.sigma_a) * grid.getMaxDensity();
	float majorant = Xmax(max_t[0], Xmax(max_t[1], max_t[2]));
	while (true)
	{
		// sample wavelength
		Vec3f pmf_wavelength;
		const int channel = sampleWavelength(ray.throughput * throughput_tracking, 
			(Vec3f(majorant) - setting.sigma_a) / majorant, sampler, pmf_wavelength);
				

		const float d_sampled = distant_sample(majorant, sampler.getNext1D());
		t += d_sampled;
		//transmit
		if (t >= t_far)
		{
			const float dist_to_surface_from_current_pos = t_far - t_near;
			const Vec3f tr = transmittance(dist_to_surface_from_current_pos, Vec3f(Xmax(max_t[0], Xmax(max_t[1], max_t[2]))));
			const Vec3f p_surface = tr;
			const Vec3f pdf = pmf_wavelength * p_surface;
			throughput_tracking *= tr / (pdf[0] + pdf[1] + pdf[2]);

			// nan check
			if (isnan(throughput_tracking[0]) ||
				isnan(throughput_tracking[1]) ||
				isnan(throughput_tracking[2])) {
				ray.throughput = Vec3f(0.0f);
			}
			else {
				ray.throughput *= throughput_tracking;
			}

			// reset ray
			ray.origin = ray(t);

			return false;
		}

		// compute russian roulette probability
		const float density = getDensity(ray(t), grid.getBoundsScale(), grid.tex3DObj);
		const Vec3f sigma_s = setting.sigma_s * density;
		const Vec3f sigma_a = setting.sigma_a * density;
		const Vec3f sigma_n = Vec3f(majorant) - sigma_a - sigma_s;
		const Vec3f P_s = sigma_s / (sigma_s + sigma_n);
		const Vec3f P_n = sigma_n / (sigma_s + sigma_n);

		// In-Scattering
		if (sampler.getNext1D() < P_s[channel])
		{
			// update throughput
			const Vec3f tr = transmittance(d_sampled, Vec3f(majorant));
			const Vec3f pdf_distance = majorant * tr;
			const Vec3f pdf = pmf_wavelength * pdf_distance * P_s;
			throughput_tracking *= (tr * sigma_s) / (pdf[0] + pdf[1] + pdf[2]);

			// nan check
			if (isnan(throughput_tracking[0]) ||
				isnan(throughput_tracking[1]) ||
				isnan(throughput_tracking[2])) {
				ray.throughput = Vec3f(0.0f);
			}
			else {
				ray.throughput *= throughput_tracking;
			}


			//make next scatter Ray
			//   localize
			Vec3f b1, b2;
			branchlessONB(ray.direction, b1, b2);
			//   sample scatter dir
			Vec3f local_scatterdir = henyey_greenstein_sample(setting.g, sampler.getNext1D(), sampler.getNext1D());
			//   reset local ray to world ray
			Vec3f scatterdir = local2world(local_scatterdir, b1, b2, ray.direction);
			//   reset ray
			ray.origin = ray(t);	// get origin first!!!
			ray.direction = scatterdir;

			return true;
		}

		// Null-Scattering
		// update throughput
		const Vec3f tr = transmittance(d_sampled, Vec3f(majorant));
		const Vec3f pdf_distance = majorant * tr;
		const Vec3f pdf = pmf_wavelength * pdf_distance * P_n;
		throughput_tracking *= (tr * sigma_n) / (pdf[0] + pdf[1] + pdf[2]);
	}
}
__gpu__ Vec3f RayTraceNEE(const Ray& ray_in, const DensityGrid& grid, const RenderSetting& setting, Sampler& sampler)
{
	Vec3f radiance(0);
	Vec3f background(0.0f);

	Ray ray = ray_in;
	ray.throughput = Vec3f(1.0f, 1.0f, 1.0f);

	auto bbox = grid.getBounds();
	uint32_t depth = 0;
	while (depth < setting.max_depth)
	{
		float t_near = 0.0f;
		float t_far = 0.0f;
		if (!bbox.intersect(ray, t_near, t_far))
		{
			radiance += ray.throughput * background * (depth != 0);
			break;
		}

		// russian roulette
		if (depth > 0) {
			const float russian_roulette_prob = Xmin(
				(ray.throughput[0] + ray.throughput[1] + ray.throughput[2]) /
				3.0f,
				1.0f);
			if (sampler.getNext1D() >= russian_roulette_prob) { break; }
			ray.throughput /= russian_roulette_prob;
		}

		// sample medium
		bool is_scatter = sampleMedium(ray, t_near, t_far, sampler, grid, setting);
		if (is_scatter)
		{
			// in-scatter--->direct light
			float costheta = dot(setting.lightdir, ray.direction);
			float nee_phase = henyey_greenstein_phase(costheta, setting.g);
			Vec3f transmittance = SunLightNEE(Ray(ray.origin, setting.lightdir), sampler, grid, setting.sigma_s, setting.sigma_a);
			radiance += ray.throughput * nee_phase * setting.l_intensity * transmittance;
		}
		depth++;
	}

	return radiance;
}
