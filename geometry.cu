#include "geometry.h"


void orthonormalBasis(const Vec3f& n, Vec3f& t, Vec3f& b)
{
#if 0
	if (std::fabs(n[0]) > std::fabs(n[1])) {
		float inv_len = 1 / std::sqrt(n[0] * n[0] + n[2] * n[2]);
		t = Vec3f(n[2] * inv_len, 0, -n[0] * inv_len);
	}
	else {
		float inv_len = 1 / std::sqrt(n[1] * n[1] + n[2] * n[2]);
		t = Vec3f(0, -n[2] * inv_len, n[1] * inv_len);
	}
	b = cross(n, t);
#elif 0
	if (std::abs(n[1]) < 0.9f) {
		t = normalize(cross(n, Vec3f(0, 1, 0)));
	}
	else {
		t = normalize(cross(n, Vec3f(0, 0, -1)));
	}
	b = normalize(cross(t, n));
#else
	float sign = std::copysign(1.0f, n[2]);	// n[2] == 0 !!!
	const float a = -1.0f / (sign + n[2]);
	const float c = n[0] * n[1] * a;
	t = Vec3f(1.0f + sign * n[0] * n[0] * a, sign * c, -sign * n[0]);
	b = Vec3f(c, sign + n[1] * n[1] * a, -n[1]);
#endif
}
