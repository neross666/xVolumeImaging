#pragma once

#include <cmath>
#include <iostream>
#include <limits>
#include <iomanip>
#include <algorithm>
#include "util.h"

constexpr float PI = 3.14159265359;

constexpr float PI_MUL_2 = 2.0f * PI;
constexpr float PI_MUL_4 = 4.0f * PI;

constexpr float PI_DIV_2 = 0.5f * PI;
constexpr float PI_DIV_4 = 0.25f * PI;

constexpr float PI_INV = 1.0f / PI;
constexpr float PI_MUL_2_INV = 1.0f / PI_MUL_2;
constexpr float PI_MUL_4_INV = 1.0f / PI_MUL_4;

constexpr float EPS = 1e-9f;
constexpr float RAY_EPS = 1e-3f;

inline float rad2deg(float rad) { return 180.0f * rad / PI; }
inline float deg2rad(float deg) { return deg / 180.0f * PI; }

template <typename T>
__twin__ T Xmax(T a, T b) {
	return (a > b) ? a : b;
}
template <typename T>
__twin__ T Xmin(T a, T b) {
	return (a < b) ? a : b;
}

#pragma region Vec2
template <typename T>
struct Vec2 {
	T v[2];

	__twin__ Vec2() { v[0] = v[1] = 0; }
	__twin__ Vec2(T x) { v[0] = v[1] = x; }
	__twin__ Vec2(T x, T y)
	{
		v[0] = x;
		v[1] = y;
	}

	__twin__ T x() const { return v[0]; }
	__twin__ T& x() { return v[0]; }

	__twin__ T y() const { return v[1]; }
	__twin__ T& y() { return v[1]; }

	__twin__ T operator[](int i) const { return v[i]; }
	__twin__ T& operator[](int i) { return v[i]; }

	__twin__ Vec2 operator-() const { return Vec2(-v[0], -v[1]); }

	__twin__ Vec2& operator+=(const Vec2& v)
	{
		for (int i = 0; i < 2; ++i) { this->v[i] += v[i]; }
		return *this;
	}
	__twin__ Vec2& operator*=(const Vec2& v)
	{
		for (int i = 0; i < 2; ++i) { this->v[i] *= v[i]; }
		return *this;
	}
	__twin__ Vec2& operator/=(const Vec2& v)
	{
		for (int i = 0; i < 2; ++i) { this->v[i] /= v[i]; }
		return *this;
	}
};

template <typename T>
__twin__ inline Vec2<T> operator+(const Vec2<T> v1, const Vec2<T> v2)
{
	return Vec2<T>(v1[0] + v2[0], v1[1] + v2[1]);
}
template <typename T>
__twin__ inline Vec2<T> operator+(const Vec2<T> v1, float k)
{
	return Vec2<T>(v1[0] + k, v1[1] + k);
}
template <typename T>
__twin__ inline Vec2<T> operator+(float k, const Vec2<T> v2)
{
	return v2 + k;
}

template <typename T>
__twin__ inline Vec2<T> operator-(const Vec2<T> v1, const Vec2<T> v2)
{
	return Vec2<T>(v1[0] - v2[0], v1[1] - v2[1]);
}
template <typename T>
__twin__ inline Vec2<T> operator-(const Vec2<T> v1, float k)
{
	return Vec2<T>(v1[0] - k, v1[1] - k);
}
template <typename T>
__twin__ inline Vec2<T> operator-(float k, const Vec2<T> v2)
{
	return Vec2<T>(k - v2[0], k - v2[1]);
}

template <typename T>
__twin__ inline Vec2<T> operator*(const Vec2<T> v1, const Vec2<T> v2)
{
	return Vec2<T>(v1[0] * v2[0], v1[1] * v2[1]);
}
template <typename T>
__twin__ inline Vec2<T> operator*(const Vec2<T> v1, float k)
{
	return Vec2<T>(v1[0] * k, v1[1] * k);
}
template <typename T>
__twin__ inline Vec2<T> operator*(float k, const Vec2<T> v2)
{
	return v2 * k;
}

template <typename T>
__twin__ inline Vec2<T> operator/(const Vec2<T> v1, const Vec2<T> v2)
{
	return Vec2<T>(v1[0] / v2[0], v1[1] / v2[1]);
}
template <typename T>
__twin__ inline Vec2<T> operator/(const Vec2<T> v1, float k)
{
	return Vec2<T>(v1[0] / k, v1[1] / k);
}
template <typename T>
__twin__ inline Vec2<T> operator/(float k, const Vec2<T> v2)
{
	return Vec2<T>(k / v2[0], k / v2[1]);
}

using Vec2i = Vec2<int>;
using Vec2f = Vec2<float>;

#pragma endregion Vec2

#pragma region Vec3
template <typename T>
struct Vec3 {
	T v[3];

	// implement Point
	static constexpr int dim = 3;

	__twin__ Vec3() { v[0] = v[1] = v[2] = 0; }
	__twin__ Vec3(T x) { v[0] = v[1] = v[2] = x; }
	__twin__ Vec3(T x, T y, T z)
	{
		v[0] = x;
		v[1] = y;
		v[2] = z;
	}

	__twin__ T operator[](int i) const { return v[i]; }
	__twin__ T& operator[](int i) { return v[i]; }

	__twin__ Vec3 operator-() const { return Vec3(-v[0], -v[1], -v[2]); }

	__twin__ Vec3& operator+=(const Vec3& v)
	{
		for (int i = 0; i < 3; ++i) { this->v[i] += v[i]; }
		return *this;
	}
	__twin__ Vec3& operator*=(const Vec3& v)
	{
		for (int i = 0; i < 3; ++i) { this->v[i] *= v[i]; }
		return *this;
	}
	__twin__ Vec3& operator/=(const Vec3& v)
	{
		for (int i = 0; i < 3; ++i) { this->v[i] /= v[i]; }
		return *this;
	}

	__twin__ const T* getPtr() const { return &v[0]; }
};

template <typename T>
__twin__ inline Vec3<T> operator+(const Vec3<T>& v1, const Vec3<T>& v2)
{
	return Vec3<T>(v1[0] + v2[0], v1[1] + v2[1], v1[2] + v2[2]);
}
template <typename T>
__twin__ inline Vec3<T> operator+(const Vec3<T>& v1, float k)
{
	return Vec3<T>(v1[0] + k, v1[1] + k, v1[2] + k);
}
template <typename T>
__twin__ inline Vec3<T> operator+(float k, const Vec3<T>& v2)
{
	return v2 + k;
}

template <typename T>
__twin__ inline Vec3<T> operator-(const Vec3<T>& v1, const Vec3<T>& v2)
{
	return Vec3<T>(v1[0] - v2[0], v1[1] - v2[1], v1[2] - v2[2]);
}
template <typename T>
__twin__ inline Vec3<T> operator-(const Vec3<T>& v1, float k)
{
	return Vec3<T>(v1[0] - k, v1[1] - k, v1[2] - k);
}
template <typename T>
__twin__ inline Vec3<T> operator-(float k, const Vec3<T>& v2)
{
	return Vec3<T>(k - v2[0], k - v2[1], k - v2[2]);
}

template <typename T>
__twin__ inline Vec3<T> operator*(const Vec3<T>& v1, const Vec3<T>& v2)
{
	return Vec3<T>(v1[0] * v2[0], v1[1] * v2[1], v1[2] * v2[2]);
}
template <typename T>
__twin__ inline Vec3<T> operator*(const Vec3<T>& v1, float k)
{
	return Vec3<T>(v1[0] * k, v1[1] * k, v1[2] * k);
}
template <typename T>
__twin__ inline Vec3<T> operator*(float k, const Vec3<T>& v2)
{
	return v2 * k;
}

template <typename T>
__twin__ inline Vec3<T> operator/(const Vec3<T>& v1, const Vec3<T>& v2)
{
	return Vec3<T>(v1[0] / v2[0], v1[1] / v2[1], v1[2] / v2[2]);
}
template <typename T>
__twin__ inline Vec3<T> operator/(const Vec3<T>& v1, float k)
{
	return Vec3<T>(v1[0] / k, v1[1] / k, v1[2] / k);
}
template <typename T>
__twin__ inline Vec3<T> operator/(float k, const Vec3<T>& v2)
{
	return Vec3<T>(k / v2[0], k / v2[1], k / v2[2]);
}

template <typename T>
__twin__ inline Vec3<T> vmin(const Vec3<T>& v1, const Vec3<T>& v2)
{
	return Vec3<T>(Xmin(v1[0], v2[0]), Xmin(v1[1], v2[1]), Xmin(v1[2], v2[2]));
}

template <typename T>
__twin__ inline Vec3<T> vmax(const Vec3<T>& v1, const Vec3<T>& v2)
{
	return Vec3<T>(Xmax(v1[0], v2[0]), Xmax(v1[1], v2[1]), Xmax(v1[2], v2[2]));
}

template <typename T>
__twin__ inline T dot(const Vec3<T>& v1, const Vec3<T>& v2)
{
	return v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2];
}

template <typename T>
__twin__ inline Vec3<T> cross(const Vec3<T>& v1, const Vec3<T>& v2)
{
	return Vec3<T>(v1[1] * v2[2] - v1[2] * v2[1], v1[2] * v2[0] - v1[0] * v2[2],
		v1[0] * v2[1] - v1[1] * v2[0]);
}

template <typename T>
__twin__ inline float length(const Vec3<T>& v) {
	return std::sqrt(dot(v, v));
}

template <typename T>
__twin__ inline float length2(const Vec3<T>& v) {
	return dot(v, v);
}

template <typename T>
__twin__ inline Vec3<T> normalize(const Vec3<T>& v) {
	auto l = length(v);
	if (l != 0)
		return v / length(v);
	return Vec3<T>(0);
}

template <typename T>
__twin__ inline Vec3<T> exp(const Vec3<T>& v) {
	return Vec3<T>(std::exp(v[0]), std::exp(v[1]), std::exp(v[2]));
}

using Vec3f = Vec3<float>;
using Vec3ui = Vec3<uint32_t>;
#pragma endregion Vec3



#pragma region Matrix44
//[comment]
// Implementation of a generic 4x4 Matrix class - Same thing here than with the Vec3 class. It uses
// a template which is maybe less useful than with vectors but it can be used to
// define the coefficients of the matrix to be either floats (the most case) or doubles depending
// on our needs.
//
// To use you can either write: Matrix44<float> m; or: Matrix44f m;
//[/comment]
template<typename T>
class Matrix44
{
public:

	T x[4][4] = { {1,0,0,0},{0,1,0,0},{0,0,1,0},{0,0,0,1} };

	__twin__ Matrix44() {}

	__twin__ Matrix44(T a, T b, T c, T d, T e, T f, T g, T h,
		T i, T j, T k, T l, T m, T n, T o, T p)
	{
		x[0][0] = a;
		x[0][1] = b;
		x[0][2] = c;
		x[0][3] = d;
		x[1][0] = e;
		x[1][1] = f;
		x[1][2] = g;
		x[1][3] = h;
		x[2][0] = i;
		x[2][1] = j;
		x[2][2] = k;
		x[2][3] = l;
		x[3][0] = m;
		x[3][1] = n;
		x[3][2] = o;
		x[3][3] = p;
	}

	__twin__ const T* operator [] (uint8_t i) const { return x[i]; }
	__twin__ T* operator [] (uint8_t i) { return x[i]; }

	// \brief transpose itself
	__twin__ Matrix44& transpose()
	{
		Matrix44 tmp(x[0][0],
			x[1][0],
			x[2][0],
			x[3][0],
			x[0][1],
			x[1][1],
			x[2][1],
			x[3][1],
			x[0][2],
			x[1][2],
			x[2][2],
			x[3][2],
			x[0][3],
			x[1][3],
			x[2][3],
			x[3][3]);
		*this = tmp;

		return *this;
	}

	// \brief set current matrix to its inverse
	__twin__ const Matrix44<T>& invert()
	{
		*this = inverse(*this);
		return *this;
	}

	friend std::ostream& operator << (std::ostream& s, const Matrix44& m)
	{
		std::ios_base::fmtflags oldFlags = s.flags();
		int width = 12; // total with of the displayed number
		s.precision(5); // control the number of displayed decimals
		s.setf(std::ios_base::fixed);

		s << "[" << std::setw(width) << m[0][0] <<
			" " << std::setw(width) << m[0][1] <<
			" " << std::setw(width) << m[0][2] <<
			" " << std::setw(width) << m[0][3] << "\n" <<

			" " << std::setw(width) << m[1][0] <<
			" " << std::setw(width) << m[1][1] <<
			" " << std::setw(width) << m[1][2] <<
			" " << std::setw(width) << m[1][3] << "\n" <<

			" " << std::setw(width) << m[2][0] <<
			" " << std::setw(width) << m[2][1] <<
			" " << std::setw(width) << m[2][2] <<
			" " << std::setw(width) << m[2][3] << "\n" <<

			" " << std::setw(width) << m[3][0] <<
			" " << std::setw(width) << m[3][1] <<
			" " << std::setw(width) << m[3][2] <<
			" " << std::setw(width) << m[3][3] << "]";

		s.flags(oldFlags);
		return s;
	}
};

template<typename T>
__twin__ Matrix44<T> transposed(const Matrix44<T>& x)
{
#if 0
	Matrix44 t;
	for (uint8_t i = 0; i < 4; ++i) {
		for (uint8_t j = 0; j < 4; ++j) {
			t[i][j] = x[j][i];
		}
	}

	return t;
#else
	return Matrix44<T>(x[0][0],
		x[1][0],
		x[2][0],
		x[3][0],
		x[0][1],
		x[1][1],
		x[2][1],
		x[3][1],
		x[0][2],
		x[1][2],
		x[2][2],
		x[3][2],
		x[0][3],
		x[1][3],
		x[2][3],
		x[3][3]);
#endif
}

template<typename T>
__twin__ Matrix44<T> inverse(const Matrix44<T>& x)
{
	int i, j, k;
	Matrix44<T> s;
	Matrix44<T> t(x);

	// Forward elimination
	for (i = 0; i < 3; i++) {
		int pivot = i;

		T pivotsize = t[i][i];

		if (pivotsize < 0)
			pivotsize = -pivotsize;

		for (j = i + 1; j < 4; j++) {
			T tmp = t[j][i];

			if (tmp < 0)
				tmp = -tmp;

			if (tmp > pivotsize) {
				pivot = j;
				pivotsize = tmp;
			}
		}

		if (pivotsize == 0) {
			// Cannot invert singular matrix
			return Matrix44<T>();
		}

		if (pivot != i) {
			for (j = 0; j < 4; j++) {
				T tmp;

				tmp = t[i][j];
				t[i][j] = t[pivot][j];
				t[pivot][j] = tmp;

				tmp = s[i][j];
				s[i][j] = s[pivot][j];
				s[pivot][j] = tmp;
			}
		}

		for (j = i + 1; j < 4; j++) {
			T f = t[j][i] / t[i][i];

			for (k = 0; k < 4; k++) {
				t[j][k] -= f * t[i][k];
				s[j][k] -= f * s[i][k];
			}
		}
	}

	// Backward substitution
	for (i = 3; i >= 0; --i) {
		T f;

		if ((f = t[i][i]) == 0) {
			// Cannot invert singular matrix
			return Matrix44<T>();
		}

		for (j = 0; j < 4; j++) {
			t[i][j] /= f;
			s[i][j] /= f;
		}

		for (j = 0; j < i; j++) {
			f = t[j][i];

			for (k = 0; k < 4; k++) {
				t[j][k] -= f * t[i][k];
				s[j][k] -= f * s[i][k];
			}
		}
	}

	return s;
}

//[comment]
// To make it easier to understand how a matrix multiplication works, the fragment of code
// included within the #if-#else statement, show how this works if you were to iterate
// over the coefficients of the resulting matrix (a). However you will often see this
// multiplication being done using the code contained within the #else-#end statement.
// It is exactly the same as the first fragment only we have litteraly written down
// as a series of operations what would actually result from executing the two for() loops
// contained in the first fragment. It is supposed to be faster, however considering
// matrix multiplicatin is not necessarily that common, this is probably not super
// useful nor really necessary (but nice to have -- and it gives you an example of how
// it can be done, as this how you will this operation implemented in most libraries).
//[/comment]
template<typename T>
__twin__ Matrix44<T> multiply(const Matrix44<T>& a, const Matrix44<T>& b)
{
	Matrix44<T> c;
#if 0
	for (uint8_t i = 0; i < 4; ++i) {
		for (uint8_t j = 0; j < 4; ++j) {
			c[i][j] = a[i][0] * b[0][j] + a[i][1] * b[1][j] +
				a[i][2] * b[2][j] + a[i][3] * b[3][j];
		}
	}
#else
	// A restric qualified pointer (or reference) is basically a promise
	// to the compiler that for the scope of the pointer, the target of the
	// pointer will only be accessed through that pointer (and pointers
	// copied from it.
	const T* __restrict ap = &a.x[0][0];
	const T* __restrict bp = &b.x[0][0];
	T* __restrict cp = &c.x[0][0];

	T a0, a1, a2, a3;

	a0 = ap[0];
	a1 = ap[1];
	a2 = ap[2];
	a3 = ap[3];

	cp[0] = a0 * bp[0] + a1 * bp[4] + a2 * bp[8] + a3 * bp[12];
	cp[1] = a0 * bp[1] + a1 * bp[5] + a2 * bp[9] + a3 * bp[13];
	cp[2] = a0 * bp[2] + a1 * bp[6] + a2 * bp[10] + a3 * bp[14];
	cp[3] = a0 * bp[3] + a1 * bp[7] + a2 * bp[11] + a3 * bp[15];

	a0 = ap[4];
	a1 = ap[5];
	a2 = ap[6];
	a3 = ap[7];

	cp[4] = a0 * bp[0] + a1 * bp[4] + a2 * bp[8] + a3 * bp[12];
	cp[5] = a0 * bp[1] + a1 * bp[5] + a2 * bp[9] + a3 * bp[13];
	cp[6] = a0 * bp[2] + a1 * bp[6] + a2 * bp[10] + a3 * bp[14];
	cp[7] = a0 * bp[3] + a1 * bp[7] + a2 * bp[11] + a3 * bp[15];

	a0 = ap[8];
	a1 = ap[9];
	a2 = ap[10];
	a3 = ap[11];

	cp[8] = a0 * bp[0] + a1 * bp[4] + a2 * bp[8] + a3 * bp[12];
	cp[9] = a0 * bp[1] + a1 * bp[5] + a2 * bp[9] + a3 * bp[13];
	cp[10] = a0 * bp[2] + a1 * bp[6] + a2 * bp[10] + a3 * bp[14];
	cp[11] = a0 * bp[3] + a1 * bp[7] + a2 * bp[11] + a3 * bp[15];

	a0 = ap[12];
	a1 = ap[13];
	a2 = ap[14];
	a3 = ap[15];

	cp[12] = a0 * bp[0] + a1 * bp[4] + a2 * bp[8] + a3 * bp[12];
	cp[13] = a0 * bp[1] + a1 * bp[5] + a2 * bp[9] + a3 * bp[13];
	cp[14] = a0 * bp[2] + a1 * bp[6] + a2 * bp[10] + a3 * bp[14];
	cp[15] = a0 * bp[3] + a1 * bp[7] + a2 * bp[11] + a3 * bp[15];

	return c;
#endif
}

template<typename T>
__twin__ Vec3<T> multVecMatrix(const Vec3<T>& src, const Matrix44<T> x)
{
	Vec3<T> dst;

	T a, b, c, w;

	a = src[0] * x[0][0] + src[1] * x[0][1] + src[2] * x[0][2] + x[0][3];
	b = src[0] * x[1][0] + src[1] * x[1][1] + src[2] * x[1][2] + x[1][3];
	c = src[0] * x[2][0] + src[1] * x[2][1] + src[2] * x[2][2] + x[2][3];
	w = src[0] * x[3][0] + src[1] * x[3][1] + src[2] * x[3][2] + x[3][3];

	dst[0] = a / w;
	dst[1] = b / w;
	dst[2] = c / w;

	return dst;
}

template<typename T>
__twin__ Vec3<T> multDirMatrix(const Vec3<T>& src, const Matrix44<T>& x)
{
	Vec3<T> dst;

	T a, b, c;

	a = src[0] * x[0][0] + src[1] * x[0][1] + src[2] * x[0][2];
	b = src[0] * x[1][0] + src[1] * x[1][1] + src[2] * x[1][2];
	c = src[0] * x[2][0] + src[1] * x[2][1] + src[2] * x[2][2];

	dst[0] = a;
	dst[1] = b;
	dst[2] = c;

	return dst;
}

using Matrix44f = Matrix44<float>;

#pragma endregion Matrix44


#pragma region Quat
template<typename T>
class Quat {
public:
	Quat() = default;
	constexpr Quat(T s, T i, T j, T k)
		: r(s), v(i, j, k) {
	}
	constexpr Quat(T s, Vec3<T> d)
		: r(s), v(d) {
	}
	constexpr Quat<T>& setAxisAngle(const Vec3<T>& axis, T radians) {
		v = normalize(axis) * std::sin(radians / 2);
		r = std::cos(radians / 2);
		return *this;
	}
	T r{ 1 }; // The real part
	Vec3<T> v{ 0,0,0 }; // The imaginary vector
};

template<typename T>
inline Quat<T> normalize(const Quat<T>& q)
{
	Quat<T> p;
	T x = q.v[0];
	T y = q.v[1];
	T z = q.v[2];
	T len = std::sqrt(q.r * q.r + x * x + y * y + z * z);
	if (len != 0) {
		p.r = q.r / len;
		p.v[0] = x / len;
		p.v[1] = y / len;
		p.v[2] = z / len;
	}
	return p;
}

template<typename T>
constexpr inline Quat<T> operator* (const Quat<T>& q1, const Quat<T>& q2) {
	return Quat<T>(
		q1.r * q2.r - dot(q1.v, q2.v), q1.r * q2.v + q1.v * q2.r + cross(q1.v, q2.v));
}

template<typename T>
constexpr Matrix44<T> quatToMatrix44(const Quat<T>& q) {
	auto normal_q = normalize(q);

	T r = normal_q.r;
	T x = normal_q.v[0];
	T y = normal_q.v[1];
	T z = normal_q.v[2];

	T xx = x * x;
	T xy = x * y;
	T xz = x * z;
	T xw = x * r;

	T yy = y * y;
	T yz = y * z;
	T yw = y * r;

	T zz = z * z;
	T zw = z * r;

	return Matrix44<T>(
		1 - 2 * (yy + zz),
		2 * (xy - zw),
		2 * (xz + yw),
		0,

		2 * (xy + zw),
		1 - 2 * (xx + zz),
		2 * (yz - xw),
		0,

		2 * (xz - yw),
		2 * (yz + xw),
		1 - 2 * (xx + yy),
		0,

		0,
		0,
		0,
		1);
}

using Quatf = Quat<float>;

#pragma endregion Quat


__twin__ inline void branchlessONB(const Vec3f& n, Vec3f& b1, Vec3f& b2)
{
	float sign = copysignf(1.0f, n[2]);
	//float sign = std::copysign(1.0f, n[2]);	// n[2] == 0 !!!
	const float a = -1.0f / (sign + n[2]);
	const float b = n[0] * n[1] * a;
	b1 = Vec3f(1.0f + sign * n[0] * n[0] * a, sign * b, -sign * n[0]);
	b2 = Vec3f(b, sign + n[1] * n[1] * a, -n[1]);
}


// transform direction from world to local
__twin__ inline Vec3f worldToLocal(const Vec3f& v, const Vec3f& lx, const Vec3f& ly,
	const Vec3f& lz)
{
	return Vec3f(dot(v, lx), dot(v, ly), dot(v, lz));
}

// transform direction from local to world
__twin__ inline Vec3f local2world(const Vec3f& local, const Vec3f& x, const Vec3f& y, const Vec3f& z)
{
	return x * local[0] + y * local[1] + z * local[2];
}

