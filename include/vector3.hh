#pragma once

#include <iostream>
#include <cmath>
#include <curand_kernel.h>

class Vector3
{
public:

    __host__ __device__ Vector3() {};
    __host__ __device__ Vector3(float _x)
        : x(_x), y(_x), z(_x)
    {}
    __host__ __device__ Vector3(float _x, float _y, float _z)
        : x(_x), y(_y), z(_z)
    {}

    __host__ __device__ inline Vector3 operator-() const { return Vector3(-x, -y, -z); }

    __host__ __device__ inline Vector3 operator+(const float f) const { return Vector3(x + f, y + f, z + f); }
    __host__ __device__ inline Vector3 operator-(const float f) const { return Vector3(x - f, y - f, z - f); }
    __host__ __device__ inline Vector3 operator*(const float f) const { return Vector3(x * f, y * f, z * f); }
    __host__ __device__ inline Vector3 operator/(const float f) const { return Vector3(x / f, y / f, z / f); }

    __host__ __device__ inline Vector3 operator+(const Vector3 &v) const { return Vector3(x + v.x, y + v.y, z + v.z); }
    __host__ __device__ inline Vector3 operator-(const Vector3 &v) const { return Vector3(x - v.x, y - v.y, z - v.z); }
    __host__ __device__ inline Vector3 operator*(const Vector3 &v)  const { return Vector3(x * v.x, y * v.y, z * v.z); }

    __host__ __device__ inline Vector3& operator+=(const Vector3 &v) { x += v.x, y += v.y, z += v.z; return *this; }
    __host__ __device__ inline Vector3& operator*=(const Vector3 &v) { x *= v.x, y *= v.y, z *= v.z; return *this; }

    __host__ __device__ inline float dot(const Vector3 &v) const;
    __host__ __device__ inline Vector3 cross(const Vector3 &v) const;

    __host__ __device__ inline float length2() const;
    __host__ __device__ inline float length() const;

    __host__ __device__ inline Vector3& normalize();

    __host__ __device__ inline Vector3& rotateX(const float angle);
    __host__ __device__ inline Vector3& rotateY(const float angle);
    __host__ __device__ inline Vector3& rotateZ(const float angle);

    friend std::ostream& operator<<(std::ostream& os, const Vector3& v)
    {
      os << "Vector3(" << v.x << ", " << v.y << ", " << v.z << ")";
      return os;
    }

    __device__ static Vector3 random(curandState* state)
    {
        float rx = curand_uniform(state);
        float ry = curand_uniform(state);
        float rz = curand_uniform(state);

        return Vector3(rx, ry, rz);
    }

    static Vector3 random()
    {
        float rx = ((float) rand() / (RAND_MAX));
        float ry = ((float) rand() / (RAND_MAX));
        float rz = ((float) rand() / (RAND_MAX));
        return Vector3(rx, ry, rz);
    }

public:
    float x;
    float y;
    float z;
};

/* Vector3 Implementation */

__host__ __device__ inline float Vector3::dot(const Vector3 &v) const
{
    return x * v.x + y * v.y + z * v.z;
}

__host__ __device__ inline  Vector3 Vector3::cross(const Vector3 &v) const
{
    return Vector3(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x);
}


__host__ __device__ inline float Vector3::length2() const
{
    return x * x + y * y + z * z;
}

__host__ __device__ inline  float Vector3::length() const
{
    return sqrtf(length2());
}

__host__ __device__ inline  Vector3& Vector3::normalize()
{ 
    float norm2 = length2();
    if(norm2 > 0)
    {
        float invNorm = 1 / sqrtf(norm2);
        x *= invNorm, y *= invNorm, z *= invNorm;
    }
    return *this;
}


__host__ __device__ inline Vector3& Vector3::rotateX(const float angle)
{
    float _y = y;
    float _z = z;

    y = _y * cosf(angle) - _z * sinf(angle);
    z = _y * sinf(angle) + _z * cosf(angle);
    return *this;
}

__host__ __device__ inline Vector3& Vector3::rotateY(const float angle)
{
    float _x = x;
    float _z = z;

    x = _z * sinf(angle) + _x * cosf(angle);
    z = _z * cosf(angle) - _x * sinf(angle);
    return *this;
}

__host__ __device__ inline Vector3& Vector3::rotateZ(const float angle)
{
    float _x = x;

    x = _x * cosf(angle) - y * sinf(angle);
    y = _x * sinf(angle) + y * cosf(angle);
    return *this;
}