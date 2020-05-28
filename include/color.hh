#pragma once

#include <algorithm>

#include "vector3.hh"

class Color
{
public:
  __host__ __device__ Color() {r = 0.0f, g = 0.0f, b = 0.0f;}
  __host__ __device__ Color(float c)
      : r(c), g(c), b(c)
  {}
  __host__ __device__ Color(float _r, float _g, float _b)
      : r(_r), g(_g), b(_b)
  {}

  __host__ __device__ inline Color operator*(float f) const { return Color(r * f, g * f, b * f); }
  __host__ __device__ inline Color operator*(const Vector3& f) const { return Color(r * f.x, g * f.y, b * f.z); }
  __host__ __device__ inline Color operator*(const Color& c) const { return Color(r * c.r, g * c.g, b * c.b); }
  __host__ __device__ inline Color operator+(const Color& c) const { return Color(r + c.r, g + c.g, b + c.b); } 
  __host__ __device__ inline Color& operator+=(const Color& c) { r += c.r, g += c.g, b += c.b; return *this; }
  __host__ __device__ inline Color& operator*=(const Color& c) { r *= c.r, g *= c.g, b *= c.b; return *this; }

  __host__ __device__ inline Color& clamp(const float min=0, const float max=255);

  float r;
  float g;
  float b;
};

__host__ __device__ inline Color& Color::clamp(const float min, const float max)
{
    r = (r < min) ? min : (r > max) ? max : r;
    g = (g < min) ? min : (g > max) ? max : g;
    b = (b < min) ? min : (b > max) ? max : b;

    return *this;
}