#pragma once

#include "vector3.hh"

struct Color
{
  Color() = default;
  Color(float c)
      : r(c), g(c), b(c)
  {}
  Color(float _r, float _g, float _b)
      : r(_r), g(_g), b(_b)
  {}

  Color operator*(float f) const { return Color(r * f, g * f, b * f); }
  Color operator*(const Vector3& f) const { return Color(r * f.x, g * f.y, b * f.z); }
  Color operator*(const Color& c) const { return Color(r * c.r, g * c.g, b * c.b); }
  Color operator+(const Color& c) const { return Color(r + c.r, g + c.g, b + c.b); } 
  Color& operator+=(const Color& c) { r += c.r, g += c.g, b += c.b; return *this; }
  Color& operator*=(const Color& c) { r *= c.r, g *= c.g, b *= c.b; return *this; }

  Color& clamp();

  float r;
  float g;
  float b;
};
