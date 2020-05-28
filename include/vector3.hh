#pragma once

#include <iostream>
#include <cmath>

class Vector3
{
public:

    Vector3() = default;
    Vector3(float _x)
        : x(_x), y(_x), z(_x)
    {}
    Vector3(float _x, float _y, float _z)
        : x(_x), y(_y), z(_z)
    {}

    Vector3 operator-() const { return Vector3(-x, -y, -z); }

    Vector3 operator+(const float f) const { return Vector3(x + f, y + f, z + f); }
    Vector3 operator-(const float f) const { return Vector3(x - f, y - f, z - f); }
    Vector3 operator*(const float f) const { return Vector3(x * f, y * f, z * f); }
    Vector3 operator/(const float f) const { return Vector3(x / f, y / f, z / f); }

    Vector3 operator+(const Vector3 &v) const { return Vector3(x + v.x, y + v.y, z + v.z); }
    Vector3 operator-(const Vector3 &v) const { return Vector3(x - v.x, y - v.y, z - v.z); }
    Vector3 operator*(const Vector3 &v)  const { return Vector3(x * v.x, y * v.y, z * v.z); }

    Vector3& operator+=(const Vector3 &v) { x += v.x, y += v.y, z += v.z; return *this; }
    Vector3& operator*=(const Vector3 &v) { x *= v.x, y *= v.y, z *= v.z; return *this; }

    float dot(const Vector3 &v) const;
    Vector3 cross(const Vector3 &v) const;

    float length2() const;
    float length() const;

    Vector3& normalize();

    Vector3& rotateX(const float angle);
    Vector3& rotateY(const float angle);
    Vector3& rotateZ(const float angle);

    friend std::ostream& operator<<(std::ostream& os, const Vector3& v)
    {
      os << "Vector3(" << v.x << ", " << v.y << ", " << v.z << ")";
      return os;
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
