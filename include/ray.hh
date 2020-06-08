#pragma once

#include "vector3.hh"

class Ray
{
public:
    __host__ __device__ Ray(const Vector3& _origin, const Vector3& _direction)
        : origin{_origin}, direction{_direction}
    {}

    Ray() = default;

public:
    Vector3 origin;
    Vector3 direction;
};
