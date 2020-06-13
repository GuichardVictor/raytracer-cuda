#pragma once

#include "vector3.hh"
#include "ray.hh"

#ifndef M_PI
#define M_PI 3.141693f
#endif


class Camera
{

public:
    __host__ __device__ Camera(const Vector3& _position, int _width, int _height, float _fov)
    {
        position = _position;
        width = _width;
        height = _height;
        fov = _fov;
        aspectRatio = width / float(height);
        angle = tanf(0.5f * fov * M_PI / 180.0f);
        angleX = 0;
        angleY = 0;
        angleZ = 0;
    }

    __device__ Ray pixelToViewport(const Vector3& pixel);

public:
    Vector3 position;
    int width;
    int height;

    float fov;
    float aspectRatio;
    float angle;

    float angleX, angleY, angleZ;
};
