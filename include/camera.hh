#pragma once

#include "vector3.hh"

class Camera
{

public:
    Camera(const Vector3& _position, int _width, int _height, float _fov);
    
    Vector3 pixelToViewport(const Vector3& pixel); 

public:
    Vector3 position;
    int width;
    int height;

    float fov;
    float aspectRatio;
    float angle;

    float angleX, angleY, angleZ;
};
