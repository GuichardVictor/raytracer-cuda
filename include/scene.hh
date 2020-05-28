#pragma once

#include "shape.hh"
#include "light.hh"
#include "color.hh"
#include "array.hh"

class Scene
{
public:
    __host__ __device__
    Scene() {}

    __host__ __device__
    Scene(Array<Shape>* objs,
          Array<Light>* _lights,
          const Color& background = Color(),
          const AmbientLight& ambient = AmbientLight())
        : objects{objs}, lights(_lights), backgroundColor{background}, ambientLight{ambient}
    {}

    __host__ __device__ void setAmbientLight(const AmbientLight& l) { ambientLight = l; }
    __host__ __device__ void setBackgroundColor(const Color& c) { backgroundColor = c; }

public:
    Array<Shape>* objects;
    Array<Light>* lights;

    Color backgroundColor;
    AmbientLight ambientLight;
};
