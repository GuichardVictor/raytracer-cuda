#pragma once

#include "vector3.hh"

class Light {
public:

    enum Type
    {
        Default,
        Ambient,
        Directional,
        Point,
        Spot,
        Area
    };

    __host__ __device__ Light() {}

    __host__ __device__ Light(const Vector3& _intensity)
        : position{}, intensity{_intensity}
    {}

    __host__ __device__ Light(const Vector3& _position, const Vector3& _intensity)
        : position{_position}, intensity{_intensity}
    {}

    __device__ virtual float attenuate(const float) const { return 1.0; };

    __device__ virtual void* copy_to_sm(Light** sm_lights, int i, void* sm_pointer) = 0;

public:
    Vector3 position;
    Vector3 intensity;
    Type type;

    int samples;
    float width;
    float height;
};

class AmbientLight : public Light
{
public:
    __host__ __device__ AmbientLight()
        : Light{}
    { type = Light::Type::Ambient; }

    __host__ __device__ AmbientLight(const Vector3& _intensity)
        : Light(_intensity)
    { type = Light::Type::Ambient; }

    __device__ float attenuate(const float) const override { return 1.0f; }

    __device__ virtual void* copy_to_sm(Light** sm_lights, int i, void* sm_pointer) override;
};

class DirectionalLight : public Light
{
public:
    __host__ __device__ DirectionalLight()
        : Light{}
    { type = Light::Type::Directional; }
    
    __host__ __device__ DirectionalLight(const Vector3& _position, const Vector3& _intensity)
        : Light(_position, _intensity)
    { type = Light::Type::Directional; }
    
    __device__ float attenuate(const float) const override { return 1.0; }

    __device__ virtual void* copy_to_sm(Light** sm_lights, int i, void* sm_pointer) override;
};

class PointLight : public Light
{
public:
    __host__ __device__ PointLight()
        : Light{}
    { type = Light::Type::Point; }

    __host__ __device__ PointLight(const Vector3& _position, const Vector3& _intensity)
        : Light(_position, _intensity)
    { type = Light::Type::Point; }

    __device__ float attenuate(const float r) const override { return 1.0 / (r * r); }

    __device__ virtual void* copy_to_sm(Light** sm_lights, int i, void* sm_pointer) override;
};

class SpotLight : public Light
{
public:
    __host__ __device__ SpotLight()
        : Light{}
    { type = Light::Type::Spot; }

    __host__ __device__ SpotLight(const Vector3& _position, const Vector3& _intensity)
        : Light(_position, _intensity)
    { type = Light::Type::Spot; }

    __device__ float attenuate(const float) const override { return 1.0; }
    __device__ float attenuate(const Vector3& Vobj, const Vector3& Vlight) const { return Vobj.dot(Vlight); }

    __device__ virtual void* copy_to_sm(Light** sm_lights, int i, void* sm_pointer) override;
};

class AreaLight : public Light
{
public:
    __host__ __device__ AreaLight()
        : Light{}
    {
      type = Light::Type::Area;
      samples = 2;
      width = 4;
      height = 4;
    }

    __host__ __device__ AreaLight(Vector3 _position, Vector3 _intensity)
        : Light(_position, _intensity)
    {
      type = Light::Type::Area;
      samples = 2;
      width = 4;
      height = 4;
    }

    __device__ float attenuate(const float) const override { return 1.0; }

    __device__ virtual void* copy_to_sm(Light** sm_lights, int i, void* sm_pointer) override;
};