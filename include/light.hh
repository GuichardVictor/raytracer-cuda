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

    Light() = default;

    Light(Vector3 _intensity)
        : position{}, intensity{_intensity}
    {}

    Light(Vector3 _position, Vector3 _intensity)
        : position{_position}, intensity{_intensity}
    {}

    virtual float attenuate(const float) const { return 1.0; };

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
    AmbientLight()
        : Light{}
    { type = Light::Type::Ambient; }
    AmbientLight(Vector3 _intensity)
        : Light(_intensity)
    { type = Light::Type::Ambient; }

    float attenuate(const float) const override { return 1.0; }
};

class DirectionalLight : public Light
{
public:
    DirectionalLight()
        : Light{}
    { type = Light::Type::Directional; }
    DirectionalLight(Vector3 _position, Vector3 _intensity)
        : Light(_position, _intensity)
    { type = Light::Type::Directional; }
    
    float attenuate(const float) const override { return 1.0; }
};

class PointLight : public Light
{
public:
    PointLight()
        : Light{}
    { type = Light::Type::Point; }
    PointLight(Vector3 _position, Vector3 _intensity)
        : Light(_position, _intensity)
    { type = Light::Type::Point; }

    float attenuate(const float r) const override { return 1.0 / (r * r); }
};

class SpotLight : public Light
{
public:
    SpotLight()
        : Light{}
    { type = Light::Type::Spot; }
    SpotLight(Vector3 _position, Vector3 _intensity)
        : Light(_position, _intensity)
    { type = Light::Type::Spot; }

    float attenuate(Vector3 Vobj, Vector3 Vlight) const { return Vobj.dot(Vlight); }
};

class AreaLight : public Light
{
public:
    AreaLight()
        : Light{}
    {
      type = Light::Type::Area;
      samples = 2;
      width = 4;
      height = 4;
    }

    AreaLight(Vector3 _position, Vector3 _intensity)
        : Light(_position, _intensity)
    {
      type = Light::Type::Area;
      samples = 2;
      width = 4;
      height = 4;
    }

    float attenuate(const float) const override { return 1.0; }
};
