#pragma once

#include "vector3.hh"
#include "color.hh"
#include "ray.hh"

#define K_EPSILON 0.00001

class Shape
{
public:
    virtual bool intersect(const Ray &ray, float &to, float &t1) = 0;
    virtual Vector3 getNormal(const Vector3&) { return Vector3(); };

public:
    Vector3 center;            // Position
    Color color;               // Surface Diffuse Color
    Color color_specular;      // Surface Specular Color
    float ka, kd, ks;          // Ambient, Diffuse, Specular Coefficents
    float shininess;
    float reflectivity;        // Reflectivity of material [0, 1]
    float transparency;        // Transparency of material [0, 1]
    float glossiness;          // Strength of glossy reflections
    float glossy_transparency; // Strength of glossy transparency
};


class Sphere : public Shape
{
public:
    Sphere(const Vector3 &_center, const float _radius, const Color &_color, 
           const float _ka, const float _kd, const float _ks, const float _shinny = 128.0,
           const float _reflectScale = 1.0, const float _transparency = 0.0)
        : radius{_radius}
      { 
          center = _center;
          color = _color;
          color_specular = Color(255);
          ka = _ka;
          kd = _kd;
          ks = _ks;
          shininess = _shinny;
          reflectivity = _reflectScale;
          transparency = _transparency;
          glossiness = 0;
          glossy_transparency = 0;
      }
    
    // Compute a ray-sphere intersection using the geometric method
    bool intersect(const Ray& ray, float &t0, float &t1) override;
    Vector3 getNormal(const Vector3 &hitPosition) override;

public:
    float radius;
};


class Triangle : public Shape
{
public:
    Triangle(const Vector3 &_v0, const Vector3 &_v1, const Vector3 &_v2, const Color &_color,
             const float _ka, const float _kd, const float _ks, const float _shinny = 128.0,
             const float _reflectScale = 1.0, const float _transparency = 0.0)
        : v0(_v0), v1(_v1), v2(_v2)
    {
        center = {};
        color = _color;
        color_specular = Color(255);
        ka = _ka;
        kd = _kd;
        ks = _ks;
        shininess = _shinny;
        reflectivity = _reflectScale;
        transparency = _transparency;
        glossiness = 0;
        glossy_transparency = 0;
    }

    bool intersect(const Ray& ray, float& t, float&);
    Vector3 getNormal(const Vector3&);

public:
    Vector3 v0;
    Vector3 v1;
    Vector3 v2;
};
