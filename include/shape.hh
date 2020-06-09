#pragma once

#include "vector3.hh"
#include "color.hh"
#include "ray.hh"

#define K_EPSILON 0.00001f

class Shape
{
public:
    __device__ virtual bool intersect(const Ray& ray, float& t0, float& t1) = 0;
    __device__ virtual Vector3 getNormal(const Vector3&) { return Vector3(); };
    __device__ virtual void* copy_to_sm(Shape** sm_shapes, int i, void* sm_pointer) = 0;

public:
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
    __host__ __device__ Sphere() {}
    
    __host__ __device__ Sphere(const Vector3 &_center, const float _radius, const Color &_color, 
           const float _ka, const float _kd, const float _ks, const float _shinny = 128.0,
           const float _reflectScale = 1.0, const float _transparency = 0.0)
      {
          center = _center; 
          radius = _radius;
          color = _color;
          color_specular = Color(255.0f);
          ka = _ka;
          kd = _kd;
          ks = _ks;
          shininess = _shinny;
          reflectivity = _reflectScale;
          transparency = _transparency;
          glossiness = 0.0f;
          glossy_transparency = 0.0f;
      }
    
    // Compute a ray-sphere intersection using the geometric method
    __device__ virtual bool intersect(const Ray& ray, float& t0, float& t1) override;
    __device__ Vector3 getNormal(const Vector3 &hitPosition) override;
    __device__ virtual void* copy_to_sm(Shape**, int i, void* sm_pointer) override;

public:
    Vector3 center;
    float radius;
};

class Triangle : public Shape
{
public:
    __host__ __device__ Triangle() {}

    __host__ __device__ Triangle(const Vector3 &_v0, const Vector3 &_v1, const Vector3 &_v2, const Color &_color,
             const float _ka, const float _kd, const float _ks, const float _shinny = 128.0,
             const float _reflectScale = 1.0, const float _transparency = 0.0)
        : v0(_v0), v1(_v1), v2(_v2)
    {
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

    __device__ bool intersect(const Ray& ray, float& t, float&);
    __device__ Vector3 getNormal(const Vector3&);
    __device__ virtual void* copy_to_sm(Shape** sm_shapes, int i, void* sm_pointer) override;

public:
    Vector3 v0;
    Vector3 v1;
    Vector3 v2;
};