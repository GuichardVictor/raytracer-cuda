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
    size_t textureID;
};

class Sphere : public Shape
{
public:
    __host__ __device__ Sphere() {}
    
    __host__ __device__ Sphere(const Vector3 &_center, const float _radius, const size_t _textureID)
      {
          center = _center; 
          radius = _radius;
          textureID = _textureID;
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

    __host__ __device__ Triangle(const Vector3& _v0, const Vector3& _v1, const Vector3& _v2, const size_t _textureID)
        : v0(_v0), v1(_v1), v2(_v2)
    {
        textureID = _textureID;
    }

    __device__ bool intersect(const Ray& ray, float& t, float&);
    __device__ Vector3 getNormal(const Vector3&);
    __device__ virtual void* copy_to_sm(Shape** sm_shapes, int i, void* sm_pointer) override;

public:
    Vector3 v0;
    Vector3 v1;
    Vector3 v2;
};