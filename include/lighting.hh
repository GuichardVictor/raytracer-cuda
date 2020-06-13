#pragma once

#include <curand_kernel.h>

#include "light.hh"
#include "vector3.hh"
#include "color.hh"
#include "shape.hh"
#include "array.hh"


namespace Lighting
{
    __device__ Color getLightingAll(const Shape& object, const Vector3& point,
									const Vector3& normal, const Vector3& view,
									Array<Light>* lights,
									Array<Shape>* objects,
									curandState* r_state,
									Texture* textures);
    
    __device__  bool getShadow(const Vector3& point, const Light& light,
                               Array<Shape>* objects);
    __device__  float getShadowFactor(const Vector3& point, const Light& light,
                                 Array<Shape>* objects,
                                 curandState* r_state);


    __device__ Color getLighting(const Shape& object, const Vector3& point,
								const Vector3& normal, const Vector3& view,
								Light* light, Texture* textures);
} // namespace Lighting;