#pragma once

#include "vector3.hh"
#include "color.hh"

class Texture
{
public:
    __host__ __device__ Texture() {};

    __host__ __device__ Texture(const Color &_color, 
           const float _ka, const float _kd, const float _ks, const float _shinny = 128.0,
           const float _reflectScale = 1.0, const float _transparency = 0.0)
	{
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