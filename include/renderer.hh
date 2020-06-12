#pragma once

#include <curand_kernel.h>

#include "camera.hh"
#include "scene.hh"
#include "color.hh"
#include "ray.hh"
#include "light.hh"

class Renderer {

public:
    __host__ __device__
    Renderer(int _width, int _height, Scene* _scene, Camera* _camera, int _max_ray_depth=5, int _ray_per_pixel=10)
        : width(_width), height(_height), scene(_scene), camera(_camera), max_ray_depth(_max_ray_depth), ray_per_pixel(_ray_per_pixel)
    {}

    __device__ Color trace(const Ray &ray, int depth, curandState* r_state);

public:
    int width;
    int height;
    Scene* scene;
    Camera* camera;

    int max_ray_depth;
    int ray_per_pixel;
};


__device__ Renderer* copy_renderer_to_sm(Renderer* g_renderer, void* sm_pointer);
__global__ void free_scene(Renderer** Renderer_ptr);
__global__ void renderScene(Color* framebuffer, Renderer** g_renderer, curandState* random_states);
__global__ void setupScene(Renderer** renderer, Scene** scene, Camera** cam, int width, int height, size_t* sm_memSize, curandState* r_state);
