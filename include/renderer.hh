#pragma once

#include "camera.hh"
#include "scene.hh"
#include "color.hh"
#include "ray.hh"
#include "light.hh"
#include "lighting.hh"

class Renderer {

public:
    Renderer(float _width, float _height, Scene _scene, Camera _camera)
        : width(_width), height(_height), scene(_scene), camera(_camera)
    {}

    void render(); 
    void renderAntiAliasing(int samples=16);

private:
    Color trace(const Ray &ray, const int depth);

public:
    int width;
    int height;
    Scene scene;
    Camera camera;

    static int MAX_RAY_DEPTH;
};
