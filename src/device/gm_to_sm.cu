#include "renderer.hh"
#include "shape.hh"
#include "lighting.hh"

#include <stdio.h>

// Lights =====================

__device__ void* AmbientLight::copy_to_sm(Light** sm_lights, int i, void* sm_pointer)
{
	AmbientLight* light = (AmbientLight*)sm_pointer;
    sm_pointer = (void*)(light + 1);

    memcpy(light, this, sizeof(AmbientLight));

    sm_lights[i] = light;
    return sm_pointer;
}

__device__ void* DirectionalLight::copy_to_sm(Light** sm_lights, int i, void* sm_pointer)
{
	DirectionalLight* light = (DirectionalLight*)sm_pointer;
    sm_pointer = (void*)(light + 1);

    memcpy(light, this, sizeof(DirectionalLight));

    sm_lights[i] = light;
    return sm_pointer;
}

__device__ void* PointLight::copy_to_sm(Light** sm_lights, int i, void* sm_pointer)
{
	PointLight* light = (PointLight*)sm_pointer;
    sm_pointer = (void*)(light + 1);

    memcpy(light, this, sizeof(PointLight));

    sm_lights[i] = light;
    return sm_pointer;
}

__device__ void* SpotLight::copy_to_sm(Light** sm_lights, int i, void* sm_pointer)
{
	SpotLight* light = (SpotLight*)sm_pointer;
    sm_pointer = (void*)(light + 1);

    memcpy(light, this, sizeof(SpotLight));

    sm_lights[i] = light;
    return sm_pointer;
}

__device__ void* AreaLight::copy_to_sm(Light** sm_lights, int i, void* sm_pointer)
{
	AreaLight* light = (AreaLight*)sm_pointer;
    sm_pointer = (void*)(light + 1);

    memcpy(light, this, sizeof(AreaLight));

    sm_lights[i] = light;
    return sm_pointer;
}


// Shapes =====================


__device__ void* Triangle::copy_to_sm(Shape** sm_shapes, int i, void* sm_pointer)
{
    Triangle* triangle = (Triangle*)sm_pointer;
    sm_pointer = (void*)(triangle + 1);

    memcpy(triangle, this, sizeof(Triangle));

    sm_shapes[i] = triangle;
    return sm_pointer;
}

__device__ void* Sphere::copy_to_sm(Shape** sm_shapes, int i, void* sm_pointer)
{
    Sphere* sphere = (Sphere*)sm_pointer;
    sm_pointer = (void*)(sphere + 1);

    memcpy(sphere, this, sizeof(Sphere));

    sm_shapes[i] = sphere;
    return sm_pointer;
}


// ============================


__device__ void* copy_lights_to_sm(Scene* sm_scene, Array<Light>* g_lights, void* sm_pointer)
{
	printf("lights copy\n");
    Array<Light>* sm_lights = (Array<Light>*)sm_pointer;
    sm_pointer = (void*)(sm_lights + 1);

    sm_lights->count = g_lights->count;

	sm_lights->list = (Light**)sm_pointer;
	sm_pointer = (void*)(sm_lights->list + g_lights->count);

	for (int i = 0; i < g_lights->count; i++) {
		sm_pointer = g_lights->list[i]->copy_to_sm(sm_lights->list, i, sm_pointer);
	}
	printf("~lights copy\n");
	sm_scene->lights = sm_lights;
	return sm_pointer;
}

/**
  * Memory layout: sm_shape -> [count, list*, list[0] ... list[n], *(list[0]) ... *(list[0])]
  **/
__device__ void* copy_shapes_to_sm(Scene* sm_scene, Array<Shape>* g_shapes, void* sm_pointer)
{
	printf("shapes copy\n");
    Array<Shape>* sm_shapes = (Array<Shape>*)sm_pointer;
    sm_pointer = (void*)(sm_shapes + 1);

    sm_shapes->count = g_shapes->count;

	sm_shapes->list = (Shape**)sm_pointer;
	sm_pointer = (void*)(sm_shapes->list + g_shapes->count);

	float t0 = INFINITY;
	float t1 = INFINITY;
	for (int i = 0; i < g_shapes->count; i++) {
		sm_pointer = g_shapes->list[i]->copy_to_sm(sm_shapes->list, i, sm_pointer);
	}
	printf("~shapes copy\n");
	sm_scene->objects = sm_shapes;
	return sm_pointer;
}

__device__ void* copy_scene_to_sm(Renderer* sm_renderer, Scene* g_scene, void* sm_pointer)
{
	printf("scene copy\n");
    Scene* sm_scene = (Scene*)sm_pointer;
    sm_pointer = (void*)(sm_scene + 1);

    sm_scene->backgroundColor = g_scene->backgroundColor;
    sm_scene->ambientLight = g_scene->ambientLight;

	sm_pointer = copy_shapes_to_sm(sm_scene, g_scene->objects, sm_pointer);
	sm_pointer = copy_lights_to_sm(sm_scene, g_scene->lights, sm_pointer);

	printf("~scene copy\n");
	sm_renderer->scene = sm_scene;
    return sm_pointer;
}

__device__ void* copy_camera_to_sm(Renderer* sm_renderer, Camera* g_camera, void* sm_pointer)
{
	printf("camera copy\n");
	Camera* sm_camera = (Camera*)sm_pointer;
	sm_pointer = (void*)(sm_camera + 1);

	sm_camera->position = g_camera->position;
	sm_camera->width = g_camera->width;
	sm_camera->height = g_camera->height;

	sm_camera->fov = g_camera->fov;
	sm_camera->aspectRatio = g_camera->aspectRatio;
	sm_camera->angle = g_camera->angle;

	sm_camera->angleX = g_camera->angleX;
	sm_camera->angleY = g_camera->angleY;
	sm_camera->angleZ = g_camera->angleZ;

	printf("~camera copy\n");
	sm_renderer->camera = sm_camera;
	return sm_pointer;
}

#include <iostream>

__device__ Renderer* copy_renderer_to_sm(Renderer* g_renderer, void* sm_pointer)
{
	printf("renderer copy\n");

    Renderer* sm_renderer = (Renderer*)sm_pointer;
    sm_pointer = (void*)(sm_renderer + 1);
	printf("here 1\n");

    sm_renderer->width = g_renderer->width;
    sm_renderer->height = g_renderer->height;
	printf("here 2\n");

	sm_pointer = copy_scene_to_sm(sm_renderer, g_renderer->scene, sm_pointer);
	printf("here 3\n");

	sm_pointer = copy_camera_to_sm(sm_renderer, g_renderer->camera, sm_pointer);
	printf("here 4\n");

    sm_renderer->max_ray_depth = g_renderer->max_ray_depth;
    sm_renderer->ray_per_pixel = g_renderer->ray_per_pixel;

	printf("~renderer copy\n");
	return sm_renderer;
}
