#include "renderer.hh"
#include "shape.hh"
#include "lighting.hh"

#include <stdio.h>

__device__ Array<Light>* copy_lights_to_sm(Array<Light>* g_lights, void** sm_pointer)
{
	printf("lights copy\n");
    Array<Light>* sm_lights = (Array<Light>*)(*sm_pointer);
    *sm_pointer = (void*)(sm_lights + 1);

    sm_lights->count = g_lights->count;

	sm_lights->list = (Light**)(*sm_pointer);

	*sm_pointer = (void*)((Light*)(*sm_pointer) + g_lights->count); // sm_pointer points after the list

	for (int i = 0; i < g_lights->count; i++) {
		// sm_pointer is moved by size of target shape, returns pointer to shape.
		sm_lights->list[i] = g_lights->list[i]->copy_to_sm(sm_pointer);
	}
	printf("~lights copy\n");
	return sm_lights;
}

/**
  * Memory layout: sm_shape -> [count, list*, list[0] ... list[n], *(list[0]) ... *(list[0])]
  **/
__device__ Array<Shape>* copy_shapes_to_sm(Array<Shape>* g_shapes, void** sm_pointer)
{
	printf("shapes copy\n");
    Array<Shape>* sm_shapes = (Array<Shape>*)(*sm_pointer);
    *sm_pointer = (void*)(sm_shapes + 1);

    sm_shapes->count = g_shapes->count;

	sm_shapes->list = (Shape**)(*sm_pointer);

	//*sm_pointer = (void*)(sizeof(Shape*) * g_shapes->count); // sm_pointer points after the list
	*sm_pointer = (void*)((Shape*)(*sm_pointer) + g_shapes->count);

	for (int i = 0; i < g_shapes->count; i++) {
		// sm_pointer is moved by size of target shape, returns pointer to shape.
		sm_shapes->list[i] = g_shapes->list[i]->copy_to_sm(sm_pointer);
	}
	printf("~shapes copy\n");
	return sm_shapes;
}

__device__ Scene* copy_scene_to_sm(Scene* g_scene, void** sm_pointer)
{
	printf("scene copy\n");
    Scene* sm_scene = (Scene*)(*sm_pointer);
    *sm_pointer = (void*)(sm_scene + 1);

    sm_scene->backgroundColor = g_scene->backgroundColor;
    sm_scene->ambientLight = g_scene->ambientLight;

    sm_scene->objects = copy_shapes_to_sm(g_scene->objects, sm_pointer);
    sm_scene->lights = copy_lights_to_sm(g_scene->lights, sm_pointer);

	printf("~scene copy\n");
    return sm_scene;
}

__device__ Camera* copy_camera_to_sm(Camera* g_camera, void** sm_pointer)
{
	printf("camera copy\n");
	Camera* sm_camera = (Camera*)(*sm_pointer);
	*sm_pointer = (void*)(sm_camera + 1);

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
	return sm_camera;
}

#include <iostream>

__device__ Renderer* copy_renderer_to_sm(Renderer* g_renderer, void** sm_pointer)
{
	printf("renderer copy\n");

    Renderer* sm_renderer = (Renderer*)(*sm_pointer);
    *sm_pointer = (void*)(sm_renderer + 1);
	printf("here 1\n");

    sm_renderer->width = g_renderer->width;
    sm_renderer->height = g_renderer->height;
	printf("here 2\n");

	sm_renderer->scene = copy_scene_to_sm(g_renderer->scene, sm_pointer);
	printf("here 3\n");

	sm_renderer->camera = copy_camera_to_sm(g_renderer->camera, sm_pointer);
	printf("here 4\n");

    sm_renderer->max_ray_depth = g_renderer->max_ray_depth;
    sm_renderer->ray_per_pixel = g_renderer->ray_per_pixel;

	printf("~renderer copy\n");
	return sm_renderer;
}
