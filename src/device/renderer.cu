#include "renderer.hh"
#include "shape.hh"
#include "lighting.hh"

#include <stdio.h>

#define MAX_DEPTH 1
#define max_depth_val 7
#define allocating_size (1 << (max_depth_val)) - 1

namespace Lighting
{
    __device__ Color getLightingAll(const Shape& object, const Vector3& point,
                                    const Vector3& normal, const Vector3& view,
                                    Array<Light>* lights,
                                    Array<Shape>* objects,
                                    curandState* r_state)
    {
        Color ambient = object.color;
        Color rayColor = ambient * object.ka;

        // Compute illumination with shadows
        for (size_t i = 0; i < lights->count; i++)
        {
            auto light = lights->list[i];
            float shadowFactor = getShadowFactor(point, *light, objects, r_state);
            rayColor += getLighting(object, point, normal, view, light) * (1.0 - shadowFactor);
        }

        return rayColor;
    }

    __device__ bool getShadow(const Vector3& point, const Light& light,
                            Array<Shape>* objects)
    {
        Vector3 shadowRayDirection = light.position - point;
        shadowRayDirection.normalize();
        Ray shadowRay(point, shadowRayDirection);

        for(size_t i = 0; i < objects->count; i++)
        {
            float t0 = INFINITY;
            float t1 = INFINITY;
            auto object = objects->list[i];
            if(object->intersect(shadowRay, t0, t1))
            {
                return true;
            }
        }
        return false;
    }

    __device__ float getShadowFactor(const Vector3& point, const Light& light,
                                            Array<Shape>* objects, curandState* r_state)
    { 
        if (light.type != Light::Type::Area)
        {
            return getShadow(point, light, objects) ? 1.0 : 0.0;
        }

        int shadowCount = 0;
        Vector3 start(light.position.x - (light.width / 2),
                      light.position.y - (light.height / 2),
                      light.position.z);
        Vector3 step(light.width / light.samples,
                     light.height / light.samples,
                     0);
        
        Vector3 lightSample;
        Vector3 jitter;

        for (int i = 0; i < light.samples; i++)
        {
            for (int j = 0; j < light.samples; j++)
            {
                jitter = Vector3::random(r_state) * step - (step / 2.0f); // Replace 0 0 0 to random vector
                lightSample = Vector3(start.x + (step.x * i) + jitter.x,
                                      start.y + (step.y * i) + jitter.y,
                                      start.z);
                
                Vector3 shadowRayDirection = lightSample - point;
                shadowRayDirection.normalize();
                Ray shadowRay(point, shadowRayDirection);

                bool inShadow = false;
                for (size_t i = 0; i < objects->count; i++)
                {
                    auto obj =objects->list[i];
                    float t0 = INFINITY; float t1 = INFINITY;

                    if (obj->intersect(shadowRay, t0, t1))
                    {
                        inShadow = true;
                        break;
                    }
                }

                if (inShadow)
                    shadowCount++;
            }
        }

        return shadowCount / (float)light.samples;
    }

    __device__ Color getLighting(const Shape& object, const Vector3& point,
            const Vector3& normal, const Vector3& view,
            Light* light)
    {
        Color rayColor;

        // Create diffuse color
        Vector3 N = normal;

        Vector3 L = light->position - point;
        float distance = L.length();
        L.normalize();
        float attenuate = light->attenuate(distance);

        float NdotL = N.dot(L);
        float intensity = fmaxf(0.0f, NdotL);
        Color diffuse = object.color * light->intensity * intensity * attenuate;

        // Create specular color
        Vector3 V = view;
        Vector3 H = L + V;
        H.normalize();

        float shinniness = object.shininess;
        float NdotH = N.dot(H);
        float specularIntensity = powf(fmaxf(0.0f, NdotH), shinniness);
        Color specular = object.color_specular * light->intensity * specularIntensity * attenuate;

        rayColor = diffuse * object.kd + specular * object.ks;   

        return rayColor;
    }

} // namespace Lighting

__device__ Ray Camera::pixelToViewport(const Vector3& pixel)
{
    float invWidth = 1 / (float)width;
    float invHeight = 1 / (float)height;

    float vx = (2 * ((pixel.x + 0.5) * invWidth) - 1) * angle * aspectRatio;
    float vy = (1 - 2 * ((pixel.y + 0.5) * invHeight)) * angle;

    Vector3 rayDirection  = Vector3(vx, vy, pixel.z);
    rayDirection.rotateX(angleX);
    rayDirection.rotateY(angleY);
    rayDirection.rotateZ(angleZ);
    rayDirection.normalize();
    
    return Ray(position, rayDirection);
}

__device__ Vector3 Triangle::getNormal(const Vector3&)
{
    Vector3 v01 = v1 - v0;
    Vector3 v02 = v2 - v0;
    Vector3 N = v01.cross(v02);
    N.normalize();
    return N;
}

__device__ bool Triangle::intersect(const Ray& ray, float& t, float&)
{
    Vector3 v0v1 = v1 - v0;
    Vector3 v0v2 = v2 - v0;

    Vector3 pvec = ray.direction.cross(v0v2);
    float det = v0v1.dot(pvec);
    
    if (fabs(det) < K_EPSILON) { return false; }

    float invDet = 1 / det;

    auto tvec = ray.origin - v0;
    float u = tvec.dot(pvec) * invDet;
    if (u < 0 || u > 1) { return false; }

    auto qvec = tvec.cross(v0v1);
    float v = ray.direction.dot(qvec) * invDet;
    if (v < 0 || u + v > 1) { return false; }

    t = v0v2.dot(qvec) * invDet;

    return true;
}

__device__ bool Sphere::intersect(const Ray& ray, float& t0, float& t1)
{
    Vector3 l = center - ray.origin;
    float tca = l.dot(ray.direction); // Closest approach
    if (tca < 0)
        return false; // Ray intersection behind ray origin
    float d2 = l.dot(l) - tca*tca;
    if (d2 > (radius * radius))
        return false; // Ray doesn't intersect
    float thc = sqrt((radius * radius)- d2); // Closest approach to surface of sphere
    t0 = tca - thc;
    t1 = tca + thc;
    return true;
}

__device__ Vector3 Sphere::getNormal(const Vector3& hitPosition)
{
    return (hitPosition - center) / radius;
}


__device__ Color Renderer::trace(const Ray &ray, curandState* r_state)
{
    // Transform recursive method into interative
    Ray ray_stack[allocating_size] = { Ray(ray.origin, ray.direction) };
    int hit_index_stack[allocating_size] = { -1 }  ;
    Color color_stack[allocating_size] = {Color(0.0f)};
    ray_stack[0] = ray;
    hit_index_stack[0] = 0;
    int threshold_last_depth = 1 << (max_depth_val - 1);
    for(size_t stack_index = 0; stack_index < allocating_size; stack_index++)
    {
        if(hit_index_stack[stack_index / 2 - 1] == -1)
        {
            continue;
        }

        Color rayColor = Color(0.0f);
        Shape* hit = nullptr;
        float tnear = INFINITY;
        Ray new_ray = ray_stack[stack_index];

        int hit_index = -1;
        for (size_t i = 0; i < scene->objects->count; i++)
        {
            float t0 = INFINITY;
            float t1 = INFINITY;
            auto obj = scene->objects->list[i];
            if (obj->intersect(new_ray, t0, t1))
            {
                if (t0 < 0) { t0 = t1; }
                if (t0 < tnear)
                {
                    tnear = t0;
                    hit = obj;
                    hit_index = i;
                }
            }
        }


        hit_index_stack[stack_index] = hit_index;
        if(hit_index == -1)
        {
            continue;
        }

        Vector3 hitPoint = new_ray.origin + new_ray.direction * tnear;
        Vector3 N = hit->getNormal(hitPoint);
        N.normalize();
        Vector3 V = camera->position - hitPoint;
        V.normalize();

        // Compute Color with all lights
        rayColor = Lighting::getLightingAll(*hit, hitPoint, N, V, scene->lights, scene->objects, r_state);

        if (isnan(rayColor.r) || isnan(rayColor.g) || isnan(rayColor.b))
        {
            printf("|ERROR|");
            //printf(" |%f, %f, %f| ", hitPoint.x, hitPoint.y, hitPoint.z);
        }

        color_stack[stack_index] = rayColor;

        
        // not transparent or depth == MAX RAY DEPTH

        if ((hit->transparency <= 0 && hit->reflectivity <= 0) || stack_index > threshold_last_depth)
        {
            hit_index_stack[stack_index] = -1;
            continue;
        }


        float bias = 1e-4f;
        bool inside = false;

        if (new_ray.direction.dot(N) > 0)
        {
            N = -N;
            inside = true;
        }

        Vector3 R = new_ray.direction - N * 2 * new_ray.direction.dot(N);
        R = R + Vector3::random(r_state) * hit->glossiness;
        R.normalize();

        Ray rRay(hitPoint + N * bias, R);

        // Compute Refracted Ray (transmission ray) and Color
        float ni = 1.0;
        float nt = 1.1;
        float nit = ni / nt;
        if (inside) { nit = 1 / nit; }
        float costheta = - N.dot(new_ray.direction);
        float k = 1 - nit * nit * (1 - costheta * costheta);
        Vector3 T = new_ray.direction * nit + N * (nit * costheta - sqrtf(k));
        T = T + Vector3::random(r_state) * hit->glossy_transparency;
        T.normalize();
        Ray refractionRay(hitPoint - N * bias, T);

        //set new rays and color for next 'recursion'
        ray_stack[2 * stack_index + 1] = rRay;
        ray_stack[2 * stack_index + 2] = refractionRay;
    }
    for(int i = allocating_size - 1; i > 0; i -= 2)
    {
        int parent_index = i / 2 - 1;
        if(hit_index_stack[parent_index] != -1)
        {
            Shape* hit = scene->objects->list[hit_index_stack[parent_index]];
            if(hit->transparency <= 0)
            {
                color_stack[parent_index] = color_stack[parent_index] + (color_stack[i - 1] * hit->reflectivity); 
                continue;
            }
            color_stack[parent_index] = (color_stack[i - 1] * hit->reflectivity) + (color_stack[i] * hit->transparency);
        }
    }
    Color ret = color_stack[0];
    return ret;
}
__device__ void simple_scene(Scene** scene_ptr)
{
    *scene_ptr = new Scene();

    auto scene = *scene_ptr;

    size_t object_count = 11;
    Shape** objects = new Shape*[object_count];

    Triangle* t0 = new Triangle( Vector3(0, -3, 0), Vector3(1, -5, 0), Vector3(-1, -5, 0), Color(165, 10, 14), 1.0, 0.5, 0.0, 128.0, 0.0);
    Triangle* t1 = new Triangle( Vector3(0, 4, -40), Vector3(10, -4, -40), Vector3(-10, -4, -40), Color(165, 10, 14), 1.0, 0.5, 0.0, 128.0, 0.0);

    Sphere* ts0 = new Sphere(Vector3(0, 4, 30), 0.2, Color(255), 0.3, 0.8, 0.5, 128.0, 1.0);
    Sphere* ts1 = new Sphere(Vector3(5, -4, 30), 0.2, Color(255), 0.3, 0.8, 0.5, 128.0, 1.0);
    Sphere* ts2 = new Sphere(Vector3(-5, -4, 30), 0.2, Color(255), 0.3, 0.8, 0.5, 128.0, 1.0);

    Sphere* s0 = new Sphere(Vector3(0, -10004, 20), 10000, Color(51, 51, 51), 0.2, 0.5, 0.0, 128.0, 0.0); // Black - Bottom Surface
    Sphere* s1 = new Sphere(Vector3(0, 0, 20), 4, Color(165, 10, 14), 0.3, 0.8, 0.5, 128.0, 0.05, 0.95); // Clear

    s1->glossy_transparency = 0.02f;
    s1->glossiness = 0.05;
    Sphere* s2 = new Sphere( Vector3(5, -1, 15), 2, Color(235, 179, 41), 0.4, 0.6, 0.4, 128.0, 1.0); // Yellow
    s2->glossiness = 0.2;
    Sphere* s3 = new Sphere( Vector3(5, 0, 25), 3, Color(6, 72, 111), 0.3, 0.8, 0.1, 128.0, 1.0);  // Blue
    s3->glossiness = 0.4;
    Sphere* s4 = new Sphere( Vector3(-3.5, -1, 10), 2, Color(8, 88, 56), 0.4, 0.6, 0.5, 64.0, 1.0); // Green
    s4->glossiness = 0.3;
    Sphere* s5 = new Sphere( Vector3(-5.5, 0, 15), 3, Color(51, 51, 51), 0.3, 0.8, 0.25, 32.0, 0.0); // Black

    // Add spheres to scene
    objects[0] = ts0;
    objects[1] = ts1;
    objects[2] = ts2;
    objects[3] = s0;
    objects[4] = s1;
    objects[5] = s2;
    objects[6] = s3;
    objects[7] = s4;
    objects[8] = s5;
    objects[9] = t0;
    objects[10] = t1;

    // Add light to scene
    scene->setAmbientLight(AmbientLight(Vector3(1.0)));
    scene->setBackgroundColor(Color(0.0f));

    size_t light_count = 2;
    Light** lights = new Light*[light_count];

    AreaLight* l0 = new AreaLight( Vector3(0, 20, 35), Vector3(1.4) );
    AreaLight* l1 = new AreaLight( Vector3(20, 20, 35), Vector3(1.8) );
    //DirectionalLight l2 = DirectionalLight( Vector3(0, 20, 35), Vector3(1.4) );
    //PointLight* l3 = new PointLight( Vector3(20, 20, 35), Vector3(2.0) );

    lights[0] = l0;
    lights[1] = l1;

    scene->objects = new Array<Shape>((Shape**)objects, object_count);
    scene->lights = new Array<Light>((Light**)lights, light_count);
}

__global__ void setupScene(Renderer** renderer, Scene** scene, Camera** cam, int width, int height)
{
    simple_scene(scene);
    float fov = 30.0f;
    *cam = new Camera(Vector3(0, 20, -20), width, height, fov);
    (*cam)->angleX = 30 * (M_PI / 180);
    *renderer = new Renderer(width, height, *scene, *cam, 7);
}

__global__ void renderScene(Color* framebuffer, Renderer** renderer_ptr, curandState* random_states)
{
    auto renderer = *renderer_ptr;

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if ((x >= renderer->width) || (y >= renderer->height)) { return; }

    auto index = y * renderer->width + x;

    // Get local random state
    curandState local_rand_state = random_states[index];
    
    // Send a ray through each pixel
    float ds = 1 / (float)renderer->ray_per_pixel;
    for (int s = 0; s < renderer->ray_per_pixel; s++)
    {
        auto r = Vector3::random(&local_rand_state);

        Ray ray = renderer->camera->pixelToViewport(Vector3(x + r.x, y + r.y, 1));

        // Sent pixel for traced ray
        framebuffer[index] += renderer->trace(ray, &local_rand_state) * ds;
    }

    random_states[index] = local_rand_state;
}