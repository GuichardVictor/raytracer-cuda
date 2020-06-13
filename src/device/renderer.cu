#include "renderer.hh"
#include "shape.hh"
#include "lighting.hh"

#include <stdio.h>

namespace Lighting
{
    __device__ Color getLightingAll(const Shape& object, const Vector3& point,
                                    const Vector3& normal, const Vector3& view,
                                    Array<Light>* lights,
                                    Array<Shape>* objects,
                                    curandState* r_state,
                                    Texture* textures)
    {
        Color ambient = textures[object.textureID].color;
        Color rayColor = ambient * textures[object.textureID].ka;

        // Compute illumination with shadows
        for (size_t i = 0; i < lights->count; i++)
        {
            auto light = lights->list[i];
            float shadowFactor = getShadowFactor(point, *light, objects, r_state);
            rayColor += getLighting(object, point, normal, view, light, textures) * (1.0 - shadowFactor);
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
            Light* light, Texture* textures)
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
        Color diffuse = textures[object.textureID].color * light->intensity * intensity * attenuate;

        // Create specular color
        Vector3 V = view;
        Vector3 H = L + V;
        H.normalize();

        float shinniness = textures[object.textureID].shininess;
        float NdotH = N.dot(H);
        float specularIntensity = powf(fmaxf(0.0f, NdotH), shinniness);
        Color specular = textures[object.textureID].color_specular * light->intensity * specularIntensity * attenuate;

        rayColor = diffuse * textures[object.textureID].kd + specular * textures[object.textureID].ks;   

        return rayColor;
    }

} // namespace Lighting

__device__ Ray Camera::pixelToViewport(const Vector3& pixel)
{
    float invWidth = 1.f / (float)width;
    float invHeight = 1.f / (float)height;

    float vx = (2.f * ((pixel.x + 0.5f) * invWidth) - 1.f) * angle * aspectRatio;
    float vy = (1.f - 2.f * ((pixel.y + 0.5f) * invHeight)) * angle;

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
    float thc = sqrtf((radius * radius)- d2); // Closest approach to surface of sphere
    t0 = tca - thc;
    t1 = tca + thc;
    return true;
}

__device__ Vector3 Sphere::getNormal(const Vector3& hitPosition)
{
    return (hitPosition - center) / radius;
}

__device__ Color Renderer::trace(const Ray &ray, int depth, curandState* r_state, Texture* textures)
{
    // Transform recursive method into interative
    Color rayColor = Color(0.0f);
    Shape* hit = nullptr;
    float tnear = INFINITY;

    for (size_t i = 0; i < scene->objects->count; i++)
    {
        float t0 = INFINITY;
        float t1 = INFINITY;
        auto obj = scene->objects->list[i];
        if (obj->intersect(ray, t0, t1))
        {
            if (t0 < 0) { t0 = t1; }
            if (t0 < tnear)
            {
                tnear = t0;
                hit = obj;
            }
        }
    }

    if (hit == nullptr)
    {
        return (depth < 1) ? Color(0.0f) : Color(0.0f);
    }

    Vector3 hitPoint = ray.origin + ray.direction * tnear;
    Vector3 N = hit->getNormal(hitPoint);
    N.normalize();
    Vector3 V = camera->position - hitPoint;
    V.normalize();

    // Compute Color with all lights
    rayColor = Lighting::getLightingAll(*hit, hitPoint, N, V, scene->lights, scene->objects, r_state, textures);

    float bias = 1e-4f;
    bool inside = false;

    if (ray.direction.dot(N) > 0)
    {
        N = -N;
        inside = true;
    }

    // not transparent or depth == MAX RAY DEPTH
    if ((textures[hit->textureID].transparency <= 0 && textures[hit->textureID].reflectivity <= 0) || depth >= max_ray_depth)
    {
        return rayColor;
    }

    // Compute Reflection Ray and Color 
    Vector3 R = ray.direction - N * 2 * ray.direction.dot(N);
    R = R + Vector3::random(r_state) * textures[hit->textureID].glossiness;
    R.normalize();

    Ray rRay(hitPoint + N * bias, R);
    Color reflectionColor = trace(rRay, depth + 1, r_state, textures); //* VdotR;
    Color refractionColor = Color();

    if (textures[hit->textureID].transparency <= 0)
    {
        return rayColor + (reflectionColor * textures[hit->textureID].reflectivity);
    }

    // Compute Refracted Ray (transmission ray) and Color
    float ni = 1.0;
    float nt = 1.1;
    float nit = ni / nt;
    if (inside) { nit = 1 / nit; }
    float costheta = - N.dot(ray.direction);
    float k = 1 - nit * nit * (1 - costheta * costheta);
    Vector3 T = ray.direction * nit + N * (nit * costheta - sqrtf(k));
    T = T + Vector3::random(r_state) * textures[hit->textureID].glossy_transparency;
    T.normalize();

    Ray refractionRay(hitPoint - N * bias, T);
    refractionColor = trace(refractionRay, depth + 1, r_state, textures);
    return (reflectionColor * textures[hit->textureID].reflectivity) + (refractionColor * textures[hit->textureID].transparency);
}

__device__ size_t random_sphere_scene(Scene** scene_ptr, int nb_sphere, curandState* r_state)
{
    size_t sm_memSize = 0;
    *scene_ptr = new Scene();
    sm_memSize += sizeof(Scene);

    auto scene = *scene_ptr;

    size_t object_count = 1 + nb_sphere;
    Shape** objects = new Shape*[object_count];
    sm_memSize += sizeof(Shape*) * object_count;

    scene->textures = new Texture[object_count];
    Texture* textures = scene->textures;

    for(int i = 0; i < nb_sphere; i++)
    {
        float r = curand_uniform(r_state);
        float g = curand_uniform(r_state);
        float b = curand_uniform(r_state);
        float coef = curand_uniform(r_state);
        float x = curand_uniform(r_state);
        float y = curand_uniform(r_state);
        float z = curand_uniform(r_state);
        Vector3 center{-11 + x * 22, -3 + y * 10, 5 + z * 15};
        Vector3 add = Vector3::random(r_state);
        textures[i] = Texture(Color(r * 255, g * 255, b * 255), 0.2, 0.5, 0.0, 128, (i % 5 == 0));
        objects[i] =  new Sphere(center, coef, i);
    }
    
    textures[object_count - 1] = Texture(Color(51, 51, 51), 0.2, 0.5, 0.0, 128.0, 0.0);
    Sphere* s0 = new Sphere(Vector3(0, -10004, 20), 10000, object_count - 1); // Black - Bottom Surface
    objects[object_count - 1] = s0;

    sm_memSize += sizeof(Sphere) * object_count;

    // Add light to scene
    scene->setAmbientLight(AmbientLight(Vector3(1.0)));
    scene->setBackgroundColor(Color(0.0f));

    size_t light_count = 2;
    Light** lights = new Light*[light_count];
    
    sm_memSize += sizeof(Light) * light_count;

    AreaLight* l0 = new AreaLight( Vector3(0, 20, 35), Vector3(1.4) );
    AreaLight* l1 = new AreaLight( Vector3(20, 20, 35), Vector3(1.8) );

    lights[0] = l0;
    lights[1] = l1;

   

    scene->objects = new Array<Shape>((Shape**)objects, object_count);
    scene->lights = new Array<Light>((Light**)lights, light_count);

    sm_memSize += sizeof(Array<Shape>);
    sm_memSize += sizeof(Array<Light>);

    return sm_memSize;
}

__device__ size_t simple_scene(Scene** scene_ptr)
{
    size_t sm_memSize = 0;

    *scene_ptr = new Scene();
    sm_memSize += sizeof(Scene);

    auto scene = *scene_ptr;

    size_t object_count = 11;

    scene->textures = new Texture[object_count];
    Texture* textures = scene->textures;

    Shape** objects = new Shape*[object_count];
    sm_memSize += sizeof(Shape*) * object_count;

    textures[0] = Texture(Color(165, 10, 14), 1.0, 0.5, 0.0, 128.0, 0.0); 
    Triangle* t0 = new Triangle( Vector3(0, -3, 0), Vector3(1, -5, 0), Vector3(-1, -5, 0), 0);
    textures[1] = Texture( Color(165, 10, 14), 1.0, 0.5, 0.0, 128.0, 0.0);
    Triangle* t1 = new Triangle( Vector3(0, 4, -40), Vector3(10, -4, -40), Vector3(-10, -4, -40), 1);
    sm_memSize += sizeof(Triangle) * 2;

    textures[2] = Texture(Color(255), 0.3, 0.8, 0.5, 128.0, 1.0);;
    Sphere* ts0 = new Sphere(Vector3(0, 4, 30), 0.2, 2);

    textures[3] = Texture(Color(255), 0.3, 0.8, 0.5, 128.0, 1.0);
    Sphere* ts1 = new Sphere(Vector3(5, -4, 30), 0.2, 3);

    textures[4] = Texture(Color(255), 0.3, 0.8, 0.5, 128.0, 1.0);
    Sphere* ts2 = new Sphere(Vector3(-5, -4, 30), 0.2, 4);
    sm_memSize += sizeof(Sphere) * 3;

    textures[5] = Texture(Color(51, 51, 51), 0.2, 0.5, 0.0, 128.0, 0.0);
    Sphere* s0 = new Sphere(Vector3(0, -10004, 20), 10000, 5); // Black - Bottom Surface

    textures[6] = Texture(Color(165, 10, 14), 0.3, 0.8, 0.5, 128.0, 0.05, 0.95);
    Sphere* s1 = new Sphere(Vector3(0, 0, 20), 4, 6); // Clear
    sm_memSize += sizeof(Sphere) * 2;

    textures[6].glossy_transparency = 0.02f;
    textures[6].glossiness = 0.05;

    textures[7] = Texture(Color(235, 179, 41), 0.4, 0.6, 0.4, 128.0, 1.0);
    Sphere* s2 = new Sphere( Vector3(5, -1, 15), 2, 7); // Yellow
    textures[7].glossiness = 0.2;

    textures[8] = Texture(Color(6, 72, 111), 0.3, 0.8, 0.1, 128.0, 1.0);
    Sphere* s3 = new Sphere( Vector3(5, 0, 25), 3, 8);  // Blue
    textures[8].glossiness = 0.4;

    textures[9] = Texture(Color(8, 88, 56), 0.4, 0.6, 0.5, 64.0, 1.0);
    Sphere* s4 = new Sphere( Vector3(-3.5, -1, 10), 2, 9); // Green
    textures[9].glossiness = 0.3;

    textures[10] = Texture(Color(51, 51, 51), 0.3, 0.8, 0.25, 32.0, 0.0);
    Sphere* s5 = new Sphere( Vector3(-5.5, 0, 15), 3, 10); // Black
    sm_memSize += sizeof(Sphere) * 4;

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
    sm_memSize += sizeof(Light*) * light_count;

    AreaLight* l0 = new AreaLight( Vector3(0, 20, 35), Vector3(1.4) );
    AreaLight* l1 = new AreaLight( Vector3(20, 20, 35), Vector3(1.8) );
    //DirectionalLight l2 = DirectionalLight( Vector3(0, 20, 35), Vector3(1.4) );
    //PointLight* l3 = new PointLight( Vector3(20, 20, 35), Vector3(2.0) );
    sm_memSize += sizeof(AreaLight) * 2;

    lights[0] = l0;
    lights[1] = l1;

    scene->objects = new Array<Shape>((Shape**)objects, object_count);
    scene->lights = new Array<Light>((Light**)lights, light_count);
    sm_memSize += sizeof(Array<Shape>);
    sm_memSize += sizeof(Array<Light>);

    return sm_memSize;
}

__global__ void setupScene(Renderer** renderer, Scene** scene, Camera** cam, int width, int height, size_t* memSize, curandState* random_states)
{
    size_t sm_memSize;
    //sm_memSize = simple_scene(scene);
    curandState local_rand_state = random_states[0];
    sm_memSize = random_sphere_scene(scene, 1000, &local_rand_state);
    float fov = 30.0f;
    *cam = new Camera(Vector3(0, 20, -20), width, height, fov);
    (*cam)->angleX = 30 * (M_PI / 180);
    sm_memSize += sizeof(Camera);

    *renderer = new Renderer(width, height, *scene, *cam, 3);
    sm_memSize += sizeof(Renderer);
    *memSize = sm_memSize;
    //random_states[0] = local_rand_state;
}

__global__ void renderScene(Color* framebuffer, Renderer** g_renderer, curandState* random_states)
{
    extern __shared__ float sm_pointer[];

    Renderer* sm_renderer = (Renderer*)sm_pointer;

    if (threadIdx.x == 0 && threadIdx.y == 0) // Only one thread per block
		sm_renderer = copy_renderer_to_sm(*g_renderer, (void*)sm_pointer);
    __syncthreads();

    auto renderer = sm_renderer;

    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if ((x >= renderer->width) || (y >= renderer->height)) { return; }

    auto index = y * renderer->width + x;

    // Get local random state
    curandState local_rand_state = random_states[index];
    
    Color pixelColor = Color(0.0f, 0.0f, 0.0f);
    // Send a ray through each pixel
    float ds = 1 / (float)renderer->ray_per_pixel;
    for (int s = 0; s < renderer->ray_per_pixel; s++)
    {
        auto r = Vector3::random(&local_rand_state);

        Ray ray = renderer->camera->pixelToViewport(Vector3(x + r.x, y + r.y, 1));

        // Sent pixel for traced ray
        pixelColor += renderer->trace(ray, 0, &local_rand_state, renderer->scene->textures) * ds;
    }

    framebuffer[index] = pixelColor;

    random_states[index] = local_rand_state;
}

__global__ void free_scene(Renderer** renderer_ptr)
{
    auto renderer = *renderer_ptr;

    for(int i = 0; i < renderer->scene->objects->count; i++)
    {
        delete renderer->scene->objects->list[i];
    }

    for(int i = 0; i < renderer->scene->lights->count; i++)
    {
        delete renderer->scene->lights->list[i];
    }

    delete renderer->scene->objects->list;
    delete renderer->scene->lights->list;
    delete renderer->scene->objects;
    delete renderer->scene->lights;
    delete renderer->scene;
    delete renderer->camera;
    delete *renderer_ptr;
}