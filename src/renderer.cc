#include "renderer.hh"
#include "image.hh"

#include <iostream>

int Renderer::MAX_RAY_DEPTH = 5;

void Renderer::render()
{
    Color *pixels = new Color[width * height];

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            int index = y * width + x;
            // Send a ray through each pixel
            Vector3 rayDirection = camera.pixelToViewport( Vector3(x, y, 1) );
            Ray ray(camera.position, rayDirection);
            // Sent pixel for traced ray
            pixels[index] = trace(ray, 0);
        }
    }

    Image::save("out.ppm", width, height, pixels);
    delete[] pixels;
}

void Renderer::renderAntiAliasing(int samples)
{
    float inv_samples = 1 / (float)samples;

    Color *pixels = new Color[width * height];

    for (int y = 0; y < height; y++)
    {
        for (int x = 0; x < width; x++)
        {
            for (int s = 0; s < samples; s++)
            {
                Vector3 r = Vector3::random();
                float jx = x + r.x;
                float jy = y + r.y;
                int index = y * width + x;

                // Send a jittered ray through each pixel
                Vector3 rayDirection = camera.pixelToViewport( Vector3(jx, jy, 1) );

                Ray ray(camera.position, rayDirection);

                // Sent pixel for traced ray
                pixels[index] += trace(ray, 0) * inv_samples;
            } 
        }
    }

    Image::save("out.ppm", width, height, pixels);
    delete[] pixels;
}

Color Renderer::trace(const Ray& ray, const int depth)
{
    Color rayColor;
    float tnear = INFINITY;
    std::shared_ptr<Shape> hit;

    // Find nearest intersection with ray and objects in scene
    for (auto& obj : scene.objects)
    {
        float t0 = INFINITY;
        float t1 = INFINITY;
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

    if (!hit)
    {
        return (depth < 1) ? scene.backgroundColor : Color();
    }

    // Normal and Direction from hit point
    Vector3 hitPoint = ray.origin + ray.direction * tnear;
    Vector3 N = hit->getNormal(hitPoint);
    N.normalize();
    Vector3 V = camera.position - hitPoint;
    V.normalize();

    // Compute Color with all lights
    rayColor = Lighting::getLighting(*hit, hitPoint, N, V, scene.lights, scene.objects);

    float bias = 1e-4;
    bool inside = false;

    if (ray.direction.dot(N) > 0)
    {
        N = -N;
        inside = true;
    }

    // not transparent or depth == MAX RAY DEPTH
    if ((hit->transparency <= 0 && hit->reflectivity <= 0) || depth >= MAX_RAY_DEPTH)
    {
        return rayColor;
    }

    // Compute Reflection Ray and Color 
    Vector3 R = ray.direction - N * 2 * ray.direction.dot(N);
    R = R + Vector3::random() * hit->glossiness;
    R.normalize();

    Ray rRay(hitPoint + N * bias, R);
    Color reflectionColor = trace(rRay,  depth + 1); //* VdotR;
    Color refractionColor = Color();

    if (hit->transparency <= 0)
    {
        return rayColor + (reflectionColor * hit->reflectivity);
    }

    // Compute Refracted Ray (transmission ray) and Color
    float ni = 1.0;
    float nt = 1.1;
    float nit = ni / nt;
    if(inside) nit = 1 / nit;
    float costheta = - N.dot(ray.direction);
    float k = 1 - nit * nit * (1 - costheta * costheta);
    Vector3 T = ray.direction * nit + N * (nit * costheta - sqrt(k));
    T = T + Vector3::random() * hit->glossy_transparency;
    T.normalize();

    Ray refractionRay(hitPoint - N * bias, T);
    refractionColor = trace(refractionRay, depth + 1);
    return (reflectionColor * hit->reflectivity) + (refractionColor * hit->transparency);
}
