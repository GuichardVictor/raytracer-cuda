#include "shape.hh"

bool Sphere::intersect(const Ray& ray, float& t0, float& t1)
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

Vector3 Sphere::getNormal(const Vector3& hitPosition)
{
    return (hitPosition - center) / radius;
}
