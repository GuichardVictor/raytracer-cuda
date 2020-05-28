#include "shape.hh"

Vector3 Triangle::getNormal(const Vector3&)
{
    Vector3 v01 = v1 - v0;
    Vector3 v02 = v2 - v0;
    Vector3 N = v01.cross(v02);
    N.normalize();
    return N;
}

bool Triangle::intersect(const Ray& ray, float& t, float&)
{
    Vector3 v0v1 = v1 - v0;
    Vector3 v0v2 = v2 - v0;

    Vector3 pvec = ray.direction.cross(v0v2);
    float det = v0v1.dot(pvec);
    
    if(std::fabs(det) < K_EPSILON) { return false; }

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
