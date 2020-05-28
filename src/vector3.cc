#include "vector3.hh"

float Vector3::dot(const Vector3 &v) const
{
    return x * v.x + y * v.y + z * v.z;
}

Vector3 Vector3::cross(const Vector3 &v) const
{
    return Vector3(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x);
}


float Vector3::length2() const
{
    return x * x + y * y + z * z;
}

float Vector3::length() const
{
    return sqrt(length2());
}

Vector3& Vector3::normalize()
{ 
    float norm2 = length2();
    if(norm2 > 0)
    {
        float invNorm = 1 / sqrt(norm2);
        x *= invNorm, y *= invNorm, z *= invNorm;
    }
    return *this;
}


Vector3& Vector3::rotateX(const float angle)
{
    float _y = y;
    float _z = z;

    y = _y * std::cos(angle) - _z * std::sin(angle);
    z = _y * std::sin(angle) + _z * std::cos(angle);
    return *this;
}

Vector3& Vector3::rotateY(const float angle)
{
    float _x = x;
    float _z = z;

    x = _z * std::sin(angle) + _x * std::cos(angle);
    z = _z * std::cos(angle) - _x * std::sin(angle);
    return *this;
}

Vector3& Vector3::rotateZ(const float angle)
{
    float _x = x;

    x = _x * std::cos(angle) - y * std::sin(angle);
    y = _x * std::sin(angle) + y * std::cos(angle);
    return *this;
}
