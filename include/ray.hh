#pragma once

class Ray
{
public:
    Ray(const Vector3& _origin, const Vector3& _direction)
        : origin{_origin}, direction{_direction}
    {}

    Vector3 origin_get() { return origin; }
    Vector3 direction_get() { return direction; }

public:
    Vector3 origin;
    Vector3 direction;
};
