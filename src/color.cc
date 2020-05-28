#include "color.hh"

float& clamp_value(float& v, float min=0, float max=255)
{
    v = (v > max) ? max : v;
    v = (v < min) ? min : v;

    return v;
}

Color& Color::clamp()
{
    clamp_value(r);
    clamp_value(g);
    clamp_value(b);

    return *this;
}
