#pragma once

#include <memory>
#include <vector>

#include "shape.hh"
#include "light.hh"
#include "color.hh"

class Scene
{
public:
    Scene() = default;
    Scene(const std::vector<std::shared_ptr<Shape>>& objs,
          const std::vector<std::shared_ptr<Light>>& _lights,
          const Color& background = Color(),
          const AmbientLight& ambient = AmbientLight())
        : objects{objs}, lights(_lights), backgroundColor{background}, ambientLight{ambient}
    {}

    void addLight(const std::shared_ptr<Light>& _light)
    { lights.push_back(_light); }

    void addObject(const std::shared_ptr<Shape>& _object)
    { objects.push_back(_object); }

    void setAmbientLight(const AmbientLight& l) { ambientLight = l; }

public:
    std::vector<std::shared_ptr<Shape>> objects;
    std::vector<std::shared_ptr<Light>> lights;

    Color backgroundColor;
    AmbientLight ambientLight;
};