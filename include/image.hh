#pragma once

#include <string>
#include "color.hh"

class Image
{
public:
    static bool save(const std::string& filename, size_t width, size_t height, Color* pixels);
};