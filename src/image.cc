#include "image.hh"
#include <fstream>

#define PIXEL_MAX 255

bool Image::save(const std::string& filename, size_t width, size_t height, Color* pixels)
{
    std::ofstream file{filename, std::ios::out | std::ios::binary};

    if (!file.is_open())
        return false;

    file << "P6\n";
    file << width << ' ' << height << '\n';
    file << PIXEL_MAX << '\n';

    for (size_t i = 0; i < width * height; i++)
    {
        auto pixel = pixels[i].clamp();
        file << (unsigned char) pixel.r;
        file << (unsigned char) pixel.g;
        file << (unsigned char) pixel.b;
    }

    return true;

}