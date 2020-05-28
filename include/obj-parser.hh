#pragma once

#include <vector>
#include <memory>
#include <string>
#include <fstream>
#include <sstream>
#include <algorithm>

#include "shape.hh"

// No object manipulation, the objparser will only collect a list of triangles
class ObjParser
{
public:
    using shared_Shape = std::shared_ptr<Shape>;

    static std::vector<shared_Shape> parse_file(const std::string& filepath)
    {
        std::ifstream file{filepath};

        std::vector<shared_Shape> shapes;

        std::vector<Vector3> vertices;

        while (parse_object(file, shapes, vertices))
            continue;

        return shapes;
    }

private:
    static bool is_comment(const std::string& line)
    {
        size_t start = 0;
        while (line[start] == ' ')
            start ++;

        return line[start] == '#';

    }

    static std::vector<std::string> split(const std::string& line, char delim=' ')
    {
        std::vector<std::string> cont;

        std::stringstream ss(line);
        std::string token;
        
        while (std::getline(ss, token, delim))
        {
            cont.push_back(token);
        }

        cont.erase(std::remove_if(cont.begin(), cont.end(),
                                  [](const std::string& x) {
                                        return x == "";
                                  }), cont.end());

        return cont;
    }

    static bool parse_object(std::ifstream& f, std::vector<shared_Shape>& shapes,
                             std::vector<Vector3>& vertices)
    {
        std::string line;

        while (std::getline(f, line))
        {
            if (is_comment(line))
                continue;

            auto line_split = split(line);
            if (line_split.size() == 0)
                continue;

            if (line_split[0] == "o")
            {
                // tell that there is a new object to parse
                return true;
            }
            else if (line_split[0] == "v")
            {
                // Vertice
                vertices.push_back({std::stof(line_split[1]),
                        std::stof(line_split[2]),
                        std::stof(line_split[3])});

            } else if (line_split[0] == "f")
            {
                // Triangle
                // Convert char to int and the vertices are 1 indexed in .obj files
                auto index_1 = std::stoi(split(line_split[1], '/')[0]) - 1;
                auto index_2 = std::stoi(split(line_split[2], '/')[0]) - 1;
                auto index_3 = std::stoi(split(line_split[3], '/')[0]) - 1;

                auto va = vertices[index_1];
                auto vb = vertices[index_2];
                auto vc = vertices[index_3];

                auto triangle = Triangle(va, vb, vc,
                                         Color(165, 10, 14), // Color
                                         1.0, 0.5, 0.0, // Ka Kd Ks
                                         128.0, 0.3, 0.0); // Shinny reflectScale Transparency

                shapes.push_back(std::make_shared<Triangle>(triangle));
            }
            // Discarding Normals and UVs, normals are computed, and UVs are not used
        }

        return false;
    }
};
