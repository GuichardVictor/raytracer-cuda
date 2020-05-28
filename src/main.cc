#include <chrono>

#include "scene.hh"
#include "renderer.hh"
#include "obj-parser.hh"

Scene simple_scene()
{
    Scene scene = Scene();
    scene.backgroundColor = Color();

    //Triangle t0 = Triangle( Vector3(0, -3, 0), Vector3(1, -5, 0), Vector3(-1, -5, 0), Color(165, 10, 14), 1.0, 0.5, 0.0, 128.0, 0.0);
    //Triangle t1 = Triangle( Vector3(0, 4, -30), Vector3(5, -4, -30), Vector3(-5, -4, -30), Color(165, 10, 14), 1.0, 0.5, 0.0, 128.0, 0.0);

    // Add triangles to scene
    //scene.addObject(std::make_shared<Triangle>(t0));
    //scene.addObject(std::make_shared<Triangle>(t1));

    Sphere ts0 = Sphere(Vector3(0, 4, 30), 0.2, Color(255), 0.3, 0.8, 0.5, 128.0, 1.0);
    Sphere ts1 = Sphere(Vector3(5, -4, 30), 0.2, Color(255), 0.3, 0.8, 0.5, 128.0, 1.0);
    Sphere ts2 = Sphere(Vector3(-5, -4, 30), 0.2, Color(255), 0.3, 0.8, 0.5, 128.0, 1.0);

    Sphere s0 = Sphere(Vector3(0, -10004, 20), 10000, Color(51, 51, 51), 0.2, 0.5, 0.0, 128.0, 0.0); // Black - Bottom Surface
    Sphere s1 = Sphere(Vector3(0, 0, 20), 4, Color(165, 10, 14), 0.3, 0.8, 0.5, 128.0, 0.05, 0.95); // Clear

    s1.glossy_transparency = 0.02;
    s1.glossiness = 0.05;
    Sphere s2 = Sphere( Vector3(5, -1, 15), 2, Color(235, 179, 41), 0.4, 0.6, 0.4, 128.0, 1.0); // Yellow
    s2.glossiness = 0.2;
    Sphere s3 = Sphere( Vector3(5, 0, 25), 3, Color(6, 72, 111), 0.3, 0.8, 0.1, 128.0, 1.0);  // Blue
    s3.glossiness = 0.4;
    Sphere s4 = Sphere( Vector3(-3.5, -1, 10), 2, Color(8, 88, 56), 0.4, 0.6, 0.5, 64.0, 1.0); // Green
    s4.glossiness = 0.3;
    Sphere s5 = Sphere( Vector3(-5.5, 0, 15), 3, Color(51, 51, 51), 0.3, 0.8, 0.25, 32.0, 0.0); // Black

    // Add spheres to scene
    scene.addObject(std::make_shared<Sphere>(ts0));
    scene.addObject(std::make_shared<Sphere>(ts1));
    scene.addObject(std::make_shared<Sphere>(ts2));

    scene.addObject(std::make_shared<Sphere>(s0));
    scene.addObject(std::make_shared<Sphere>(s1));  // Red
    scene.addObject(std::make_shared<Sphere>(s2));  // Yellow
    scene.addObject(std::make_shared<Sphere>(s3));  // Blue
    scene.addObject(std::make_shared<Sphere>(s4));  // Green
    scene.addObject(std::make_shared<Sphere>(s5));  // Black

    // Add light to scene
    //scene.setAmbientLight ( AmbientLight( Vector3(1.0) ) );
    AreaLight l0 = AreaLight( Vector3(0, 20, 35), Vector3(1.4) );
    AreaLight l1 = AreaLight( Vector3(20, 20, 35), Vector3(1.8) );
    //DirectionalLight l2 = DirectionalLight( Vector3(0, 20, 35), Vector3(1.4) );
    //PointLight l3 = PointLight( Vector3(20, 20, 35), Vector3(2.0) );  

    scene.addLight(std::make_shared<Light>(l0));
    scene.addLight(std::make_shared<Light>(l1));
    //scene.addLight(std::make_shared<Light>(l3));

    return scene;
}


Scene object_scene(const std::string& file)
{
    auto objects = ObjParser::parse_file(file);
    Scene scene = Scene();

    for (auto& obj : objects)
    {
        scene.addObject(obj);
    }

    scene.setAmbientLight(AmbientLight(Vector3(1.0)));

    AreaLight l0 = AreaLight(Vector3(0, 3, 0), Vector3(1.4));
    AreaLight l1 = AreaLight(Vector3(-5, 0, -5), Vector3(1.8));
    //DirectionalLight l2 = DirectionalLight( Vector3(0, 20, 35), Vector3(1.4) );

    scene.addLight(std::make_shared<Light>(l0));
    scene.addLight(std::make_shared<Light>(l1));
    //scene.addLight(std::make_shared<Light>(l2));

    return scene;
}


int main(int argc, char* argv[])
{
    argc = argc;
    argv = argv;
    int width = 1280;
    int height = 720;
    float fov = 30.0;

    int raysPerPixels = 1;
    int max_ray_depth = 3;

    // Create Scene
    std::cout << "Creating Scene: ";
    auto scene = simple_scene();

    std::cout << scene.objects.size() << " objects" << std::endl;

    // Create Camera
    Camera camera = Camera(Vector3(0, 20, -20), width, height, fov);
    camera.angleX = 30 * (M_PI / 180);

    // Create Renderer
    Renderer::MAX_RAY_DEPTH = max_ray_depth;
    Renderer r = Renderer(width, height, scene, camera);

    std::cout << "Render Settings: " << "[rays per pixels: " << raysPerPixels
              << ", depth: " << max_ray_depth << "], Output: [Dimension: "
              << width << "x" << height << "]" << std::endl;

    // Render Scene
    auto start = std::chrono::high_resolution_clock::now();
    r.renderAntiAliasing(raysPerPixels);
    auto end = std::chrono::high_resolution_clock::now();

    auto duration_ms = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
    std::cout << "Render Time: " << duration_ms.count() / 1000.0 << " s" << std::endl;
    return 0;
}
