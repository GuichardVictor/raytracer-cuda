#include <iostream>
#include <time.h>
#include <fstream>

#include "renderer.hh"
#include "vector3.hh"
#include "util.hh"

// limited version of checkCudaErrors from helper_cuda.h in CUDA examples
#define checkCudaErrors(val) check_cuda( (val), #val, __FILE__, __LINE__ )

void check_cuda(cudaError_t result, char const *const func, const char *const file, int const line) {
    if (result) {
        std::cerr << "CUDA error = " << static_cast<unsigned int>(result) << " at " <<
            file << ":" << line << " '" << func << "' \n";
        // Make sure we call CUDA Device Reset before exiting
        cudaDeviceReset();
        exit(99);
    }
}

int main()
{
    // Render Settings
    int width = 1280;
    int height = 720;

    // Cuda Settings
    int tx = 8; // BlockX size
    int ty = 8; // BlockY size

    int num_pixels = width * height;
    int buffer_size = num_pixels * sizeof(Color);

    // Allocate FrameBuffer
    Color* frameBuffer;
    checkCudaErrors(cudaMallocManaged((void **)&frameBuffer, buffer_size));

    // Allocate Random States
    curandState* d_rand_state;
    checkCudaErrors(cudaMalloc((void **)&d_rand_state, num_pixels*sizeof(curandState)));

    // Creating Scene
    Scene** scene;
    checkCudaErrors(cudaMalloc((void **)&scene, sizeof(Scene*)));

    Renderer** renderer;
    checkCudaErrors(cudaMalloc((void **)&renderer, sizeof(Renderer*)));

    Camera** camera;
    checkCudaErrors(cudaMalloc((void **)&camera, sizeof(Camera*)));


    // Scene Setup
    std::cerr << "Scene setup.." << std::endl;
    size_t* sm_memSize;
    checkCudaErrors(cudaMallocManaged((void **)&sm_memSize, sizeof(size_t)));
    setupScene<<<1,1>>>(renderer, scene, camera, width, height, sm_memSize);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());
    std::cerr << "  sm_memSize: " << *sm_memSize << std::endl;
    *sm_memSize = *sm_memSize * 3;


    // Setting Up Rendering
    dim3 blocks(width/tx+1,height/ty+1);
    dim3 threads(tx,ty);

    std::cerr << "Random init..." << std::endl;
    random_init<<<blocks, threads>>>(width, height, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    // Render our buffer

    std::cerr << "Rendering a " << width << "x" << height << " image with ";
    std::cerr << "in " << tx << "x" << ty << " blocks." << std::endl;

    clock_t start, stop;
    start = clock();

    renderScene<<<blocks, threads, *sm_memSize>>>(frameBuffer, renderer, d_rand_state);
    checkCudaErrors(cudaGetLastError());
    checkCudaErrors(cudaDeviceSynchronize());

    stop = clock();
    double timer_seconds = ((double)(stop - start)) / CLOCKS_PER_SEC;
    std::cerr << "took " << timer_seconds << " seconds." << std::endl;

    std::ofstream file{"file.ppm", std::ios::out | std::ios::binary};

    file << "P3\n" << width << " " << height << "\n255\n";
    for (int j = 0; j < height; j++) {
        for (int i = 0; i < width; i++) {
            size_t pixel_index = j * width + i;
            auto color = frameBuffer[pixel_index];
            //color = color.clamp();
            int ir = int(color.r);
            int ig = int(color.g);
            int ib = int(color.b);
            file << ir << " " << ig << " " << ib << "\n";
        }
    }
    
    // Free render buffers
    checkCudaErrors(cudaFree(d_rand_state));
    checkCudaErrors(cudaFree(frameBuffer));

    // Free scene
    // Missing: kernel to free inside scene pointers
    checkCudaErrors(cudaFree(scene));
    checkCudaErrors(cudaFree(renderer));
    checkCudaErrors(cudaFree(camera));

    // Reset Device
    cudaDeviceReset();
}