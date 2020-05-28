#include "util.hh"

#define RANDOM_SEED 8080

__global__ void random_init(int width, int height, curandState *rand_state)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    int y = threadIdx.y + blockIdx.y * blockDim.y;
    if((x >= width) || (y >= height)) { return; }
    int pixel_index = y * width + x;
    //Each thread gets same seed, a different sequence number, no offset
    curand_init(RANDOM_SEED, pixel_index, 0, &rand_state[pixel_index]);
}