#pragma once

#include <curand_kernel.h>

__global__ void random_init(int width, int height, curandState *rand_state);