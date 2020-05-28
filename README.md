# GPGPU Raytracer Project

## Branches

* master: baseline of CUDA accelerated raytracer
* cuda: cuda raytracer
* cpu: cpu raytrace

The `master` branch is protected and won't be used except to display this readme and the baseline of cuda version.

## How to

Compile project (CPU & CUDA)

```sh
$ mkdir build && cd build
$ cmake ..
$ make # Your build system used in the cmake command
```

## Output

This raytracer output a `PPM` file.

## Result (Example)

| 9 Spheres (different Mat)          | CPU (AMD 3700x) | CUDA 10 (gtx 1660 super 7.5cp) |
|------------------------------------|-----------------|--------------------------------|
| 1280x720, 3 depth, 1 ray per pixel | 2s              | 0.003s                         |
