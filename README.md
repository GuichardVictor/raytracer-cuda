# CUDA Accelerated Raytracer

## How to

Compile

```sh
$ mkdir build && cd build
$ cmake ..
$ make # Your build system used in the cmake command
```

## Output

This raytracer output a `PPM` file.

## Result

| 9 Spheres (different Mat)          | CPU (AMD 3700x) | CUDA 10 (gtx 1660 super 7.5cp) |
|------------------------------------|-----------------|--------------------------------|
| 1280x720, 3 depth, 1 ray per pixel | 2s              | 0.003s                         |