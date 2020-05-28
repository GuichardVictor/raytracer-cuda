cmake_minimum_required(VERSION 3.14)
project(raytracer-cuda CXX CUDA)
set(CMAKE_CXX_STANDARD 17)

find_package(CUDA QUIET REQUIRED)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode=arch=compute_75,code=sm_75 -dc")
set(CMAKE_CUDA_FLAGS_RELEASE "${CMAKE_CUDA_FLAGS_RELEASE} -O3")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -g -G")

include_directories ("${CMAKE_SOURCE_DIR}/src/")
include_directories ("${CMAKE_SOURCE_DIR}/include/")

file(GLOB_RECURSE SOURCES "src/*.cc" "src/*.cu")

add_executable(raytracer-cuda ${SOURCES})
set_target_properties(raytracer-cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(raytracer-cuda)