cmake_minimum_required(VERSION 3.14)
project(raytracer)
set(CMAKE_CXX_STANDARD 17)

add_compile_options(-Wall -Wextra -Werror -pedantic
    -I${CMAKE_SOURCE_DIR}/include/ -O3 -fno-stack-protector -z execstack)

include_directories ("${CMAKE_SOURCE_DIR}/src/")
include_directories ("${CMAKE_SOURCE_DIR}/include/")

file(GLOB_RECURSE SOURCES "src/*.cc")

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

add_executable(raytracer ${SOURCES})

target_link_libraries(raytracer)
