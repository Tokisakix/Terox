cmake_minimum_required(VERSION 3.18)
project(terox._C)

enable_language(CUDA)

add_subdirectory(extern/pybind11)

find_package(Python COMPONENTS NumPy)

pybind11_add_module(cpp_function terox/_C/cpp_function.cpp)
pybind11_add_module(cuda_function terox/_C/cuda_function.cu)
