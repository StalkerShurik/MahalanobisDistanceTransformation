cmake_minimum_required(VERSION 3.24)
project(MahalanobisDistanceTransformation)

add_subdirectory(pybind11)
pybind11_add_module(mahalanobis_transformation library.cpp)
