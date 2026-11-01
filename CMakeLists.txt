cmake_minimum_required(VERSION 3.24)
project(ZeroGR LANGUAGES CXX CUDA)

include(cmake/Dependencies.cmake)

add_subdirectory(src)
add_subdirectory(examples)
