cmake_minimum_REQUIRED(VERSION 2.8)

project(holodepth)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/CMake)

add_subdirectory(src)
add_subdirectory(scripts)
