cmake_minimum_required(VERSION 2.8)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/../bin)

project (holovideo)

add_subdirectory(render_algorithms)
add_subdirectory(remote)
