cmake_minimum_REQUIRED(VERSION 2.8)

project(dscp4)

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/CMake)

find_package(Boost 1.54 COMPONENTS "filesystem" "iostreams" "program_options" "thread" REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})
if(BOOST_DISABLE_AUTOLINK)
	add_definitions(-DBOOST_ALL_NO_LIB)
endif(BOOST_DISABLE_AUTOLINK)

find_package(GLUT REQUIRED)
find_package(OpenGL REQUIRED)
find_package(CUDA REQUIRED)
find_package(assimp REQUIRED)
find_package(X11)

add_subdirectory(libdscp4)
add_subdirectory(dscp4)

