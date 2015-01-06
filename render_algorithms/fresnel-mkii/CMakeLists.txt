### CMakeLists.txt for CUDA
 
cmake_minimum_required(VERSION 2.8)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)
find_package(CUDA QUIET REQUIRED)

# Pass options to NVCC
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-O3 -gencode arch=compute_20,code=sm_20)

include_directories(${OpenGL_INCLUDE_DIR})
include_directories(${GLEW_INCLUDE_DIR})
include_directories(${GLUT_INCLUDE_DIR})
include_directories(${CUDA_TOOLKIT_INCLUDE})

# For compilation ...
# Specify target & source files to compile it from
cuda_add_executable(
    fresnel-mkii
	fresnel-mkii.cu
    )
	
target_link_libraries(fresnel-mkii ${OPENGL_gl_LIBRARY} ${OPENGL_glu_LIBRARY} ${GLEW_LIBRARY} ${GLUT_LIBRARY})