### CMakeLists.txt for CUDA
 
cmake_minimum_required(VERSION 2.8)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(GLEW REQUIRED)
find_package(CUDA QUIET REQUIRED)

# Pass options to NVCC
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};
		-O3
		-gencode=arch=compute_11,code=\"sm_11,compute_11\"
		-gencode=arch=compute_20,code=\"sm_20,compute_20\"
		-gencode=arch=compute_30,code=\"sm_30,compute_30\"
		-gencode=arch=compute_35,code=\"sm_35,compute_35\"
		-gencode=arch=compute_37,code=\"sm_37,compute_37\"
		-gencode=arch=compute_50,code=\"sm_50,compute_50\")

include_directories(${OpenGL_INCLUDE_DIR})
include_directories(${GLEW_INCLUDE_DIR})
include_directories(${GLUT_INCLUDE_DIR})
include_directories(${CUDA_TOOLKIT_INCLUDE})
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc)

# For compilation ...
# Specify target & source files to compile it from
cuda_add_executable(
    fresnel-mkii
	fresnel-mkii.cu
    )
	
target_link_libraries(fresnel-mkii ${OPENGL_gl_LIBRARY} ${OPENGL_glu_LIBRARY} ${GLEW_LIBRARY} ${GLUT_LIBRARY})
