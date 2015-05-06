### CMakeLists.txt for CUDA
cmake_minimum_required(VERSION 2.8)

project(fresnel-mkii)

find_package(OpenGL QUIET REQUIRED)
find_package(GLUT QUIET REQUIRED)
find_package(GLEW QUIET REQUIRED)
find_package(CUDA QUIET REQUIRED)

# Do not set compute_13 architecture for CUDA 7
# CUDA 7 has removed support for compute_13
# Which means no FX5800 support!
if(${CUDA_VERSION_MAJOR} GREATER 6)
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};
	-gencode=arch=compute_20,code=\"sm_20,compute_20\"
	-gencode=arch=compute_30,code=\"sm_30,compute_30\"
	-gencode=arch=compute_35,code=\"sm_35,compute_35\"
	-gencode=arch=compute_50,code=\"sm_50,compute_50\"
	-gencode=arch=compute_52,code=\"sm_52,compute_52\")
else()
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};
	-gencode=arch=compute_11,code=\"sm_11,compute_11\"
	-gencode=arch=compute_12,code=\"sm_12,compute_12\"
	-gencode=arch=compute_13,code=\"sm_13,compute_13\"
	-gencode=arch=compute_20,code=\"sm_20,compute_20\"
	-gencode=arch=compute_30,code=\"sm_30,compute_30\"
	-gencode=arch=compute_35,code=\"sm_35,compute_35\"
	-gencode=arch=compute_50,code=\"sm_50,compute_50\")
endif()

include_directories(${OpenGL_INCLUDE_DIR})
include_directories(${GLEW_INCLUDE_DIR})
include_directories(${GLUT_INCLUDE_DIR})
include_directories(${CUDA_TOOLKIT_INCLUDE})
include_directories(${CUDA_TOOLKIT_ROOT_DIR}/samples/common/inc)

file(GLOB CUDA_FILES
	"*.h"
	"*.cu"
)

# For compilation ...
# Specify target & source files to compile it from
cuda_add_executable(
    ${PROJECT_NAME}
	${CUDA_FILES}
	OPTIONS
	RELEASE -O3 --use_fast_math -DNDEBUG
	DEBUG -g -G -DDEBUG
    )
	
if(${CUDA_VERSION_MAJOR} GREATER 6)
	message(WARNING "${PROJECT_NAME} set to not compile for sm_11, sm_12 & sm_13 architectures.\nsm_11, sm_12 and sm_13 have been deprecated and removed from CUDA 7 (No FX5800 support).\nYou must install CUDA 6.5 if you want to compile for kepler GPUs.")
endif()
	
target_link_libraries(${PROJECT_NAME} ${OPENGL_gl_LIBRARY} ${OPENGL_glu_LIBRARY} ${GLEW_LIBRARY} ${GLUT_LIBRARY})
