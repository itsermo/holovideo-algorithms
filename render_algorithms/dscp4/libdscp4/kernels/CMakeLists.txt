cmake_minimum_REQUIRED(VERSION 2.8)

project(dscp4-fringe)

if(WITH_CUDA)
	set(DSCP4_FRINGE_CUDA_LIB "${PROJECT_NAME}-cuda" PARENT_SCOPE)
	
	set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_12,code=sm_12)

	#set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)

	set(BUILD_SHARED_LIBS OFF)

	#set(CUDA_SEPARABLE_COMPILATION ON)

	include_directories(${CUDA_TOOLKIT_INCLUDE})

	file(GLOB CUDA_FILES
		"*cuda.h"
		"*.cu"
	)

	# For compilation ...
	# Specify target & source files to compile it from
	cuda_add_library(
		${PROJECT_NAME}-cuda
		${CUDA_FILES}
		OPTIONS -DSTUFF="blah blah"
		RELEASE -O3 --use_fast_math -DNDEBUG
		DEBUG -g -DDEBUG
		)
endif()