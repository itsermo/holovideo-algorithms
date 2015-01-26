cmake_minimum_REQUIRED(VERSION 2.8)

if(WITH_CUDA)
	project(dscp4-fringe-cuda)

	set(DSCP4_FRINGE_CUDA_LIB "${PROJECT_NAME}" PARENT_SCOPE)
	set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE ON)
	set(CUDA_HOST_COMPILATION_CPP OFF)
	if(CMAKE_COMPILER_IS_GNUCC)
		set(CUDA_NVCC_FLAGS -Xcompiler -fpic)
	endif()
	
	set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};
		-gencode=arch=compute_13,code=\"sm_13,compute_13\"
		-gencode=arch=compute_20,code=\"sm_20,compute_20\"
		-gencode=arch=compute_30,code=\"sm_30,compute_30\"
		-gencode=arch=compute_35,code=\"sm_35,compute_35\"
		-gencode=arch=compute_50,code=\"sm_50,compute_50\")

	set(BUILD_SHARED_LIBS OFF)

	include_directories(${CUDA_TOOLKIT_INCLUDE})

	file(GLOB CUDA_FILES
		"*cuda.h"
		"*.cu"
	)

	# For compilation ...
	# Specify target & source files to compile it from
	cuda_add_library(
		${PROJECT_NAME}
		${CUDA_FILES}
		OPTIONS
		RELEASE -O3 --use_fast_math -DNDEBUG
		DEBUG -g -G -DDEBUG
		)
		
	#target_link_libraries(${PROJECT_NAME} ${CUDA_LIBRARIES})
endif()

if(WITH_OPENCL)
	project(dscp4-fringe-opencl)
	set(DSCP4_FRINGE_OPENCL_LIB "${PROJECT_NAME}" PARENT_SCOPE)
	set(BUILD_SHARED_LIBS OFF)
	
	
	file(GLOB OPENCL_FILES
		"*opencl.h"
		"*opencl.cpp"
		"*.cl"
	)
	
	add_library(${PROJECT_NAME} STATIC ${OPENCL_FILES})

	if(CMAKE_COMPILER_IS_GNUCC)
  		SET_TARGET_PROPERTIES(${PROJECT_NAME} PROPERTIES COMPILE_FLAGS "-fPIC")
	endif()
	
	target_include_directories(${PROJECT_NAME} PUBLIC ${OPENCL_INCLUDE_DIRS} ${OpenGL_INCLUDE_DIR})

	target_link_libraries(${PROJECT_NAME} LINK_PUBLIC ${OPENCL_LIBRARIES} ${OPENGL_gl_LIBRARY})
	
	if(WITH_LIBLOG4CXX)
		add_definitions(-DDSCP4_HAVE_LOG4CXX)
		target_include_directories(${PROJECT_NAME} PUBLIC ${LIBLOG4CXX_INCLUDE_DIRS})
		target_link_libraries(${PROJECT_NAME} LINK_PUBLIC ${LIBLOG4CXX_LIBRARIES})
	endif()

	
endif()
