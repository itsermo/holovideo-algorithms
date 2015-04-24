cmake_minimum_REQUIRED(VERSION 2.8)

add_subdirectory(kernels)

find_package(Boost 1.54 COMPONENTS "system" "filesystem" QUIET)
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})
if(BOOST_DISABLE_AUTOLINK)
	add_definitions(-DBOOST_ALL_NO_LIB)
endif(BOOST_DISABLE_AUTOLINK)

include_directories(${SDL2_INCLUDE_DIR})
include_directories(${OpenGL_INCLUDE_DIR})
include_directories(${GLEW_INCLUDE_DIR})
include_directories(${GLM_INCLUDE_DIRS})

add_definitions(-DGLM_FORCE_RADIANS)

if(X11_FOUND)
	add_definitions(-DDSCP4_HAVE_X11)
	include_directories(${X11_INCLUDE_DIR})
endif()

if(WITH_LIBLOG4CXX)
	add_definitions(-DDSCP4_HAVE_LOG4CXX)
	include_directories(${LIBLOG4CXX_INCLUDE_DIRS})
endif()

if(WITH_CUDA)
	add_definitions(-DDSCP4_HAVE_CUDA)
	include_directories(${CUDA_TOOLKIT_INCLUDE})
endif()

if(WITH_OPENCL)
	add_definitions(-DDSCP4_HAVE_OPENCL)
endif()

if(WITH_PNG)
	add_definitions(-DDSCP4_HAVE_PNG)
	include_directories(${ZLIB_INCLUDE_DIRS} ${PNG_INCLUDE_DIRS})
endif()

file(GLOB DSCP4_LIB_SRC
	"*.h"
	"*.hpp"
	"*.cpp"
)

set(DSCP4_LIB dscp4-lib)

if(BUILD_DSCP4_STATIC)
	add_definitions(-DDSCP4_STATIC)
	add_library(${DSCP4_LIB} STATIC ${DSCP4_LIB_SRC})
else()
	add_library(${DSCP4_LIB} SHARED ${DSCP4_LIB_SRC})
endif()

target_include_directories(${DSCP4_LIB} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

set_target_properties(${DSCP4_LIB} PROPERTIES OUTPUT_NAME libdscp4 CLEAN_DIRECT_OUTPUT 1)

target_link_libraries(${DSCP4_LIB} ${Boost_LIBRARIES} ${OPENGL_gl_LIBRARY} ${GLEW_LIBRARY} ${SDL2_LIBRARY})

if(WITH_CUDA)
	target_link_libraries(${DSCP4_LIB} ${DSCP4_FRINGE_CUDA_LIB})
endif()

if(WITH_OPENCL)
	target_link_libraries(${DSCP4_LIB} ${DSCP4_FRINGE_OPENCL_LIB})
endif()

if(WITH_LIBLOG4CXX)
	target_link_libraries(${DSCP4_LIB} ${LIBLOG4CXX_LIBRARIES})
endif()

if(X11_FOUND)
	add_definitions(-DDSCP4_ENABLE_X11)
	target_link_libraries(${DSCP4_LIB} ${X11_LIBRARIES})
endif()

if(WITH_PNG)
	target_link_libraries(${DSCP4_LIB} ${PNG_LIBRARIES} ${ZLIB_LIBRARIES})
endif()

file(GLOB SHADER_FILES
	"shaders/*.frag"
	"shaders/*.vert"
)

file(GLOB KERNEL_FILES
	"kernels/*.cu"
	"kernels/*.cl"
)

install(TARGETS ${DSCP4_LIB} DESTINATION lib)
install(FILES dscp4.h DESTINATION include)
install(FILES ${SHADER_FILES} DESTINATION share/${PROJECT_NAME}/shaders)
install(FILES ${KERNEL_FILES} DESTINATION share/${PROJECT_NAME}/kernels)

set(DSCP4_LIB_INSTALL_PATH "${CMAKE_INSTALL_PREFIX}/lib" PARENT_SCOPE)
set(SHADERS_INSTALL_PATH "${CMAKE_INSTALL_PREFIX}/share/${PROJECT_NAME}/shaders" PARENT_SCOPE)
set(KERNELS_INSTALL_PATH "${CMAKE_INSTALL_PREFIX}/share/${PROJECT_NAME}/kernels" PARENT_SCOPE)
