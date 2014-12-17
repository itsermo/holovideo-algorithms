cmake_minimum_REQUIRED(VERSION 2.8)

find_package(Boost 1.54 COMPONENTS "filesystem" "iostreams" "program_options" "thread" REQUIRED)
include_directories(${Boost_INCLUDE_DIRS})
link_directories(${Boost_LIBRARY_DIRS})
if(BOOST_DISABLE_AUTOLINK)
	add_definitions(-DBOOST_ALL_NO_LIB)
endif(BOOST_DISABLE_AUTOLINK)

include_directories(${GLUT_INCLUDE_DIR})
include_directories(${OpenGL_INCLUDE_DIR})
include_directories(${CUDA_TOOLKIT_INCLUDE})

if(X11_FOUND)
	include_directories(${X11_INCLUDE_DIR})
endif()

file(GLOB DSCP4_LIB_SRC
	"*.h"
	"*.hpp"
	"*.cpp"
)

set(DSCP4_LIB ${PROJECT_NAME}-lib)

add_library(${DSCP4_LIB} ${DSCP4_LIB_SRC})

target_include_directories(${DSCP4_LIB} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

set_target_properties(${DSCP4_LIB} PROPERTIES OUTPUT_NAME ${PROJECT_NAME} CLEAN_DIRECT_OUTPUT 1)

target_link_libraries(${DSCP4_LIB} ${Boost_LIBRARIES} ${OPENGL_gl_LIBRARY} ${OPENGL_glu_LIBRARY} ${GLUT_glut_LIBRARY} ${CUDA_LIBRARIES})



if(X11_FOUND)
	add_definitions(-DDSCP4_ENABLE_X11)
	target_link_libraries(${DSCP4_BIN} ${X11_LIBRARIES})
endif()