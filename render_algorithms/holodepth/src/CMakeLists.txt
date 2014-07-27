cmake_minimum_REQUIRED(VERSION 2.8)

project(holodepth)

OPTION(WITH_REMOTEQT "Enables RemoteQT GUI controls and feedback" ON)
OPTION(WITH_KINECT "Sets the Kinect code to enabled.  Requires WITH_VIEWS_FROM_CLOUD to be ON." OFF)
OPTION(WITH_VIEWS_FROM_CLOUD "Renders from point clouds." OFF)
OPTION(WITH_GLEW "Enables GL extension wrangler." OFF)

set(WITH_KINECT_MODE "0" CACHE FILEPATH "Selects the kinect mode. 1-4, 1 is single Kinect.")

IF(WITH_KINECT)
	SET(WITH_VIEWS_FROM_CLOUD ON)
	ADD_DEFINITIONS(-DKINECT_MODE=${KINECT_MODE})
ENDIF()

IF(WITH_REMOTEQT)
	ADD_DEFINITIONS(-DREMOTEQT_GUI)
ENDIF()

IF(WITH_VIEWS_FROM_CLOUD)
	ADD_DEFINITIONS(-DVIEWS_FROM_CLOUD)
ENDIF()

IF(WITH_GLEW)
	INCLUDE_DIRECTORIES(${GLEW_INCLUDE_DIR})
	ADD_DEFINITIONS(-DUSE_GLEW)
ENDIF()

find_package(OpenGL REQUIRED)
find_package(GLee REQUIRED)
find_package(Cg REQUIRED)
find_package(GLEW)
find_package(X11)
find_package(Threads)

INCLUDE_DIRECTORIES(${CG_INCLUDE_DIR})
INCLUDE_DIRECTORIES(${GLEE_INCLUDE_DIR})

if(WIN32)
get_filename_component(CG_LIBRARY_DIR ${CG_GL_LIBRARY} DIRECTORY)
LINK_DIRECTORIES(${CG_LIBRARY_DIR})
endif()

file(GLOB holodepth_SRC
	"*.h"
	"*.cpp"
)
SET (holodepth_BIN ${PROJECT_NAME})

#ADD_EXECUTABLE(${holosuite_BIN} ${SRC_FILES})
add_executable (${holodepth_BIN} ${holodepth_SRC})

target_link_libraries(${holodepth_BIN} ${OPENGL_GL_LIBRARY} ${OPENGL_GLU_LIBRARY} ${GLUT_glut_LIBRARY} ${GLEE_LIB} ${CG_LIBRARY} ${CG_GL_LIBRARY})

if(X11_FOUND)
	target_link_libraries(${holodepth_BIN} ${X11_LIBRARY})
endif()

if(GLEW_FOUND)
	target_link_libraries(${holodepth_BIN} ${GLEW_LIBRARIES})
endif()

if(Threads_FOUND)
	target_link_libraries(${holodepth_BIN} ${CMAKE_THREAD_LIBS_INIT})
endif()