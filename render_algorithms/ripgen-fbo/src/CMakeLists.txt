cmake_minimum_REQUIRED(VERSION 2.8)

project(ripgen-fbo)

find_package(TIFF REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(libfreenect REQUIRED)
find_package(VRPN REQUIRED)
find_package(X11 REQUIRED)
find_package(XercesC REQUIRED)
find_package(GLUT REQUIRED)
find_package(GLU REQUIRED)

file(GLOB ripgen-fbo_SRC
	"*.h"
	"*.cpp"
)

add_executable (ripgen-fbo ${ripgen-fbo_SRC})

target_link_libraries(ripgen-fbo tiff GL GLU glut GLEW X11 freenect vrpn xerces-c)
