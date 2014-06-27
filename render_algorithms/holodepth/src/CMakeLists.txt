cmake_minimum_REQUIRED(VERSION 2.8)

project(holodepth)

find_package(Threads REQUIRED)
find_package(OpenGL REQUIRED)
find_package(X11 REQUIRED)
find_package(GLU REQUIRED)
find_package(GLee REQUIRED)
find_package(GLUT REQUIRED)
find_package(Cg REQUIRED)


file(GLOB holodepth_SRC
	"*.h"
	"*.cpp"
)

add_executable (holodepth ${holodepth_SRC})

target_link_libraries(holodepth GL GLU glut glee pthread Cg CgGL X11)
