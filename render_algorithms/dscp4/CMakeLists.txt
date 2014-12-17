cmake_minimum_REQUIRED(VERSION 2.8)

#Macro for getting the windows version
macro(get_WIN32_WINNT version)
    if (WIN32 AND CMAKE_SYSTEM_VERSION)
        set(ver ${CMAKE_SYSTEM_VERSION})
        string(REPLACE "." "" ver ${ver})
        string(REGEX REPLACE "([0-9])" "0\\1" ver ${ver})

        set(${version} "0x${ver}")
    endif()
endmacro()

project(dscp4)

if(CMAKE_COMPILER_IS_GNUCXX)
	include(CheckCXXCompilerFlag)
	CHECK_CXX_COMPILER_FLAG("-std=c++11" COMPILER_SUPPORTS_CXX11)
	CHECK_CXX_COMPILER_FLAG("-std=c++0x" COMPILER_SUPPORTS_CXX0X)
	if(COMPILER_SUPPORTS_CXX11)
	    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
	elseif(COMPILER_SUPPORTS_CXX0X)
	    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0x")
	else()
		message(STATUS "The compiler ${CMAKE_CXX_COMPILER} has no C++11 support. Please use a different C++ compiler.")
	endif()
endif()

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/CMake)

option(BOOST_DISABLE_AUTOLINK "Defines BOOST_ALL_NO_LIB macro to turn off boost auto linking" OFF)
option(BUILD_DSCP4_APP "Builds the executable application for testing the DSCP4 algorithm library (libdscp4)" ON)

find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_package(CUDA REQUIRED)
find_package(X11)

add_subdirectory(libdscp4)

if(${BUILD_DSCP4_APP})
	find_package(assimp REQUIRED)
	add_subdirectory(dscp4)
endif()

#Need to define windows version macro for various reasons
if(WIN32)
	get_WIN32_WINNT(winver)
	add_definitions(-D_WIN32_WINNT=${winver})
endif()
