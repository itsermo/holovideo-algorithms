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

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang" OR "${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
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

find_package(OpenGL REQUIRED)
find_package(GLEW REQUIRED)
find_package(GLM REQUIRED)
find_package(SDL2 REQUIRED)
find_package(CUDA)
find_package(OpenCL)
find_package(Log4Cxx)
find_package(X11)
find_package(PNG)

option(BOOST_DISABLE_AUTOLINK "Defines BOOST_ALL_NO_LIB macro to turn off boost auto linking" ${WIN32})
option(BUILD_DSCP4_APP "Builds the executable application for testing the DSCP4 algorithm library (libdscp4)" ON)
option(BUILD_DSCP4_STATIC "Builds the DSCP4 lib as a static lib" ${WIN32})
option(WITH_CUDA "Enables CUDA compute building" ${CUDA_FOUND})
option(WITH_LIBLOG4CXX "Uses Log4Cxx library for logging status and error messages" ${LIBLOG4CXX_FOUND})
option(WITH_TRACE_LOG "Enables trace log to resolve code performance and behavior, such as showing FPS calculations in the log. Requires liblog4cxx" OFF)
option(WITH_OPENCL "Enables OpenCL compute building" ${OPENCL_FOUND})
option(WITH_PNG "Enables screenshot functionality (saving render buffer to PNG)" ${PNG_FOUND})

if(NOT WITH_CUDA AND NOT WITH_OPENCL)
	message(WARNING "You haven't enabled support for OpenCL or CUDA, use the -DWITH_CUDA=YES and/or -DWITH_OPENCL=YES, otherwise you will not be able to compute holograms.")
endif()
	
if(LIBLOG4CXX_FOUND)
	option(LIBLOG4CXX_USING_STATICLIB "Defines LOG4CXX_STATIC for use with static LOG4CXX libraries" ${WIN32})
endif()

if(GLEW_FOUND)
	option(GLEW_USING_STATICLIB "Defines GLEW_STATIC for use with static GLEW libraries" ${WIN32})
endif()

if(GLEW_USING_STATICLIB)
	add_definitions(-DGLEW_STATIC)
endif()

if(LIBLOG4CXX_USING_STATICLIB)
	add_definitions(-DLOG4CXX_STATIC)
endif()

if(WITH_TRACE_LOG)
	add_definitions(-DDSCP4_ENABLE_TRACE_LOG)
endif()

# use, i.e. don't skip the full RPATH for the build tree
SET(CMAKE_SKIP_BUILD_RPATH FALSE)

# when building, don't use the install RPATH already
# (but later on when installing)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE) 

SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)


# the RPATH to be used when installing, but only if it's not a system directory
LIST(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
IF("${isSystemDir}" STREQUAL "-1")
   SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
ENDIF("${isSystemDir}" STREQUAL "-1")

add_subdirectory(libdscp4)

if(${BUILD_DSCP4_APP})
	find_package(assimp REQUIRED)
	add_subdirectory(dscp4)
else()
	set(DSCP4_APP_BIN_INSTALL_PATH "NOT_INSTALLED")
	set(MODELS_INSTALL_PATH "NOT_INSTALLED")
endif()

#Need to define windows version macro for various reasons
if(WIN32)
	get_WIN32_WINNT(winver)
	add_definitions(-D_WIN32_WINNT=${winver})
endif()

if(WITH_OPENCL)
	set(DEFAULT_COMPUTE_METHOD "opencl")
elseif(WITH_CUDA)
	set(DEFAULT_COMPUTE_METHOD "cuda")
else()
	set(DEFAULT_COMPUTE_METHOD "none")
endif()

configure_file(dscp4.conf.in dscp4.conf)

if(UNIX)
	if(BUILD_DSCP4_APP)
		configure_file(dscp4/assets/dscp4.desktop.in ${DSCP4_APP_BIN}/dscp4.desktop)
		install(FILES ${CMAKE_BINARY_DIR}/${DSCP4_APP_BIN}/dscp4.desktop DESTINATION /usr/local/share/applications)
	endif()
	install(FILES ${CMAKE_BINARY_DIR}/dscp4.conf DESTINATION /etc/dscp4)
elseif(WIN32)
	install(FILES ${CMAKE_BINARY_DIR}/dscp4.conf DESTINATION $ENV{PROGRAMDATA}/dscp4)
endif()


# uninstall target
configure_file(
    "${CMAKE_CURRENT_SOURCE_DIR}/cmake_uninstall.cmake.in"
    "${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake"
    IMMEDIATE @ONLY)

add_custom_target(uninstall
    COMMAND ${CMAKE_COMMAND} -P ${CMAKE_CURRENT_BINARY_DIR}/cmake_uninstall.cmake)
