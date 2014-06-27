cmake_minimum_required(VERSION 2.8)

PROJECT ( remoteqt CXX C )

#FIND_PACKAGE ( FindCUDA )
#FIND_PACKAGE ( FindGLUT )
#FIND_PACKAGE ( FindOpenGL )


#find VRPN client library
#list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake)
#FIND_PACKAGE(VRPN REQUIRED)


FIND_PACKAGE(Qt4 REQUIRED)

find_package(PCL 1.4 REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

#to find headers from MOC compile
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})

#extra headers from the ripgen project
INCLUDE_DIRECTORIES(../render_algorithms/ripgen-fbo/src)

#sequencer from sSpace project
#INCLUDE_DIRECTORIES(../zSpaceViewer/Source)

SET(RemoteQT_SOURCES 
	main.cpp 
	RemoteQT.cpp 
	)

SET(RemoteQT_HEADERS 
	RemoteQT.h)

SET(RemoteQT_FORMS RemoteQT.ui)

QT4_WRAP_CPP(RemoteQT_HEADERS_MOC ${RemoteQT_HEADERS})
QT4_WRAP_UI(RemoteQT_FORMS_HEADERS ${RemoteQT_FORMS})

INCLUDE(${QT_USE_FILE})
ADD_DEFINITIONS(${QT_DEFINITIONS})



# packages                                                                                                                                                                                                 
find_package(CUDA)

set(CUDA_HOST_COMPILER /usr/bin/gcc)
INCLUDE_DIRECTORIES(/usr/local/cuda/samples/common/inc/)

#set(libfreenect_DIR /opt/libfreenect/build)
find_package(libfreenect)
#INCLUDE_DIRECTORIES(/opt/libfreenect/include)

# nvcc flags                                                                                                                                                                                               
#set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_13,code=sm_13)


set(CUDA_64_BIT_DEVICE_CODE true)
set(CUDA_VERBOSE_BUILD ON)

#cuda_compile(HELPER_O helper.cu)
#cuda_compile(DRIVER_O driver.cu OPTIONS -G)
cuda_compile(DRIVER_O uploader.cu)
#cuda_add_executable(driver ${HELPER_O} ${DRIVER_O})


cuda_add_executable(

#ADD_EXECUTABLE(
	RemoteQT 
	
	${DRIVER_O}
	
	
	${RemoteQT_SOURCES} 
	${RemoteQT_FORMS_HEADERS}
	${RemoteQT_HEADERS_MOC}
	JPCLSharedmemAdaptor.cpp
	JVRPNClient.cpp
	../render_algorithms/ripgen-fbo/src/JSharedMemory.cpp
	../render_algorithms/ripgen-fbo/src/JHolovideoDisplay.cpp
	../render_algorithms/ripgen-fbo/src/JSceneSlice.cpp
)

TARGET_LINK_LIBRARIES(
	RemoteQT 
	${QT_LIBRARIES}
	${PCL_LIBRARIES}
	vrpn
)

add_subdirectory(scripts)
