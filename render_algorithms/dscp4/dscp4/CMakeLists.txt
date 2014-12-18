cmake_minimum_REQUIRED(VERSION 2.8)

#OPTION(WITH_REMOTEQT "Enables RemoteQT GUI controls and feedback" ON)



include_directories(${Boost_INCLUDE_DIR})
include_directories(${assimp_INCLUDE_DIRS})

if(WITH_LIBLOG4CXX)
	add_definitions(-DDSCP4_HAVE_LOG4CXX)
	include_directories(${LIBLOG4CXX_INCLUDE_DIRS})
endif()

file(GLOB DSCP4_SRC
	"*.hpp"
	"*.cpp"
)

set(DSCP4_BIN ${PROJECT_NAME})

add_executable(${DSCP4_BIN} ${DSCP4_SRC})

set_target_properties(${DSCP4_BIN} PROPERTIES OUTPUT_NAME ${PROJECT_NAME} CLEAN_DIRECT_OUTPUT 1)

target_link_libraries(${DSCP4_BIN} LINK_PUBLIC ${PROJECT_NAME}-lib ${Boost_LIBRARIES} ${assimp_LIBRARIES})

if(WITH_LIBLOG4CXX)
	target_link_libraries(${DSCP4_BIN} LINK_PUBLIC ${LIBLOG4CXX_LIBRARIES})
endif()