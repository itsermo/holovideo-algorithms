cmake_minimum_REQUIRED(VERSION 2.8)

option(BOOST_DISABLE_AUTOLINK "Defines BOOST_ALL_NO_LIB macro to turn off boost auto linking" OFF)
#OPTION(WITH_REMOTEQT "Enables RemoteQT GUI controls and feedback" ON)

include_directories(${Boost_INCLUDE_DIR})
include_directories(${assimp_INCLUDE_DIRS})

file(GLOB DSCP4_SRC
	"*.hpp"
	"*.cpp"
)

set(DSCP4_BIN ${PROJECT_NAME})

add_executable(${DSCP4_BIN} ${DSCP4_SRC})

set_target_properties(${DSCP4_BIN} PROPERTIES OUTPUT_NAME ${PROJECT_NAME} CLEAN_DIRECT_OUTPUT 1)

target_link_libraries(${DSCP4_BIN} ${PROJECT_NAME}-lib ${Boost_LIBRARIES} ${assimp_LIBRARIES})

