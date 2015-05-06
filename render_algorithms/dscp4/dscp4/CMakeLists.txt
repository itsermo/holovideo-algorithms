cmake_minimum_REQUIRED(VERSION 2.8)

find_package(Boost 1.54 COMPONENTS "system" "program_options" "filesystem" QUIET)
include_directories(${Boost_INCLUDE_DIR})
if(BOOST_DISABLE_AUTOLINK)
	add_definitions(-DBOOST_ALL_NO_LIB)
endif(BOOST_DISABLE_AUTOLINK)

include_directories(${assimp_INCLUDE_DIRS})

if(BUILD_DSCP4_STATIC)
	add_definitions(-DDSCP4_STATIC)
endif()

if(WITH_LIBLOG4CXX)
	add_definitions(-DDSCP4_HAVE_LOG4CXX)
	include_directories(${LIBLOG4CXX_INCLUDE_DIRS})
endif()

if(X11_FOUND)
	add_definitions(-DDSCP4_HAVE_X11)
endif()

file(GLOB DSCP4_SRC
	"*.hpp"
	"*.cpp"
)

set(DSCP4_APP_BIN ${PROJECT_NAME})
set(DSCP4_APP_BIN ${PROJECT_NAME} PARENT_SCOPE)

add_executable(${DSCP4_APP_BIN} ${DSCP4_SRC})

set_target_properties(${DSCP4_APP_BIN} PROPERTIES OUTPUT_NAME ${PROJECT_NAME} CLEAN_DIRECT_OUTPUT 1)

target_link_libraries(${DSCP4_APP_BIN} LINK_PUBLIC ${PROJECT_NAME}-lib ${Boost_LIBRARIES} ${assimp_LIBRARIES})

if(WITH_LIBLOG4CXX)
	target_link_libraries(${DSCP4_APP_BIN} LINK_PUBLIC ${LIBLOG4CXX_LIBRARIES})
endif()

if(WITH_CUDA)
	add_definitions(-DDSCP4_HAVE_CUDA)
endif()

if(WITH_OPENCL)
	add_definitions(-DDSCP4_HAVE_OPENCL)
endif()

file(GLOB ASSET_FILES
	"assets/*.xpm"
)

install(TARGETS ${DSCP4_APP_BIN} DESTINATION bin)
install(FILES ${ASSET_FILES} DESTINATION share/${PROJECT_NAME}/assets)

set(DSCP4_APP_BIN_INSTALL_PATH "${CMAKE_INSTALL_PREFIX}/bin" PARENT_SCOPE)
set(ASSETS_INSTALL_PATH "${CMAKE_INSTALL_PREFIX}/share/${PROJECT_NAME}/assets" PARENT_SCOPE)

