# Reference: http://qt-project.org/doc/qt-5.0/qtdoc/cmake-manual.html

cmake_minimum_required(VERSION 2.8.12)
cmake_policy(SET CMP0020 NEW)

set(DSCP4_QT_APP_BIN "dscp4-qt")

project(${DSCP4_QT_APP_BIN})

set(DSCP4_QT_APP_BIN ${PROJECT_NAME} PARENT_SCOPE)

set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/cmake)
set(CMAKE_INCLUDE_CURRENT_DIR ON)
set(CMAKE_AUTOMOC ON)

find_package(Qt5Widgets REQUIRED)

find_package(Boost 1.54 COMPONENTS "system" "program_options" "filesystem" REQUIRED)
include_directories(${Boost_INCLUDE_DIR})
if(BOOST_DISABLE_AUTOLINK)
	add_definitions(-DBOOST_ALL_NO_LIB)
endif(BOOST_DISABLE_AUTOLINK)

include_directories(${assimp_INCLUDE_DIRS})

if(WITH_LIBLOG4CXX)
	add_definitions(-DDSCP4_HAVE_LOG4CXX)
	include_directories(${LIBLOG4CXX_INCLUDE_DIRS})
endif()

if(X11_FOUND)
	add_definitions(-DDSCP4_HAVE_X11)
endif()

file(GLOB SOURCE_FILES
    src/*.cc
	src/*.h
	src/*.cpp
	src/*.hpp
	../dscp4/DSCP4ProgOptions.cpp
	../dscp4/DSCP4ProgOptions.hpp
)

include_directories(../dscp4 ../libdscp4)

if (WIN32)
    set(SOURCE_FILES ${SOURCE_FILES} win/dscp4-qt.rc)
endif ()

set(UI_FILES
    ui/mainwindow.ui
)
source_group("UI Files" FILES ${UI_FILES})

qt5_wrap_ui(UI_HEADERS ${UI_FILES})
source_group("Generated UI Headers" FILES ${UI_HEADERS})

qt5_add_resources(RESOURCE_FILES dscp4-qt.qrc)

if (APPLE)
    set(SOURCE_FILES ${SOURCE_FILES} AppIcon.icns)
    set_source_files_properties(AppIcon.icns PROPERTIES MACOSX_PACKAGE_LOCATION Resources)
endif ()

add_executable(${DSCP4_QT_APP_BIN} MACOSX_BUNDLE
    ${SOURCE_FILES}
    ${UI_HEADERS}
    ${UI_FILES}
    ${RESOURCE_FILES}
)

set(MACOSX_BUNDLE_BUNDLE_NAME "dscp4-qt")
set(MACOSX_BUNDLE_BUNDLE_VERSION "1.0")
set(MACOSX_BUNDLE_LONG_VERSION_STRING "1.0")
set(MACOSX_BUNDLE_SHORT_VERSION_STRING "1.0")
set(MACOSX_BUNDLE_COPYRIGHT "Copyright (c) MIT")
set(MACOSX_BUNDLE_GUI_IDENTIFIER "edu.mit.media.obmg.dscp4")
set(MACOSX_BUNDLE_ICON_FILE AppIcon.icns)

qt5_use_modules(${DSCP4_QT_APP_BIN} Widgets)

target_link_libraries(${DSCP4_QT_APP_BIN} dscp4-lib ${Boost_LIBRARIES} ${assimp_LIBRARIES})

if(WITH_LIBLOG4CXX)
	target_link_libraries(${DSCP4_QT_APP_BIN} ${LIBLOG4CXX_LIBRARIES})
endif()

if (MSVC)
    set_target_properties(${DSCP4_QT_APP_BIN} PROPERTIES
        WIN32_EXECUTABLE YES
        LINK_FLAGS "/ENTRY:mainCRTStartup"
    )
endif ()

file(GLOB RES_FILES
	"res/*.xpm"
	"res/*.png"
)

install(TARGETS ${DSCP4_QT_APP_BIN} DESTINATION bin)
install(FILES ${RES_FILES} DESTINATION share/dscp4/assets)

set(DSCP4_QT_APP_BIN_INSTALL_PATH "${CMAKE_INSTALL_PREFIX}/bin" PARENT_SCOPE)

