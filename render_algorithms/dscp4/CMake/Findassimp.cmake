FIND_PATH(
	assimp_INCLUDE_DIRS
	NAMES postprocess.h scene.h version.h config.h cimport.h
	PATHS /usr/local/include/
)

FIND_LIBRARY(
	assimp_LIBRARY
	NAMES assimp
	PATHS /usr/local/lib/
)

find_library( assimp_LIBRARY_DEBUG
  NAMES assimpd
  )
mark_as_advanced( assimp_LIBRARY_DEBUG )

if(assimp_LIBRARY_DEBUG)
LIST (APPEND assimp_LIBRARIES optimized ${assimp_LIBRARY} debug ${assimp_LIBRARY_DEBUG})
else()
set(assimp_LIBRARIES ${assimp_LIBRARY} )
endif()

IF (assimp_INCLUDE_DIRS AND assimp_LIBRARIES)
    SET(assimp_FOUND TRUE)
ENDIF (assimp_INCLUDE_DIRS AND assimp_LIBRARIES)



IF (assimp_FOUND)
    IF (NOT assimp_FIND_QUIETLY)
        MESSAGE(STATUS "Found asset importer library: ${assimp_LIBRARIES}")
    ENDIF (NOT assimp_FIND_QUIETLY)
ELSE (assimp_FOUND)
    IF (assimp_FIND_REQUIRED)
        MESSAGE(FATAL_ERROR "Could not find asset importer library")
    ENDIF (assimp_FIND_REQUIRED)
ENDIF (assimp_FOUND)

