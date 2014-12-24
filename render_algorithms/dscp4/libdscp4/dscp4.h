#pragma once

#ifdef WIN32
#ifdef DSCP4_STATIC
#define DSCP4_API
#else
#define DSCP4_API __declspec(dllexport)
#endif
#else
#define DSCP4_API
#endif

extern "C" {
	DSCP4_API bool dscp4_InitRenderer();
	DSCP4_API void dscp4_DeinitRenderer();
	DSCP4_API void dscp4_AddMesh(const char *id, unsigned int numVertices, float *vertices, float *normals = 0, float *colors = 0);
	DSCP4_API void dscp4_RemoveMesh(const char *id);
	DSCP4_API void dscp4_AddPointCloud(const char *id, unsigned int numPoints, float *points);
	DSCP4_API void dscp4_RemovePointCloud(const char *id);
	DSCP4_API void dscp4_TranslateObject(const char *id, float x, float y, float z);
	DSCP4_API void dscp4_ScaleObject(const char* id, float x, float y, float z);
	DSCP4_API void dscp4_RotateObject(const char* id, float x, float y, float z);
}
