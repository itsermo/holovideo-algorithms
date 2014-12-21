#pragma once

extern "C" {
	bool dscp4_InitRenderer();
	void dscp4_DeinitRenderer();
	void dscp4_AddMesh(const char *id, unsigned int numVertices, float *vertices, float *colors = NULL);
	void dscp4_RemoveMesh(const char *id);
	void dscp4_AddPointCloud(const char *id, unsigned int numPoints, float *points);
	void dscp4_RemovePointCloud(const char *id);
	void dscp4_TranslateObject(const char *id, float x, float y, float z);
	void dscp4_ScaleObject(const char* id, float x, float y, float z);
	void dscp4_RotateObject(const char* id, float x, float y, float z);
}
