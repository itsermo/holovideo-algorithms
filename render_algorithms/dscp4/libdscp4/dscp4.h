#pragma once

extern "C" {
	bool InitRenderer();
	void DeinitRenderer();
	void AddMesh(const char *id, unsigned int numVertices, float *vertices, char *colors = NULL);
}
