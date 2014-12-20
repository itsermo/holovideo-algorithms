#pragma once

extern "C" {
	bool InitRenderer();
	void DeinitRenderer();
	void AddMesh(float *vertices, int numVertices);
}
