#include "DSCP4Render.hpp"

dscp4::DSCP4Render *g_renderContext = nullptr;

extern "C"
{

	bool dscp4_InitRenderer()
	{
		g_renderContext = new dscp4::DSCP4Render;
		return g_renderContext->init();
	}

	void dscp4_DeinitRenderer()
	{
		g_renderContext->deinit();
		delete g_renderContext;
		g_renderContext = nullptr;
	}

	void dscp4_AddMesh(const char *id, unsigned int numVertices, float *vertices, float *colors = NULL)
	{
		g_renderContext->addMesh(id, numVertices, vertices, colors);
	}

	void dscp4_RemoveMesh(const char *id)
	{
		g_renderContext->removeMesh(id);
	}

	void dscp4_AddPointCloud(const char *id, unsigned int numPoints, float *points)
	{

	}

	void dscp4_RemovePointCloud(const char *id)
	{

	}
}