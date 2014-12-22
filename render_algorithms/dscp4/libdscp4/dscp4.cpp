#include "dscp4.h"
#include "DSCP4Render.hpp"

dscp4::DSCP4Render *g_renderContext = nullptr;

extern "C"
{

	DSCP4_API bool dscp4_InitRenderer()
	{
		g_renderContext = new dscp4::DSCP4Render;
		return g_renderContext->init();
	}

	DSCP4_API void dscp4_DeinitRenderer()
	{
		g_renderContext->deinit();
		delete g_renderContext;
		g_renderContext = nullptr;
	}

	DSCP4_API void dscp4_AddMesh(const char *id, unsigned int numVertices, float *vertices, float *colors)
	{
		g_renderContext->addMesh(id, numVertices, vertices, colors);
	}

	DSCP4_API void dscp4_RemoveMesh(const char *id)
	{
		g_renderContext->removeMesh(id);
	}

	DSCP4_API void dscp4_AddPointCloud(const char *id, unsigned int numPoints, float *points)
	{

	}

	DSCP4_API void dscp4_RemovePointCloud(const char *id)
	{

	}
}