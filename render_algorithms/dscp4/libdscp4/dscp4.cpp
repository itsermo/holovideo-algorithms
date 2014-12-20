#include "DSCP4Render.hpp"

dscp4::DSCP4Render *g_renderContext = nullptr;

extern "C"
{

	bool InitRenderer()
	{
		g_renderContext = new dscp4::DSCP4Render;
		return g_renderContext->init();
	}

	void DeinitRenderer()
	{
		g_renderContext->deinit();
		delete g_renderContext;
		g_renderContext = nullptr;
	}

	void AddMesh(const char *id, unsigned int numVertices, float *vertices, char *colors = NULL)
	{
		g_renderContext->addMesh(id, numVertices, vertices, colors);
	}
}