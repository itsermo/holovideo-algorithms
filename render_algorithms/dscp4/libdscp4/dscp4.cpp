#include "dscp4.h"
#include "DSCP4Render.hpp"

extern "C"
{

	DSCP4_API dscp4_context_t dscp4_CreateContext()
	{
		return (dscp4_context_t*)(new dscp4::DSCP4Render());
	}

	DSCP4_API void dscp4_DestroyContext(dscp4_context_t* renderContext)
	{
		delete (dscp4_context_t*)*renderContext;
		*renderContext = nullptr;
	}

	DSCP4_API bool dscp4_InitRenderer(dscp4_context_t context)
	{
		return ((dscp4::DSCP4Render*)context)->init();
	}

	DSCP4_API void dscp4_DeinitRenderer(dscp4_context_t renderContext)
	{
		((dscp4::DSCP4Render*)renderContext)->deinit();
	}

	DSCP4_API void dscp4_SetRenderMode(dscp4_context_t renderContext, render_mode_t renderMode)
	{
		((dscp4::DSCP4Render*)renderContext)->setRenderMode(renderMode);
	}


	DSCP4_API void dscp4_SetShadeModel(dscp4_context_t renderContext, shade_model_t shadeModel)
	{
		((dscp4::DSCP4Render*)renderContext)->setShadingModel(shadeModel);
	}

	DSCP4_API void dscp4_SetAutoScaleEnabled(dscp4_context_t renderContext, bool autoScaleEnabled)
	{
		((dscp4::DSCP4Render*)renderContext)->setAutoScaleEnabled(autoScaleEnabled);
	}

	DSCP4_API void dscp4_AddMesh(dscp4_context_t renderContext, const char *id, unsigned int numVertices, float *vertices, float *normals, float *colors)
	{
		((dscp4::DSCP4Render*)renderContext)->addMesh(id, numVertices, vertices, normals, colors);
	}

	DSCP4_API void dscp4_RemoveMesh(dscp4_context_t renderContext, const char *id)
	{
		((dscp4::DSCP4Render*)renderContext)->removeMesh(id);
	}

	DSCP4_API void dscp4_AddPointCloud(dscp4_context_t renderContext, const char *id, unsigned int numPoints, float *points, float pointSize)
	{

	}

	DSCP4_API void dscp4_RemovePointCloud(dscp4_context_t renderContext, const char *id)
	{

	}

	DSCP4_API void dscp4_TranslateObject(dscp4_context_t renderContext, const char *id, float x, float y, float z)
	{

	}

	DSCP4_API void dscp4_ScaleObject(dscp4_context_t renderContext, const char* id, float x, float y, float z)
	{

	}

	DSCP4_API void dscp4_RotateObject(dscp4_context_t renderContext, const char* id, float x, float y, float z);
}
