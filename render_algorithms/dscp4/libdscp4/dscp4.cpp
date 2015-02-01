#include "dscp4.h"
#include "DSCP4Render.hpp"

extern "C"
{

	DSCP4_API dscp4_context_t dscp4_CreateContext(
		render_options_t *render_options,
		algorithm_options_t *algorithm_options,
		display_options_t display_options,
		unsigned int verbosity)
	{
		return (dscp4_context_t*)(new dscp4::DSCP4Render(
			render_options,
			algorithm_options,
			display_options,
			verbosity)
			);
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


	DSCP4_API void dscp4_SetShadeModel(dscp4_context_t renderContext, shader_model_t shadeModel)
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
		((dscp4::DSCP4Render*)renderContext)->translateMesh(id, x, y, z);
	}

	DSCP4_API void dscp4_ScaleObject(dscp4_context_t renderContext, const char* id, float x, float y, float z)
	{
		((dscp4::DSCP4Render*)renderContext)->scaleMesh(id, x, y, z);
	}

	DSCP4_API void dscp4_RotateObject(dscp4_context_t renderContext, const char* id, float angle, float x, float y, float z)
	{
		((dscp4::DSCP4Render*)renderContext)->rotateMesh(id, angle, x, y, z);
	}

	DSCP4_API void dscp4_SetSpinOn(dscp4_context_t renderContext, int spinOn)
	{
		((dscp4::DSCP4Render*)renderContext)->setSpinOn(spinOn);
	}

	DSCP4_API int dscp4_GetSpinOn(dscp4_context_t renderContext)
	{
		return ((dscp4::DSCP4Render*)renderContext)->getSpinOn();
	}

	DSCP4_API void dscp4_SaveFrameBufferToPNG(dscp4_context_t renderContext)
	{
		((dscp4::DSCP4Render*)renderContext)->saveScreenshotPNG();
	}

	DSCP4_API void dscp4_ForceRedraw(dscp4_context_t renderContext)
	{
		((dscp4::DSCP4Render*)renderContext)->Update();
	}

}
