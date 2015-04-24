#include "dscp4.h"
#include "DSCP4Render.hpp"

extern "C"
{
	DSCP4_API dscp4_context_t dscp4_CreateContext(
		render_options_t *render_options,
		algorithm_options_t *algorithm_options,
		display_options_t display_options,
		unsigned int verbosity, void * logAppender)
	{
		return (dscp4_context_t*)(new dscp4::DSCP4Render(
			render_options,
			algorithm_options,
			display_options,
			verbosity, logAppender)
			);
	}

	DSCP4_API dscp4_context_t dscp4_CreateContextDefault()
	{
		return (dscp4_context_t*)(new dscp4::DSCP4Render);
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

	DSCP4_API void dscp4_SetEventCallback(dscp4_context_t renderContext, dscp4_event_cb_t eventCallback, void * parent)
	{
		((dscp4::DSCP4Render*)renderContext)->setEventCallback(eventCallback, parent);
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

	DSCP4_API void dscp4_AddMesh(dscp4_context_t renderContext, const char *id, unsigned int numIndecies, unsigned int numVertices, float *vertices, float *normals, float *colors)
	{
		((dscp4::DSCP4Render*)renderContext)->addMesh(id, numIndecies, numVertices, vertices, normals, colors);
	}

	DSCP4_API void dscp4_AddPointCloud(
		dscp4_context_t renderContext,
		const char *id,
		unsigned int numPoints,
		void *cloudData
		)
	{
		((dscp4::DSCP4Render*)renderContext)->addPointCloud(id, numPoints, cloudData);
	}

	DSCP4_API void dscp4_RemoveMesh(dscp4_context_t renderContext, const char *id)
	{
		((dscp4::DSCP4Render*)renderContext)->removeMesh(id);
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

	DSCP4_API void dscp4_SetCameraView(dscp4_context_t renderContext, camera_t cameraView)
	{
		((dscp4::DSCP4Render*)renderContext)->setCameraView(
			dscp4::Camera{ 
		glm::vec3(cameraView.eye.x, cameraView.eye.y, cameraView.eye.z),
		glm::vec3(cameraView.center.x, cameraView.center.y, cameraView.center.z),
		glm::vec3(cameraView.up.x, cameraView.up.y, cameraView.up.z) });
	}

	DSCP4_API void dscp4_GetCameraView(dscp4_context_t renderContext, camera_t *cameraView)
	{
		auto cam = ((dscp4::DSCP4Render*)renderContext)->getCameraView();
		cameraView->eye.x = cam.eye.x;
		cameraView->eye.y = cam.eye.y;
		cameraView->eye.z = cam.eye.z;
		cameraView->center.x = cam.center.x;
		cameraView->center.y = cam.center.y;
		cameraView->center.z = cam.center.z;
		cameraView->up.x = cam.up.x;
		cameraView->up.y = cam.up.y;
		cameraView->up.z = cam.up.z;
	}

	DSCP4_API void dscp4_GetRotateViewAngleX(dscp4_context_t renderContext, float* rotateViewAngleX)
	{
		*rotateViewAngleX = ((dscp4::DSCP4Render*)renderContext)->getRotateViewAngleX();
	}

	DSCP4_API void dscp4_GetRotateViewAngleY(dscp4_context_t renderContext, float* rotateViewAngleY)
	{
		*rotateViewAngleY = ((dscp4::DSCP4Render*)renderContext)->getRotateViewAngleY();
	}

	DSCP4_API void dscp4_GetRotateViewAngleZ(dscp4_context_t renderContext, float* rotateViewAngleZ)
	{
		*rotateViewAngleZ = ((dscp4::DSCP4Render*)renderContext)->getRotateViewAngleZ();
	}

	DSCP4_API void dscp4_SetRotateViewAngleX(dscp4_context_t renderContext, float angleX)
	{
		((dscp4::DSCP4Render*)renderContext)->setRotateViewAngleX(angleX);
	}

	DSCP4_API void dscp4_SetRotateViewAngleY(dscp4_context_t renderContext, float angleY)
	{
		((dscp4::DSCP4Render*)renderContext)->setRotateViewAngleY(angleY);
	}

	DSCP4_API void dscp4_SetRotateViewAngleZ(dscp4_context_t renderContext, float angleZ)
	{
		((dscp4::DSCP4Render*)renderContext)->setRotateViewAngleZ(angleZ);
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
		((dscp4::DSCP4Render*)renderContext)->saveScreenshot();
	}

	DSCP4_API void dscp4_ForceRedraw(dscp4_context_t renderContext)
	{
		((dscp4::DSCP4Render*)renderContext)->Update();
	}

}
