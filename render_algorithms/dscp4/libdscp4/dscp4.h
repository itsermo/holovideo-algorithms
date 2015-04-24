#ifndef DSCP4_H
#define DSCP4_H


#ifdef WIN32
#ifdef DSCP4_STATIC
#define DSCP4_API
#else
#define DSCP4_API __declspec(dllexport)
#endif
#else
#define DSCP4_API
#endif

#include "dscp4_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

	DSCP4_API dscp4_context_t dscp4_CreateContext(
		render_options_t *render_options,
		algorithm_options_t *algorithm_options,
		display_options_t display_options,
		unsigned int verbosity, void * logAppender = nullptr);

	DSCP4_API dscp4_context_t dscp4_CreateContextDefault();

	DSCP4_API void dscp4_DestroyContext(dscp4_context_t* renderContext);

	DSCP4_API bool dscp4_InitRenderer(dscp4_context_t renderContext);
	DSCP4_API void dscp4_DeinitRenderer(dscp4_context_t renderContext);

	DSCP4_API void dscp4_SetEventCallback(dscp4_context_t renderContext, dscp4_event_cb_t eventCallback, void * parent = 0);

	DSCP4_API void dscp4_SetRenderMode(dscp4_context_t renderContext, render_mode_t renderMode);
	DSCP4_API void dscp4_SetShaderModel(dscp4_context_t renderContext, shader_model_t shadeModel);
	DSCP4_API void dscp4_SetAutoScaleEnabled(dscp4_context_t renderContext, bool autoScaleEnabled);
	
	DSCP4_API void dscp4_AddMesh(
		dscp4_context_t renderContext,
		const char *id,
		unsigned int numIndecies,
		unsigned int numVertices,
		float *vertices,
		float *normals = 0,
		float *colors = 0
		);

	DSCP4_API void dscp4_AddPointCloud(
		dscp4_context_t renderContext,
		const char *id,
		unsigned int numPoints,
		void *cloudData
		);

	DSCP4_API void dscp4_RemoveMesh(dscp4_context_t renderContext, const char *id);

	DSCP4_API void dscp4_TranslateObject(dscp4_context_t renderContext, const char *id, float x, float y, float z);
	DSCP4_API void dscp4_ScaleObject(dscp4_context_t renderContext, const char* id, float x, float y, float z);
	DSCP4_API void dscp4_RotateObject(dscp4_context_t renderContext, const char* id, float angle, float x, float y, float z);

	DSCP4_API void dscp4_SetCameraView(dscp4_context_t renderContext, camera_t cameraView);
	DSCP4_API void dscp4_GetCameraView(dscp4_context_t renderContext, camera_t *cameraView);

	DSCP4_API void dscp4_GetRotateViewAngleX(dscp4_context_t renderContext, float* rotateViewAngleX);
	DSCP4_API void dscp4_GetRotateViewAngleY(dscp4_context_t renderContext, float* rotateViewAngleY);
	DSCP4_API void dscp4_GetRotateViewAngleZ(dscp4_context_t renderContext, float* rotateViewAngleZ);
	DSCP4_API void dscp4_SetRotateViewAngleX(dscp4_context_t renderContext, float angleX);
	DSCP4_API void dscp4_SetRotateViewAngleY(dscp4_context_t renderContext, float angleY);
	DSCP4_API void dscp4_SetRotateViewAngleZ(dscp4_context_t renderContext, float angleZ);

	DSCP4_API void dscp4_SetSpinOn(dscp4_context_t renderContext, int spinOn);
	DSCP4_API int  dscp4_GetSpinOn(dscp4_context_t renderContext);

	DSCP4_API void dscp4_SaveFrameBufferToPNG(dscp4_context_t renderContext);
	DSCP4_API void dscp4_ForceRedraw(dscp4_context_t renderContext);

#ifdef __cplusplus
};
#endif

#endif