#ifndef DSCP4_FRINGE_OPENCL_H__
#define DSCP4_FRINGE_OPENCL_H__

#include "../dscp4_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

	typedef struct
	{
		dscp4_fringe_context_t * fringe_context;
		void * stereogram_rgba_opencl_resource;
		void * stereogram_depth_opencl_resource;
		void ** fringe_opencl_resources;
		void * framebuffer_opencl_output;
		void * cl_context;
		unsigned int num_gpus;
		void * gpu_properties;
		void * kernel;
		void * program;
		void * command_queue;
		void * logger;
		bool have_cl_gl_depth_images_extension;
		bool have_cl_gl_sharing_extension;
		bool have_cl_double_precision_extension;
	} dscp4_fringe_opencl_context_t;

	dscp4_fringe_opencl_context_t* dscp4_fringe_opencl_CreateContext(dscp4_fringe_context_t *fringeContext, int *glContext);

	void dscp4_fringe_opencl_DestroyContext(dscp4_fringe_opencl_context_t** openclContext);

	void dscp4_fringe_opencl_ComputeFringe(dscp4_fringe_opencl_context_t* openclContext);

#ifdef __cplusplus
};
#endif

#endif
