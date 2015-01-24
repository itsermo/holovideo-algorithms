#ifndef DSCP4_FRINGE_CUDA_H__
#define DSCP4_FRINGE_CUDA_H__

#include <stdbool.h>
#include "../dscp4_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

	typedef struct
	{
		dscp4_fringe_context_t * fringe_context;
		struct cudaGraphicsResource * stereogram_rgba_cuda_resource;
		struct cudaGraphicsResource * stereogram_depth_cuda_resource;
		struct cudaGraphicsResource ** fringe_cuda_resources;
		unsigned char * spec_buffer;
		float * wafel_positions;
		unsigned char * wafel_buffers;
		struct cudaDeviceProp * gpu_properties;
		int num_gpus;
	} dscp4_fringe_cuda_context_t;

	dscp4_fringe_cuda_context_t* dscp4_fringe_cuda_CreateContext(dscp4_fringe_context_t *fringeContext);

	void dscp4_fringe_cuda_DestroyContext(dscp4_fringe_cuda_context_t** cudaContext);

	void dscp4_fringe_cuda_ComputeFringe(dscp4_fringe_cuda_context_t* cudaContext);

#ifdef __cplusplus
};
#endif

#endif