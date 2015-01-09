#ifndef DSCP4_FRINGE_H__
#define DSCP4_FRINGE_H__

#include <stdbool.h>
#include "../dscp4_defs.h"

#ifdef __cplusplus
extern "C" {
#endif

	typedef struct
	{
		dscp4_fringe_context_t fringe_context;
		int num_gpus;
	} dscp4_fringe_cuda_context_t;

	dscp4_fringe_cuda_context_t* dscp4_fringe_cuda_CreateContext(dscp4_fringe_context_t fringeContext);

	void dscp4_fringe_cuda_DestroyContext(dscp4_fringe_cuda_context_t** cudaContext);

	char * dscp4_fringe_cuda_HelloWorld();

	void dscp4_fringe_cuda_ComputeFringe();

#ifdef __cplusplus
};
#endif

#endif