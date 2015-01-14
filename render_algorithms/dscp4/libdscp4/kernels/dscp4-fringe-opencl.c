#ifndef DSCP4_FRINGE_H__
#define DSCP4_FRINGE_H__

#include "dscp4-fringe-opencl.h"
#include <CL/cl.h>
#include <CL/cl_gl.h>

#ifdef __cplusplus
extern "C" {
#endif

	dscp4_fringe_opencl_context_t* dscp4_fringe_opencl_CreateContext(dscp4_fringe_context_t *fringeContext)
	{
		dscp4_fringe_opencl_context_t * context = (dscp4_fringe_opencl_context_t*)malloc(sizeof(dscp4_fringe_opencl_context_t));

		cl_platform_id platform_id = NULL;
		cl_device_id device_id = NULL;
		cl_uint ret_num_devices;
		cl_uint ret_num_platforms;
		cl_int ret = clGetPlatformIDs(1, &platform_id, &ret_num_platforms);

		ret = clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1,
			&device_id, &ret_num_devices);

		context->num_gpus = ret_num_devices;

		context->cl_context = clCreateContext(NULL, context->num_gpus, &device_id, NULL, NULL, &ret);
		return context;

	}

	void dscp4_fringe_opencl_DestroyContext(dscp4_fringe_opencl_context_t** openclContext)
	{

	}

	void dscp4_fringe_opencl_ComputeFringe(dscp4_fringe_opencl_context_t* openclContext)
	{


	}

#ifdef __cplusplus
};
#endif

#endif