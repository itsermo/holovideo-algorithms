#include "dscp4-fringe-opencl.h"

#ifdef DSCP4_HAVE_LOG4CXX
#include <log4cxx/logger.h>
#else
#define LOG4CXX_TRACE(logger, expression)    
#define LOG4CXX_DEBUG(logger, expression)    
#define LOG4CXX_INFO(logger, expression)   
#define LOG4CXX_WARN(logger, expression)    
#define LOG4CXX_ERROR(logger, expression)    
#define LOG4CXX_FATAL(logger, expression) 
#endif

#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <math.h>
#include <stdio.h>

#include <CL/cl.h>
#include <CL/cl_gl.h>

#if WIN32
#include <Windows.h>
#include <wingdi.h>
#elif __linux__
#include <GL/glx.h>
#endif

#include <GL/gl.h>

#ifdef DSCP4_HAVE_LOG4CXX
static log4cxx::LoggerPtr DSCP4_OPENCL_LOGGER = log4cxx::Logger::getLogger("edu.mit.media.obmg.holovideo.dscp4.opencl");
#endif

#define CHECK_OPENCL_RC(rc, what)								\
if (rc != CL_SUCCESS)											\
{																\
	LOG4CXX_ERROR(DSCP4_OPENCL_LOGGER, what)					\
	LOG4CXX_ERROR(DSCP4_OPENCL_LOGGER, "OpenCL Error: " << rc)	\
}																\

#ifdef __cplusplus
extern "C" {
#endif


	//int IsExtensionSupported(
	//	const char* support_str, const char* ext_string, size_t ext_buffer_size)
	//{
	//	size_t offset = 0;
	//	const char* space_substr = strnstr(ext_string + offset, " ", ext_buffer_size - offset);
	//	size_t space_pos = space_substr ? space_substr - ext_string : 0;
	//	while (space_pos < ext_buffer_size)
	//	{
	//		if (strncmp(support_str, ext_string + offset, space_pos) == 0)
	//		{
	//			// Device supports requested extension!
	//			printf("INFO OpenCL -- Found extension support '%s'!\n", support_str);
	//			return 1;
	//		}
	//		// Keep searching -- skip to next token string
	//		offset = space_pos + 1;
	//		space_substr = strnstr(ext_string + offset, " ", ext_buffer_size - offset);
	//		space_pos = space_substr ? space_substr - ext_string : 0;
	//	}
	//	printf("ERROR OpenCL -- Extension not supported '%s'!\n", support_str);
	//	return 0;
	//}

	dscp4_fringe_opencl_context_t* dscp4_fringe_opencl_CreateContext(dscp4_fringe_context_t *fringeContext, int *glContext)
	{

		cl_int ret = CL_SUCCESS;

		dscp4_fringe_opencl_context_t * context = new dscp4_fringe_opencl_context_t;
		*context = { 0 };

		LOG4CXX_DEBUG(DSCP4_OPENCL_LOGGER, "Creating OpenCL Context");

		const unsigned int num_output_buffers = fringeContext->display_options.num_heads / fringeContext->display_options.num_heads_per_gpu;

#if defined (__APPLE__) || defined(MACOSX)
		static const char* CL_GL_SHARING_EXT = "cl_APPLE_gl_sharing";
#else
		static const char* CL_GL_SHARING_EXT = "cl_khr_gl_sharing";
#endif
		// Get string containing supported device extensions
		std::ifstream programFileStream;
		std::string programString;
		
		LOG4CXX_DEBUG(DSCP4_OPENCL_LOGGER, "Reading OpenCL kernel from " << fringeContext->algorithm_options.opencl_kernel_filename)
		programFileStream.open(fringeContext->algorithm_options.opencl_kernel_filename);

		if (!programFileStream.is_open())
		{
			LOG4CXX_ERROR(DSCP4_OPENCL_LOGGER, "Could not find OpenCL kernel file " << fringeContext->algorithm_options.opencl_kernel_filename);
			dscp4_fringe_opencl_DestroyContext(&context);
			return NULL;
		}

		programString = std::string((std::istreambuf_iterator<char>(programFileStream)), (std::istreambuf_iterator<char>()));
		programFileStream.close();

		context->fringe_context = fringeContext;

		cl_platform_id platformID = NULL;
		cl_device_id deviceID = NULL;
		cl_uint numDevices;
		cl_uint numPlatforms;
		CHECK_OPENCL_RC(clGetPlatformIDs(1, &platformID, &numPlatforms), "Getting OpenCL platform IDs")

		LOG4CXX_DEBUG(DSCP4_OPENCL_LOGGER, "Found " << numPlatforms << " OpenCL platforms")

		CHECK_OPENCL_RC(clGetDeviceIDs(platformID, CL_DEVICE_TYPE_GPU, 1, &deviceID, &numDevices), "Getting OpenCL device ID for GPUs")

		size_t extensionSize;
		clGetDeviceInfo(deviceID, CL_DEVICE_EXTENSIONS, 0, NULL, &extensionSize);
		char *extensions = new char[extensionSize];
		clGetDeviceInfo(deviceID, CL_DEVICE_EXTENSIONS, extensionSize, extensions, NULL);

		std::stringstream extensionsStringStream(extensions);
		std::string extension;
		while (std::getline(extensionsStringStream, extension, ' '))
		{
			if (extension == "cl_khr_gl_depth_images")
				context->have_cl_gl_depth_images_extension = true;
			else if (extension == "cl_khr_gl_sharing")
				context->have_cl_gl_sharing_extension = true;
		}

		delete[] extensions;

		LOG4CXX_DEBUG(DSCP4_OPENCL_LOGGER, "Found " << numDevices << " OpenCL GPU devices")

		context->num_gpus = numDevices;

#ifdef _WIN32
		cl_context_properties properties[] = {
			CL_GL_CONTEXT_KHR, (cl_context_properties)wglGetCurrentContext(),
			CL_WGL_HDC_KHR, (cl_context_properties)wglGetCurrentDC(),
			CL_CONTEXT_PLATFORM, (cl_context_properties)platformID,
			0
		};
#elif defined(__linux__)
		cl_context_properties properties[] = {
			CL_GL_CONTEXT_KHR, (cl_context_properties)glXGetCurrentContext(),
			CL_GLX_DISPLAY_KHR, (cl_context_properties)glXGetCurrentDisplay(),
			CL_CONTEXT_PLATFORM, (cl_context_properties)platformID,
			0
		};
#elif defined(__APPLE__)
		CGLContextObj glContext = CGLGetCurrentContext();
		CGLShareGroupObj shareGroup = CGLGetShareGroup(glContext);
		cl_context_properties properties[] = {
			CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE,
			(cl_context_properties)shareGroup,
		};
#endif

		context->cl_context = clCreateContext(properties, 1, &deviceID, NULL, NULL, &ret);
		CHECK_OPENCL_RC(ret, "Could not create OpenCL context")
		context->command_queue = clCreateCommandQueue((cl_context)context->cl_context, deviceID, 0, &ret);
		CHECK_OPENCL_RC(ret, "Could not create OpenCL command queue")

		context->stereogram_rgba_opencl_resource = clCreateFromGLTexture2D((cl_context)context->cl_context, CL_MEM_READ_ONLY, GL_TEXTURE_2D, 0, fringeContext->stereogram_gl_fbo_color, &ret);
		CHECK_OPENCL_RC(ret, "Could not create OpenCL stereogram RGBA memory resource from OpenGL")

		if (context->have_cl_gl_depth_images_extension)
			context->stereogram_depth_opencl_resource = clCreateFromGLTexture2D((cl_context)context->cl_context, CL_MEM_READ_ONLY, GL_TEXTURE_2D, 0, fringeContext->stereogram_gl_fbo_depth, &ret);
		else
			context->stereogram_depth_opencl_resource = clCreateFromGLBuffer((cl_context)context->cl_context, CL_MEM_READ_ONLY, fringeContext->stereogram_gl_depth_buf_in, &ret);

		CHECK_OPENCL_RC(ret, "Could not create OpenCL stereogram DEPTH memory resource from OpenGL")

		context->fringe_opencl_resources = new void*[num_output_buffers];

		for (unsigned int i = 0; i < num_output_buffers; i++)
		{
			context->fringe_opencl_resources[i] = (cl_mem)clCreateFromGLTexture2D((cl_context)context->cl_context, CL_MEM_WRITE_ONLY, GL_TEXTURE_2D, 0, fringeContext->fringe_gl_tex_out[i], &ret);
			CHECK_OPENCL_RC(ret, "Could not create OpenCL fringe output texture memory resource " << i << " from OpenGL")
		}

		const char * sourceStr = programString.c_str();
		size_t sourceSize = programString.size();
		context->program = clCreateProgramWithSource((cl_context)context->cl_context, 1, &sourceStr, &sourceSize, &ret);
		CHECK_OPENCL_RC(ret, "Could not create OpenCL program from source")

		ret = clBuildProgram((cl_program)context->program, 1, &deviceID, NULL, NULL, NULL);
		CHECK_OPENCL_RC(ret, "Could not build OpenCL program from source")

		if (context->have_cl_gl_depth_images_extension)
			context->kernel = clCreateKernel((cl_program)context->program, "test_hologram2", &ret);
		else
			context->kernel = clCreateKernel((cl_program)context->program, "test_hologram", &ret);

		CHECK_OPENCL_RC(ret, "Could not create OpenCL kernel object")

		//context->kernel_global_worksize[0] = 64 - (context->fringe_context->algorithm_options.num_wafels_per_scanline % 64) + context->fringe_context->algorithm_options.num_wafels_per_scanline;
		//context->kernel_global_worksize[1] = context->fringe_context->algorithm_options.num_scanlines;
		//context->kernel_local_worksize[0] = 64;
		//context->kernel_local_worksize[1] = 1;

		ret = clSetKernelArg((cl_kernel)context->kernel, 1, sizeof(cl_mem), &(context->stereogram_rgba_opencl_resource));
		ret = clSetKernelArg((cl_kernel)context->kernel, 2, sizeof(cl_mem), &(context->stereogram_depth_opencl_resource));
		ret = clSetKernelArg((cl_kernel)context->kernel, 4, sizeof(cl_uint), &context->fringe_context->algorithm_options.num_wafels_per_scanline);
		ret = clSetKernelArg((cl_kernel)context->kernel, 5, sizeof(cl_uint), &context->fringe_context->algorithm_options.num_scanlines);
		ret = clSetKernelArg((cl_kernel)context->kernel, 6, sizeof(cl_uint), &context->fringe_context->algorithm_options.cache.stereogram_res_x);
		ret = clSetKernelArg((cl_kernel)context->kernel, 7, sizeof(cl_uint), &context->fringe_context->algorithm_options.cache.stereogram_res_y);
		ret = clSetKernelArg((cl_kernel)context->kernel, 8, sizeof(cl_uint), &context->fringe_context->algorithm_options.cache.stereogram_tile_x);
		ret = clSetKernelArg((cl_kernel)context->kernel, 9, sizeof(cl_uint), &context->fringe_context->algorithm_options.cache.stereogram_tile_y);
		ret = clSetKernelArg((cl_kernel)context->kernel, 10, sizeof(cl_uint), &context->fringe_context->algorithm_options.cache.output_buffer_res_x);
		ret = clSetKernelArg((cl_kernel)context->kernel, 11, sizeof(cl_uint), &context->fringe_context->algorithm_options.cache.output_buffer_res_y);

		return context;
	}

	void dscp4_fringe_opencl_DestroyContext(dscp4_fringe_opencl_context_t** openclContext)
	{
		cl_int ret = 0;

		LOG4CXX_DEBUG(DSCP4_OPENCL_LOGGER, "Destroying OpenCL context")

		ret = clFlush((cl_command_queue)(*openclContext)->command_queue);
		CHECK_OPENCL_RC(ret, "Could not flush OpenCL command queue")
		ret = clFinish((cl_command_queue)(*openclContext)->command_queue);
		CHECK_OPENCL_RC(ret, "Could not finish OpenCL command queue")
		ret = clReleaseKernel((cl_kernel)(*openclContext)->kernel);
		CHECK_OPENCL_RC(ret, "Could not release OpenCL kernel")
		ret = clReleaseProgram((cl_program)(*openclContext)->program);
		CHECK_OPENCL_RC(ret, "Could not release OpenCL program")


		const unsigned int num_output_buffers =
			(*openclContext)->fringe_context->display_options.num_heads /
			(*openclContext)->fringe_context->display_options.num_heads_per_gpu;

		for (unsigned int i = 0; i < num_output_buffers; i++)
		{
			ret = clReleaseMemObject((cl_mem)(*openclContext)->fringe_opencl_resources[i]);
			CHECK_OPENCL_RC(ret, "Could not release fringe texture " << i << " OpenCL memory resource")

		}

		delete[] (*openclContext)->fringe_opencl_resources;

		ret = clReleaseMemObject((cl_mem)(*openclContext)->stereogram_rgba_opencl_resource);
		CHECK_OPENCL_RC(ret, "Could not release stereogram RGBA OpenCL memory resource")
		ret = clReleaseMemObject((cl_mem)(*openclContext)->stereogram_depth_opencl_resource);
		CHECK_OPENCL_RC(ret, "Could not release stereogram DEPTH OpenCL memory resource")


		ret = clReleaseCommandQueue((cl_command_queue)(*openclContext)->command_queue);
		CHECK_OPENCL_RC(ret, "Could not release OpenCL command queue")

		ret = clReleaseContext((cl_context)(*openclContext)->cl_context);
		CHECK_OPENCL_RC(ret, "Could not release fringe OpenCL context")

		delete *openclContext;
		*openclContext = NULL;
	}

	void dscp4_fringe_opencl_ComputeFringe(dscp4_fringe_opencl_context_t* openclContext)
	{
		const unsigned int num_output_buffers = openclContext->fringe_context->algorithm_options.cache.num_output_buffers;

		cl_int ret = 0;

		cl_event *event = new cl_event[num_output_buffers * 3];

		glFinish();

		for (unsigned int i = 0; i < num_output_buffers; i++)
		{
			ret = clSetKernelArg((cl_kernel)openclContext->kernel, 0, sizeof(cl_mem), &(openclContext->fringe_opencl_resources[i]));
			ret = clSetKernelArg((cl_kernel)openclContext->kernel, 3, sizeof(cl_uint), &i);

			ret = clEnqueueAcquireGLObjects((cl_command_queue)openclContext->command_queue, 1, (const cl_mem*)&openclContext->stereogram_rgba_opencl_resource, 0, NULL, NULL);
			ret = clEnqueueAcquireGLObjects((cl_command_queue)openclContext->command_queue, 1, (const cl_mem*)&openclContext->stereogram_depth_opencl_resource, 0, NULL, NULL);
			ret = clEnqueueAcquireGLObjects((cl_command_queue)openclContext->command_queue, 1, (const cl_mem*)&openclContext->fringe_opencl_resources[i], 0, NULL, &event[i*num_output_buffers]);

			ret = clEnqueueNDRangeKernel((cl_command_queue)openclContext->command_queue, (cl_kernel)openclContext->kernel, 2, NULL,
				(const size_t*)openclContext->fringe_context->algorithm_options.cache.opencl_global_workgroup_size, (const size_t*)openclContext->fringe_context->algorithm_options.opencl_local_workgroup_size, 1, &event[i*num_output_buffers], &event[i*num_output_buffers+1]);

			ret = clEnqueueReleaseGLObjects((cl_command_queue)openclContext->command_queue, 1, (const cl_mem*)&openclContext->stereogram_rgba_opencl_resource, 0, NULL, NULL);
			ret = clEnqueueReleaseGLObjects((cl_command_queue)openclContext->command_queue, 1, (const cl_mem*)&openclContext->fringe_opencl_resources[i], 1, &event[i*num_output_buffers+1], &event[i*num_output_buffers+2]);
			ret = clEnqueueReleaseGLObjects((cl_command_queue)openclContext->command_queue, 1, (const cl_mem*)&openclContext->stereogram_depth_opencl_resource, 0, NULL, NULL);
		}

		for (unsigned int i = 0; i < num_output_buffers; i++)
		{
			ret = clWaitForEvents(1, &event[i* num_output_buffers + 2]);
		}

		delete[] event;
	}

#ifdef __cplusplus
};
#endif
