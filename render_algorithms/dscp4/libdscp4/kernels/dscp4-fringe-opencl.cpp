#include "dscp4-fringe-opencl.h"

#ifdef DSCP4_HAVE_LOG4CXX
#include <log4cxx/logger.h>
#include <log4cxx/appender.h>
#include <log4cxx/patternlayout.h>
#include <log4cxx/consoleappender.h>
#include <log4cxx/basicconfigurator.h>
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

#ifndef __APPLE__
#include <CL/cl.h>
#include <CL/cl_gl.h>
#else
#include <OpenCL/cl.h>
#include <OpenCL/cl_gl.h>
#include <OpenCL/cl_gl_ext.h>
#endif

#if WIN32
#include <Windows.h>
#include <wingdi.h>
#elif __linux__
#include <GL/glx.h>
#endif

#ifndef __APPLE__
#include <GL/gl.h>
#else
#include <OpenGL/gl.h>
#include <OpenGL/CGLTypes.h>
#include <OpenGL/CGLDevice.h>
#include <OpenGL/CGLCurrent.h>
#endif

#ifdef DSCP4_HAVE_LOG4CXX
static log4cxx::LoggerPtr DSCP4_OPENCL_LOGGER = log4cxx::Logger::getLogger("edu.mit.media.obmg.holovideo.dscp4.opencl");
#endif

#define CHECK_OPENCL_RC(rc, what)												\
if (rc != CL_SUCCESS)															\
{																				\
	LOG4CXX_ERROR(DSCP4_OPENCL_LOGGER, what << ": " << clGetErrorString(rc))	\
}																				\

#ifdef __cplusplus
extern "C" {
#endif

	const char *clGetErrorString(cl_int error);

	dscp4_fringe_opencl_context_t* dscp4_fringe_opencl_CreateContext(dscp4_fringe_context_t *fringeContext, int *glContext, void * logAppender)
	{

		cl_int ret = CL_SUCCESS;

#ifdef DSCP4_HAVE_LOG4CXX
		if (!logAppender)
		{
			log4cxx::BasicConfigurator::resetConfiguration();

#ifdef WIN32
			log4cxx::PatternLayoutPtr logLayoutPtr = new log4cxx::PatternLayout(L"%-5p %m%n");
#else
			log4cxx::PatternLayoutPtr logLayoutPtr = new log4cxx::PatternLayout("%-5p %m%n");
#endif

			log4cxx::ConsoleAppenderPtr logAppenderPtr = new log4cxx::ConsoleAppender(logLayoutPtr);
			log4cxx::BasicConfigurator::configure(logAppenderPtr);
		}
		else
		{
			if (DSCP4_OPENCL_LOGGER->getParent() != nullptr && DSCP4_OPENCL_LOGGER->getParent()->getAllAppenders().size() == 0 && DSCP4_OPENCL_LOGGER->getAllAppenders().size() == 0)
				DSCP4_OPENCL_LOGGER->addAppender((log4cxx::Appender*)logAppender);
		}
#endif
		dscp4_fringe_opencl_context_t * context = new dscp4_fringe_opencl_context_t;
		*context = { 0 };

		context->fringe_context = fringeContext;


		LOG4CXX_DEBUG(DSCP4_OPENCL_LOGGER, "Creating OpenCL Context");

		const unsigned int num_fringe_buffers = fringeContext->display_options.num_heads / fringeContext->display_options.num_heads_per_gpu;

#if defined (__APPLE__) || defined(MACOSX)
		static const char* CL_GL_SHARING_EXT = "cl_APPLE_gl_sharing";
#else
		static const char* CL_GL_SHARING_EXT = "cl_khr_gl_sharing";
#endif
		// Get string containing supported device extensions
		std::ifstream programFileStream;
		std::string programString;
		
		LOG4CXX_DEBUG(DSCP4_OPENCL_LOGGER, "Reading OpenCL kernel from " << fringeContext->kernel_file_path)
		programFileStream.open(fringeContext->kernel_file_path);

		if (!programFileStream.is_open())
		{
			LOG4CXX_ERROR(DSCP4_OPENCL_LOGGER, "Could not find OpenCL kernel file " << fringeContext->algorithm_options->opencl_kernel_filename);
			delete[] context;
			return NULL;
		}

		programString = std::string((std::istreambuf_iterator<char>(programFileStream)), (std::istreambuf_iterator<char>()));
		programFileStream.close();

		cl_platform_id platformID = NULL;
		cl_device_id deviceID = NULL;
		cl_uint numDevices;
		cl_uint numPlatforms;
		CHECK_OPENCL_RC(clGetPlatformIDs(1, &platformID, &numPlatforms), "Getting OpenCL platform IDs")
		if (!platformID)
		{
			dscp4_fringe_opencl_DestroyContext(&context);
			return nullptr;
		}

		LOG4CXX_DEBUG(DSCP4_OPENCL_LOGGER, "Found " << numPlatforms << " OpenCL platforms")

		CHECK_OPENCL_RC(clGetDeviceIDs(platformID, CL_DEVICE_TYPE_GPU, 1, &deviceID, &numDevices), "Getting OpenCL device ID for GPUs")
		if (!deviceID || numDevices == 0)
		{
			dscp4_fringe_opencl_DestroyContext(&context);
			return nullptr;
		}

		size_t extensionSize;
		CHECK_OPENCL_RC(clGetDeviceInfo(deviceID, CL_DEVICE_EXTENSIONS, 0, NULL, &extensionSize), "Getting OpenCL device extension string size")

		char *extensions = new char[extensionSize];
		CHECK_OPENCL_RC(clGetDeviceInfo(deviceID, CL_DEVICE_EXTENSIONS, extensionSize, extensions, NULL), "Getting OpenCL extensions")

		std::stringstream extensionsStringStream(extensions);
		std::string extension;
		while (std::getline(extensionsStringStream, extension, ' '))
		{
			if (extension == "cl_khr_gl_depth_images")
				context->have_cl_gl_depth_images_extension = true;
			else if (extension == "cl_khr_gl_sharing")
				context->have_cl_gl_sharing_extension = true;
			else if (extension == "cl_khr_fp64" || extension == "cl_amd_fp64")
				context->have_cl_double_precision_extension = true;
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
		CGLContextObj glContextApple = CGLGetCurrentContext();
		CGLShareGroupObj shareGroup = CGLGetShareGroup(glContextApple);
		cl_context_properties properties[] = {
			CL_CONTEXT_PROPERTY_USE_CGL_SHAREGROUP_APPLE,
			(cl_context_properties)shareGroup,
		};
#endif

		context->cl_context = clCreateContext(properties, 1, &deviceID, NULL, NULL, &ret);
		CHECK_OPENCL_RC(ret, "Could not create OpenCL context")
		if (!context->cl_context)
		{
			dscp4_fringe_opencl_DestroyContext(&context);
			return nullptr;
		}

		context->command_queue = clCreateCommandQueue((cl_context)context->cl_context, deviceID, 0, &ret);
		CHECK_OPENCL_RC(ret, "Could not create OpenCL command queue")
		if (!context->command_queue)
		{
			dscp4_fringe_opencl_DestroyContext(&context);
			return nullptr;
		}

		context->stereogram_rgba_opencl_resource = clCreateFromGLTexture2D((cl_context)context->cl_context, CL_MEM_READ_ONLY, GL_TEXTURE_2D, 0, fringeContext->stereogram_gl_fbo_color, &ret);
		CHECK_OPENCL_RC(ret, "Could not create OpenCL stereogram RGBA memory resource from OpenGL")
		if (!context->stereogram_rgba_opencl_resource)
		{
			dscp4_fringe_opencl_DestroyContext(&context);
			return nullptr;
		}

		if (context->have_cl_gl_depth_images_extension)
			context->stereogram_depth_opencl_resource = clCreateFromGLTexture2D((cl_context)context->cl_context, CL_MEM_READ_ONLY, GL_TEXTURE_2D, 0, fringeContext->stereogram_gl_fbo_depth, &ret);
		else
			context->stereogram_depth_opencl_resource = clCreateFromGLBuffer((cl_context)context->cl_context, CL_MEM_READ_ONLY, fringeContext->stereogram_gl_depth_pbo_in, &ret);

		CHECK_OPENCL_RC(ret, "Could not create OpenCL stereogram DEPTH memory resource from OpenGL")
		if (!context->stereogram_depth_opencl_resource)
		{
			dscp4_fringe_opencl_DestroyContext(&context);
			return nullptr;
		}

		context->fringe_opencl_resources = new void*[num_fringe_buffers];

		for (unsigned int i = 0; i < num_fringe_buffers; i++)
		{
			context->fringe_opencl_resources[i] = (cl_mem)clCreateFromGLTexture2D((cl_context)context->cl_context, CL_MEM_READ_WRITE, GL_TEXTURE_2D, 0, fringeContext->fringe_gl_tex_out[i], &ret);
			CHECK_OPENCL_RC(ret, "Could not create OpenCL fringe output texture memory resource " << i << " from OpenGL")
			if (!context->fringe_opencl_resources[i])
			{
				dscp4_fringe_opencl_DestroyContext(&context);
				return nullptr;
			}
		}

		const char * sourceStr = programString.c_str();
		size_t sourceSize = programString.size();
		context->program = clCreateProgramWithSource((cl_context)context->cl_context, 1, &sourceStr, &sourceSize, &ret);
		CHECK_OPENCL_RC(ret, "Could not create OpenCL program from source")
		if (!context->program)
		{
			dscp4_fringe_opencl_DestroyContext(&context);
			return nullptr;
		}

		std::string buildOptions;

#ifndef _DEBUG
		// Optimizations (they don't appear to do much, just 2-3 fps increase)
		buildOptions += "-cl-mad-enable -cl-no-signed-zeros -cl-unsafe-math-optimizations -cl-finite-math-only -cl-fast-relaxed-math ";
#endif

		// Use double precision, where possible
		if (context->have_cl_double_precision_extension)
			buildOptions += " -D CONFIG_USE_DOUBLE";
		else
			LOG4CXX_WARN(DSCP4_OPENCL_LOGGER, "Your OpenCL device does not support double precision computation extensions. Continuing, but your results will be totally fucked up")

		// Use depth texture if OpenCL supports it, otherwise use PBO copied from depth texture
		if (context->have_cl_gl_depth_images_extension)
			buildOptions += " -D CONFIG_USE_DEPTH_TEXTURE";

		ret = clBuildProgram((cl_program)context->program, 1, &deviceID, buildOptions.c_str(), NULL, NULL);
		CHECK_OPENCL_RC(ret, "Could not build OpenCL program from source")
		if (ret == CL_BUILD_PROGRAM_FAILURE)
		{
			size_t len;
			clGetProgramBuildInfo((cl_program)context->program, deviceID, CL_PROGRAM_BUILD_LOG, 0, NULL, &len);

			char * log = new char[len];
			clGetProgramBuildInfo((cl_program)context->program, deviceID, CL_PROGRAM_BUILD_LOG, len, log, &len);

			LOG4CXX_ERROR(DSCP4_OPENCL_LOGGER, "OpenCL Build Log:\n" << log)
			delete[] log;

			dscp4_fringe_opencl_DestroyContext(&context);
			return nullptr;
		}
		else if (ret != CL_SUCCESS)
		{
			LOG4CXX_ERROR(DSCP4_OPENCL_LOGGER, "Undefined OpenCL kernel build error")
			dscp4_fringe_opencl_DestroyContext(&context);
			return nullptr;
		}

		context->kernel = clCreateKernel((cl_program)context->program, "computeFringe", &ret);
		CHECK_OPENCL_RC(ret, "Could not create OpenCL kernel object")

		if (!context->kernel)
		{
			dscp4_fringe_opencl_DestroyContext(&context);
			return nullptr;
		}

		// the Mark IV architecture specifications call for a buffer that is slightly bigger
		// than the buffer that the OS can see, so our algorithm in OpenCL will
		// store to this buffer, then we must appropriately copy data to
		// a buffer of appropriate size. also zero the data and put alpha at 255
		cl_uchar * specBuffer = new cl_uchar[context->fringe_context->display_options.head_res_x_spec * context->fringe_context->display_options.head_res_y_spec * context->fringe_context->display_options.num_heads * 4];
		for (unsigned int i = 0; i < context->fringe_context->display_options.head_res_x_spec * context->fringe_context->display_options.head_res_y_spec * context->fringe_context->display_options.num_heads; i++)
		{
			specBuffer[i * 4] = 0;
			specBuffer[i * 4 + 1] = 0;
			specBuffer[i * 4 + 2] = 0;
			specBuffer[i * 4 + 3] = 255;
		}

		context->framebuffer_opencl_output = clCreateBuffer(
			(cl_context)context->cl_context,
			CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
			context->fringe_context->display_options.head_res_x_spec
			* context->fringe_context->display_options.head_res_y_spec
			* context->fringe_context->display_options.num_heads
			* 4,
			specBuffer,
			&ret);
		CHECK_OPENCL_RC(ret, "Could not create OpenCL output buffer")

		delete[] specBuffer;

		if (!context->framebuffer_opencl_output)
		{
			dscp4_fringe_opencl_DestroyContext(&context);
			return nullptr;
		}

		ret = clSetKernelArg((cl_kernel)context->kernel, 0, sizeof(cl_mem), &(context->framebuffer_opencl_output));
		if (ret != CL_SUCCESS)
		{
			LOG4CXX_ERROR(DSCP4_OPENCL_LOGGER, "Could not set argument 0 (framebuffer output) in OpenCL kernel")
			dscp4_fringe_opencl_DestroyContext(&context);
			return nullptr;
		}

		ret = clSetKernelArg((cl_kernel)context->kernel, 1, sizeof(cl_mem), &(context->stereogram_rgba_opencl_resource));
		if (ret != CL_SUCCESS)
		{
			LOG4CXX_ERROR(DSCP4_OPENCL_LOGGER, "Could not set argument 1 (stereogram RGBA) in OpenCL kernel")
				dscp4_fringe_opencl_DestroyContext(&context);
			return nullptr;
		}

		ret = clSetKernelArg((cl_kernel)context->kernel, 2, sizeof(cl_mem), &(context->stereogram_depth_opencl_resource));
		if (ret != CL_SUCCESS)
		{
			LOG4CXX_ERROR(DSCP4_OPENCL_LOGGER, "Could not set argument 2 (stereogram DEPTH) in OpenCL kernel")
				dscp4_fringe_opencl_DestroyContext(&context);
			return nullptr;
		}

		ret = clSetKernelArg((cl_kernel)context->kernel, 3, sizeof(cl_uint), &context->fringe_context->algorithm_options->num_wafels_per_scanline);
		if (ret != CL_SUCCESS)
		{
			LOG4CXX_ERROR(DSCP4_OPENCL_LOGGER, "Could not set argument 3 (number of wafels per scanline) in OpenCL kernel")
				dscp4_fringe_opencl_DestroyContext(&context);
			return nullptr;
		}

		ret = clSetKernelArg((cl_kernel)context->kernel, 4, sizeof(cl_uint), &context->fringe_context->algorithm_options->num_scanlines);
		if (ret != CL_SUCCESS)
		{
			LOG4CXX_ERROR(DSCP4_OPENCL_LOGGER, "Could not set argument 4 (number of scanlines) in OpenCL kernel")
				dscp4_fringe_opencl_DestroyContext(&context);
			return nullptr;
		}

		ret = clSetKernelArg((cl_kernel)context->kernel, 5, sizeof(cl_uint), &context->fringe_context->algorithm_options->cache.stereogram_res_x);
		if (ret != CL_SUCCESS)
		{
			LOG4CXX_ERROR(DSCP4_OPENCL_LOGGER, "Could not set argument 5 (stereogram X resolution) in OpenCL kernel")
				dscp4_fringe_opencl_DestroyContext(&context);
			return nullptr;
		}

		ret = clSetKernelArg((cl_kernel)context->kernel, 6, sizeof(cl_uint), &context->fringe_context->algorithm_options->cache.stereogram_res_y);
		if (ret != CL_SUCCESS)
		{
			LOG4CXX_ERROR(DSCP4_OPENCL_LOGGER, "Could not set argument 6 (stereogram Y resolution) in OpenCL kernel")
				dscp4_fringe_opencl_DestroyContext(&context);
			return nullptr;
		}

		ret = clSetKernelArg((cl_kernel)context->kernel, 7, sizeof(cl_uint), &context->fringe_context->algorithm_options->cache.stereogram_num_tiles_x);
		if (ret != CL_SUCCESS)
		{
			LOG4CXX_ERROR(DSCP4_OPENCL_LOGGER, "Could not set argument 7 (number of tiles in X direction in stereogram) in OpenCL kernel")
				dscp4_fringe_opencl_DestroyContext(&context);
			return nullptr;
		}

		ret = clSetKernelArg((cl_kernel)context->kernel, 8, sizeof(cl_uint), &context->fringe_context->algorithm_options->cache.stereogram_num_tiles_y);
		if (ret != CL_SUCCESS)
		{
			LOG4CXX_ERROR(DSCP4_OPENCL_LOGGER, "Could not set argument 8 (number of tiles in Y direction in stereogram) in OpenCL kernel")
				dscp4_fringe_opencl_DestroyContext(&context);
			return nullptr;
		}

		ret = clSetKernelArg((cl_kernel)context->kernel, 9, sizeof(cl_uint), &context->fringe_context->algorithm_options->cache.fringe_buffer_res_x);
		if (ret != CL_SUCCESS)
		{
			LOG4CXX_ERROR(DSCP4_OPENCL_LOGGER, "Could not set argument 9 (output buffer X resolution) in OpenCL kernel")
				dscp4_fringe_opencl_DestroyContext(&context);
			return nullptr;
		}

		ret = clSetKernelArg((cl_kernel)context->kernel, 10, sizeof(cl_uint), &context->fringe_context->algorithm_options->cache.fringe_buffer_res_y);
		if (ret != CL_SUCCESS)
		{
			LOG4CXX_ERROR(DSCP4_OPENCL_LOGGER, "Could not set argument 10 (output buffer Y resolution) in OpenCL kernel")
				dscp4_fringe_opencl_DestroyContext(&context);
			return nullptr;
		}

		ret = clSetKernelArg((cl_kernel)context->kernel, 15, context->fringe_context->algorithm_options->num_wafels_per_scanline*sizeof(cl_float), NULL);
		if (ret != CL_SUCCESS)
		{
			LOG4CXX_ERROR(DSCP4_OPENCL_LOGGER, "Could not set argument 15 (wafel position buffer size) in OpenCL kernel")
				dscp4_fringe_opencl_DestroyContext(&context);
			return nullptr;
		}

		ret = clSetKernelArg((cl_kernel)context->kernel, 22, sizeof(cl_uint), &context->fringe_context->algorithm_options->cache.num_samples_per_wafel);
		if (ret != CL_SUCCESS)
		{
			LOG4CXX_ERROR(DSCP4_OPENCL_LOGGER, "Could not set argument 22 (number of samples per wafel) in OpenCL kernel")
				dscp4_fringe_opencl_DestroyContext(&context);
			return nullptr;
		}

		ret = clSetKernelArg((cl_kernel)context->kernel, 23, sizeof(cl_float), &context->fringe_context->algorithm_options->cache.sample_pitch);
		if (ret != CL_SUCCESS)
		{
			LOG4CXX_ERROR(DSCP4_OPENCL_LOGGER, "Could not set argument 23 (sample pitch) in OpenCL kernel")
				dscp4_fringe_opencl_DestroyContext(&context);
			return nullptr;
		}

		ret = clSetKernelArg((cl_kernel)context->kernel, 24, sizeof(cl_float), &context->fringe_context->algorithm_options->cache.z_span);
		if (ret != CL_SUCCESS)
		{
			LOG4CXX_ERROR(DSCP4_OPENCL_LOGGER, "Could not set argument 24 (Z-span) in OpenCL kernel")
				dscp4_fringe_opencl_DestroyContext(&context);
			return nullptr;
		}

		ret = clSetKernelArg((cl_kernel)context->kernel, 26, sizeof(cl_uint), &context->fringe_context->display_options.num_aom_channels);
		if (ret != CL_SUCCESS)
		{
			LOG4CXX_ERROR(DSCP4_OPENCL_LOGGER, "Could not set argument 26 (number of AOM channels) in OpenCL kernel")
				dscp4_fringe_opencl_DestroyContext(&context);
			return nullptr;
		}

		ret = clSetKernelArg((cl_kernel)context->kernel, 27, sizeof(cl_uint), &context->fringe_context->display_options.head_res_y_spec);
		if (ret != CL_SUCCESS)
		{
			LOG4CXX_ERROR(DSCP4_OPENCL_LOGGER, "Could not set argument 27 (the GPU head Y spec resolution) in OpenCL kernel")
				dscp4_fringe_opencl_DestroyContext(&context);
			return nullptr;
		}

		ret = clSetKernelArg((cl_kernel)context->kernel, 28, sizeof(cl_uint), &context->fringe_context->algorithm_options->cache.num_fringe_buffers);
		if (ret != CL_SUCCESS)
		{
			LOG4CXX_ERROR(DSCP4_OPENCL_LOGGER, "Could not set argument 28 (the number of fringe output buffers) in OpenCL kernel")
				dscp4_fringe_opencl_DestroyContext(&context);
			return nullptr;
		}

		return context;
	}

	void dscp4_fringe_opencl_DestroyContext(dscp4_fringe_opencl_context_t** openclContext)
	{
		cl_int ret = CL_SUCCESS;

		LOG4CXX_DEBUG(DSCP4_OPENCL_LOGGER, "Destroying OpenCL context")

		if ((*openclContext)->command_queue)
		{
			ret = clFlush((cl_command_queue)(*openclContext)->command_queue);
			CHECK_OPENCL_RC(ret, "Could not flush OpenCL command queue")
		}

		if ((*openclContext)->command_queue)
		{
			ret = clFinish((cl_command_queue)(*openclContext)->command_queue);
			CHECK_OPENCL_RC(ret, "Could not finish OpenCL command queue")
		}
		
		if ((*openclContext)->kernel)
		{
			ret = clReleaseKernel((cl_kernel)(*openclContext)->kernel);
			CHECK_OPENCL_RC(ret, "Could not release OpenCL kernel")
		}
		
		if ((*openclContext)->program)
		{
			ret = clReleaseProgram((cl_program)(*openclContext)->program);
			CHECK_OPENCL_RC(ret, "Could not release OpenCL program")
		}

		if ((*openclContext)->fringe_opencl_resources)
		{
			for (unsigned int i = 0; i < (*openclContext)->fringe_context->algorithm_options->cache.num_fringe_buffers; i++)
			{
				ret = clReleaseMemObject((cl_mem)(*openclContext)->fringe_opencl_resources[i]);
				CHECK_OPENCL_RC(ret, "Could not release fringe texture " << i << " OpenCL memory resource")
			}

			delete[](*openclContext)->fringe_opencl_resources;
		}

		if ((*openclContext)->stereogram_rgba_opencl_resource)
		{
			ret = clReleaseMemObject((cl_mem)(*openclContext)->stereogram_rgba_opencl_resource);
			CHECK_OPENCL_RC(ret, "Could not release stereogram RGBA OpenCL memory resource")
		}

		if ((*openclContext)->stereogram_depth_opencl_resource)
		{
			ret = clReleaseMemObject((cl_mem)(*openclContext)->stereogram_depth_opencl_resource);
			CHECK_OPENCL_RC(ret, "Could not release stereogram DEPTH OpenCL memory resource")
		}

		if ((*openclContext)->framebuffer_opencl_output)
		{
			ret = clReleaseMemObject((cl_mem)(*openclContext)->framebuffer_opencl_output);
			CHECK_OPENCL_RC(ret, "Could not release mega buffer OpenCL memory resource")
		}

		if ((cl_command_queue)(*openclContext)->command_queue)
		{
			ret = clReleaseCommandQueue((cl_command_queue)(*openclContext)->command_queue);
			CHECK_OPENCL_RC(ret, "Could not release OpenCL command queue")
		}

		if ((cl_context)(*openclContext)->cl_context)
		{
			ret = clReleaseContext((cl_context)(*openclContext)->cl_context);
			CHECK_OPENCL_RC(ret, "Could not release fringe OpenCL context")
		}

		delete *openclContext;
		*openclContext = NULL;
	}

	void dscp4_fringe_opencl_ComputeFringe(dscp4_fringe_opencl_context_t* openclContext)
	{
		const size_t region[3] = { openclContext->fringe_context->algorithm_options->cache.fringe_buffer_res_x, openclContext->fringe_context->algorithm_options->cache.fringe_buffer_res_y / openclContext->fringe_context->display_options.num_heads_per_gpu, 1 };
		const size_t outputBufferSize = openclContext->fringe_context->algorithm_options->cache.fringe_buffer_res_x * openclContext->fringe_context->algorithm_options->cache.fringe_buffer_res_y * 4;
		
		const unsigned int num_gpus = 1;

		cl_int ret = 0;

		cl_event *event = new cl_event[num_gpus * 3];

		glFinish();

		for (unsigned int i = 0; i < num_gpus; i++)
		{
			ret = clSetKernelArg((cl_kernel)openclContext->kernel, 11, sizeof(cl_float), &openclContext->fringe_context->algorithm_options->red_gain);
			ret = clSetKernelArg((cl_kernel)openclContext->kernel, 12, sizeof(cl_float), &openclContext->fringe_context->algorithm_options->green_gain);
			ret = clSetKernelArg((cl_kernel)openclContext->kernel, 13, sizeof(cl_float), &openclContext->fringe_context->algorithm_options->blue_gain);

			ret = clSetKernelArg((cl_kernel)openclContext->kernel, 14, sizeof(cl_float), &openclContext->fringe_context->algorithm_options->cache.reference_beam_angle_rad);
			ret = clSetKernelArg((cl_kernel)openclContext->kernel, 16, sizeof(cl_float), &openclContext->fringe_context->algorithm_options->cache.k_r);
			ret = clSetKernelArg((cl_kernel)openclContext->kernel, 17, sizeof(cl_float), &openclContext->fringe_context->algorithm_options->cache.k_g);
			ret = clSetKernelArg((cl_kernel)openclContext->kernel, 18, sizeof(cl_float), &openclContext->fringe_context->algorithm_options->cache.k_b);
			ret = clSetKernelArg((cl_kernel)openclContext->kernel, 19, openclContext->have_cl_double_precision_extension ? sizeof(cl_double) : sizeof(cl_float), &openclContext->fringe_context->algorithm_options->cache.upconvert_const_r);
			ret = clSetKernelArg((cl_kernel)openclContext->kernel, 20, openclContext->have_cl_double_precision_extension ? sizeof(cl_double) : sizeof(cl_float), &openclContext->fringe_context->algorithm_options->cache.upconvert_const_g);
			ret = clSetKernelArg((cl_kernel)openclContext->kernel, 21, openclContext->have_cl_double_precision_extension ? sizeof(cl_double) : sizeof(cl_float), &openclContext->fringe_context->algorithm_options->cache.upconvert_const_b);
			ret = clSetKernelArg((cl_kernel)openclContext->kernel, 25, sizeof(cl_float), &openclContext->fringe_context->algorithm_options->cache.z_offset);

			ret = clEnqueueAcquireGLObjects((cl_command_queue)openclContext->command_queue, 1, (const cl_mem*)&openclContext->stereogram_rgba_opencl_resource, 0, NULL, NULL);
			ret = clEnqueueAcquireGLObjects((cl_command_queue)openclContext->command_queue, 1, (const cl_mem*)&openclContext->stereogram_depth_opencl_resource, 0, NULL, NULL);

			for (int f = 0; f < openclContext->fringe_context->display_options.num_heads / openclContext->fringe_context->display_options.num_heads_per_gpu; f++)
				ret = clEnqueueAcquireGLObjects((cl_command_queue)openclContext->command_queue, 1, (const cl_mem*)&openclContext->fringe_opencl_resources[f], 0, NULL, &event[i*num_gpus]);

			ret = clEnqueueNDRangeKernel((cl_command_queue)openclContext->command_queue, (cl_kernel)openclContext->kernel, 2, NULL,
				(const size_t*)openclContext->fringe_context->algorithm_options->cache.opencl_global_workgroup_size, (const size_t*)openclContext->fringe_context->algorithm_options->opencl_local_workgroup_size, 1, &event[i*num_gpus], &event[i*num_gpus+1]);
			if (ret != CL_SUCCESS)
			{
				LOG4CXX_ERROR(DSCP4_OPENCL_LOGGER, "Could not enqueue OpenCL kernel")
				delete[] event;
				return;
			}

			ret = clFlush((cl_command_queue)(openclContext)->command_queue);
			if (ret != CL_SUCCESS)
			{
				LOG4CXX_ERROR(DSCP4_OPENCL_LOGGER, "Could not flush OpenCL command queue")
				delete[] event;
				return;
			}

			ret = clFinish((cl_command_queue)(openclContext)->command_queue);
			if (ret != CL_SUCCESS)
			{
				LOG4CXX_ERROR(DSCP4_OPENCL_LOGGER, "Could not finish OpenCL command queue")
				delete[] event;
				return;
			}

			for (unsigned int j = 0; j < openclContext->fringe_context->display_options.num_heads; j++)
			{
				unsigned int numGPUS = openclContext->fringe_context->display_options.num_heads / openclContext->fringe_context->display_options.num_heads_per_gpu;
				unsigned int which_gpu = j % numGPUS;

				size_t dst_origin[3] = { 0,
					((which_gpu + 1) % openclContext->fringe_context->display_options.num_heads_per_gpu) == 0 ?
					(openclContext->fringe_context->display_options.num_heads_per_gpu - (j % openclContext->fringe_context->display_options.num_heads_per_gpu) - 1) * openclContext->fringe_context->display_options.head_res_y :
					(j % openclContext->fringe_context->display_options.num_heads_per_gpu) * openclContext->fringe_context->display_options.head_res_y,
					0 };

				ret = clEnqueueCopyBufferToImage((cl_command_queue)openclContext->command_queue,
					(cl_mem)openclContext->framebuffer_opencl_output,
					(cl_mem)openclContext->fringe_opencl_resources[which_gpu],
					j * openclContext->fringe_context->display_options.head_res_x_spec * openclContext->fringe_context->display_options.head_res_y_spec * 4,
					dst_origin,
					region,
					1,
					&event[i*num_gpus + 1], &event[i*num_gpus + 2]);
			}

			ret = clEnqueueReleaseGLObjects((cl_command_queue)openclContext->command_queue, 1, (const cl_mem*)&openclContext->stereogram_rgba_opencl_resource, 0, NULL, NULL);
			for (int f = 0; f < openclContext->fringe_context->display_options.num_heads / openclContext->fringe_context->display_options.num_heads_per_gpu; f++)
				ret = clEnqueueReleaseGLObjects((cl_command_queue)openclContext->command_queue, 1, (const cl_mem*)&openclContext->fringe_opencl_resources[f], 1, &event[i*num_gpus + 1], &event[i*num_gpus + 2]);
			ret = clEnqueueReleaseGLObjects((cl_command_queue)openclContext->command_queue, 1, (const cl_mem*)&openclContext->stereogram_depth_opencl_resource, 0, NULL, NULL);
		}

		for (unsigned int i = 0; i < num_gpus; i++)
		{
			ret = clWaitForEvents(1, &event[i* num_gpus + 2]);
		}

		delete[] event;
	}

	const char *clGetErrorString(cl_int error)
	{
		switch (error){
			// run-time and JIT compiler errors
		case 0: return "CL_SUCCESS";
		case -1: return "CL_DEVICE_NOT_FOUND";
		case -2: return "CL_DEVICE_NOT_AVAILABLE";
		case -3: return "CL_COMPILER_NOT_AVAILABLE";
		case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
		case -5: return "CL_OUT_OF_RESOURCES";
		case -6: return "CL_OUT_OF_HOST_MEMORY";
		case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
		case -8: return "CL_MEM_COPY_OVERLAP";
		case -9: return "CL_IMAGE_FORMAT_MISMATCH";
		case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
		case -11: return "CL_BUILD_PROGRAM_FAILURE";
		case -12: return "CL_MAP_FAILURE";
		case -13: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
		case -14: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
		case -15: return "CL_COMPILE_PROGRAM_FAILURE";
		case -16: return "CL_LINKER_NOT_AVAILABLE";
		case -17: return "CL_LINK_PROGRAM_FAILURE";
		case -18: return "CL_DEVICE_PARTITION_FAILED";
		case -19: return "CL_KERNEL_ARG_INFO_NOT_AVAILABLE";

			// compile-time errors
		case -30: return "CL_INVALID_VALUE";
		case -31: return "CL_INVALID_DEVICE_TYPE";
		case -32: return "CL_INVALID_PLATFORM";
		case -33: return "CL_INVALID_DEVICE";
		case -34: return "CL_INVALID_CONTEXT";
		case -35: return "CL_INVALID_QUEUE_PROPERTIES";
		case -36: return "CL_INVALID_COMMAND_QUEUE";
		case -37: return "CL_INVALID_HOST_PTR";
		case -38: return "CL_INVALID_MEM_OBJECT";
		case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
		case -40: return "CL_INVALID_IMAGE_SIZE";
		case -41: return "CL_INVALID_SAMPLER";
		case -42: return "CL_INVALID_BINARY";
		case -43: return "CL_INVALID_BUILD_OPTIONS";
		case -44: return "CL_INVALID_PROGRAM";
		case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
		case -46: return "CL_INVALID_KERNEL_NAME";
		case -47: return "CL_INVALID_KERNEL_DEFINITION";
		case -48: return "CL_INVALID_KERNEL";
		case -49: return "CL_INVALID_ARG_INDEX";
		case -50: return "CL_INVALID_ARG_VALUE";
		case -51: return "CL_INVALID_ARG_SIZE";
		case -52: return "CL_INVALID_KERNEL_ARGS";
		case -53: return "CL_INVALID_WORK_DIMENSION";
		case -54: return "CL_INVALID_WORK_GROUP_SIZE";
		case -55: return "CL_INVALID_WORK_ITEM_SIZE";
		case -56: return "CL_INVALID_GLOBAL_OFFSET";
		case -57: return "CL_INVALID_EVENT_WAIT_LIST";
		case -58: return "CL_INVALID_EVENT";
		case -59: return "CL_INVALID_OPERATION";
		case -60: return "CL_INVALID_GL_OBJECT";
		case -61: return "CL_INVALID_BUFFER_SIZE";
		case -62: return "CL_INVALID_MIP_LEVEL";
		case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
		case -64: return "CL_INVALID_PROPERTY";
		case -65: return "CL_INVALID_IMAGE_DESCRIPTOR";
		case -66: return "CL_INVALID_COMPILER_OPTIONS";
		case -67: return "CL_INVALID_LINKER_OPTIONS";
		case -68: return "CL_INVALID_DEVICE_PARTITION_COUNT";

			// extension errors
		case -1000: return "CL_INVALID_GL_SHAREGROUP_REFERENCE_KHR";
		case -1001: return "CL_PLATFORM_NOT_FOUND_KHR";
		case -1002: return "CL_INVALID_D3D10_DEVICE_KHR";
		case -1003: return "CL_INVALID_D3D10_RESOURCE_KHR";
		case -1004: return "CL_D3D10_RESOURCE_ALREADY_ACQUIRED_KHR";
		case -1005: return "CL_D3D10_RESOURCE_NOT_ACQUIRED_KHR";
		default: return "Unknown OpenCL error";
		}
	}

#ifdef __cplusplus
};
#endif
