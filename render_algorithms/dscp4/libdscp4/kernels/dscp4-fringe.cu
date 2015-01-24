#include "dscp4-fringe-cuda.h"

#include <cuda.h>

#ifdef WIN32
#include <Windows.h>
#endif

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

#include <stdio.h>
#include <math.h>

texture<uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat> viewset_color_in;
texture<uchar4, cudaTextureType2D, cudaReadModeNormalizedFloat> * framebuffer_tex_out;

__global__ void computeFringe(
	unsigned char* framebuffer_out,
	const float* viewset_depth_in,
	const unsigned int num_wafels_per_scanline,
	const unsigned int num_scanlines,
	const unsigned int viewset_res_x,
	const unsigned int viewset_res_y,
	const unsigned int viewset_num_tiles_x,
	const unsigned int viewset_num_tiles_y,
	const unsigned int framebuffer_res_x,
	const unsigned int framebuffer_res_y,
	unsigned char* wafel_buffer,
	float* wafel_position,
	const float K_R,
	const float K_G,
	const float K_B,
	const float UPCONVERT_CONST_R,
	const float UPCONVERT_CONST_G,
	const float UPCONVERT_CONST_B,
	const unsigned int NUM_SAMPLES_PER_WAFEL,
	const float SAMPLE_PITCH,
	const float Z_SPAN,
	const float Z_OFFSET,
	const unsigned int NUM_AOM_CHANNELS,
	const unsigned int HEAD_RES_Y_SPEC,
	const unsigned int NUM_BUFFERS
	);

dscp4_fringe_cuda_context_t* dscp4_fringe_cuda_CreateContext(dscp4_fringe_context_t* fringeContext)
{
	cudaError_t error = cudaSuccess;
	dscp4_fringe_cuda_context_t* cudaContext = (dscp4_fringe_cuda_context_t*)malloc(sizeof(dscp4_fringe_cuda_context_t));
	cudaContext->fringe_context = fringeContext;

	error = cudaGetDeviceCount(&cudaContext->num_gpus);

	
	if (error != cudaSuccess)
	{
		printf("ERROR Could not get CUDA device count -- Are there any CUDA devices present?");
		free(cudaContext);
		return NULL;
	}

	cudaContext->gpu_properties = (struct cudaDeviceProp*)malloc(sizeof(struct cudaDeviceProp)*cudaContext->num_gpus);

	for (int i = 0; i < cudaContext->num_gpus; i++)
	{
		error = cudaGLSetGLDevice(i);
		error = cudaGetDeviceProperties(&cudaContext->gpu_properties[i], i);
	}

	error = cudaGraphicsGLRegisterImage(&cudaContext->stereogram_rgba_cuda_resource, cudaContext->fringe_context->stereogram_gl_fbo_color, GL_TEXTURE_2D, cudaGraphicsRegisterFlagsReadOnly);
	if (error != cudaSuccess)
		printf("ERROR Could not register viewset OpenGL RGBA texture");

	error = cudaGraphicsGLRegisterBuffer(&cudaContext->stereogram_depth_cuda_resource, cudaContext->fringe_context->stereogram_gl_depth_buf_in, cudaGraphicsRegisterFlagsReadOnly);
	if (error != cudaSuccess)
		printf("ERROR Could not register viewset OpenGL DEPTH texture with CUDA");

	cudaContext->fringe_cuda_resources = (struct cudaGraphicsResource**)malloc(sizeof(void*)*cudaContext->fringe_context->algorithm_options.cache.num_fringe_buffers);

	error = cudaMalloc((void**)framebuffer_tex_out, sizeof(framebuffer_tex_out) * cudaContext->fringe_context->algorithm_options.cache.num_fringe_buffers);
	if (error)
		printf("ERROR Could not alloc CUDA framebuffer textures");

	for (unsigned int i = 0; i < cudaContext->fringe_context->algorithm_options.cache.num_fringe_buffers; i++)
	{
		//error = cudaGraphicsGLRegisterBuffer(&cudaContext->fringe_cuda_resources[i], cudaContext->fringe_context->fringe_gl_buf_out[i], cudaGraphicsRegisterFlagsWriteDiscard);
		error = cudaGraphicsGLRegisterImage(&cudaContext->fringe_cuda_resources[i], cudaContext->fringe_context->fringe_gl_tex_out[i], GL_TEXTURE_2D, cudaGraphicsRegisterFlagsWriteDiscard);
		if (error)
			printf("ERROR Could not register CUDA image framebuffer texture objects");
	}

	error = cudaMalloc(&cudaContext->spec_buffer, cudaContext->fringe_context->display_options.head_res_x_spec * cudaContext->fringe_context->display_options.head_res_y_spec * cudaContext->fringe_context->display_options.num_heads * 4);
	if (error != cudaSuccess)
		printf("ERROR Could not alloc CUDA megabuffer");

	error = cudaMalloc(&cudaContext->wafel_buffers, cudaContext->fringe_context->algorithm_options.cache.num_samples_per_wafel * cudaContext->fringe_context->algorithm_options.num_wafels_per_scanline * cudaContext->fringe_context->algorithm_options.num_scanlines * sizeof(unsigned char));
	error = cudaMalloc(&cudaContext->wafel_positions, cudaContext->fringe_context->algorithm_options.cache.num_samples_per_wafel * cudaContext->fringe_context->algorithm_options.num_wafels_per_scanline * cudaContext->fringe_context->algorithm_options.num_scanlines * sizeof(float));

	return cudaContext;
};

void dscp4_fringe_cuda_DestroyContext(dscp4_fringe_cuda_context_t** cudaContext)
{

	cudaFree((*cudaContext)->wafel_buffers);
	cudaFree((*cudaContext)->wafel_positions);
	cudaFree((*cudaContext)->spec_buffer);

	cudaFree(framebuffer_tex_out);

	for (unsigned int i = 0; i < (*cudaContext)->fringe_context->display_options.num_heads / 2; i++)
	{
		cudaGraphicsUnregisterResource((*cudaContext)->fringe_cuda_resources[i]);
	}

	cudaGraphicsUnregisterResource((*cudaContext)->stereogram_depth_cuda_resource);
	cudaGraphicsUnregisterResource((*cudaContext)->stereogram_rgba_cuda_resource);

	if (*cudaContext != NULL)
	{
		if ((*cudaContext)->fringe_cuda_resources)
		{
			free((*cudaContext)->fringe_cuda_resources);
			(*cudaContext)->fringe_cuda_resources = NULL;
		}

		if ((*cudaContext)->gpu_properties)
		{
			free((*cudaContext)->gpu_properties);
			(*cudaContext)->gpu_properties = NULL;
		}

		free(*cudaContext);
		*cudaContext = NULL;
	}
};

void dscp4_fringe_cuda_ComputeFringe(dscp4_fringe_cuda_context_t* cudaContext)
{
	//void **output;
	//size_t * outputSizes;

	cudaChannelFormatDesc rgbaTexDesc;
	rgbaTexDesc.x = 8;
	rgbaTexDesc.y = 8;
	rgbaTexDesc.z = 8;
	rgbaTexDesc.w = 8;
	rgbaTexDesc.f = cudaChannelFormatKindUnsigned;

	cudaArray_t viewsetRGBAArray;
	cudaArray_t * framebufferArrays = NULL;
	void * viewsetDepthArray = NULL;
	size_t viewsetDepthArraySize;

	cudaError_t error = cudaSuccess;

	error = cudaGraphicsMapResources(1, &cudaContext->stereogram_rgba_cuda_resource, 0);
	if(error != cudaSuccess)
		printf("ERROR Mapping stereogram RGBA CUDA graphics resource\n");

	error = cudaGraphicsMapResources(1, &cudaContext->stereogram_depth_cuda_resource, 0);
	if(error != cudaSuccess)
		printf("ERROR Mapping stereogram DEPTH CUDA graphics resource\n");

	error = cudaGraphicsSubResourceGetMappedArray(&viewsetRGBAArray, cudaContext->stereogram_rgba_cuda_resource, 0, 0);
	if (error != cudaSuccess)
		printf("ERROR Mapping stereogram COLOR texture CUDA graphics resource\n");

	error = cudaBindTextureToArray(&viewset_color_in, viewsetRGBAArray, &rgbaTexDesc);

	error = cudaGraphicsResourceGetMappedPointer((void**)&viewsetDepthArray, &viewsetDepthArraySize, cudaContext->stereogram_depth_cuda_resource);
	if(error != cudaSuccess)
		printf("ERROR Getting stereogram DEPTH CUDA graphics resource mapped pointer\n");


	//output = (void**)malloc(sizeof(void*)* cudaContext->fringe_context->algorithm_options.cache.num_fringe_buffers);
	//outputSizes = (size_t*)malloc(sizeof(size_t)* cudaContext->fringe_context->algorithm_options.cache.num_fringe_buffers);

	cudaMalloc((void**)framebufferArrays, sizeof(cudaArray_t)* cudaContext->fringe_context->algorithm_options.cache.num_fringe_buffers);

	for (unsigned int i = 0; i < cudaContext->fringe_context->algorithm_options.cache.num_fringe_buffers; i++)
	{
		//error = cudaGraphicsMapResources(1, (cudaGraphicsResource_t*)(&cudaContext->fringe_cuda_resources[i]), 0);
		//if(error != cudaSuccess)
		//	printf("ERROR Mapping CUDA fringe texture buffer %i\n", i);


		//error = cudaGraphicsResourceGetMappedPointer(&output[i], &outputSizes[i], cudaContext->fringe_cuda_resources[i]);
		//if (error != cudaSuccess)
		//	printf("ERROR Getting fringe texture buffer %i CUDA mapped pointer\n", i);

		error = cudaGraphicsSubResourceGetMappedArray(&framebufferArrays[i], cudaContext->fringe_cuda_resources[i],0,0);
		if (error != cudaSuccess)
			printf("ERROR Getting framebuffer %i CUDA array\n", i);

		error = cudaBindTextureToArray(&framebuffer_tex_out[i], framebufferArrays[i], &rgbaTexDesc);
		if (error != cudaSuccess)
			printf("ERROR Binding framebuffer texture %i  to CUDA array\n", i);

	}

	//// run kernel here
	//for (unsigned int i = 0; i < cudaContext->fringe_context->algorithm_options.cache.num_fringe_buffers; i++)
	//{



		dim3 threadsPerBlock(
			cudaContext->fringe_context->algorithm_options.cuda_block_dimensions[0],
			cudaContext->fringe_context->algorithm_options.cuda_block_dimensions[1]
			);
		dim3 numBlocks(
			cudaContext->fringe_context->algorithm_options.cache.cuda_number_of_blocks[0],
			cudaContext->fringe_context->algorithm_options.cache.cuda_number_of_blocks[1]
			);

		computeFringe << <numBlocks, threadsPerBlock >> >(
			(unsigned char*)cudaContext->spec_buffer,
			(const float*)viewsetDepthArray,
			cudaContext->fringe_context->algorithm_options.num_wafels_per_scanline,
			cudaContext->fringe_context->algorithm_options.num_scanlines,
			cudaContext->fringe_context->algorithm_options.cache.stereogram_res_x,
			cudaContext->fringe_context->algorithm_options.cache.stereogram_res_y,
			cudaContext->fringe_context->algorithm_options.cache.stereogram_num_tiles_x,
			cudaContext->fringe_context->algorithm_options.cache.stereogram_num_tiles_y,
			cudaContext->fringe_context->algorithm_options.cache.fringe_buffer_res_x,
			cudaContext->fringe_context->algorithm_options.cache.fringe_buffer_res_y,
			cudaContext->wafel_buffers,
			cudaContext->wafel_positions,
			cudaContext->fringe_context->algorithm_options.cache.k_r,
			cudaContext->fringe_context->algorithm_options.cache.k_g,
			cudaContext->fringe_context->algorithm_options.cache.k_b,
			cudaContext->fringe_context->algorithm_options.cache.upconvert_const_r,
			cudaContext->fringe_context->algorithm_options.cache.upconvert_const_g,
			cudaContext->fringe_context->algorithm_options.cache.upconvert_const_b,
			cudaContext->fringe_context->algorithm_options.cache.num_samples_per_wafel,
			cudaContext->fringe_context->algorithm_options.cache.sample_pitch,
			cudaContext->fringe_context->algorithm_options.cache.z_span,
			cudaContext->fringe_context->algorithm_options.cache.z_offset,
			cudaContext->fringe_context->display_options.num_aom_channels,
			cudaContext->fringe_context->display_options.head_res_y_spec,
			cudaContext->fringe_context->algorithm_options.cache.num_fringe_buffers
			);
	//}g

	error = cudaUnbindTexture(&viewset_color_in);

	//write texture outputs here

	for (unsigned int i = 0; i < cudaContext->fringe_context->display_options.num_heads; i++)
	{
		cudaMemcpyToArray(framebufferArrays[i], 0, 0, cudaContext->spec_buffer, cudaContext->fringe_context->algorithm_options.cache.fringe_buffer_res_x * cudaContext->fringe_context->algorithm_options.cache.fringe_buffer_res_y *4, cudaMemcpyDeviceToDevice);
	}

	for (unsigned int i = 0; i < cudaContext->fringe_context->algorithm_options.cache.num_fringe_buffers; i++)
	{
		cudaUnbindTexture(framebuffer_tex_out[i]);
		error = cudaGraphicsUnmapResources(1, (cudaGraphicsResource_t*)(&cudaContext->fringe_cuda_resources[i]), 0);
		if(error != cudaSuccess)
			printf("ERROR Unmapping CUDA fringe buffer %i resource\n", i);

	}

	error = cudaGraphicsUnmapResources(1, &cudaContext->stereogram_rgba_cuda_resource, 0);
	if(error != cudaSuccess)
		printf("ERROR Unmapping viewset RGBA CUDA graphics resource\n");

	error = cudaGraphicsUnmapResources(1, &cudaContext->stereogram_depth_cuda_resource, 0);
	if(error != cudaSuccess)
		printf("ERROR Unmapping viewset DEPTH CUDA graphics resource\n");


	//free(output);
	//free(outputSizes);
	free(framebufferArrays);
};

__global__ void computeFringe(
	unsigned char* framebuffer_out,
	const float* viewset_depth_in,
	const unsigned int num_wafels_per_scanline,
	const unsigned int num_scanlines,
	const unsigned int viewset_res_x,
	const unsigned int viewset_res_y,
	const unsigned int viewset_num_tiles_x,
	const unsigned int viewset_num_tiles_y,
	const unsigned int framebuffer_res_x,
	const unsigned int framebuffer_res_y,
	unsigned char* wafel_buffer,
	float* wafel_position,
	const float K_R,
	const float K_G,
	const float K_B,
	const float UPCONVERT_CONST_R,
	const float UPCONVERT_CONST_G,
	const float UPCONVERT_CONST_B,
	const unsigned int NUM_SAMPLES_PER_WAFEL,
	const float SAMPLE_PITCH,
	const float Z_SPAN,
	const float Z_OFFSET,
	const unsigned int NUM_AOM_CHANNELS,
	const unsigned int HEAD_RES_Y_SPEC,
	const unsigned int NUM_BUFFERS
	)
{

	int x = (blockIdx.x * blockDim.x) + threadIdx.x;
	int y = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (x < num_wafels_per_scanline && y < num_scanlines)
	{
		// offset of wafel samples/position buffer
		const unsigned int wafel_offset = x*y*NUM_SAMPLES_PER_WAFEL;
		const float num_views = (viewset_num_tiles_x * viewset_num_tiles_y);

		for (int i = 0; i < NUM_SAMPLES_PER_WAFEL; i++)
		{
			wafel_position[i + wafel_offset] = (-(float)ceil((double)(num_wafels_per_scanline / 2.f)) + i) * SAMPLE_PITCH + x;
		}

		for (unsigned int color_chan = 0; color_chan < 3; color_chan++)
		{
			x = (blockIdx.x * blockDim.x) + threadIdx.x;
			y = (blockIdx.y * blockDim.y) + threadIdx.y;

			float k = (color_chan == 0 ? K_R : color_chan == 1 ? K_G : K_B);
			float up_const = (color_chan == 0 ? UPCONVERT_CONST_R : color_chan == 1 ? UPCONVERT_CONST_G : UPCONVERT_CONST_B);

			for (unsigned int vy = 0; vy < viewset_num_tiles_y; vy++)
			{
				for (unsigned int vx = 0; vx < viewset_num_tiles_x; vx++)
				{
					float d = (viewset_depth_in[y * viewset_res_x + x] - 0.5) * Z_SPAN + Z_OFFSET;
					float4 color = tex2D(viewset_color_in, x, y);
					float c = 255.f*(color_chan == 0 ? color.x : color_chan == 1 ? color.y : color.z);

					for (int i = 0; i < NUM_SAMPLES_PER_WAFEL; i++)
					{
						wafel_buffer[wafel_offset + i] += c / num_views * cos(k * sqrt(pow((float)((int)wafel_position[wafel_offset + i] - (int)x), (float)2) + pow(d, (float)2)) - d + wafel_position[wafel_offset+i] * up_const);
					}
					//framebuffer_out[(y * framebuffer_res_x * 4) + (x * 4)] = viewset_depth_in[y * viewset_res_x + x] * 255.f;
					//framebuffer_out[(y * framebuffer_res_x * 4) + (x * 4 + 1)] = 0;
					//framebuffer_out[(y * framebuffer_res_x * 4) + (x * 4 + 2)] = 0;
					x += num_wafels_per_scanline;
				}
				x = (blockIdx.x * blockDim.x) + threadIdx.x;
				y += num_scanlines;
			}
		}

		x = (blockIdx.x * blockDim.x) + threadIdx.x;
		y = (blockIdx.y * blockDim.y) + threadIdx.y;

		int which_frame_buf = (y % NUM_AOM_CHANNELS);
		int which_hololine = y / NUM_AOM_CHANNELS;
//		int which_frameline = (float)x / (framebuffer_res_x / NUM_SAMPLES_PER_WAFEL);
//		int which_wafel = x - (which_frameline * (framebuffer_res_x / NUM_SAMPLES_PER_WAFEL));

		for (int i = 0; i < NUM_SAMPLES_PER_WAFEL; i++)
		{
			framebuffer_out[
				which_frame_buf / NUM_BUFFERS * framebuffer_res_x * HEAD_RES_Y_SPEC * 4
					+ which_hololine * (((NUM_SAMPLES_PER_WAFEL * num_wafels_per_scanline) / framebuffer_res_x) * framebuffer_res_x * 4)
					+ NUM_SAMPLES_PER_WAFEL * 4 * x
					+ which_frame_buf % 3 + 4 * i
			] = wafel_buffer[wafel_offset + i];
		}

	}
}
