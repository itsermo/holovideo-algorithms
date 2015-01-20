#include "dscp4-fringe-cuda.h"

#include <cuda.h>

#ifdef WIN32
#include <Windows.h>
#endif

#include <cuda_gl_interop.h>
#include <cuda_runtime.h>

const int N = 16;
const int blocksize = 16;

#include <stdio.h>
#include <math.h>

__global__ void computeFringe(
	void * fringeDataOut,
	const void * rgbaIn,
	const void * depthIn,
	const unsigned int which_buffer,
	const unsigned int num_wafels_per_scanline,
	const unsigned int num_scanlines,
	const unsigned int stereogram_res_x,
	const unsigned int stereogram_res_y,
	const unsigned int stereogram_num_tiles_x,
	const unsigned int stereogram_num_tiles_y,
	const unsigned int fringe_buffer_res_x,
	const unsigned int fringe_buffer_res_y
	);

__global__ void hello(char *a, int *b)
{
	a[threadIdx.x] += b[threadIdx.x];
}

char * dscp4_fringe_cuda_HelloWorld()
{
	char *a = (char*)malloc(N);
	a[0] = 'H';
	a[1] = 'e';
	a[2] = 'l';
	a[3] = 'l';
	a[4] = 'o';
	a[5] = ' ';
	a[6] = '\0';
	a[7] = '\0';
	a[8] = '\0';
	a[9] = '\0';
	a[10] = '\0';
	a[11] = '\0';
	//a[12] = '\0';
	//a[13] = '\0';
	//a[14] = '\0';
	//a[15] = '\0';

	//strcpy	"Hello \0\0\0\0\0\0";
	int b[N] = { 15, 10, 6, 0, -11, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

	char *ad;
	int *bd;
	const int csize = N*sizeof(char);
	const int isize = N*sizeof(int);

	//printf("%s", a);

	cudaMalloc((void**)&ad, csize);
	cudaMalloc((void**)&bd, isize);
	cudaMemcpy(ad, a, csize, cudaMemcpyHostToDevice);
	cudaMemcpy(bd, b, isize, cudaMemcpyHostToDevice);

	dim3 dimBlock(blocksize, 1);
	dim3 dimGrid(1, 1);
	hello << <dimGrid, dimBlock >> >(ad, bd);
	cudaMemcpy(a, ad, csize, cudaMemcpyDeviceToHost);
	cudaFree(ad);
	cudaFree(bd);

	return a;
	//printf("%s\n", a);
	//return EXIT_SUCCESS;
};

dscp4_fringe_cuda_context_t* dscp4_fringe_cuda_CreateContext(dscp4_fringe_context_t* fringeContext)
{
	cudaError_t error = cudaSuccess;
	dscp4_fringe_cuda_context_t* cudaContext = (dscp4_fringe_cuda_context_t*)malloc(sizeof(dscp4_fringe_cuda_context_t));
	cudaContext->fringe_context = fringeContext;

	error = cudaGetDeviceCount(&cudaContext->num_gpus);
	
	if (error != cudaSuccess)
	{
		free(cudaContext);
		return NULL;
	}

	cudaContext->gpu_properties = (struct cudaDeviceProp*)malloc(sizeof(struct cudaDeviceProp)*cudaContext->num_gpus);

	for (int i = 0; i < cudaContext->num_gpus; i++)
	{
		error = cudaGLSetGLDevice(i);
		error = cudaGetDeviceProperties(&cudaContext->gpu_properties[i], i);
	}


	error = cudaGraphicsGLRegisterBuffer(&cudaContext->stereogram_depth_cuda_resource, cudaContext->fringe_context->stereogram_gl_depth_buf_in, cudaGraphicsRegisterFlagsReadOnly);
	error = cudaGraphicsGLRegisterBuffer(&cudaContext->stereogram_rgba_cuda_resource, cudaContext->fringe_context->stereogram_gl_rgba_buf_in, cudaGraphicsRegisterFlagsReadOnly);


	//error = cudaGraphicsGLRegisterImage(&cudaContext->fringe_cuda_resources, cudaContext->fringe_context->fringe_gl_buf_out[0], GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard);

	cudaContext->fringe_cuda_resources = (struct cudaGraphicsResource**)malloc(sizeof(void*)*cudaContext->fringe_context->display_options.num_heads / 2);

	for (unsigned int i = 0; i < cudaContext->fringe_context->display_options.num_heads / 2; i++)
	{
		error = cudaGraphicsGLRegisterBuffer(&cudaContext->fringe_cuda_resources[i], cudaContext->fringe_context->fringe_gl_buf_out[i], cudaGraphicsRegisterFlagsWriteDiscard);
	}

	return cudaContext;
};

void dscp4_fringe_cuda_DestroyContext(dscp4_fringe_cuda_context_t** cudaContext)
{

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
	void **output;
	size_t * outputSizes;

	void * rgbaPtr;
	void * depthPtr;
	size_t rgbaSize;
	size_t depthSize;

	cudaError_t error = cudaSuccess;

	error = cudaGraphicsMapResources(1, &cudaContext->stereogram_rgba_cuda_resource, 0);
	if(error != cudaSuccess)
		printf("ERROR Mapping stereogram RGBA CUDA graphics resource\n");

	error = cudaGraphicsMapResources(1, &cudaContext->stereogram_depth_cuda_resource, 0);
	if(error != cudaSuccess)
		printf("ERROR Mapping stereogram DEPTH CUDA graphics resource\n");

	error = cudaGraphicsResourceGetMappedPointer((void**)&rgbaPtr, &rgbaSize, cudaContext->stereogram_rgba_cuda_resource);
	if(error != cudaSuccess)
		printf("ERROR Getting stereogram RGBA CUDA graphics resource mapped pointer\n");

	error = cudaGraphicsResourceGetMappedPointer((void**)&depthPtr, &depthSize, cudaContext->stereogram_depth_cuda_resource);
	if(error != cudaSuccess)
		printf("ERROR Getting stereogram DEPTH CUDA graphics resource mapped pointer\n");


	output = (void**)malloc(sizeof(void*)* cudaContext->fringe_context->display_options.num_heads / 2);
	outputSizes = (size_t*)malloc(sizeof(size_t) * cudaContext->fringe_context->display_options.num_heads / 2);

	for (unsigned int i = 0; i < cudaContext->fringe_context->display_options.num_heads / 2; i++)
	{
		error = cudaGraphicsMapResources(1, (cudaGraphicsResource_t*)(&cudaContext->fringe_cuda_resources[i]), 0);
		if(error != cudaSuccess)
			printf("ERROR Mapping CUDA fringe texture buffer %i\n", i);

		error = cudaGraphicsResourceGetMappedPointer(&output[i], &outputSizes[i], cudaContext->fringe_cuda_resources[i]);
		if(error != cudaSuccess)
			printf("ERROR Getting fringe texture buffer %i CUDA mapped pointer\n",i);

	}

	//for (unsigned int i = 0; i < cudaContext->fringe_context->algorithm_options.cache.stereogram_res_y; i++)
	//{
	//	cudaMemcpy((unsigned char*)output[1] + i * 3552 * 4, (unsigned char*)rgbaPtr + i*cudaContext->fringe_context->algorithm_options.cache.stereogram_res_x * 4, cudaContext->fringe_context->algorithm_options.cache.stereogram_res_x * 4, cudaMemcpyDeviceToDevice);
	//	//cudaMemcpy((unsigned char*)output[1] + i * 3552 * 4, depthPtr, cudaContext->fringe_context->algorithm_options.cache.stereogram_res_x * 4, cudaMemcpyDeviceToDevice);
	//	//cudaMemcpy((unsigned char*)output[2] + i * 3552 * 4, rgbaPtr, cudaContext->fringe_context->algorithm_options.cache.stereogram_res_x * 4, cudaMemcpyDeviceToDevice);
	//	//cudaMemcpy((char*)output[0] + i * 3552 * 4, depthPtr, cudaContext->fringe_context->algorithm_options.cache.stereogram_res_x * 4, cudaMemcpyDeviceToDevice);
	//	//error = cudaMemset((char*)output[0] + i*3552*4, 255, cudaContext->fringe_context->algorithm_options.cache.stereogram_res_x * 4);
	//	//error = cudaMemset((char*)output[1] + i*3552*4, 127, cudaContext->fringe_context->algorithm_options.cache.stereogram_res_x * 4);
	//	error = cudaMemset((char*)output[2] + i*3552*4, 0, cudaContext->fringe_context->algorithm_options.cache.stereogram_res_x * 4);
	//}

	
	// run kernel here
	for (unsigned int i = 0; i < cudaContext->fringe_context->algorithm_options.cache.num_fringe_buffers; i++)
	{

		dim3 threadsPerBlock(8, 4);
		dim3 numBlocks(cudaContext->fringe_context->algorithm_options.num_wafels_per_scanline / threadsPerBlock.x,
			cudaContext->fringe_context->algorithm_options.num_scanlines / threadsPerBlock.y);
		computeFringe << <numBlocks, threadsPerBlock >> >(
			output[i],
			rgbaPtr,
			depthPtr,
			i,
			cudaContext->fringe_context->algorithm_options.num_wafels_per_scanline,
			cudaContext->fringe_context->algorithm_options.num_scanlines,
			cudaContext->fringe_context->algorithm_options.cache.stereogram_res_x,
			cudaContext->fringe_context->algorithm_options.cache.stereogram_res_y,
			cudaContext->fringe_context->algorithm_options.cache.stereogram_num_tiles_x,
			cudaContext->fringe_context->algorithm_options.cache.stereogram_num_tiles_y,
			cudaContext->fringe_context->algorithm_options.cache.fringe_buffer_res_x,
			cudaContext->fringe_context->algorithm_options.cache.fringe_buffer_res_y
			);
	}

	//write texture outputs here

	for (unsigned int i = 0; i < cudaContext->fringe_context->display_options.num_heads / 2; i++)
	{
		error = cudaGraphicsUnmapResources(1, (cudaGraphicsResource_t*)(&cudaContext->fringe_cuda_resources[i]), 0);
		if(error != cudaSuccess)
			printf("ERROR Unmapping CUDA fringe buffer %i resource\n", i);

	}

	error = cudaGraphicsUnmapResources(1, &cudaContext->stereogram_rgba_cuda_resource, 0);
	if(error != cudaSuccess)
		printf("ERROR Unmapping stereogram RGBA CUDA graphics resource\n");

	error = cudaGraphicsUnmapResources(1, &cudaContext->stereogram_depth_cuda_resource, 0);
	if(error != cudaSuccess)
		printf("ERROR Unmapping stereogram DEPTH CUDA graphics resource\n");


	free(output);
	free(outputSizes);
};

__global__ void computeFringe(
	void * fringeDataOut,
	const void * rgbaIn,
	const void * depthIn,
	const unsigned int which_buffer,
	const unsigned int num_wafels_per_scanline,
	const unsigned int num_scanlines,
	const unsigned int stereogram_res_x,
	const unsigned int stereogram_res_y,
	const unsigned int stereogram_num_tiles_x,
	const unsigned int stereogram_num_tiles_y,
	const unsigned int fringe_buffer_res_x,
	const unsigned int fringe_buffer_res_y
	)
{
	int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	int j = (blockIdx.y * blockDim.y) + threadIdx.y;

	if (i < num_wafels_per_scanline && j < num_scanlines)
	{
		((unsigned char*)fringeDataOut)[(j * fringe_buffer_res_x * 4) + (i * 4)] = ((unsigned char*)rgbaIn)[j * num_wafels_per_scanline * stereogram_num_tiles_x * 4 + i * 4];
		((unsigned char*)fringeDataOut)[(j * fringe_buffer_res_x * 4) + (i * 4) + 1] = ((unsigned char*)rgbaIn)[j * num_wafels_per_scanline * stereogram_num_tiles_x * 4 + i * 4 + 1];
		((unsigned char*)fringeDataOut)[(j * fringe_buffer_res_x * 4) + (i * 4) + 2] = ((unsigned char*)rgbaIn)[j * num_wafels_per_scanline * stereogram_num_tiles_x * 4 + i * 4 + 2];
		((unsigned char*)fringeDataOut)[(j * fringe_buffer_res_x * 4) + (i * 4) + 3] = 0;
		//((unsigned char*)fringeDataOut)[j * 3552 * 4 + i * 4+1] = ((unsigned char*)rgbaIn)[j * 693 * 4 + i * 4];
		//((unsigned char*)fringeDataOut)[j * 3552 * 4 + i * 4 + 1] = ((unsigned char*)rgbaIn)[j * 693 * 4 + i * 4 + 1];
		//((unsigned char*)fringeDataOut)[j * 3552 * 4 + i * 4 + 2] = ((unsigned char*)rgbaIn)[j * 693 * 4 + i * 4 + 2];
		//((unsigned char*)fringeDataOut)[j * 3552 * 4 + i * 4 + 3] = 255;
		//((unsigned char*)fringeDataOut)[j * 3552 * 4 + i] = ((unsigned char*)rgbaIn)[j * 693 * 4 + i];
		//((unsigned char*)fringeDataOut)[j * 3552 * 4 + i+1] = ((unsigned char*)rgbaIn)[j * 693 * 4 + i+1];

		//((unsigned char*)fringeDataOut)[j * 3552 * 4 + i+3] = ((unsigned char*)rgbaIn)[j * 693 * 4 + i+3];

		((int*)fringeDataOut)[i] = 0;
		((int*)fringeDataOut)[j] = 0;
	}
}
