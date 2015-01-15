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

__global__ void hello(char *a, int *b)
{
	a[threadIdx.x] += b[threadIdx.x];
}

__global__ void computeFringe(void * fringeDataOut, void * rgbaIn, void * depthIn)
{
	unsigned int i = (blockIdx.x * blockDim.x) + threadIdx.x;
	unsigned int j = (blockIdx.y * blockDim.y) + threadIdx.y;
	((int*)fringeDataOut)[i] = 0;
	((int*)fringeDataOut)[j] = 0;
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

	error = cudaGraphicsGLRegisterBuffer(&cudaContext->stereogram_rgba_cuda_resource, cudaContext->fringe_context->stereogram_gl_rgba_buf_in, cudaGraphicsRegisterFlagsReadOnly);
	error = cudaGraphicsGLRegisterBuffer(&cudaContext->stereogram_depth_cuda_resource, cudaContext->fringe_context->stereogram_gl_depth_buf_in, cudaGraphicsRegisterFlagsReadOnly);

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
	// The total number of wafels in one frame
	const int NUM_WAFELS = cudaContext->fringe_context->algorithm_options.num_wafels_per_scanline *
		cudaContext->fringe_context->algorithm_options.num_scanlines;
	
	// The size (in bytes) per wafel
	const size_t WAFEL_SIZE = cudaContext->fringe_context->display_options.head_res_x *
		cudaContext->fringe_context->display_options.head_res_y *
		cudaContext->fringe_context->display_options.num_heads * sizeof(char) * 3 /
		NUM_WAFELS;

	void **output;
	size_t * outputSizes;

	void * rgbaPtr;
	void * depthPtr;
	size_t rgbaSize;
	size_t depthSize;

	cudaError_t error = cudaSuccess;

	error = cudaGraphicsMapResources(1, &cudaContext->stereogram_rgba_cuda_resource, 0);
	error = cudaGraphicsMapResources(1, &cudaContext->stereogram_depth_cuda_resource, 0);

	cudaGraphicsResourceGetMappedPointer((void**)&rgbaPtr, &rgbaSize, cudaContext->stereogram_rgba_cuda_resource);
	cudaGraphicsResourceGetMappedPointer((void**)&depthPtr, &depthSize, cudaContext->stereogram_depth_cuda_resource);

	output = (void**)malloc(sizeof(void*)* cudaContext->fringe_context->display_options.num_heads / 2);
	outputSizes = (size_t*)malloc(sizeof(size_t) * cudaContext->fringe_context->display_options.num_heads / 2);

	for (int i = 0; i < cudaContext->fringe_context->display_options.num_heads / 2; i++)
	{
		error = cudaGraphicsMapResources(1, (cudaGraphicsResource_t*)(&cudaContext->fringe_cuda_resources[i]), 0);
		error = cudaGraphicsResourceGetMappedPointer(&output[i], &outputSizes[i], cudaContext->fringe_cuda_resources[i]);
	}

	//error = cudaMemset(output[0], 255, outputSizes[0]);

	// run kernel here
	dim3 threadsPerBlock(
		cudaContext->fringe_context->algorithm_options.num_wafels_per_scanline,
		cudaContext->fringe_context->algorithm_options.num_scanlines);
	dim3 numBlocks(cudaContext->fringe_context->algorithm_options.num_wafels_per_scanline * 4 / threadsPerBlock.x,
		cudaContext->fringe_context->algorithm_options.num_scanlines * 4 / threadsPerBlock.y);
	computeFringe <<<numBlocks, threadsPerBlock >>>(output[0], rgbaPtr, depthPtr);


	//write texture outputs here

	for (int i = 0; i < cudaContext->fringe_context->display_options.num_heads / 2; i++)
	{
		error = cudaGraphicsUnmapResources(1, (cudaGraphicsResource_t*)(&cudaContext->fringe_cuda_resources[i]), 0);
	}

	error = cudaGraphicsUnmapResources(1, &cudaContext->stereogram_rgba_cuda_resource, 0);
	error = cudaGraphicsUnmapResources(1, &cudaContext->stereogram_depth_cuda_resource, 0);

	free(output);
	free(outputSizes);
};