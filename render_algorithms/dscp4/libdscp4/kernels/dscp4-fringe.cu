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
	((char*)rgbaIn)[threadIdx.y * 693 * 4 + threadIdx.x] = 255 - ((char*)rgbaIn)[threadIdx.y * 693 * 4 + threadIdx.x];
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

	//cudaContext->fringe_cuda_resources = (struct cudaGraphicsResource**)malloc(sizeof(void*)*cudaContext->fringe_context->display_options.num_heads / 2);

	//for (unsigned int i = 0; i < cudaContext->fringe_context->display_options.num_heads / 2; i++)
	//{
	//	error = cudaGraphicsGLRegisterImage(&cudaContext->fringe_cuda_resources[i], cudaContext->fringe_context->fringe_gl_buf_out[i], GL_TEXTURE_2D, cudaGraphicsMapFlagsWriteDiscard);
	//}

	return cudaContext;
};

void dscp4_fringe_cuda_DestroyContext(dscp4_fringe_cuda_context_t** cudaContext)
{
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
	cudaArray *output;

	unsigned int * rgbaPtr;
	unsigned int * depthPtr;
	size_t rgbaSize;
	size_t depthSize;

	cudaError_t error = cudaSuccess;

	error = cudaGraphicsMapResources(1, &cudaContext->stereogram_rgba_cuda_resource, 0);
	error = cudaGraphicsMapResources(1, &cudaContext->stereogram_depth_cuda_resource, 0);

	cudaGraphicsResourceGetMappedPointer((void**)&rgbaPtr, &rgbaSize, cudaContext->stereogram_rgba_cuda_resource);
	cudaGraphicsResourceGetMappedPointer((void**)&depthPtr, &depthSize, cudaContext->stereogram_depth_cuda_resource);



	//for (int i = 0; i < cudaContext->fringe_context->display_options.num_heads / 2; i++)
	//{

	//	error = cudaGraphicsMapResources(1, (cudaGraphicsResource_t*)(&cudaContext->fringe_cuda_resources[i]), 0);

	//	error = cudaGraphicsSubResourceGetMappedArray(&output, cudaContext->fringe_cuda_resources[i], 0, 0);


	//}

	// run kernel here

	dim3 dimBlock(blocksize, 1);
	dim3 dimGrid(1, 1);
	computeFringe << <dimGrid, 1024 >> >(NULL, rgbaPtr, depthPtr);

	//write texture outputs here

	//for (int i = 0; i < cudaContext->fringe_context->display_options.num_heads / 2; i++)
	//{
	//	error = cudaGraphicsUnmapResources(1, (cudaGraphicsResource_t*)(&cudaContext->fringe_cuda_resources[i]), 0);
	//}

	error = cudaGraphicsUnmapResources(1, &cudaContext->stereogram_rgba_cuda_resource, 0);
	error = cudaGraphicsUnmapResources(1, &cudaContext->stereogram_depth_cuda_resource, 0);
	
};