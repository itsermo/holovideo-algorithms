#include "dscp4-fringe-cuda.h"
#include <cuda.h>

const int N = 16;
const int blocksize = 16;

#include <stdio.h>

__global__ void hello(char *a, int *b)
{
	a[threadIdx.x] += b[threadIdx.x];
}


void dscp4_fringe_cuda_HelloWorld()
{
	char a[N] = "Hello \0\0\0\0\0\0";
	int b[N] = { 15, 10, 6, 0, -11, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0 };

	char *ad;
	int *bd;
	const int csize = N*sizeof(char);
	const int isize = N*sizeof(int);

	printf("%s", a);

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

	printf("%s\n", a);
	//return EXIT_SUCCESS;
};

dscp4_fringe_cuda_context_t* dscp4_fringe_cuda_CreateContext(dscp4_fringe_context_t fringeContext)
{
	dscp4_fringe_cuda_context_t* context = (dscp4_fringe_cuda_context_t*)malloc(sizeof(dscp4_fringe_cuda_context_t));
	context->fringe_context = fringeContext;

	return context;
};

void dscp4_fringe_cuda_DestroyContext(dscp4_fringe_cuda_context_t** cudaContext)
{
	if (*cudaContext != NULL)
	{
		free(*cudaContext);
		*cudaContext = NULL;
	}
};

void dscp4_fringe_cuda_ComputeFringe()
{

};