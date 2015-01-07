#include "dscp4-fringe-cuda.h"

__global__ void addOne(int* x)
{
	//*x = add1(*x);
	*x = 7;
}

__device__ int add1(int x)
{
	return x + 1;
}

__device__ int add2(int x)
{
	return add1(add1(x));
}