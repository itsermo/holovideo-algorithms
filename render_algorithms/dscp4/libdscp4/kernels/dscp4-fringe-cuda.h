#if !defined (DSCP4_FRINGE_H__)
#define DSCP4_FRINGE_H__

#ifndef __CUDACC__
#define __global__
#define __device__
#endif

__global__ void addOne(int* x);
__device__ int add1(int x);
__device__ int add2(int x);

#endif