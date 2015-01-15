__kernel void turn_red(__write_only image2d_t bmp)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int2 coords = (int2)(x, y);
	//Attention to RGBA order
	float4 val = (float4)(1.0f, 0.0f, 0.0f, 1.0f);

	write_imagef(bmp, coords, val);
}

__kernel void compute_fringe(__global const int *A, __global const int *B, __global int *C) {

	// Get the index of the current element to be processed
	int i = get_global_id(0);

	// Do the operation
	C[i] = A[i] + B[i];
}