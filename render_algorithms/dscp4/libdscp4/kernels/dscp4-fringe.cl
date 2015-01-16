__constant sampler_t sampler =
CLK_NORMALIZED_COORDS_FALSE
| CLK_ADDRESS_CLAMP_TO_EDGE
| CLK_FILTER_NEAREST;

__kernel void turn_red(__write_only image2d_t fringe, __read_only image2d_t color, __read_only image2d_t depth, uint which_buffer)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int2 coords = (int2)(x, y);
	//Attention to RGBA order
	float4 val;

	if(which_buffer == 0)
		val = (float4)(1.0f, 0.0f, 0.0f, 1.0f);
	else if(which_buffer == 1)
		val = (float4)(0.0f, 1.0f, 0.0f, 1.0f);
	else
		val = (float4)(0.0f, 0.0f, 1.0f, 1.0f);
	
	val = read_imagef(color, sampler, coords);
	//val = read_imagef(depth, sampler, coords);

	//val[0] = 1.f - d[0];

	write_imagef(fringe, coords, val);
}

__kernel void compute_fringe(__global const int *A, __global const int *B, __global int *C) {

	// Get the index of the current element to be processed
	int i = get_global_id(0);

	// Do the operation
	C[i] = A[i] + B[i];
}