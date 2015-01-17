__constant sampler_t sampler =
CLK_NORMALIZED_COORDS_FALSE
| CLK_ADDRESS_CLAMP_TO_EDGE
| CLK_FILTER_NEAREST;

__kernel void test_hologram(__write_only image2d_t fringe,
	__read_only image2d_t color,
	__global float* depth,
	uint which_buffer,
	uint num_wafels_per_scanline,
	uint num_scanlines,
	uint stereogram_res_x,
	uint stereogram_res_y,
	uint stereogram_tile_x,
	uint stereogram_tile_y,
	uint buffer_res_x,
	uint buffer_res_y	
	)
{
	int x = get_global_id(0);
	int y = get_global_id(1);

	if (x < num_wafels_per_scanline && y < num_scanlines)
	{
		int2 coords = (int2)(x, y);
		//Attention to RGBA order
		float4 val;
		float d = depth[y * stereogram_res_x + x];

		if (which_buffer == 0)
			val = (float4)(1.0f, 0.0f, 0.0f, 1.0f);
		else if (which_buffer == 1)
			val = (float4)(0.0f, 1.0f, 0.0f, 1.0f);
		else
			val = (float4)(0.0f, 0.0f, 1.0f, 1.0f);

		//val = read_imagef(color, sampler, coords);
		//float4 d = read_imagef(depth, sampler, coords);

		val.x = 1.f - d;

		write_imagef(fringe, coords, val);
	}
}

__kernel void compute_hologram(__global const int *A, __global const int *B, __global int *C) {

	// Get the index of the current element to be processed
	int i = get_global_id(0);

	// Do the operation
	C[i] = A[i] + B[i];
}