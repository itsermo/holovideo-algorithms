__constant sampler_t sampler =
CLK_NORMALIZED_COORDS_FALSE
| CLK_ADDRESS_CLAMP_TO_EDGE
| CLK_FILTER_NEAREST;

#ifndef M_PI
#define M_PI 3.14159265358979323846264338327950288f
#endif
#define K_R 2 * M_PI / 0.0000000633f
#define K_G 2 * M_PI / 0.0000000532f
#define K_B 2 * M_PI / 0.0000000445f
#define PIXELS_PER_HOLOLINE 355200
#define PIXEL_CLOCK_RATE 400000000
#define HOLOGRAM_PLANE_WIDTH 0.15f
#define TEMPORAL_UPCONVERT_R 225000000
#define TEMPORAL_UPCONVERT_G 290000000
#define TEMPORAL_UPCONVERT_B 350000000


// Computes the DSCP hologram, where stereogram
// depth component is stored in PBO instead of texture
// (NVIDIA does not have the OpenCL "cl_khr_gl_depth_images"
//  extension, which enables usage of depth texture directly)
__kernel void computeFringe(
	__write_only image2d_t frame_buffer_out,
	__read_only image2d_t viewSet_color_in,
	__global __read_only float* viewSet_depth_in,
	uint which_buffer,
	uint num_wafels_per_scanline,
	uint num_scanlines,
	uint viewSet_res_x,
	uint viewSet_res_y,
	uint viewSet_num_tiles_x,
	uint viewSet_num_tiles_y,
	uint framebuffer_res_x,
	uint framebuffer_res_y
	)
{
	int2 coords = (int2)(get_global_id(0), get_global_id(1));

	if (coords.x < num_wafels_per_scanline && coords.y < num_scanlines)
	{
		for (uint color_chan = 0; color_chan < 3; color_chan++)
		{
			coords.x = get_global_id(0);
			coords.y = get_global_id(1);

			float wafel = 0.f;

			for (uint vy = 0; vy < viewSet_num_tiles_y; vy++)
			{
				for (uint vx = 0; vx < viewSet_num_tiles_x; vx++)
				{
					//Attention to RGBA order
					float4 val;
					float d = viewSet_depth_in[coords.y * stereogram_res_x + coords.x];

					if (which_buffer == 0)
						val = (float4)(1.0f, 0.0f, 0.0f, 1.0f);
					else if (which_buffer == 1)
						val = (float4)(0.0f, 1.0f, 0.0f, 1.0f);
					else
						val = (float4)(0.0f, 0.0f, 1.0f, 1.0f);

					//val = read_imagef(color, sampler, coords);
					//float4 d = read_imagef(depth, sampler, coords);

					val.x = cos(sqrt(1.f - d*d + d*d)*TWO_PI_COLOR_R - d + d*sin(30.f) - 0.5f);

					write_imagef(frame_buffer_out, coords, val);
					coords.x += num_wafels_per_scanline;
				}

				coords.x = get_global_id(0);
				coords.y += num_scanlines;
			}
		}
	}
}

// Computes the DSCP hologram, using depth and color
// textures directly from the render buffer (faster)
__kernel void computeFringe2(
	__write_only image2d_t fringe_buffer_out,
	__read_only image2d_t stereogram_color_in,
	__read_only image2d_t stereogram_depth_in,
	uint which_fringe_buffer,
	uint num_wafels_per_scanline,
	uint num_scanlines,
	uint stereogram_res_x,
	uint stereogram_res_y,
	uint stereogram_tile_x,
	uint stereogram_tile_y,
	uint fringe_res_x,
	uint fringe_res_y
	)
{
	int x = get_global_id(0);
	int y = get_global_id(1);

	if (x < num_wafels_per_scanline && y < num_scanlines)
	{
		int2 coords = (int2)(x, y);
		//Attention to RGBA order
		float4 val;
		//float d = depth[y * stereogram_res_x + x];

		if (which_fringe_buffer == 0)
			val = (float4)(1.0f, 0.0f, 0.0f, 1.0f);
		else if (which_fringe_buffer == 1)
			val = (float4)(0.0f, 1.0f, 0.0f, 1.0f);
		else
			val = (float4)(0.0f, 0.0f, 1.0f, 1.0f);

		//val = read_imagef(color, sampler, coords);
		float4 d = read_imagef(stereogram_color_in, sampler, coords);

		val.x = 1.f - d.y;

		write_imagef(fringe_buffer_out, coords, val);
	}
}