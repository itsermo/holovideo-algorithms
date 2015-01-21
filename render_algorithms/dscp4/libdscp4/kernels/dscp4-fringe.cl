__constant sampler_t sampler =
CLK_NORMALIZED_COORDS_FALSE
| CLK_ADDRESS_CLAMP_TO_EDGE
| CLK_FILTER_NEAREST;

__constant float m_pi = 3.14159265358979323846264338327950288f;
__constant float two_pi_color_r = 2.f * m_pi / 0.0000350;
__constant float two_pi_color_g = 2.f * m_pi / 0.0000450;
__constant float two_pi_color_b = 2.f * m_pi / 0.0000650;


// Computes the DSCP hologram, where stereogram
// depth component is stored in PBO instead of texture
// (NVIDIA does not have the OpenCL "cl_khr_gl_depth_images"
//  extension, which enables usage of depth texture directly)
__kernel void computeFringe(
	__write_only image2d_t fringe_buffer_out,
	__read_only image2d_t stereogram_color_in,
	__global __read_only float* stereogram_depth_in,
	uint which_buffer,
	uint num_wafels_per_scanline,
	uint num_scanlines,
	uint stereogram_res_x,
	uint stereogram_res_y,
	uint stereogram_num_tiles_x,
	uint stereogram_num_tiles_y,
	uint fringe_res_x,
	uint fringe_res_y
	)
{
	int2 coords = (int2)(get_global_id(0), get_global_id(1));

	if (coords.x < num_wafels_per_scanline && coords.y < num_scanlines)
	{
		for (uint color_chan = 0; color_chan < 3; color_chan++)
		{
			coords.x = get_global_id(0);
			coords.y = get_global_id(1);
			for (uint sy = 0; sy < 4; sy++)
			{
				for (uint sx = 0; sx < 4; sx++)
				{
					//Attention to RGBA order
					float4 val;
					float d = stereogram_depth_in[coords.y * stereogram_res_x + coords.x];

					if (which_buffer == 0)
						val = (float4)(1.0f, 0.0f, 0.0f, 1.0f);
					else if (which_buffer == 1)
						val = (float4)(0.0f, 1.0f, 0.0f, 1.0f);
					else
						val = (float4)(0.0f, 0.0f, 1.0f, 1.0f);

					//val = read_imagef(color, sampler, coords);
					//float4 d = read_imagef(depth, sampler, coords);

					val.x = cos(sqrt(1.f - d*d)*m_pi);


					//coords.y += sy * num_scanlines;

					write_imagef(fringe_buffer_out, coords, val);
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
		float4 d = read_imagef(stereogram_depth_in, sampler, coords);

		val.x = 1.f - d.x;

		write_imagef(fringe_buffer_out, coords, val);
	}
}