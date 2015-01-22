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
#define PIXEL_PITCH HOLOGRAM_PLANE_WIDTH / PIXELS_PER_HOLOLINE
#define Z_SPAN 0.5f
#define Z_OFFSET 0.f
#define PIXELS_PER_WAFEL 592
#define THETA 30.f*M_PI/180.f
#define UPCONVERT_CONST_R (sin(THETA) + 2 * M_PI / K_R * PIXELS_PER_HOLOLINE * TEMPORAL_UPCONVERT_R / (PIXEL_CLOCK_RATE * HOLOGRAM_PLANE_WIDTH))
#define UPCONVERT_CONST_G (sin(THETA) + 2 * M_PI / K_G * PIXELS_PER_HOLOLINE * TEMPORAL_UPCONVERT_G / (PIXEL_CLOCK_RATE * HOLOGRAM_PLANE_WIDTH))
#define UPCONVERT_CONST_B (sin(THETA) + 2 * M_PI / K_B * PIXELS_PER_HOLOLINE * TEMPORAL_UPCONVERT_B / (PIXEL_CLOCK_RATE * HOLOGRAM_PLANE_WIDTH))
#define NUM_HOLO_CHANNELS 18


// Computes the DSCP hologram, where stereogram
// depth component is stored in PBO instead of texture
// (NVIDIA does not have the OpenCL "cl_khr_gl_depth_images"
//  extension, which enables usage of depth texture directly)
__kernel void computeFringe(
	__write_only image2d_t framebuffer_0_out,
	__write_only image2d_t framebuffer_1_out,
	__write_only image2d_t framebuffer_2_out,
	__read_only image2d_t viewSet_color_in,
	__global __read_only float* viewSet_depth_in,
	uint num_wafels_per_scanline,
	uint num_scanlines,
	uint viewSet_res_x,
	uint viewSet_res_y,
	uint viewSet_num_tiles_x,
	uint viewSet_num_tiles_y,
	uint framebuffer_res_x,
	uint framebuffer_res_y,
	__local char* wafel_buffer,
	__local char* wafel_position
	)
{
	int2 coords = (int2)(get_global_id(0), get_global_id(1));

	for (int i = 0; i < PIXELS_PER_WAFEL; i++)
	{ 
		wafel_position[i] = (-300 + i) * PIXEL_PITCH + coords.x;
	}

	if (coords.x < num_wafels_per_scanline && coords.y < num_scanlines)
	{
		for (uint color_chan = 0; color_chan < 3; color_chan++)
		{
			coords.x = get_global_id(0);
			coords.y = get_global_id(1);

			float wafel = 0.f;
			float k = (color_chan == 0 ? K_R : color_chan == 1 ? K_G : K_B);
			float up_const = (color_chan == 0 ? UPCONVERT_CONST_R : color_chan == 1 ? UPCONVERT_CONST_G : UPCONVERT_CONST_B);

			for (uint vy = 0; vy < viewSet_num_tiles_y; vy++)
			{

				for (uint vx = 0; vx < viewSet_num_tiles_x; vx++)
				{
					float4 color = read_imagef(viewSet_color_in, sampler, coords);

					float c = 255.f * (color_chan == 0 ? color.x : color_chan == 1 ? color.y : color.z);
					float d = (viewSet_depth_in[coords.y * viewSet_res_x + coords.x] - 0.5) * Z_SPAN + Z_OFFSET;


					for (int i = 0; i < PIXELS_PER_WAFEL; i++)
					{ 
						wafel_buffer[i] += c * cos(k * sqrt( pow((float)((int)wafel_position[i] - (int)coords.x), (float)2) + pow(d,(float)2)) - d + wafel_position[i] * up_const);
					}
					////Attention to RGBA order
					//float4 val;
					//float d = viewSet_depth_in[coords.y * viewSet_res_x + coords.x];

					//if (which_buffer == 0)
					//	val = (float4)(1.0f, 0.0f, 0.0f, 1.0f);
					//else if (which_buffer == 1)
					//	val = (float4)(0.0f, 1.0f, 0.0f, 1.0f);
					//else
					//	val = (float4)(0.0f, 0.0f, 1.0f, 1.0f);

					////val = read_imagef(color, sampler, coords);
					////float4 d = read_imagef(depth, sampler, coords);

					//val.x = cos(sqrt(1.f - d*d + d*d)*K_R - d + d*sin(30.f) - 0.5f);

					//write_imagef(frame_buffer_out, coords, val);
					coords.x += num_wafels_per_scanline;
				}

				coords.x = get_global_id(0);
				coords.y += num_scanlines;
			}
		}

		/*int x = get_global_id(0);
		int y = get_global_id(1);*/

		coords.x = get_global_id(0);
		coords.y = get_global_id(1);

		int which_frame_buf = (coords.y % NUM_HOLO_CHANNELS);
		int hololine = coords.y / NUM_HOLO_CHANNELS;
		int frameline = (float)coords.x / (framebuffer_res_x / PIXELS_PER_WAFEL);
		int wafel = coords.x - frameline * framebuffer_res_x / PIXELS_PER_WAFEL;

		int which_rgba = coords.y % 3;

		for (int i = 0; i < PIXELS_PER_WAFEL; i++)
		{
			float4 val = (float4)(0.f,0.f,0.f,0.f);
			int2 frame_coords = (int2)(0, 0);

			switch (which_frame_buf)
			{
			case 0:
			case 1:
			case 2:
			case 9:
			case 10:
			case 11:

				frame_coords.y = which_frame_buf < 9 ? 0 : framebuffer_res_y / 2;

				frame_coords.y += PIXELS_PER_HOLOLINE / framebuffer_res_x * hololine + frameline;

				frame_coords.x = wafel * PIXELS_PER_WAFEL;

				//val = read_imagef(framebuffer_0_out, sampler, frame_coords);

				break;
			case 3:
			case 4:
			case 5:
			case 12:
			case 13:
			case 14:

				frame_coords.y = which_frame_buf < 9 ? 0 : framebuffer_res_y / 2;

				frame_coords.y += PIXELS_PER_HOLOLINE / framebuffer_res_x * hololine + frameline;

				frame_coords.x = wafel * PIXELS_PER_WAFEL;

				//val = read_imagef(framebuffer_1_out, sampler, frame_coords);

				break;
			case 6:
			case 7:
			case 8:
			case 15:
			case 16:
			case 17:
				frame_coords.y = which_frame_buf < 9 ? 0 : framebuffer_res_y / 2;

				frame_coords.y += PIXELS_PER_HOLOLINE / framebuffer_res_x * hololine + frameline;

				frame_coords.x = wafel * PIXELS_PER_WAFEL;

				//val = read_imagef(framebuffer_2_out, sampler, frame_coords);
				break;
			}

			switch (which_rgba)
			{
				//blue
			case 0:
				val.z = wafel_buffer[i] / 255.f;
				break;
			case 1: //green
				val.y = wafel_buffer[i] / 255.f;
				break;
			case 2: //red
				val.x = wafel_buffer[i] / 255.f;
				break;
			}

			switch (which_frame_buf)
			{
			case 0:
			case 1:
			case 2:
			case 9:
			case 10:
			case 11:

				write_imagef(framebuffer_0_out, frame_coords, val);

				break;
			case 3:
			case 4:
			case 5:
			case 12:
			case 13:
			case 14:

				write_imagef(framebuffer_1_out, frame_coords, val);

				break;
			case 6:
			case 7:
			case 8:
			case 15:
			case 16:
			case 17:

				write_imagef(framebuffer_2_out, frame_coords, val);

				break;
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