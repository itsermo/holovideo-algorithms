__constant sampler_t sampler =
CLK_NORMALIZED_COORDS_FALSE
| CLK_ADDRESS_CLAMP_TO_EDGE
| CLK_FILTER_NEAREST;

//#ifndef M_PI
//#define M_PI 3.14159265358979323846264338327950288f
//#endif
//#define K_R 2 * M_PI / 0.000000633f
//#define K_G 2 * M_PI / 0.000000532f
//#define K_B 2 * M_PI / 0.000000445f
//#define PIXELS_PER_HOLOLINE 355200
//#define PIXEL_CLOCK_RATE 400000000
//#define HOLOGRAM_PLANE_WIDTH 0.15f
//#define TEMPORAL_UPCONVERT_R 225000000
//#define TEMPORAL_UPCONVERT_G 290000000
//#define TEMPORAL_UPCONVERT_B 350000000
//#define PIXEL_PITCH HOLOGRAM_PLANE_WIDTH / PIXELS_PER_HOLOLINE
//#define Z_SPAN 0.5f
//#define Z_OFFSET 0.f
//#define PIXELS_PER_WAFEL 592
//#define THETA 30.f*M_PI/180.f
////#define THETA 0.f
//#define UPCONVERT_CONST_R (sin(THETA) + 2 * M_PI / K_R * PIXELS_PER_HOLOLINE * TEMPORAL_UPCONVERT_R / (PIXEL_CLOCK_RATE * HOLOGRAM_PLANE_WIDTH))
//#define UPCONVERT_CONST_G (sin(THETA) + 2 * M_PI / K_G * PIXELS_PER_HOLOLINE * TEMPORAL_UPCONVERT_G / (PIXEL_CLOCK_RATE * HOLOGRAM_PLANE_WIDTH))
//#define UPCONVERT_CONST_B (sin(THETA) + 2 * M_PI / K_B * PIXELS_PER_HOLOLINE * TEMPORAL_UPCONVERT_B / (PIXEL_CLOCK_RATE * HOLOGRAM_PLANE_WIDTH))
//#define NUM_HOLO_CHANNELS 18


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

//__kernel void computeFringeTestBuffer(
//	__global unsigned char* framebuffer_out,
//	__read_only image2d_t viewset_color_in,
//	__global __read_only float* viewset_depth_in,
//	uint num_wafels_per_scanline,
//	uint num_scanlines,
//	uint viewset_res_x,
//	uint viewset_res_y,
//	uint viewset_num_tiles_x,
//	uint viewset_num_tiles_y,
//	uint framebuffer_res_x,
//	uint framebuffer_res_y,
//	__local unsigned char* wafel_buffer,
//	__local float* wafel_position
//	)
//{
//	int x = get_global_id(0);
//	int y = get_global_id(1);
//
//	const int framebuffer_size = framebuffer_res_x * framebuffer_res_y * 4;
//
//	if (x < num_wafels_per_scanline && y < num_scanlines)
//	{
//
//		for (uint color_chan = 0; color_chan < 3; color_chan++)
//		{
//			x = get_global_id(0);
//			y = get_global_id(1);
//
//			float wafel = 0.f;
//			//float k = (color_chan == 0 ? K_R : color_chan == 1 ? K_G : K_B);
//			//float up_const = (color_chan == 0 ? UPCONVERT_CONST_R : color_chan == 1 ? UPCONVERT_CONST_G : UPCONVERT_CONST_B);
//
//			for (uint vy = 0; vy < viewset_num_tiles_y; vy++)
//			{
//
//				for (uint vx = 0; vx < viewset_num_tiles_x; vx++)
//				{
//					float d = (viewset_depth_in[y * viewset_res_x + x] - 0.5) * Z_SPAN + Z_OFFSET;
//					float4 c = read_imagef(viewset_color_in, sampler, (int2)(x, y));
//
//					for(int i = 0; i < 3; i++)
//					{
//						framebuffer_out[i*framebuffer_size + y*framebuffer_res_x * 4 + 4 * x] = floor(c.x * 255.f);
//						framebuffer_out[i*framebuffer_size + y*framebuffer_res_x * 4 + 4 * x + 1] = floor(c.y * 255.f);
//						framebuffer_out[i*framebuffer_size + y*framebuffer_res_x * 4 + 4 * x + 2] = floor(c.z * 255.f);
//						framebuffer_out[i*framebuffer_size + y*framebuffer_res_x * 4 + 4 * x + 3] = 255;
//					//framebuffer_out[y*framebuffer_res_x * 3 + x + 3] = 0;
//					}
//					x += num_wafels_per_scanline;
//				}
//
//				x = get_global_id(0);
//				y += num_scanlines;
//
//			}
//		}
//
//	}
//}

//__kernel void computeFringe3(
//	__global unsigned char* framebuffer_out,
//	__read_only image2d_t viewset_color_in,
//	__global __read_only float* viewset_depth_in,
//	const uint num_wafels_per_scanline,
//	const uint num_scanlines,
//	const uint viewset_res_x,
//	const uint viewset_res_y,
//	const uint viewset_num_tiles_x,
//	const uint viewset_num_tiles_y,
//	const uint framebuffer_res_x,
//	const uint framebuffer_res_y,
//	__local unsigned char* wafel_buffer,
//	__local float* wafel_position
//	)
//{
//	int x = get_global_id(0);
//	int y = get_global_id(1);
//
//	if (x < num_wafels_per_scanline && y < num_scanlines)
//	{
//		const int framebuffer_size = framebuffer_res_x * framebuffer_res_y * 4;
//
//		for (int i = 0; i < PIXELS_PER_WAFEL; i++)
//		{
//			wafel_position[i] = (-ceil((float)num_wafels_per_scanline/2.f) + i) * PIXEL_PITCH + x;
//		}
//
//		for (uint color_chan = 0; color_chan < 3; color_chan++)
//		{
//			x = get_global_id(0);
//			y = get_global_id(1);
//
//			float wafel = 0.f;
//			float k = (color_chan == 0 ? K_R : color_chan == 1 ? K_G : K_B);
//			float up_const = (color_chan == 0 ? UPCONVERT_CONST_R : color_chan == 1 ? UPCONVERT_CONST_G : UPCONVERT_CONST_B);
//
//			for (uint vy = 0; vy < viewset_num_tiles_y; vy++)
//			{
//				for (uint vx = 0; vx < viewset_num_tiles_x; vx++)
//				{
//					float d = (viewset_depth_in[y * viewset_res_x + x] - 0.5) * Z_SPAN + Z_OFFSET;
//					float4 color = read_imagef(viewset_color_in, sampler, (int2)(x, y));
//					float c = 255.f*(color_chan == 0 ? color.x : color_chan == 1 ? color.y : color.z);
//
//					//framebuffer_out[color_chan*framebuffer_res_x * 2600 * 4 + y*framebuffer_res_x * 4 + 4 * x] = floor(color.x * 255.f);
//					//framebuffer_out[color_chan*framebuffer_res_x * 2600 * 4 + y * framebuffer_res_x * 4 + 4 * x + 1] = floor(color.y * 255.f);
//					//framebuffer_out[color_chan*framebuffer_res_x * 2600 * 4 + y * framebuffer_res_x * 4 + 4 * x + 2] = floor(color.z * 255.f);
//					//framebuffer_out[color_chan*framebuffer_res_x * 2600 * 4 + y * framebuffer_res_x * 4 + 4 * x + 3] = 255;
//
//					for (int i = 0; i < PIXELS_PER_WAFEL; i++)
//					{
//						wafel_buffer[i] += c / 16 * cos(k * sqrt(pow((float)((int)wafel_position[i] - (int)x), (float)2) + pow(d, (float)2)) - d + wafel_position[i] * up_const);
//					}
//
//					x += num_wafels_per_scanline;
//				}
//
//				x = get_global_id(0);
//				y += num_scanlines;
//
//			}
//
//
//		}
//
//		int2 coords = (int2)(get_global_id(0), get_global_id(1));
//
//
//		int which_frame_buf = (coords.y % NUM_HOLO_CHANNELS);
//		int which_hololine = coords.y / NUM_HOLO_CHANNELS;
//		int which_frameline = (float)coords.x / (framebuffer_res_x / PIXELS_PER_WAFEL);
//		int which_wafel = coords.x - (which_frameline * (framebuffer_res_x / PIXELS_PER_WAFEL));
//
//		for (int i = 0; i < PIXELS_PER_WAFEL; i++)
//		{
//			framebuffer_out[which_frame_buf / 3 * framebuffer_res_x * 2600 * 4 + which_hololine * (100 * framebuffer_res_x * 4) + PIXELS_PER_WAFEL * 4 * coords.x + which_frame_buf % 3 + 4 * i] = wafel_buffer[i];
//		}
//		
//	}
//}

__kernel void computeFringeVar(
	__global unsigned char* framebuffer_out,
	__read_only image2d_t viewset_color_in,
	__global __read_only float* viewset_depth_in,
	const uint num_wafels_per_scanline,
	const uint num_scanlines,
	const uint viewset_res_x,
	const uint viewset_res_y,
	const uint viewset_num_tiles_x,
	const uint viewset_num_tiles_y,
	const uint framebuffer_res_x,
	const uint framebuffer_res_y,
	__local unsigned char* wafel_buffer,
	__local float* wafel_position,
	const float K_R,
	const float K_G,
	const float K_B,
	const float UPCONVERT_CONST_R,
	const float UPCONVERT_CONST_G,
	const float UPCONVERT_CONST_B,
	const unsigned int NUM_SAMPLES_PER_WAFEL,
	const float SAMPLE_PITCH,
	const float Z_SPAN,
	const float Z_OFFSET,
	const unsigned int NUM_AOM_CHANNELS,
	const unsigned int HEAD_RES_Y_SPEC,
	const unsigned int NUM_BUFFERS
	)
{
	int x = get_global_id(0);
	int y = get_global_id(1);

	if (x < num_wafels_per_scanline && y < num_scanlines)
	{
		const float num_views = (viewset_num_tiles_x * viewset_num_tiles_y);

		for (int i = 0; i < NUM_SAMPLES_PER_WAFEL; i++)
		{
			wafel_position[i] = (-ceil((float)num_wafels_per_scanline / 2.f) + i) * SAMPLE_PITCH + x;
		}

		for (uint color_chan = 0; color_chan < 3; color_chan++)
		{
			x = get_global_id(0);
			y = get_global_id(1);

			float wafel = 0.f;
			float k = (color_chan == 0 ? K_R : color_chan == 1 ? K_G : K_B);
			float up_const = (color_chan == 0 ? UPCONVERT_CONST_R : color_chan == 1 ? UPCONVERT_CONST_G : UPCONVERT_CONST_B);

			for (uint vy = 0; vy < viewset_num_tiles_y; vy++)
			{
				for (uint vx = 0; vx < viewset_num_tiles_x; vx++)
				{
					float d = (viewset_depth_in[y * viewset_res_x + x] - 0.5) * Z_SPAN + Z_OFFSET;
					float4 color = read_imagef(viewset_color_in, sampler, (int2)(x, y));
					float c = 255.f*(color_chan == 0 ? color.x : color_chan == 1 ? color.y : color.z);

					//framebuffer_out[color_chan*framebuffer_res_x * 2600 * 4 + y*framebuffer_res_x * 4 + 4 * x] = floor(color.x * 255.f);
					//framebuffer_out[color_chan*framebuffer_res_x * 2600 * 4 + y * framebuffer_res_x * 4 + 4 * x + 1] = floor(color.y * 255.f);
					//framebuffer_out[color_chan*framebuffer_res_x * 2600 * 4 + y * framebuffer_res_x * 4 + 4 * x + 2] = floor(color.z * 255.f);
					//framebuffer_out[color_chan*framebuffer_res_x * 2600 * 4 + y * framebuffer_res_x * 4 + 4 * x + 3] = 255;

					for (int i = 0; i < NUM_SAMPLES_PER_WAFEL; i++)
					{
						wafel_buffer[i] += c / num_views * cos(k * sqrt(pow((float)((int)wafel_position[i] - (int)x), (float)2) + pow(d, (float)2)) - d + wafel_position[i] * up_const);
					}
					x += num_wafels_per_scanline;
				}
				x = get_global_id(0);
				y += num_scanlines;
			}
		}

		int2 coords = (int2)(get_global_id(0), get_global_id(1));

		int which_frame_buf = (coords.y % NUM_AOM_CHANNELS);
		int which_hololine = coords.y / NUM_AOM_CHANNELS;
		int which_frameline = (float)coords.x / (framebuffer_res_x / NUM_SAMPLES_PER_WAFEL);
//		int which_wafel = coords.x - (which_frameline * (framebuffer_res_x / NUM_SAMPLES_PER_WAFEL));

		for (int i = 0; i < NUM_SAMPLES_PER_WAFEL; i++)
		{
			framebuffer_out[
				which_frame_buf / NUM_BUFFERS * framebuffer_res_x * HEAD_RES_Y_SPEC * 4
					+ which_hololine * (((NUM_SAMPLES_PER_WAFEL * num_wafels_per_scanline) / framebuffer_res_x) * framebuffer_res_x * 4)
					+ NUM_SAMPLES_PER_WAFEL * 4 * coords.x
					+ which_frame_buf % 3 + 4 * i
			] = wafel_buffer[i];
		}
	}
}