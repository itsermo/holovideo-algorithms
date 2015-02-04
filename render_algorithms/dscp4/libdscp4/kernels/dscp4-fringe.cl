#if CONFIG_USE_DOUBLE

#if defined(cl_khr_fp64)  // Khronos extension available?
#pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)  // AMD extension available?
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
#endif

// double
typedef double real_t;
typedef double2 real2_t;
#define FFT_PI 3.14159265358979323846
#define FFT_SQRT_1_2 0.70710678118654752440

#else

// float
typedef float real_t;
typedef float2 real2_t;
#define FFT_PI       3.14159265359f
#define FFT_SQRT_1_2 0.707106781187f

#endif

__constant sampler_t sampler =
CLK_NORMALIZED_COORDS_FALSE
| CLK_ADDRESS_CLAMP_TO_EDGE
| CLK_FILTER_NEAREST;


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
	const real_t SPATIAL_UPCONVERT_CONST_R,
	const real_t SPATIAL_UPCONVERT_CONST_G,
	const real_t SPATIAL_UPCONVERT_CONST_B,
	const unsigned int NUM_SAMPLES_PER_WAFEL,
	const float SAMPLE_PITCH,
	const float Z_SPAN,
	const float Z_OFFSET,
	const unsigned int NUM_AOM_CHANNELS,
	const unsigned int HEAD_RES_Y_SPEC,
	const unsigned int NUM_BUFFERS
	)
{
	const unsigned int global_x = get_global_id(0);
	const unsigned int global_y = get_global_id(1);

	if (global_x < num_wafels_per_scanline && global_y < num_scanlines)
	{
		int x = global_x;
		int y = global_y;
		
		const unsigned int num_views = (viewset_num_tiles_x * viewset_num_tiles_y);

		for (int i = 0; i < NUM_SAMPLES_PER_WAFEL; i++)
		{
			wafel_position[i] = -(float)NUM_SAMPLES_PER_WAFEL * SAMPLE_PITCH / 2.f + i*(float)SAMPLE_PITCH;
			wafel_buffer[i] = 0;
		}

		for (unsigned int color_chan = 0; color_chan < 3; color_chan++)
		{
			x = global_x;
			y = global_y;

			float k = (color_chan == 0 ? K_R : color_chan == 1 ? K_G : K_B);
			double spatial_up_const = (color_chan == 0 ? SPATIAL_UPCONVERT_CONST_R : color_chan == 1 ? SPATIAL_UPCONVERT_CONST_G : SPATIAL_UPCONVERT_CONST_B);

			for (unsigned int vy = 0, idx = 0; vy < viewset_num_tiles_y; vy++)
			{
				for (unsigned int vx = 0; vx < viewset_num_tiles_x; vx++, idx++)
				{
					// Check later
					float d = (viewset_depth_in[y * viewset_res_x + x] - 0.5f) * Z_SPAN + Z_OFFSET;
					float temp_x = d * tan(30.f * 3.141592654f / 180.f * (idx - num_views / 2)) + NUM_SAMPLES_PER_WAFEL * SAMPLE_PITCH / 2;
					float4 color = read_imagef(viewset_color_in, sampler, (int2)(x,y));
					unsigned char c = (color_chan == 0 ? 255.f*color.x : color_chan == 1 ? 255.f*color.y : 255.f*color.z);

					for (int i = 0; i < NUM_SAMPLES_PER_WAFEL; i++)
					{
						//wafel_buffer[wafel_offset + i] += c / num_views * cos(k * (float)sqrt(pow((double)((double)wafel_position[wafel_offset + i] - (double)x), (double)2) + pow((double)d, (double)2)) - d + wafel_position[wafel_offset + i] * up_const);
						//wafel_buffer[i] += c * cos(k * sqrt((wafel_position[i] - temp_x)*(wafel_position[i] - temp_x) + d*d) - d + (global_x * NUM_SAMPLES_PER_WAFEL * SAMPLE_PITCH + temp_x) * (sin(30.f*3.141592654 / 180.f) + 2.f * 3.141592654 / k * spatial_up_const));
						double mycos = cos(k * sqrt((wafel_position[i] - temp_x)*(wafel_position[i] - temp_x) + d*d) - d + (global_x * NUM_SAMPLES_PER_WAFEL * SAMPLE_PITCH + temp_x) * (sin(30.f*3.141592654 / 180.f) + 2.f * 3.141592654 / k * spatial_up_const));
						wafel_buffer[i] += (unsigned char)((double)c * (mycos + 1.f) / 2.f);
						//wafel_buffer[i] = c;
					}

					x += num_wafels_per_scanline;
				}
				x = global_x;
				y += num_scanlines;
			}

		}

		int which_frame_buf = (global_y % NUM_AOM_CHANNELS);
		int which_hololine = global_y / NUM_AOM_CHANNELS;
	
		for (int i = 0; i < NUM_SAMPLES_PER_WAFEL; i++)
		{
			const unsigned int idx = which_frame_buf / NUM_BUFFERS * framebuffer_res_x * HEAD_RES_Y_SPEC * 4
				+ which_hololine * (((NUM_SAMPLES_PER_WAFEL * num_wafels_per_scanline) / framebuffer_res_x) * framebuffer_res_x * 4)
				+ NUM_SAMPLES_PER_WAFEL * 4 * global_x
				+ which_frame_buf % 3 + 4 * i;

			framebuffer_out[idx] = wafel_buffer[i];

			if (which_frame_buf % 3 == 2)
				framebuffer_out[idx + 1] = 255;
		}
	}


}