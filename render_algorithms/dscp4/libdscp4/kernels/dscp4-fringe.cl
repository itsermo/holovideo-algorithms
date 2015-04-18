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

__kernel void computeFringe(
	__global unsigned char* framebuffer_out,	//0
	__read_only image2d_t viewset_color_in,		//1
#ifdef CONFIG_USE_DEPTH_TEXTURE
	__read_only image2d_t viewset_depth_in,		//2
#else
	__global __read_only float* viewset_depth_in,
#endif
	const uint num_wafels_per_scanline,			//3
	const uint num_scanlines,					//4
	const uint viewset_res_x,					//5
	const uint viewset_res_y,					//6
	const uint viewset_num_tiles_x,				//7
	const uint viewset_num_tiles_y,				//8
	const uint framebuffer_res_x,				//9
	const uint framebuffer_res_y,				//10
	const float redGain,						//11
	const float greenGain,						//12
	const float blueGain,						//13
	const float REF_BEAM_ANGLE_RAD,				//14
	__local float* wafel_position,				//15
	const float K_R,							//16
	const float K_G,							//17
	const float K_B,							//18
	const real_t SPATIAL_UPCONVERT_CONST_R,		//19
	const real_t SPATIAL_UPCONVERT_CONST_G,		//20
	const real_t SPATIAL_UPCONVERT_CONST_B,		//21
	const unsigned int NUM_SAMPLES_PER_WAFEL,	//22
	const float SAMPLE_PITCH,					//23
	const float Z_SPAN,							//24
	const float Z_OFFSET,						//25
	const unsigned int NUM_AOM_CHANNELS,		//26
	const unsigned int HEAD_RES_Y_SPEC,			//27
	const unsigned int NUM_BUFFERS				//28
	)
{
	const unsigned int global_x = get_global_id(0);
	const unsigned int global_y = get_global_id(1);

	if (global_x < num_wafels_per_scanline && global_y < num_scanlines)
	{
		int x = global_x;
		int y = global_y;
		
		const unsigned int num_views = (viewset_num_tiles_x * viewset_num_tiles_y);

		unsigned char wafel_buffer[1024];
		for (int i = 0; i < NUM_SAMPLES_PER_WAFEL; i++)
		{
			wafel_position[i] = -(float)NUM_SAMPLES_PER_WAFEL * SAMPLE_PITCH * 0.5f + i*(float)SAMPLE_PITCH;
			wafel_buffer[i] = 0;
		}

		for (unsigned int color_chan = 0; color_chan < 3; color_chan++)
		{
			x = global_x;
			y = global_y;

			const float k = (color_chan == 0 ? K_R : color_chan == 1 ? K_G : K_B);
			const real_t spatial_upconvert_const = (color_chan == 0 ? SPATIAL_UPCONVERT_CONST_R : color_chan == 1 ? SPATIAL_UPCONVERT_CONST_G : SPATIAL_UPCONVERT_CONST_B);

			for (unsigned int vy = 0, idx = 0; vy < viewset_num_tiles_y; vy++)
			{
				for (unsigned int vx = 0; vx < viewset_num_tiles_x; vx++, idx++)
				{
					// Check later
#ifdef CONFIG_USE_DEPTH_TEXTURE
					float4 depth = read_imagef(viewset_depth_in, sampler, (int2)(x,y));
					float d = (depth.x - 0.5f) * Z_SPAN + Z_OFFSET;
#else
					float d = (viewset_depth_in[y * viewset_res_x + x] - 0.5f) * Z_SPAN + Z_OFFSET;
#endif
					float temp_x = d * native_tan(REF_BEAM_ANGLE_RAD * (idx - num_views *0.5f)) + NUM_SAMPLES_PER_WAFEL * SAMPLE_PITCH *0.5f;
					float4 color = read_imagef(viewset_color_in, sampler, (int2)(x,y));
					unsigned char c = (color_chan == 0 ? 255.f*color.x*redGain : color_chan == 1 ? 255.f*color.y*greenGain : 255.f*color.z*blueGain);

					if(c != 0)
						for (int i = 0; i < NUM_SAMPLES_PER_WAFEL; i++)
						{
							real_t mycos = native_cos(k * native_sqrt(pow(wafel_position[i] - temp_x, 2) + pow(d, 2)) - d + (global_x * NUM_SAMPLES_PER_WAFEL * SAMPLE_PITCH + temp_x) * (native_sin(REF_BEAM_ANGLE_RAD) + (float)(2.f * 3.14159265359f / k * spatial_upconvert_const)));
							wafel_buffer[i] += (unsigned char)(c * (mycos + 1.f)*0.5f);
						}

					x += num_wafels_per_scanline;
				}
				x = global_x;
				y += num_scanlines;
			}

		}

		int which_frame_buf = (global_y % NUM_AOM_CHANNELS);
		int which_hololine = global_y / NUM_AOM_CHANNELS;

		unsigned int offset = which_frame_buf / NUM_BUFFERS * framebuffer_res_x * HEAD_RES_Y_SPEC * 4
			+ which_hololine * (((NUM_SAMPLES_PER_WAFEL * num_wafels_per_scanline) / framebuffer_res_x) * framebuffer_res_x * 4)
			+ NUM_SAMPLES_PER_WAFEL * 4 * global_x
			+ which_frame_buf % 3;

		for (int i = 0; i < NUM_SAMPLES_PER_WAFEL; i++)
		{
			const unsigned int idx = offset + 4 * i;
			framebuffer_out[idx] = wafel_buffer[i];

			if (which_frame_buf % 3 == 2)
				framebuffer_out[idx + 1] = 255;
		}
	}


}