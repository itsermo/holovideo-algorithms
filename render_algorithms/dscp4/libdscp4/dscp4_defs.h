#ifndef DSCP4_DEFS_H
#define DSCP4_DEFS_H

// These are some default values you can reference, which are
// only called by the default constructor of C++ DSCP4Render class
//
// Changing these will only affect the default DSCP4Render class constructor,
// DSCP4 program will load and overwrite these values from dscp4.conf file
//
// These will ONLY BE USED in the case that there is NO dscp4.conf file present,
// and/or no values passed to the command line
//
// In other words, if you are using the DSCP4 program, changing these will not do anything
#ifdef WIN32
#define DSCP4_DEFAULT_RENDER_SHADERS_PATH 			"C:\\Program Files\\dscp4\\share\\dscp4\\shaders"
#define DSCP4_DEFAULT_RENDER_KERNELS_PATH			"C:\\Program Files\\dscp4\\share\\dscp4\\kernels"
#define DSCP4_DEFAULT_RENDER_MODELS_PATH			"C:\\Program Files\\dscp4\\share\\dscp4\\models"
#else
#define DSCP4_DEFAULT_RENDER_SHADERS_PATH 			"/usr/local/dscp4/shaders"
#define DSCP4_DEFAULT_RENDER_KERNELS_PATH			"/usr/local/dscp4/kernels"
#define DSCP4_DEFAULT_RENDER_MODELS_PATH			"/usr/local/dscp4/models"
#endif
#define DSCP4_DEFAULT_RENDER_SHADER_FILENAME_PREFIX	"pointlight"
#define DSCP4_DEFAULT_RENDER_SHADER_MODEL			DSCP4_SHADER_MODEL_FLAT
#define DSCP4_DEFAULT_RENDER_RENDER_MODE 			DSCP4_RENDER_MODE_STEREOGRAM_VIEWING
#define DSCP4_DEFAULT_RENDER_LIGHT_POS_X 			-4.0f
#define DSCP4_DEFAULT_RENDER_LIGHT_POS_Y 			4.0f
#define DSCP4_DEFAULT_RENDER_LIGHT_POS_Z 			2.0f
#define DSCP4_DEFAULT_RENDER_AUTOSCALE_ENABLED 		false
#define DSCP4_DEFAULT_ALGORITHM_OPENCL_KERNEL_FILENAME	"dscp4-fringe.cl"
#define DSCP4_DEFAULT_ALGORITHM_COMPUTE_METHOD		DSCP4_COMPUTE_METHOD_CUDA
#define DSCP4_DEFAULT_ALGORITHM_NUM_VIEWS_X 		16
#define DSCP4_DEFAULT_ALGORITHM_NUM_VIEWS_Y 		1
#define DSCP4_DEFAULT_ALGORITHM_NUM_WAFELS 			693
#define DSCP4_DEFAULT_ALGORITHM_NUM_SCANLINES 		460
#define DSCP4_DEFAULT_ALGORITHM_FOV_X 				30.f
#define DSCP4_DEFAULT_ALGORITHM_FOV_Y 				30.f
#define DSCP4_DEFAULT_ALGORITHM_Z_NEAR 				0.00001f
#define DSCP4_DEFAULT_ALGORITHM_Z_FAR				2.25f
#define DSCP4_DEFAULT_ALGORITHM_OPENCL_WORKSIZE_X	32
#define DSCP4_DEFAULT_ALGORITHM_OPENCL_WORKSIZE_Y	4
#define DSCP4_DEFAULT_ALGORITHM_CUDA_BLOCK_DIM_X	8
#define DSCP4_DEFAULT_ALGORITHM_CUDA_BLOCK_DIM_Y	8
#define DSCP4_DEFAULT_ALGORITHM_REF_BEAM_ANGLE		30.f
#define DSCP4_DEFAULT_ALGORITHM_TEMP_UPCONVERT_R	225000000
#define DSCP4_DEFAULT_ALGORITHM_TEMP_UPCONVERT_G	290000000
#define DSCP4_DEFAULT_ALGORITHM_TEMP_UPCONVERT_B	350000000
#define DSCP4_DEFAULT_ALGORITHM_WAVELENGTH_R		0.0000000633
#define DSCP4_DEFAULT_ALGORITHM_WAVELENGTH_G		0.0000000532
#define DSCP4_DEFAULT_ALGORITHM_WAVELENGTH_B		0.0000000445
#define DSCP4_DEFAULT_ALGORITHM_GAIN_R				1.f
#define DSCP4_DEFAULT_ALGORITHM_GAIN_G				1.f
#define DSCP4_DEFAULT_ALGORITHM_GAIN_B				1.f
#define DSCP4_DEFAULT_DISPLAY_NAME					"MIT Mark IV"
#define DSCP4_DEFAULT_DISPLAY_X11_ENV_VAR			":0"
#define DSCP4_DEFAULT_DISPLAY_NUM_HEADS				6
#define DSCP4_DEFAULT_DISPLAY_NUM_HEADS_PER_GPU		2
#define DSCP4_DEFAULT_DISPLAY_HEAD_RES_X			3552
#define DSCP4_DEFAULT_DISPLAY_HEAD_RES_Y			2476
#define DSCP4_DEFAULT_DISPLAY_HEAD_RES_X_SPEC		3552
#define DSCP4_DEFAULT_DISPLAY_HEAD_RES_Y_SPEC		2476
#define DSCP4_DEFAULT_LOG_VERBOSITY					3
#define DSCP4_DEFAULT_DISPLAY_HOLOGRAM_PLANE_WIDTH	0.15f
#define DSCP4_DEFAULT_DISPLAY_NUM_SAMPLES_PER_HOLOLINE 355200
#define DSCP4_DEFAULT_DISPLAY_NUM_AOM_CHANNELS		18
#define DSCP4_DEFAULT_DISPLAY_PIXEL_CLOCK_RATE		400000000



#ifndef __cplusplus
#include <stdbool.h>
#endif

#include <stdlib.h>

#ifdef __cplusplus
extern "C"{
#endif

	typedef void* dscp4_context_t;

	typedef enum {
		DSCP4_SIMPLE_OBJECT_TYPE_SPHERE = 0,
		DSCP4_SIMPLE_OBJECT_TYPE_CUBE = 1,
		DSCP4_SIMPLE_OBJECT_TYPE_PYRAMID = 2
	} simple_object_t;

	typedef enum {
		DSCP4_RENDER_MODE_MODEL_VIEWING = 0,
		DSCP4_RENDER_MODE_STEREOGRAM_VIEWING = 1,
		DSCP4_RENDER_MODE_AERIAL_DISPLAY = 2,
		DSCP4_RENDER_MODE_HOLOVIDEO_FRINGE = 3
	} render_mode_t;

	typedef enum {
		DSCP4_SHADER_MODEL_OFF = 0,
		DSCP4_SHADER_MODEL_FLAT = 1,
		DSCP4_SHADER_MODEL_SMOOTH = 2
	} shader_model_t;

	typedef enum {
		DSCP4_COMPUTE_METHOD_NONE = -1,
		DSCP4_COMPUTE_METHOD_CUDA = 0,
		DSCP4_COMPUTE_METHOD_OPENCL = 1
	} compute_method_t;

	typedef struct
	{
		float x, y, z;
	} point3f_t;

	typedef struct
	{
		float x, y, z, w;
	} point4f_t;

	typedef struct
	{
		float r, g, b;
	} color3f_t;

	typedef struct
	{
		float r, g, b, a;
	} color4f_t;

	typedef struct
	{
		unsigned char r, g, b;
	} color3b_t;

	typedef struct
	{
		unsigned char r, g, b, a;
	} color4b_t;

	typedef struct
	{
		//axis/angle, where w is angle
		point4f_t rotate;
		point3f_t scale, translate;
	} mesh_transform_t;

	typedef struct
	{
		point3f_t eye, center, up;
	} camera_t;

	typedef struct
	{
		unsigned int gl_vertex_buf_id, gl_normal_buf_id, gl_color_buf_id;
		mesh_transform_t transform;
		//w is radius squared, xyz is center
		point4f_t bounding_sphere;
		unsigned int num_vertices, num_points_per_vertex, num_color_channels;
		unsigned int vertex_stride, color_stride;

		// determines whether we are dealing with points, lines, triangles, or quads
		unsigned int num_indecies;

		bool is_pcl_cloud;
	} mesh_header_t;

	typedef struct
	{
		mesh_header_t info;
		void *vertices;
		void *colors;
		void *normals;
	} mesh_t;

	typedef struct
	{
		const char * shaders_path;
		const char * kernels_path;
		const char * shader_filename_prefix;
		render_mode_t render_mode;
		shader_model_t shader_model;
		float light_pos_x, light_pos_y, light_pos_z;
		bool auto_scale_enabled;
	} render_options_t;

	// values that are computed once, based on algorithm
	// parameters and display options, to set up the
	// stereogram grid and also compute fringe pattern
	typedef struct
	{
		// the number of output buffers,
		// this is normally number of heads / number of heads per gpu
		unsigned int num_fringe_buffers;

		// the output buffer resolution in the X dimension (in pixels)
		unsigned int fringe_buffer_res_x;

		// the output buffer resolution in the Y dimension (in pixels)
		unsigned int fringe_buffer_res_y;

		// the width of the entire stereogram, in pixels
		unsigned int stereogram_res_x;

		// the height of the entire stereogram, in pixels
		unsigned int stereogram_res_y;

		// the number of views in the X dimension, in the stereogram grid
		unsigned int stereogram_num_tiles_x;

		// the number of views in the Y dimension, in the stereogram grid
		unsigned int stereogram_num_tiles_y;

		// the kernel is run on an entire hologram frame, that means
		// global size needs to be { num_wafels_per_scanline, num_scanlines},
		// but the size must be a multiple of local workgroup size
		size_t opencl_global_workgroup_size[2];

		// number of blocks is a function of the hologram frame dimensions
		// and the block dimensions
		unsigned int cuda_number_of_blocks[2];

		// number of bytes in a hololine / num_wafels_per_scanline
		unsigned int num_samples_per_wafel;

		// reference beam, in radians
		float reference_beam_angle_rad;

		// the pitch of one byte of a wafel (hologram plane width/bytes_per_hololine)
		float sample_pitch;

		// the k constant for each color
		float k_r;
		float k_g;
		float k_b;

		// the upconvert constant values for r,g,b SSB
		double upconvert_const_r;
		double upconvert_const_g;
		double upconvert_const_b;

		// the model Z shift
		float z_span;
		float z_offset;

	} algorithm_cache_t;

	typedef struct
	{
		unsigned int num_views_x, num_views_y, num_wafels_per_scanline, num_scanlines;
		float fov_x, fov_y;
		float reference_beam_angle;

		unsigned int temporal_upconvert_red;
		unsigned int temporal_upconvert_green;
		unsigned int temporal_upconvert_blue;

		float wavelength_red;
		float wavelength_green;
		float wavelength_blue;

		float red_gain;
		float green_gain;
		float blue_gain;

		float z_near;
		float z_far;

		compute_method_t compute_method;
		const char * opencl_kernel_filename;
		size_t opencl_local_workgroup_size[2];
		unsigned int cuda_block_dimensions[2];
		algorithm_cache_t cache;
	} algorithm_options_t;

	typedef struct
	{
		// friendly name of the display
		const char * name;

		// the environment variable for X11 display (usually :0)
		const char* x11_env_var;

		// the number of display ports (DVI/VGA/DP/etc.)
		unsigned int num_heads;

		// the number of heads per gpu
		unsigned int num_heads_per_gpu;
		
		// the horizontal modeline resolution of a single head, as determined by OS
		unsigned int head_res_x;
			
		// the vertical modeline resolution of a single head, as determined by OS
		unsigned int head_res_y;
		
		// the horizontal resolution as determined by the display spec
		unsigned int head_res_x_spec;
		
		// the vertical resolution as determined by the display spec
		unsigned int head_res_y_spec; 
		
		// the number of channels on the AOM device
		unsigned int num_aom_channels;

		// the number of samples in one hololine (formerly called "pixels per hololine")		
		unsigned int num_samples_per_hololine;
		
		// the hologram plane width in meters
		float hologram_plane_width;

		// the display pixel clock rate in Hz
		unsigned int pixel_clock_rate;
	} display_options_t;

	typedef struct
	{
		algorithm_options_t *algorithm_options;
		display_options_t display_options;
		const char * kernel_file_path;
		unsigned int view_gl_fbo;
		unsigned int view_gl_fbo_color;
		unsigned int view_gl_fbo_depth;
		unsigned int stereogram_gl_fbo;
		unsigned int stereogram_gl_fbo_color;
		unsigned int stereogram_gl_fbo_depth;
		unsigned int stereogram_gl_depth_pbo_in;
		unsigned int * fringe_gl_tex_out;
	} dscp4_fringe_context_t;

	typedef enum
	{
		DSCP4_CALLBACK_TYPE_STOPPED,
		DSCP4_CALLBACK_TYPE_NEW_FRAME
	} callback_type_t;

	typedef struct
	{
		double compute_fps;
		double render_fps;
		int x_res;
		int y_res;
		void * buffer;
	} frame_data_t;

	typedef void(*dscp4_event_cb_t)(const callback_type_t evt, void * parent, void *data);


#ifdef __cplusplus
};
#endif

#endif
