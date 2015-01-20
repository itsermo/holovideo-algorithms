#ifndef DSCP4_DEFS_H
#define DSCP4_DEFS_H

// These are some default values you can reference, which are
// only called by the default constructor of C++ DSCP4Render class
//
// Changing these will only affect the default constructor,
// DSCP4 program will load and overwrite these values from dscp4.conf file
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
#define DSCP4_DEFAULT_RENDER_RENDER_MODE 			DSCP4_RENDER_MODE_HOLOVIDEO_FRINGE
#define DSCP4_DEFAULT_RENDER_LIGHT_POS_X 			-0.7f
#define DSCP4_DEFAULT_RENDER_LIGHT_POS_Y 			0.7f
#define DSCP4_DEFAULT_RENDER_LIGHT_POS_Z 			0.5f
#define DSCP4_DEFAULT_RENDER_AUTOSCALE_ENABLED 		true
#define DSCP4_DEFAULT_ALGORITHM_NUM_VIEWS_X 		16
#define DSCP4_DEFAULT_ALGORITHM_NUM_VIEWS_Y 		1
#define DSCP4_DEFAULT_ALGORITHM_NUM_WAFELS 			693
#define DSCP4_DEFAULT_ALGORITHM_NUM_SCANLINES 		460
#define DSCP4_DEFAULT_ALGORITHM_FOV_X 				30.f
#define DSCP4_DEFAULT_ALGORITHM_FOV_Y 				30.f
#define DSCP4_DEFAULT_DISPLAY_NAME					"MIT Mark IV"
#define DSCP4_DEFAULT_DISPLAY_NUM_HEADS				6
#define DSCP4_DEFAULT_DISPLAY_NUM_HEADS_PER_GPU		2
#define DSCP4_DEFAULT_DISPLAY_HEAD_RES_X			3552
#define DSCP4_DEFAULT_DISPLAY_HEAD_RES_Y			2476
#define DSCP4_DEFAULT_LOG_VERBOSITY					3
#define DSCP4_DEFAULT_COMPUTE_METHOD				DSCP4_COMPUTE_METHOD_CUDA
#define DSCP4_DEFAULT_ALGORITHM_OPENCL_KERNEL_FILENAME	"dscp4-fringe.cl"
#define DSCP4_DEFAULT_ALGORITHM_OPENCL_WORKSIZE_X	64
#define DSCP4_DEFAULT_ALGORITHM_OPENCL_WORKSIZE_Y	64
#define DSCP4_DEFAULT_ALGORITHM_CUDA_BLOCK_DIM_X	8
#define DSCP4_DEFAULT_ALGORITHM_CUDA_BLOCK_DIM_Y	4

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
		unsigned int gl_vertex_buf_id, gl_normal_buf_id, gl_color_buf_id;
		mesh_transform_t transform;
		//w is radius squared, xyz is center
		point4f_t bounding_sphere;
		unsigned int num_vertices, num_points_per_vertex, num_color_channels;
		unsigned int vertex_stride, color_stride;
		bool is_point_cloud;
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
	} algorithm_cache_t;

	typedef struct
	{
		unsigned int num_views_x, num_views_y, num_wafels_per_scanline, num_scanlines;
		float fov_x, fov_y;
		compute_method_t compute_method;
		const char * opencl_kernel_filename;
		size_t opencl_local_workgroup_size[2];
		unsigned int cuda_block_dimensions[2];
		algorithm_cache_t cache;
	} algorithm_options_t;

	typedef struct
	{
		const char * name;
		unsigned int num_heads, num_heads_per_gpu, head_res_x, head_res_y;
	} display_options_t;

	typedef struct
	{
		algorithm_options_t algorithm_options;
		display_options_t display_options;
		const char * kernel_file_path;
		unsigned int view_gl_fbo;
		unsigned int view_gl_fbo_color;
		unsigned int view_gl_fbo_depth;
		unsigned int stereogram_gl_fbo;
		unsigned int stereogram_gl_fbo_color;
		unsigned int stereogram_gl_fbo_depth;
		unsigned int stereogram_gl_rgba_buf_in;
		unsigned int stereogram_gl_depth_buf_in;
		unsigned int * fringe_gl_buf_out;
		unsigned int * fringe_gl_tex_out;
	} dscp4_fringe_context_t;

#ifdef __cplusplus
};
#endif

#endif
