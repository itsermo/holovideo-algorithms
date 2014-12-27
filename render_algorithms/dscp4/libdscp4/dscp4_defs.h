#ifndef DSCP4_DEFS_H
#define DSCP4_DEFS_H

extern "C"{

	typedef void* dscp4_context_t;

	typedef enum {
		DSCP4_SIMPLE_OBJECT_TYPE_SPHERE = 0,
		DSCP4_SIMPLE_OBJECT_TYPE_CUBE = 1,
		DSCP4_SIMPLE_OBJECT_TYPE_PYRAMID = 2
	} simple_object_t;

	typedef enum {
		DSCP4_RENDER_MODE_MODEL_VIEWING = 0,
		DSCP4_RENDER_MODE_STEREOGRAM_VIEWING = 1,
		DSCP4_RENDER_MODE_HOLOVIDEO_FRINGE = 2
	} render_mode_t;

	typedef enum {
		DSCP4_SHADER_MODEL_OFF = 0,
		DSCP4_SHADER_MODEL_FLAT = 1,
		DSCP4_SHADER_MODEL_SMOOTH = 2
	} shader_model_t;

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
		const char * shader_filename_prefix;
		render_mode_t render_mode;
		shader_model_t shader_model;
		float light_pos_x, light_pos_y, light_pos_z;
		bool auto_scale_enabled;
	} render_options_t;

	typedef struct
	{
		unsigned int num_views_x, num_views_y, num_wafels_per_scanline, num_scanlines;
	} algorithm_options_t;

	typedef struct
	{
		const char * name;
		unsigned int num_heads, head_res_x, head_res_y;
	} display_options_t;
};

#endif