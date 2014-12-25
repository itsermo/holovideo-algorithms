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
		DSCP4_SHADE_MODEL_OFF = 0,
		DSCP4_SHADE_MODEL_FLAT = 1,
		DSCP4_SHADE_MODEL_SMOOTH = 2
	} shade_model_t;

	typedef struct
	{
		float x, y, z;
	} point3f_t;

	typedef struct
	{
		float x, y, z;
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
		float center_x, center_y, center_z, sq_radius;
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

};

#endif