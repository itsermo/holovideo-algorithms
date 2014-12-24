#ifndef DSCP4_DEFS_H
#define DSCP4_DEFS_H

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

#endif