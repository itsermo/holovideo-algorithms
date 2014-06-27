/*
 * obj.c -- obj model loader
 * last modification: aug. 14, 2007
 *
 * Copyright (c) 2005-2007 David HENRY
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or
 * sell copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR
 * ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF
 * CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 * WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 * gcc -Wall -ansi -lGL -lGLU -lglut obj.c -o obj
 */

#ifdef __APPLE__
#include <GLUT/glut.h>
#else
#include <GL/glut.h>
#endif
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


/* Function prototype */
GLuint createDL(void);
  


/* Vectors */
typedef float vec3_t[3];
typedef float vec4_t[4];

/* Vertex */
struct obj_vertex_t
{
  vec4_t xyzw;
};

/* Texture coordinates */
struct obj_texCoord_t
{
  vec3_t uvw;
};

/* Normal vector */
struct obj_normal_t
{
  vec3_t ijk;
};

/* Polygon */
struct obj_face_t
{
  GLenum type;        /* primitive type */
  int num_elems;      /* number of vertices */

  int *vert_indices;  /* vertex indices */
  int *uvw_indices;   /* texture coordinate indices */
  int *norm_indices;  /* normal vector indices */
};

/* OBJ model structure */
struct obj_model_t
{
  int num_verts;                     /* number of vertices */
  int num_texCoords;                 /* number of texture coords. */
  int num_normals;                   /* number of normal vectors */
  int num_faces;                     /* number of polygons */

  int has_texCoords;                 /* has texture coordinates? */
  int has_normals;                   /* has normal vectors? */

  struct obj_vertex_t *vertices;     /* vertex list */
  struct obj_texCoord_t *texCoords;  /* tex. coord. list */
  struct obj_normal_t *normals;      /* normal vector list */
  struct obj_face_t *faces;          /* model's polygons */
};

/*** An OBJ model ***/
static struct obj_model_t objfile;

 // Free resources allocated for the model.
void FreeModel (struct obj_model_t *mdl);

 // Allocate resources for the model after first pass.
int MallocModel (struct obj_model_t *mdl);

 // Load an OBJ model from file -- first pass.
 //Get the number of triangles/vertices/texture coords for
 //allocating buffers.
 
int FirstPass (FILE *fp, struct obj_model_t *mdl);

 // Load an OBJ model from file -- first pass.
 // This time, read model data and feed buffers.
int SecondPass (FILE *fp, struct obj_model_t *mdl) ;

 // Load an OBJ model from file, in two passes.
int ReadOBJModel (const char *filename, struct obj_model_t *mdl);

/**
 * Draw the OBJ model.
 */
void
RenderOBJModel (struct obj_model_t *mdl)
;