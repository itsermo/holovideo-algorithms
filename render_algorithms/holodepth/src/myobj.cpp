/*
 *  myobj.c
 *  Dblfrustum
 *
 *  Created by Quinn Smithwick on 9/30/08.
 *  Copyright 2008 __MyCompanyName__. All rights reserved.
 *
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
#include "myobj.h"
/* Texture id for the demo */
//static GLuint texId;

/* Function prototype */
GLuint createDL(void);
  
/* Scalar */
//float tz=0;
float ty=0;
float tx=0;
//GLuint DLid;

/**
 * Free resources allocated for the model.
 */
void
FreeModel (struct obj_model_t *mdl)
{
  int i;

  if (mdl)
    {
      if (mdl->vertices)
	{
	  free (mdl->vertices);
	  mdl->vertices = NULL;
	}

      if (mdl->texCoords)
	{
	  free (mdl->texCoords);
	  mdl->texCoords = NULL;
	}

      if (mdl->normals)
	{
	  free (mdl->normals);
	  mdl->normals = NULL;
	}

      if (mdl->faces)
	{
	  for (i = 0; i < mdl->num_faces; ++i)
	    {
	      if (mdl->faces[i].vert_indices)
		free (mdl->faces[i].vert_indices);

	      if (mdl->faces[i].uvw_indices)
		free (mdl->faces[i].uvw_indices);

	      if (mdl->faces[i].norm_indices)
		free (mdl->faces[i].norm_indices);
	    }

	  free (mdl->faces);
	  mdl->faces = NULL;
	}
    }
}

/**
 * Allocate resources for the model after first pass.
 */
int
MallocModel (struct obj_model_t *mdl)
{
  if (mdl->num_verts)
    {
      mdl->vertices = (struct obj_vertex_t *)
	malloc (sizeof (struct obj_vertex_t) * mdl->num_verts);
      if (!mdl->vertices)
	return 0;
    }

  if (mdl->num_texCoords)
    {
      mdl->texCoords = (struct obj_texCoord_t *)
	malloc (sizeof (struct obj_texCoord_t) * mdl->num_texCoords);
      if (!mdl->texCoords)
	return 0;
    }

  if (mdl->num_normals)
    {
      mdl->normals = (struct obj_normal_t *)
	malloc (sizeof (struct obj_normal_t) * mdl->num_normals);
      if (!mdl->normals)
	return 0;
    }

  if (mdl->num_faces)
    {
      mdl->faces = (struct obj_face_t *)
	calloc (mdl->num_faces, sizeof (struct obj_face_t));
      if (!mdl->faces)
	return 0;
    }

  return 1;
}

/**
 * Load an OBJ model from file -- first pass.
 * Get the number of triangles/vertices/texture coords for
 * allocating buffers.
 */
int
FirstPass (FILE *fp, struct obj_model_t *mdl)
{
  int v, t, n;
  char buf[256];

  while (!feof (fp))
    {
      /* Read whole line */
      fgets (buf, sizeof (buf), fp);

      switch (buf[0])
	{
	case 'v':
	  {
	    if (buf[1] == ' ')
	      {
		/* Vertex */
		mdl->num_verts++;
	      }
	    else if (buf[1] == 't')
	      {
		/* Texture coords. */
		mdl->num_texCoords++;
	      }
	    else if (buf[1] == 'n')
	      {
		/* Normal vector */
		mdl->num_normals++;
	      }
	    else
	      {
		printf ("Warning: unknown token \"%s\"! (ignoring)\n", buf);
	      }

	    break;
	  }

	case 'f':
	  {
	    /* Face */
	    if (sscanf (buf + 2, "%d/%d/%d", &v, &n, &t) == 3)
	      {
		mdl->num_faces++;
		mdl->has_texCoords = 1;
		mdl->has_normals = 1;
	      }
	    else if (sscanf (buf + 2, "%d//%d", &v, &n) == 2)
	      {
		mdl->num_faces++;
		mdl->has_texCoords = 0;
		mdl->has_normals = 1;
	      }
	    else if (sscanf (buf + 2, "%d/%d", &v, &t) == 2)
	      {
		mdl->num_faces++;
		mdl->has_texCoords = 1;
		mdl->has_normals = 0;
	      }
	    else if (sscanf (buf + 2, "%d", &v) == 1)
	      {
		mdl->num_faces++;
		mdl->has_texCoords = 0;
		mdl->has_normals = 0;
	      }
	    else
	      {
		/* Should never be there or the model is very crappy */
		fprintf (stderr, "Error: found face with no vertex!\n");
	      }

	    break;
	  }

	case 'g':
	  {
	    /* Group */
	    /*	fscanf (fp, "%s", buf); */
	    break;
	  }

	default:
	  break;
	}
    }

  /* Check if informations are valid */
  if ((mdl->has_texCoords && !mdl->num_texCoords) ||
      (mdl->has_normals && !mdl->num_normals))
    {
      fprintf (stderr, "error: contradiction between collected info!\n");
      return 0;
    }

  if (!mdl->num_verts)
    {
      fprintf (stderr, "error: no vertex found!\n");
      return 0;
    }

  printf ("first pass results: found\n");
  printf ("   * %i vertices\n", mdl->num_verts);
  printf ("   * %i texture coords.\n", mdl->num_texCoords);
  printf ("   * %i normal vectors\n", mdl->num_normals);
  printf ("   * %i faces\n", mdl->num_faces);
  printf ("   * has texture coords.: %s\n", mdl->has_texCoords ? "yes" : "no");
  printf ("   * has normals: %s\n", mdl->has_normals ? "yes" : "no");

  return 1;
}

/**
 * Load an OBJ model from file -- first pass.
 * This time, read model data and feed buffers.
 */
int
SecondPass (FILE *fp, struct obj_model_t *mdl)
{
  struct obj_vertex_t *pvert = mdl->vertices;
  struct obj_texCoord_t *puvw = mdl->texCoords;
  struct obj_normal_t *pnorm = mdl->normals;
  struct obj_face_t *pface = mdl->faces;
  char buf[128], *pbuf;
  int i;

  while (!feof (fp))
    {
      /* Read whole line */
      fgets (buf, sizeof (buf), fp);

      switch (buf[0])
	{
	case 'v':
	  {
	    if (buf[1] == ' ')
	      {
		/* Vertex */
		if (sscanf (buf + 2, "%f %f %f %f",
			    &pvert->xyzw[0], &pvert->xyzw[1],
			    &pvert->xyzw[2], &pvert->xyzw[3]) != 4)
		  {
		    if (sscanf (buf + 2, "%f %f %f", &pvert->xyzw[0],
				&pvert->xyzw[1], &pvert->xyzw[2] ) != 3)
		      {
			fprintf (stderr, "Error reading vertex data!\n");
			return 0;
		      }
		    else
		      {
			pvert->xyzw[3] = 1.0;
		      }
		  }

		pvert++;
	      }
	    else if (buf[1] == 't')
	      {
		/* Texture coords. */
		if (sscanf (buf + 2, "%f %f %f", &puvw->uvw[0],
			    &puvw->uvw[1], &puvw->uvw[2]) != 3)
		  {
		    if (sscanf (buf + 2, "%f %f", &puvw->uvw[0],
				&puvw->uvw[1]) != 2)
		      {
			if (sscanf (buf + 2, "%f", &puvw->uvw[0]) != 1)
			  {
			    fprintf (stderr, "Error reading texture coordinates!\n");
			    return 0;
			  }
			else
			  {
			    puvw->uvw[1] = 0.0;
			    puvw->uvw[2] = 0.0;
			  }
		      }
		    else
		      {
			puvw->uvw[2] = 0.0;
		      }
		  }

		puvw++;
	      }
	    else if (buf[1] == 'n')
	      {
		//  printf("normals");
		/* Normal vector */
		if (sscanf (buf + 2, "%f %f %f", &pnorm->ijk[0],
			    &pnorm->ijk[1], &pnorm->ijk[2]) != 3)
		  {
		    fprintf (stderr, "Error reading normal vectors!\n");
		    return 0;
		  }

		pnorm++;
	      }

	    break;
	  }

	case 'f':
	  {
	    pbuf = buf;
	    pface->num_elems = 0;

	    /* Count number of vertices for this face */
	    while (*pbuf)
	      {
		if (*pbuf == ' ')
		  pface->num_elems++;

		pbuf++;
	      }

	    /* Select primitive type */
	    if (pface->num_elems < 3)
	      {
		fprintf (stderr, "Error: a face must have at least 3 vertices!\n");
		return 0;
	      }
	    else if (pface->num_elems == 3)
	      {
		pface->type = GL_TRIANGLES;
	      }
	    else if (pface->num_elems == 4)
	      {
		pface->type = GL_QUADS;
	      }
	    else
	      {
		pface->type = GL_POLYGON;
	      }

	    /* Memory allocation for vertices */
	    pface->vert_indices = (int *)malloc (sizeof (int) * pface->num_elems);

	    if (mdl->has_texCoords)
	      pface->uvw_indices = (int *)malloc (sizeof (int) * pface->num_elems);

	    if (mdl->has_normals)
	      pface->norm_indices = (int *)malloc (sizeof (int) * pface->num_elems);

	    /* Read face data */
	    pbuf = buf;
	    i = 0;

	    for (i = 0; i < pface->num_elems; ++i)
	      {
		pbuf = strchr (pbuf, ' ');
		pbuf++; /* Skip space */

		/* Try reading vertices */
		if (sscanf (pbuf, "%d/%d/%d",
			    &pface->vert_indices[i],
			    &pface->uvw_indices[i],
			    &pface->norm_indices[i]) != 3)
		  {
		    if (sscanf (pbuf, "%d//%d", &pface->vert_indices[i],
				&pface->norm_indices[i]) != 2)
		      {
			if (sscanf (pbuf, "%d/%d", &pface->vert_indices[i],
				    &pface->uvw_indices[i]) != 2)
			  {
			    sscanf (pbuf, "%d", &pface->vert_indices[i]);
			  }
		      }
		  }

		/* Indices must start at 0 */
		pface->vert_indices[i]--;

		if (mdl->has_texCoords)
		  pface->uvw_indices[i]--;

		if (mdl->has_normals)
		  pface->norm_indices[i]--;
	      }

	    pface++;
	    break;
	  }
	}
    }

  printf ("second pass results: read\n");
  printf ("   * %i vertices\n", pvert - mdl->vertices);
  printf ("   * %i texture coords.\n", puvw - mdl->texCoords);
  printf ("   * %i normal vectors\n", pnorm - mdl->normals);
  printf ("   * %i faces\n", pface - mdl->faces);

  return 1;
}

/**
 * Load an OBJ model from file, in two passes.
 */
int
ReadOBJModel (const char *filename, struct obj_model_t *mdl)
{
  FILE *fp;

  fp = fopen (filename, "r");
  if (!fp)
    {
      fprintf (stderr, "Error: couldn't open \"%s\"!\n", filename);
      return 0;
    }

  /* reset model data */
  memset (mdl, 0, sizeof (struct obj_model_t));

  /* first pass: read model info */
  if (!FirstPass (fp, mdl))
    {
      fclose (fp);
      return 0;
    }

  rewind (fp);

  /* memory allocation */
  if (!MallocModel (mdl))
    {
      fclose (fp);
      FreeModel (mdl);
      return 0;
    }

  /* second pass: read model data */
  if (!SecondPass (fp, mdl))
    {
      fclose (fp);
      FreeModel (mdl);
      return 0;
    }

  fclose (fp);
  return 1;
}

/**
 * Draw the OBJ model.
 */
void
RenderOBJModel (struct obj_model_t *mdl)
{
  int i, j;
  for (i = 0; i < mdl->num_faces; ++i)
    {
      glBegin (mdl->faces[i].type);  
	for (j = 0; j < mdl->faces[i].num_elems; ++j)
	  {
	    if (mdl->has_texCoords){
	      glTexCoord3fv (mdl->texCoords[mdl->faces[i].uvw_indices[j]].uvw);
		  }

	    if (mdl->has_normals){
	      glNormal3fv (mdl->normals[mdl->faces[i].norm_indices[j]].ijk);
		}
		
		
		glVertex4fv (mdl->vertices [mdl->faces[i].vert_indices[j]].xyzw);
	  }
	glEnd();
		 
    }
	
	
}



