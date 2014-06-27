#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "setupglew.h"
#include <GL/gl.h>
#include <GL/glu.h>
#include <tiffio.h>
#include <string.h>
#include <netinet/in.h>

#include "orientation.h"
#include "light.h"
#include "texture.h"
#include "texturecoord.h"
#include "material.h"
#include "drawable.h"
#include "triangles.h"
#include "primitives.h"
#include "object.h"
#include "model.h"
#include "render.h"
#include "holoren.h"
#include "parser.h"
#include "utils.h"


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// triangleList

triangleList::triangleList(int _n)
{
	n = _n;
	verts = new float[3*3*n];
	norms = new float[3*3*n];
	
	useTexCoords = 0;
	tCoords = NULL;
}

triangleList::triangleList(char *fname, int _useTexCoords)
{
	n = 0;
	verts = norms = NULL;
	
	useTexCoords = _useTexCoords;
	tCoords = NULL;
	
	readAliasTrifile(fname);
}

triangleList::triangleList(char *fname, int _useTexCoords, int type)
{
	n = 0;

	verts = norms = NULL;
	
	useTexCoords = _useTexCoords;
	tCoords = NULL;
	if (type == DRAWABLE_RAW) {
		readRawTrifile(fname);
	} else {
		readAliasTrifile(fname);
	}
}

triangleList::~triangleList()
{
	if(verts) delete [] verts;
	if(norms) delete [] norms;
	if(tCoords) delete [] tCoords;
}

void triangleList::setNTriangles(int _n, int copy)
{
	float *v = new float[3*3*_n];
	float *r = new float[3*3*_n];
	float *t = NULL;
	
	if(useTexCoords) t = new float[3*2*_n];
	
	if(copy)
	{
		memcpy(v, verts, sizeof(float)*3*3*n);
		memcpy(r, norms, sizeof(float)*3*3*n);
		if(useTexCoords) memcpy(t, tCoords, sizeof(float)*2*3*n);
	}
	
	if(verts) delete [] verts;
	if(norms) delete [] norms;
	if(tCoords) delete [] tCoords;
	
	verts = v;
	norms = r;
	tCoords = t;
	
	n = _n;
}

void triangleList::draw(textureCoordsConf *texCoords)
{
	int i;

	if(useTexCoords)
	{	
		glBegin(GL_TRIANGLES);
		for(i = 0; i < n; i++)
		{
			texCoords->activateVert(i*3*3+0, &verts[i*3*3+0], &tCoords[i*3*2+0]);
			glNormal3fv(&norms[i*3*3+0]);
			glVertex3fv(&verts[i*3*3+0]);

			texCoords->activateVert(i*3*3+1, &verts[i*3*3+3], &tCoords[i*3*2+2]);

			glNormal3fv(&norms[i*3*3+3]);
			glVertex3fv(&verts[i*3*3+3]);
	
			texCoords->activateVert(i*3*3+2, &verts[i*3*3+6], &tCoords[i*3*2+4]);

			glNormal3fv(&norms[i*3*3+6]);
			glVertex3fv(&verts[i*3*3+6]);
		}
		glEnd();
	}
	else
	{	
		glBegin(GL_TRIANGLES);
		for(i = 0; i < n; i++)
		{
			texCoords->activateVert(i*3*3+0, &verts[i*3*3+0], NULL);

			glNormal3fv(&norms[i*3*3+0]);
			glVertex3fv(&verts[i*3*3+0]);

			
			//printf("(%f, %f, %f)\n",verts[i*3*3+0],verts[i*3*3+1],verts[i*3*3+2]);

			texCoords->activateVert(i*3*3+1, &verts[i*3*3+3], NULL);

			glNormal3fv(&norms[i*3*3+3]);
			glVertex3fv(&verts[i*3*3+3]);
	
			texCoords->activateVert(i*3*3+2, &verts[i*3*3+6], NULL);


			glNormal3fv(&norms[i*3*3+6]);
			glVertex3fv(&verts[i*3*3+6]);
		}
		glEnd();
	}
}

void triangleList::setTriangle(int tri, float *v1, float *v2, float *v3, float *n1, float *n2, float *n3, float *t1, float *t2, float *t3)
{
	memcpy((void *) &verts[tri*3*3 + 0], (void *) v1, sizeof(float)*3);
	memcpy((void *) &verts[tri*3*3 + 3], (void *) v2, sizeof(float)*3);
	memcpy((void *) &verts[tri*3*3 + 6], (void *) v3, sizeof(float)*3);
	
	memcpy((void *) &norms[tri*3*3 + 0], (void *) n1, sizeof(float)*3);
	memcpy((void *) &norms[tri*3*3 + 3], (void *) n2, sizeof(float)*3);
	memcpy((void *) &norms[tri*3*3 + 6], (void *) n3, sizeof(float)*3);

	
	if(useTexCoords)
	{
		memcpy((void *) &tCoords[tri*2*3 + 0], (void *) t1, sizeof(float)*2);
		memcpy((void *) &tCoords[tri*2*3 + 2], (void *) t2, sizeof(float)*2);
		memcpy((void *) &tCoords[tri*2*3 + 4], (void *) t3, sizeof(float)*2);
	}
}
void triangleList::readRawTrifile(char *fname)
{
	FILE *f;
	int i, ntriangles;
	float v0[3];
	float v1[3];
	float v2[3];
	float n[3];
	float t[2];
	float sc;
	t[0] = t[1] =0;
	//Find # of triangles
	f = fopen(fname, "r");
	if(!f)
	{
		printf("object file not found: %s", fname);
		setNTriangles(0);
		return;
	}
	i=0;
	int r;
	printf("prescanning %s \n", fname);
	while(2 < fscanf(f, " %f %f %f %f %f %f %f %f %f ", v0, v0, v0, v0, v0, v0, v0, v0, v0)) {
		i++;
	}
	
	setNTriangles(i, 0);

	fclose(f);
	
	printf("done prescanning %s. Found %d faces\n", fname, i);

	f = fopen(fname, "r");
	if(!f)
	{
		printf("object file not found: %s", fname);
		setNTriangles(0);
		return;
	}
	ntriangles = i;
	for(i=0;i<ntriangles;i++) {
		fscanf(f, " %f %f %f ", v0, v0+1, v0+2);
		fscanf(f, " %f %f %f ", v1, v1+1, v1+2);
		fscanf(f, " %f %f %f ", v2, v2+1, v2+2);
		#define X 0
		#define Y 1
		#define Z 2
		
		//compute face normal
		n[X] = (v1[Y]-v0[Y])*(v2[Z]-v0[Z]) - (v1[Z]-v0[Z])*(v2[Y]-v0[Y]);
		n[Y] = (v1[Z]-v0[Z])*(v2[X]-v0[X]) - (v1[X]-v0[X])*(v2[Z]-v0[Z]);
		n[Z] = (v1[X]-v0[X])*(v2[Y]-v0[Y]) - (v1[Y]-v0[Y])*(v2[X]-v0[X]);
		
		sc = 1.0/sqrt(n[X]*n[X]+n[Y]*n[Y]+n[Z]*n[Z]);
		
		n[X] *= sc;
		n[Y] *= sc;
		n[Z] *= sc;
		
		setTriangle(i, v0, v1, v2, n, n, n, t, t, t);
		//printf("recorded triangle with (%f, %f, %f)\n", v0[0], v0[1], v0[2]);
	}
	fclose(f);
	printf("Done parsing %s\n", fname);

}
void triangleList::readAliasTrifile(char *fname)
{
	int i, j, k, count, totalcount = 0;
	FILE *f;
	float start;
	float data[33];
	float vert[3][3], norm[3][3], tex[3][2];

	/* scan through the file once to get the total number of triangles */
	f = fopen(fname, "rb");
	if(!f)
	{
		printf("object file not found: %s", fname);
		setNTriangles(0);
		return;
	}
	
	fread(&i, sizeof(int), 1, f); /* alias magic key = 12332 */
	if(ntohl(i) != 123322)
	{
		fclose(f);
		printf("not a valid Alias triangle file: %s", fname);
		setNTriangles(0);
		return;
	}

	FILE *t = fopen("debug2.dat", "a");
		
	while(!feof(f))
	{
		
		fread(&start, sizeof(float), 1, f);
		fprintf(t, "%d start: %f ", ftell(f), start);
		start = htonf(start);
		fprintf(t, "%f\n", start);
//		if(_isnan(start)) continue;

		if(start == -99999.0)
		{
			fprintf(t, "%s\n", get_string(f));
			continue;
		}
		
		if(start == 99999.0)
		{
			fprintf(t, "%s\n", get_string(f));
			fread(&count, sizeof(int), 1, f);
			fprintf(t, "--COUNT: %d ", count);
			count = ntohl(count);
			fprintf(t, "%d\n", count);
			if(count != 0) fprintf(t, "%s\n", get_string(f));
		}

		for(i = 0; i < count; i++)
		{
			fread(data, sizeof(float)*33, 1, f);
		}		
		totalcount += count;
	}
	fclose(f);	
		
	fprintf(t, "***TOTALCOUNT: %d\n", totalcount);
	fclose(t);
	setNTriangles(totalcount, 0);
	totalcount = 0;
		
	/* scan through and get the triangle data */
	f = fopen(fname, "rb");
	fread(&i, sizeof(int), 1, f); /* grab the magic key again */
	
	printf("loading alias triangle file: %s\n", fname);
		
	while(!feof(f))
	{
		fread(&start, sizeof(float), 1, f);
		start = htonf(start);
//		if(_isnan(start)) continue;

		if(start == -99999.0)
		{
			get_string(f);
			continue;
		}
		
		if(start == 99999.0)
		{
			get_string(f);
			fread(&count, sizeof(int), 1, f);
			count = htonl(count);

			if(count != 0) get_string(f);
		}

		for(i = 0; i < count; i++)
		{
			fread(data, sizeof(float)*33, 1, f);
					
			for(j = 0; j < 3; j++)
			{
				for(k = 0; k < 3; k++) norm[j][k] = htonf(data[11*j + k]);
				for(k = 0; k < 3; k++) vert[j][k] = htonf(data[11*j + 3 + k]);
				for(k = 0; k < 2; k++) tex[j][k] = htonf(data[11*j + 9 + k]);
			}					
						
			setTriangle(totalcount + i, vert[0], vert[1], vert[2], norm[0], norm[1], norm[2], tex[0], tex[1], tex[2]);
		}
		
		totalcount += count;
	}

	fclose(f);
}
