#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "setupglew.h"
#include <GL/gl.h>
#include <GL/glu.h>
#include <tiffio.h>
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

//endian swap float
float htonf(float a)
{

	int ia =  *((int *) (&a));
	int ta = ntohl(ia);
	float bleh = *((float *) (&ta));
	return bleh;
}

char *get_string(FILE *f)
{
	static char tmp[1024], *ptr;
	ptr = tmp;
	do
		fread(ptr, sizeof(char), 1, f);
	while(*ptr++ != '\0');
	return tmp;
}

void error(char *errmsg, char *opt)
{
	fprintf(stderr, "Error: %s%s\n", errmsg, opt);
	exit(1);
}

void multMatrix(float a[3][3], float r[3])
{
	float b[3];
	
	b[0] = a[0][0]*r[0] + a[0][1]*r[1] + a[0][2]*r[2];
	b[1] = a[1][0]*r[0] + a[1][1]*r[1] + a[1][2]*r[2];
	b[2] = a[2][0]*r[0] + a[2][1]*r[1] + a[2][2]*r[2];
	
	r[0] = b[0];
	r[1] = b[1];
	r[2] = b[2];
}

void transformVect(orientation *o, float n[3], int trans)
{
	float rot[3][3];
	int i, j;
	
	//x rot	
	for(i = 0; i < 3; i++)
		for(j = 0; j < 3; j++) rot[i][j] = 0.0;		

	rot[0][0] = 1.0;
	rot[1][1] = (float) cos(o->rotate[0]);
	rot[1][2] = (float) -sin(o->rotate[0]);
	rot[2][1] = (float) sin(o->rotate[0]);
	rot[2][2] = (float) cos(o->rotate[0]);
	
	multMatrix(rot, n);

	//y rot	


	for(i = 0; i < 3; i++)
		for(j = 0; j < 3; j++) rot[i][j] = 0.0;		

	rot[0][0] = (float) cos(o->rotate[1]);
	rot[0][2] = (float) sin(o->rotate[1]);
	rot[2][0] = (float) -sin(o->rotate[1]);
	rot[2][2] = (float) cos(o->rotate[1]);
	rot[1][1] = 1.0;
	
	multMatrix(rot, n);

	//z rot	
	for(i = 0; i < 3; i++)
		for(j = 0; j < 3; j++) rot[i][j] = 0.0;		

	rot[0][0] = (float) cos(o->rotate[2]);
	rot[0][1] = (float) -sin(o->rotate[2]);
	rot[1][0] = (float) sin(o->rotate[2]);
	rot[1][1] = (float) cos(o->rotate[2]);
	rot[2][2] = 1.0;
	
	multMatrix(rot, n);
	
	if(trans)
	{
		for(i = 0; i < 3; i++) n[i] += o->translate[i];
	}
}

void inverseTransformVect(orientation *o, float n[3], int trans)
{
	float rot[3][3];
	int i, j;
	
	if(trans)
	{
		for(i = 0; i < 3; i++) n[i] -= o->translate[i];
	}
	
	//z rot
	for(i = 0; i < 3; i++)
		for(j = 0; j < 3; j++) rot[i][j] = 0.0;		

	rot[0][0] = (float) cos(-o->rotate[2]);
	rot[0][1] = (float) -sin(-o->rotate[2]);

	rot[1][0] = (float) sin(-o->rotate[2]);
	rot[1][1] = (float) cos(-o->rotate[2]);
	rot[2][2] = 1.0;
	
	multMatrix(rot, n);

	//y rot
	for(i = 0; i < 3; i++)
		for(j = 0; j < 3; j++) rot[i][j] = 0.0;		

	rot[0][0] = (float) cos(-o->rotate[1]);
	rot[0][2] = (float) sin(-o->rotate[1]);
	rot[2][0] = (float) -sin(-o->rotate[1]);
	rot[2][2] = (float) cos(-o->rotate[1]);
	rot[1][1] = 1.0;
	
	multMatrix(rot, n);

	//x rot
	for(i = 0; i < 3; i++)
		for(j = 0; j < 3; j++) rot[i][j] = 0.0;		

	rot[0][0] = 1.0;
	rot[1][1] = (float) cos(-o->rotate[0]);
	rot[1][2] = (float) -sin(-o->rotate[0]);
	rot[2][1] = (float) sin(-o->rotate[0]);
	rot[2][2] = (float) cos(-o->rotate[0]);
	
	multMatrix(rot, n);
}
