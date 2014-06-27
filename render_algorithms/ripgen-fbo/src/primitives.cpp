#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "setupglew.h"
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glut.h>
#include <tiffio.h>

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
// primitives

void primitiveSquare::draw(textureCoordsConf *texCoords)
{
	float verts[12] = {-1.0, 1.0, 0.0,
							1.0, 1.0, 0.0,
							1.0, -1.0, 0.0,
							-1.0, -1.0, 0.0};
							
	float texCoord[8] = {1.0, 1.0,
								0.0, 1.0,
								0.0, 0.0,
								1.0, 0.0};

	glBegin(GL_QUADS);
	glNormal3f(0.0, 0.0, 1.0);
	for(int i = 0; i < 4; i++)
        {
            texCoords->activateVert(i, &(verts[i*3]), &(texCoord[i*2]));
		glVertex3fv(&(verts[i*3]));
	}
	
	glEnd();
}

void primitiveTriangle::draw(textureCoordsConf *texCoords)
{
	float verts[9] = {0.0, 1.0, 0.0,
							1.0, -1.0, 0.0,
							-1.0, -1.0, 0.0};
							
	float texCoord[6] = {0.5, 1.0,
								1.0, 0.0,
								0.0, 0.0};

	glBegin(GL_TRIANGLES);
	glNormal3f(0.0, 0.0, 1.0);
	
	for(int i = 0; i < 3; i++)
	{
		texCoords->activateVert(i, &(verts[i*3]), &(texCoord[i*2]));
		glVertex3fv(&(verts[i*3]));
	}
	
	glEnd();
}

void primitiveCube::draw(textureCoordsConf *texCoords)
{
	float vertsA[12] = {-1.0, 1.0, -1.0,
								1.0, 1.0, -1.0,
								1.0, -1.0, -1.0,
								-1.0, -1.0, -1.0};

	float vertsB[12] = {-1.0, 1.0, 1.0,
								1.0, 1.0, 1.0,
								1.0, -1.0, 1.0,
								-1.0, -1.0, 1.0};

	float vertsC[12] = {1.0, -1.0, 1.0,
								1.0, 1.0, 1.0,
								1.0, 1.0, -1.0,
								1.0, -1.0, -1.0};

	float vertsD[12] = {-1.0, -1.0, 1.0,
								-1.0, 1.0, 1.0,
								-1.0, 1.0, -1.0,
								-1.0, -1.0, -1.0};
																
	float vertsE[12] = {-1.0, 1.0, 1.0,
								1.0, 1.0, 1.0,
								1.0, 1.0, -1.0,
								-1.0, 1.0, -1.0};
								
	float vertsF[12] = {-1.0, -1.0, 1.0,
								1.0, -1.0, 1.0,
								1.0, -1.0, -1.0,
								-1.0, -1.0, -1.0};
								
	float texCoord[8] = {1.0, 1.0,
								0.0, 1.0,
								0.0, 0.0,
								1.0, 0.0};

	int i;
																
	glBegin(GL_QUADS);
	
	glNormal3f(0.0, 0.0, -1.0);
	for(i = 0; i < 4; i++)
	{
		texCoords->activateVert(i, &(vertsA[i*3]), &(texCoord[i*2]));
		glVertex3fv(&(vertsA[i*3]));
	}
		
	glNormal3f(0.0, 0.0, 1.0);
	for(i = 0; i < 4; i++)
	{
		texCoords->activateVert(i, &(vertsB[i*3]), &(texCoord[i*2]));
		glVertex3fv(&(vertsB[i*3]));
	}

	glNormal3f(1.0, 0.0, 0.0);
	for(i = 0; i < 4; i++)
	{
		texCoords->activateVert(i, &(vertsC[i*3]), &(texCoord[i*2]));
		glVertex3fv(&(vertsC[i*3]));
	}

	glNormal3f(-1.0, 0.0, 0.0);
	for(i = 0; i < 4; i++)
	{
		texCoords->activateVert(i, &(vertsD[i*3]), &(texCoord[i*2]));
		glVertex3fv(&(vertsD[i*3]));
	}

	glNormal3f(0.0, 1.0, 0.0);
	for(i = 0; i < 4; i++)
	{
		texCoords->activateVert(i, &(vertsE[i*3]), &(texCoord[i*2]));
		glVertex3fv(&(vertsE[i*3]));
	}

	glNormal3f(0.0, -1.0, 0.0);
	for(i = 0; i < 4; i++)
	{
		texCoords->activateVert(i, &(vertsF[i*3]), &(texCoord[i*2]));
		glVertex3fv(&(vertsF[i*3]));
	}
				
	glEnd();
}

void primitiveSphere::draw(textureCoordsConf *texCoords)
{
	glutSolidSphere(1.0, 16, 16);
}
