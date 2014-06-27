#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "setupglew.h"
#include <GL/gl.h>
#include <GL/glu.h>
#include <tiffio.h>
#include <string.h>

#include <math.h>

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

extern GLenum *glTextureNums;


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// textureCoordConf

textureCoordConf::textureCoordConf()
{
	xTile = GL_REPEAT;
	yTile = GL_REPEAT;
	space = GL_OBJECT_PLANE;
	combination = GL_DECAL;
	
	fName[0] = '\0';
	
	format = TEX_COORD_TCD;
	mode = TEX_COORD_GENERATE;
	
	verts = NULL;
	nVerts = 0;
	
	texS[0] = 1.0;
	texS[1] = texS[2] = texS[3] = 0.0;
	
	texT[0] = texT[1] = texT[3] = 0.0;
	texT[2] = 1.0;
	
	coordOrient = new orientation();
	ang0 = 0.0;
	nWraps = 1.0;
	yBottom = 0.0;
	yTop = 1.0;
}

textureCoordConf::~textureCoordConf()
{
	if(verts) delete [] verts;
	if(coordOrient) delete coordOrient;
}

void textureCoordConf::init(char *path)
{
	if(mode == TEX_COORD_FILE)
	{	
		switch(format)
		{
			case TEX_COORD_TCD:
				_readTCDFile(path);
				break;
		}
	}
	else //mode == TEX_COORD_GEN
	{
	}
}

int textureCoordConf::useCoords()
{
	return (mode == TEX_COORD_FILE) || (mode == TEX_COORD_CYLIN) || (mode = TEX_COORD_EMBED);
}

float *textureCoordConf::vertCoords(int v, float *vert, float *embCoord)
{
	if(mode == TEX_COORD_FILE)
	{
		return &(verts[v*2]);
	}
	else if(mode == TEX_COORD_CYLIN)
	{
		return _cylindricalTextureCoords(vert);
	}
	else if(mode == TEX_COORD_EMBED)
	{
		return embCoord;
	}
	
	return NULL;
}

float *textureCoordConf::_cylindricalTextureCoords(float *vert)
{
	float norm[3];
	float angle;
	
	for(int i = 0; i < 3; i++) norm[i] = vert[i];
	
	inverseTransformVect(coordOrient, norm, 1);
	
	angle = (float) (atan2(norm[2], norm[0]) + ang0);
	
	tVert[0] = angle/((float) 2.0*(float) M_PI*nWraps);
	tVert[1] = (norm[1] - yBottom) / (yTop - yBottom);

	return tVert;
}

void textureCoordConf::_readTCDFile(char *path)
{
    FILE *f;
	char completeFName[512];
	strcpy(completeFName, path);
	strcat(completeFName, fName);

	if ( (f = fopen(completeFName, "r")) == NULL) {
        printf("couldn't open %s: exiting.\n", completeFName);
        exit (0);
    }
	
	fscanf(f, "%d\n", &nVerts);
	verts = new float[nVerts*2];
	
	for(int i = 0; i < nVerts; i++)
	{
		fscanf(f, "%f %f\n", &(verts[i*2+0]), &(verts[i*2+1]));
	}

	fclose(f);
}

void textureCoordConf::activate()
{
	glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, combination);
	
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, xTile);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, yTile);

	if(mode != TEX_COORD_CUBEMAP_GENERATE)	glEnable(GL_TEXTURE_2D);
	
	if(mode == TEX_COORD_GENERATE)
	{	
		glEnable(GL_TEXTURE_GEN_S);
		glEnable(GL_TEXTURE_GEN_T);
		
		glTexGeni(GL_S, GL_TEXTURE_GEN_MODE, space);
		glTexGeni(GL_T, GL_TEXTURE_GEN_MODE, space);
		glTexGenfv(GL_S, (space==GL_OBJECT_LINEAR)?GL_OBJECT_PLANE:GL_EYE_PLANE, texS);
		glTexGenfv(GL_T, (space==GL_OBJECT_LINEAR)?GL_OBJECT_PLANE:GL_EYE_PLANE, texT);
	}
	else if(mode == TEX_COORD_CUBEMAP_GENERATE)
	{
	}
}

void textureCoordConf::deactivate()
{
	if(mode != TEX_COORD_CUBEMAP_GENERATE) glDisable(GL_TEXTURE_2D);
	
	if(mode == TEX_COORD_GENERATE)
	{
		glDisable(GL_TEXTURE_GEN_S);
		glDisable(GL_TEXTURE_GEN_T);
	}
	else if(mode == TEX_COORD_CUBEMAP_GENERATE)
	{
	}
}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// textureCoordsConf

textureCoordsConf::textureCoordsConf()
{
	nTextureCoords = 0;
	texCoords = NULL;
}

textureCoordsConf::~textureCoordsConf()
{
	if(texCoords)
	{
		for(int i = 0; i < nTextureCoords; i++) delete texCoords[i];
		delete [] texCoords;
	}
}

void textureCoordsConf::setNTextureCoords(int n)
{
	if(texCoords)
	{
		for(int i = 0; i < nTextureCoords; i++) delete texCoords[i];
		delete [] texCoords;
	}
	
	nTextureCoords = n;
	texCoords = new textureCoordConf*[n];
	for(int i = 0; i < n; i++) texCoords[i] = new textureCoordConf();
}

void textureCoordsConf::init(char *path)
{
	for(int i = 0; i < nTextureCoords; i++) texCoords[i]->init(path);
}
