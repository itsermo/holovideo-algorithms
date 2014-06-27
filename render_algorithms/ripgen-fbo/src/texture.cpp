#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "setupglew.h"
#include <GL/gl.h>
#include <GL/glu.h>
#include <tiffio.h>
#include <string.h>

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
// texture

textureConf::textureConf()
{
	fName[0] = '\0';
	texW = texH = 0;
	texData = NULL;
	texName = 0;
	
	mode = TEX_NORMAL;
	for(int i = 0; i < 6; i++)
	{
		cmFName[i][0] = '\0';
		cmTexData[i] = NULL;
	}
}
	
textureConf::~textureConf()
{
	if(texData) _TIFFfree(texData);
	for(int i = 0; i < 6; i++) if(cmTexData[i]) _TIFFfree(cmTexData[i]);
}

void textureConf::init(char *path)
{
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
	glGenTextures(1, &texName);
	
	if(mode == TEX_NORMAL)
	{
		_importTiff(path, fName, texData);

		if(texData)
		{		
			glBindTexture(GL_TEXTURE_2D, texName);
	
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
			glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, texW, texH, 0, GL_RGBA, GL_UNSIGNED_BYTE, texData);
		}
	}
	else //mode == TEX_CUBEMAP
	{
	}
}

void textureConf::activate()
{
	if(mode == TEX_NORMAL)
	{
		glBindTexture(GL_TEXTURE_2D, texName);
	}
	else //mode == TEX_CUBEMAP
	{
	}
}

void textureConf::deactivate()
{
	if(mode == TEX_NORMAL)
	{
	}
	else //mode == TEX_CUBEMAP
	{
	}
}

void textureConf::_importTiff(char *path, char *fname, unsigned int *&texdata)
{
	char completeFName[512];
	TIFF *tif;
		
	strcpy(completeFName, path);
	strcat(completeFName, fname);
	
	printf("loading texture file in tiff format: %s\n", completeFName);
	tif = TIFFOpen(completeFName, "r");
			
	if(!tif)
	{
		printf("cannot find texture data file: %s\n", completeFName);
		texdata = NULL;
		texW = texH = 0;
	}
	else
	{			
		TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &texW);
		TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &texH);
		texdata = (unsigned int *) _TIFFmalloc(texW*texH*sizeof(uint32));
		if(!TIFFReadRGBAImage(tif, texW, texH, (uint32 *) texdata, 0))
		{
			printf("invalid tiff file for texture data: %s", completeFName);
			texW = texH = 0;
			texdata = NULL;
		}
		TIFFClose(tif);	
	}
}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// textures

texturesConf::texturesConf()
{
	nTextures = 0;
	textures = NULL;
}
	
texturesConf::~texturesConf()
{
	if(textures)
	{
		for(int i = 0; i < nTextures; i++) delete textures[i];
		delete [] textures;
	}
}
	
void texturesConf::setNTextures(int n)
{
	if(textures)
	{
		for(int i = 0; i < nTextures; i++) delete textures[i];
		delete [] textures;
	}	
		
	nTextures = n;
	textures = new textureConf*[n];
	for(int i = 0; i < n; i++) textures[i] = new textureConf();
}

void texturesConf::init(char *path)
{
	for(int i = 0; i < nTextures; i++) textures[i]->init(path);
}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// texturesVal

texturesValConf::texturesValConf()
{
	ntextures = 0;
	textures = NULL;
	texturesIndex = NULL;
}

texturesValConf::~texturesValConf()
{
	if(textures) delete [] textures;
	if(texturesIndex) delete [] texturesIndex;
}

void texturesValConf::setNTextures(int n)
{
	if(textures) delete [] textures;
	if(texturesIndex) delete [] texturesIndex;
	
	ntextures = n;
	textures = new textureConf*[n];
	texturesIndex = new int[n];
}

void texturesValConf::init(texturesConf *texs)
{
	for(int i = 0; i < ntextures; i++) if((texturesIndex[i] < texs->nTextures) && (texturesIndex[i] >= 0)) textures[i] = texs->textures[texturesIndex[i]];
}

void texturesValConf::activate(textureCoordsConf *texCoords)
{
	for(int i = 0; i < ntextures; i++)
	{
		if(textures[i]) textures[i]->activate();
		if((i < texCoords->nTextureCoords) && texCoords->texCoords[i]) texCoords->texCoords[i]->activate();
	}
}

void texturesValConf::deactivate(textureCoordsConf *texCoords)
{
	for(int i = ntextures-1; i >= 0; i--)
	{
		if(textures[i]) textures[i]->deactivate();
		if((i < texCoords->nTextureCoords) && texCoords->texCoords[i]) texCoords->texCoords[i]->deactivate();
	}
}

//this has to be in here to compile because i'm lazy
void textureCoordsConf::activateVert(int v, float *vert, float *embCoord)
{
	float *t;
	
	for(int i = 0; i < nTextureCoords; i++)
        {
            if(texCoords[i]->useCoords())
                {
                    t = texCoords[i]->vertCoords(v, vert, embCoord);
                    if(t) glTexCoord2fv(t);
                }
        }
}
