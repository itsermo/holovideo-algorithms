#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "setupglew.h"
#include <GL/gl.h>
#include <GL/glu.h>
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
// material

materialConf::materialConf()
{
	ambient[0] = ambient[1] = ambient[2] = ambient[3] = 0.0;
	diffuse[0] = diffuse[1] = diffuse[2] = diffuse[3] = 0.0;
	specular[0] = specular[1] = specular[2] = specular[3] = 0.0;
	shininess = 0.0;
	emission[0] = emission[1] = emission[2] = 0.0;
	emission[3] = 1.0;
	
	transparency = 0;
	
	textures = new texturesValConf();
}

materialConf::~materialConf()
{
	delete textures;
}

void materialConf::init(texturesConf *texs)
{
	textures->init(texs);
}

void materialConf::activate(int state, textureCoordsConf *texCoords)
{
	glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT, ambient);
	glMaterialfv(GL_FRONT_AND_BACK, GL_DIFFUSE, diffuse);
	glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, specular);
	glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, shininess);
	glMaterialfv(GL_FRONT_AND_BACK, GL_EMISSION, emission);
		
	textures->activate(texCoords);
}

void materialConf::deactivate(textureCoordsConf *texCoords)
{
	textures->deactivate(texCoords);
}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// materials

materialsConf::materialsConf()
{
	nMaterials = 0;
	materials = NULL;
}

materialsConf::~materialsConf()
{
	if(materials)
	{
		for(int i = 0; i < nMaterials; i++) delete materials[i];
		delete [] materials;
	}
}

void materialsConf::setNMaterials(int n)
{
	if(materials)
	{
		for(int i = 0; i < nMaterials; i++) delete materials[i];
		delete [] materials;


	}

	nMaterials = n;
	materials = new materialConf*[n];
	for(int i = 0; i < n; i++) materials[i] = new materialConf();
}

void materialsConf::init(texturesConf *texs)
{
	for(int i = 0; i < nMaterials; i++) materials[i]->init(texs);
}
