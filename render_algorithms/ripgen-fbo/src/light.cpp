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

static GLenum glLightNums[] = { GL_LIGHT0, GL_LIGHT1, GL_LIGHT2, GL_LIGHT3, GL_LIGHT4, GL_LIGHT5, GL_LIGHT6, GL_LIGHT7 };


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// light

lightConf::lightConf()
{
	position[0] = position[1] = position[2] = position[3] = 0.0;
	direction[0] = direction[1] = direction[2] = direction[3] = 0.0;
	ambient[0] = ambient[1] = ambient[2] = ambient[3] = 0.0;
	diffuse[0] = diffuse[1] = diffuse[2] = diffuse[3] = 0.0;
	specular[0] = specular[1] = specular[2] = specular[3] = 0.0;
	spotExponent = 0.0;
	spotCutoff = 180.0;
	constantAttenuation = 1.0;
	linearAttenuation = 0.0;
	quadraticAttenuation = 0.0;
}

lightConf::~lightConf() {}

void lightConf::init(GLenum _lNum)
{
	lNum = _lNum;
	for(int i = 0; i < 4; i++) rDirection[i] = -direction[i];
	
	glLightfv(lNum, GL_AMBIENT, ambient);
	glLightfv(lNum, GL_DIFFUSE, diffuse);
	glLightfv(lNum, GL_SPECULAR, specular);
	glLightfv(lNum, GL_POSITION, position);
	glLightfv(lNum, GL_SPOT_DIRECTION, direction);
	glLightf(lNum, GL_SPOT_EXPONENT, spotExponent);
	glLightf(lNum, GL_SPOT_CUTOFF, spotCutoff);
	glLightf(lNum, GL_CONSTANT_ATTENUATION, constantAttenuation);
	glLightf(lNum, GL_LINEAR_ATTENUATION, linearAttenuation);
	glLightf(lNum, GL_QUADRATIC_ATTENUATION, quadraticAttenuation);
	glEnable(lNum);
}

void lightConf::activate(int state)
{
	glLightfv(lNum, GL_POSITION, position);
	
	if(state == NORMAL) glLightfv(lNum, GL_SPOT_DIRECTION, direction);
	else glLightfv(lNum, GL_SPOT_DIRECTION, rDirection);
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// lighting

lightingConf::lightingConf()
{
	nLights = 0;
	globalAmbient[0] = globalAmbient[1] = globalAmbient[2] = globalAmbient[3] = 0.0;
	lights = NULL;
}

lightingConf::~lightingConf()
{
	if(lights)
	{
		for(int i = 0; i < nLights; i++) delete lights[i];
		delete [] lights;
	}
}

void lightingConf::setNLights(int n)
{
	if(lights)
	{
		for(int i = 0; i < nLights; i++) delete lights[i];
		delete [] lights;
	}


	nLights = n;
	lights = new lightConf*[n];
	for(int i = 0; i < n; i++) lights[i] = new lightConf();
}
	
void lightingConf::init()
{
	glLightModelfv(GL_LIGHT_MODEL_AMBIENT, globalAmbient);
	
	for(int i = 0; i < nLights; i++) lights[i]->init(glLightNums[i]);
}

void lightingConf::activate(int state)
{
	for(int i = 0; i < nLights; i++) lights[i]->activate(state);
}
