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
#include "RIP.h"

#ifdef M_TWO_PI
#undef M_TWO_PI
#endif
#define M_TWO_PI (2.0*M_PI)


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// model

modelConf::modelConf()
{
	objects = new objectsConf;
	orient = new orientation;
	path[0] = '\0';
}

modelConf::~modelConf()
{
	if(objects) delete objects;
	if(orient) delete orient;
}

void modelConf::init(materialsConf *mats)
{
	objects->init(mats, path);
}

void modelConf::activate(int state)
{					
	glPushMatrix();
	orient->activate();

    //------ added by WJP to rock the model
    //------ back and forth and slowly roll it around x.
    //---
#ifdef MODEL_ANIMATE
    static double count = 0;
    static double xspin = 0.0;
    float rocklength = 2000.0;
    float rockamount;
    double t;
    
    t = (double)count/rocklength;
    rockamount = 10.0 * cos (M_TWO_PI * (t-floorf(t)));    
	glRotatef(rockamount, 0.0, 1.0, 0.0);
	glRotatef(xspin, 0.0, 0.0, 1.0);
    count++;
    xspin += 0.010;
#endif    

	objects->activate(state);	
    // glRotatef (40.0, 0.0, 0.0, 1.0);
    //glRotatef (40.0, 1.0, 0.0, 0.0);
	glPopMatrix();
}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// models

modelsConf::modelsConf()
{
	nModels = 0;
	models = NULL;
	orient = new orientation;
}

modelsConf::~modelsConf()
{
	if(models)
	{
		for(int i = 0; i < nModels; i++) delete models[i];
		delete [] models;
	}
	if(orient) delete orient;
}

void modelsConf::setNModels(int n)
{
	if(models)
	{
		for(int i = 0; i < nModels; i++) delete models[i];
		delete [] models;
	}

	nModels = n;
	models = new modelConf*[n];
	for(int i = 0; i < n; i++) models[i] = new modelConf();
}
	
void modelsConf::init(materialsConf *mats)
{
	for(int i = 0; i < nModels; i++) models[i]->init(mats);
}

void modelsConf::activate(int state)
{
	glPushMatrix();
	orient->activate();

	for(int i = 0; i < nModels; i++) models[i]->activate(state);

	glPopMatrix();
}
