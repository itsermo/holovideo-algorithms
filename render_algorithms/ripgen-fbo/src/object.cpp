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
// object

objectConf::objectConf()
{
	fName[0] = '\0';
	material = 0;
	orient = new orientation();
	drawType = DRAWABLE_ALIAS;
	drawObject = NULL;
	useTexCoords = 0;
	
	texCoords = new textureCoordsConf();
}

objectConf::~objectConf()
{
	if(orient) delete orient;
	if(drawObject) delete drawObject;
	if(texCoords) delete texCoords;
}

void objectConf::init(materialsConf *mats, char *path)
{
	dln = glGenLists(1);
	mat = mats->materials[material];
	char completeFName[512];
	strcpy(completeFName, path);
	strcat(completeFName, fName);

	switch(drawType)
	{
		case DRAWABLE_ALIAS:
			drawObject = new triangleList(completeFName, useTexCoords);
			break;
		case DRAWABLE_RAW:
			drawObject = new triangleList(completeFName, 0, DRAWABLE_RAW);
			break;
		case DRAWABLE_SQUARE:
			drawObject = new primitiveSquare();
			break;
		case DRAWABLE_TRIANGLE:
			drawObject = new primitiveTriangle();
			break;
		case DRAWABLE_CUBE:
			drawObject = new primitiveCube();
			break;
		case DRAWABLE_SPHERE:
			drawObject = new primitiveSphere();
			break;

		default:
			drawObject = NULL;
			break;
	};
	
	if(texCoords) texCoords->init(path);

    // WJP 12/13/04: these three lines were commented out
	glNewList(dln, GL_COMPILE);
	if(drawObject) drawObject->draw(texCoords);
	glEndList();
}

void objectConf::activate(int state)
{
    glPushMatrix();
    orient->activate();
    mat->activate(state, texCoords);
    // WJP 12/13/04: this line was commented out
    if(drawObject) glCallList(dln);
    // WJP 12/13/04: this line was NOT commented out.
    //if(drawObject) drawObject->draw(texCoords);
    mat->deactivate(texCoords);
    glPopMatrix();
}


//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// objects

objectsConf::objectsConf()
{
	nObjects = 0;
	objects = NULL;
}

objectsConf::~objectsConf()
{
	if(objects)
	{
		for(int i = 0; i < nObjects; i++) delete objects[i];
		delete [] objects;
	}
}

void objectsConf::setNObjects(int n)
{
	if(objects)
	{
		for(int i = 0; i < nObjects; i++) delete objects[i];
		delete [] objects;
	}

	nObjects = n;
	objects = new objectConf*[n];
	for(int i = 0; i < n; i++) objects[i] = new objectConf();
}

void objectsConf::init(materialsConf *mats, char *path)
{
	for(int i = 0; i < nObjects; i++) objects[i]->init(mats, path);
}

void objectsConf::activate(int state)
{
	int i;

	for(i = 0; i < nObjects; i++)
        {
            if(!objects[i]->mat->transparency) objects[i]->activate(state);
        }
	
	/* set the z-buffer to read only and draw all the transparent objects */

    glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    //glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);
    //glBlendFunc(GL_SRC_ALPHA_SATURATE,GL_ONE);
    //glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);
    glEnable(GL_POLYGON_SMOOTH);
    glEnable(GL_BLEND);
    glDepthMask(GL_FALSE);

	for(i = 0; i < nObjects; i++)
	{
		if(objects[i]->mat->transparency) objects[i]->activate(state);
	}

	/* return rendering to normal */
    glDisable(GL_POLYGON_SMOOTH);
    glDisable(GL_BLEND);
    glDepthMask(GL_TRUE);
}
