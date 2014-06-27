#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <GL/glew.h>
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
#include "string.h"

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// holoConf

holoConf::holoConf()
{
	ren = new renderConf();
	lighting = new lightingConf();
	materials = new materialsConf();
	textures = new texturesConf();
	models = new modelsConf();

	texPath[0] = '\0';

	mButtons[0] = GLUT_UP;
	mButtons[1] = GLUT_UP;
	mButtons[2] = GLUT_UP;
}

holoConf::~holoConf()
{
	if(ren) delete ren;
	if(lighting) delete lighting;
	if(materials) delete materials;
	if(textures) delete textures;
	if(models) delete models;
}




void holoConf::config(char *fName)
{
	if(fName)
	{	
		_parseConfigFile(fName);
		strcpy(loadedFile, fName);
	}
	
}

void holoConf::init()
{
	printf("initializing the renderer\n");
	ren->init();
	printf("initializing lighting\n");
	lighting->init();
	printf("initializing materials\n");
	materials->init(textures);
	printf("initializing textures\n");
	textures->init(texPath);
	printf("initializing models\n");
	models->init(materials);

	printf("holoren initialization complete\n");
}

void holoConf::mouse(int button, int state, int x, int y)
{
	mButtons[button] = state;
	mLX = x;
	mLY = y;
}



void holoConf::spin(int x, int y)
{
        models->orient->rotate[0] += x;
		models->orient->rotate[1] += y;
		if(models->orient->rotate[0] > 360)
			models->orient->rotate[0] -= 360;
		
		if(models->orient->rotate[0] < 360)
			models->orient->rotate[0] += 360;
		
		if(models->orient->rotate[1] > 360)
			models->orient->rotate[1] -= 360;
		
		if(models->orient->rotate[1] < -360)
			models->orient->rotate[1] += 360;		
}

void holoConf::motion(int x, int y)
{
	if(!x && !y) return;
	
	//printf("%d, %d\n", x, y);
	static int dX, dY;

	dX = x - mLX;
	dY = y - mLY;
	if(mButtons[GLUT_LEFT_BUTTON] == GLUT_DOWN)
	{

		models->orient->translate[0] += dX/200.0;
		models->orient->translate[1] += dY/200.0;
		
		glutPostRedisplay();
	}
	else if(mButtons[GLUT_RIGHT_BUTTON] == GLUT_DOWN)
	{


		models->orient->translate[0] += dX/200.0;
		models->orient->translate[2] += dY/200.0;

		glutPostRedisplay();
	}
	else if(mButtons[GLUT_MIDDLE_BUTTON] == GLUT_DOWN)
	{

		models->orient->rotate[1] += dX/20.0;
		models->orient->rotate[0] += dY/20.0;

		glutPostRedisplay();
	}

		mLX = x;
		mLY = y;
}

void holoConf::keyboard(unsigned char key, int x, int y)
{
	switch(key) {
	case '+':
		models->orient->translate[2] += .5;
		glutPostRedisplay();
		break;
	case '-':
		models->orient->translate[2] -= .5;
		glutPostRedisplay();
		break;
	case '4':
		models->orient->translate[0] -= .5;
		glutPostRedisplay();
		break;
	case '6':
		models->orient->translate[0] += .5;
		glutPostRedisplay();
		break;
	case '8':
		models->orient->translate[1] -= .5;
		glutPostRedisplay();
		break;
	case '2':
		models->orient->translate[1] += .5;
		glutPostRedisplay();
		break;
	case 'r':
		models->orient->translate[0] = 0;
		models->orient->translate[1] = 0;
		models->orient->translate[2] = 0;
		models->orient->rotate[0] = 0;
		models->orient->rotate[1] = 0;
		models->orient->rotate[2] = 0;
		glutPostRedisplay();
		break;
	};
}

void holoConf::render(int camx)
{
	glClear(GL_COLOR_BUFFER_BIT);
	ren->activate(camx, NORMAL, this); //JB: See render.cpp
	if(ren->doubleCamera) ren->activate(camx, PSEUDO, this);
	
	/*
	int cx, cy;
	cx = glutGet(GLUT_WINDOW_WIDTH)/2;
	cy = glutGet(GLUT_WINDOW_HEIGHT)/2;
	mLX = cx;
	mLY = cy;
	glutWarpPointer(mLX,mLY);
	*/
}

