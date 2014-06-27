#ifndef _OBJECT_H
#define _OBJECT_H

#include "drawable.h"
#include "material.h"

#define DRAWABLE_ALIAS 0
#define DRAWABLE_SQUARE 1
#define DRAWABLE_TRIANGLE 2
#define DRAWABLE_CUBE 3
#define DRAWABLE_SPHERE 4
#define DRAWABLE_RAW 5

struct objectConf
{
	char fName[512];
	int material;
	orientation *orient;	
	drawable *drawObject;
	materialConf *mat;
	int dln;
	int drawType;
	int useTexCoords;
	textureCoordsConf *texCoords;

	objectConf();
	~objectConf();
	void init(materialsConf *mats, char *path);
	void activate(int state);
};

struct objectsConf
{
	int nObjects;
	objectConf **objects;

	objectsConf();
	~objectsConf();
	void setNObjects(int n);
	void init(materialsConf *mats, char *path);
	void activate(int state);
};

#endif
