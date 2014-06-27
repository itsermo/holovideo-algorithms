#ifndef _HOLOREN_H
#define _HOLOREN_H

#include "flyRender.h"

struct renderConf;
struct lightingConf;
struct materialsConf;
struct modelsConf;
struct texturesConf;

struct holoConf : Render
{
	renderConf *ren;
	lightingConf *lighting;
	materialsConf *materials;
	modelsConf *models;
	texturesConf *textures;
	
	char loadedFile[1024];
	
	char texPath[512];

	int mButtons[3];
	int mLX, mLY; //location of mousedown
	
	holoConf();
	virtual ~holoConf();

	void config(char *fName);
	void init();

	void _parseConfigFile(char *fName);


	void render(int camx);
	void mouse(int button, int state, int x, int y);
	void motion(int x, int y);
     void spin(int x, int y);
	void keyboard(unsigned char key, int x, int y);
};

#endif
