#ifndef _RENDER_H
#define _RENDER_H

#define NORMAL 0
#define PSEUDO 1
#define FREE 2

struct holoConf;
struct orientation;

struct renderConf
{
	int doubleCamera;
	int recenter;

	int screenX, screenY;
	float holoX, holoY;
	
	float eyeZ;
	float cameraPlaneX;
	float halfCameraPlaneX;
	float cameraPlaneXInc;
	int viewsX;

	float farClip;
	float nearClip;

	renderConf();
	~renderConf();
	void init();
	void activate(int camx, int state, holoConf *conf);
	
protected:
	void placeCamera(int camx, int state);
};

#endif
