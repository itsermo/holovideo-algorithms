/*
 *  JZCameraCloud.h
 *  HoloDepth-Xcode
 *
 *  Created by James Barabas on 12/4/10.
 *  Copyright 2010 MIT Media Lab. All rights reserved.
 *
 */

#define SWISSRANGER_RESX 176
#define SWISSRANGER_RESY 144

#ifdef __APPLE__
#include <OpenGL/gl.h>
#include <OpenGL/glu.h>
#else
#if USE_GLEW
#include <GL/glew.h>
#endif
#include <GL/gl.h>
#include <GL/glu.h>
#define GL_GLEXT_PROTOTYPES
#include <GL/glext.h>

#endif

class JZCameraCloud
{
public:
	JZCameraCloud();
	~JZCameraCloud();
	void setCenterOfProjection(float x, float y, float z);

	bool loadFromFile(char* filename);
	
	void buildFlatLumaGLTexture(); //take point cloud data and render luma to a gl texture 
	void buildFlatDepthGLTexture();
	bool computeAnglesForPixels();

	
	float cop[3]; //center of projection position in world frame
	int resx; //horizontal camera resolution
	int resy; //vertical camera resolution
	//point cloud
	float *xs;
	float *ys;
	float *zs;
	float *ls; //luma
	GLuint lumtextureGL;
	GLuint ztextureGL;
	float gain; // for pre-scaling 16bit luma
	float zmax;
	float *la; //buffer for storing a luminance-alpha copy of luma image
	float *xangles; //left-right angle (for each pixel) relative to camera (0 for pixels at center column of frame)
};
