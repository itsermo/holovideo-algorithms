/*
 *  JZCameraCloud.cpp
 *  HoloDepth-Xcode
 *
 *  Created by James Barabas on 12/4/10.
 *  Copyright 2010 MIT Media Lab. All rights reserved.
 *
 */

#include "stdlib.h"
#include "math.h"
#include "stdio.h"
#include "JZCameraCloud.h"

JZCameraCloud::JZCameraCloud() 
{
	cop[0] = cop[1] = cop[2] = 0;
	resx = SWISSRANGER_RESX;
	resy = SWISSRANGER_RESY;
	xs = (float*)malloc(resx*resy*sizeof(float));
	ys = (float*)malloc(resx*resy*sizeof(float));
	zs = (float*)malloc(resx*resy*sizeof(float));
	ls = (float*)malloc(resx*resy*sizeof(float));
	lumtextureGL = 0;
	ztextureGL = 0;
	gain = 1/2048.0; //1/largest camera luminance value (max is 2^16) we care about (for normalization)
	zmax = 1 ; //distance in m that maps to 1.0 in openGL depth texture (depth range)
	la = (float*)malloc(resx*resy*2*sizeof(float));
	xangles = (float*)malloc(resx*resy*sizeof(float));

}

JZCameraCloud::~JZCameraCloud()
{

	free(xs);
	free(ys);
	free(zs);
	free(ls);
	free(la);
	free(xangles);

}

//compute horizontal angles. of pixels relative to camera axis
bool JZCameraCloud::computeAnglesForPixels()
{
	for(int i=0;i<resx*resy;i++)
	{
		xangles[i] = atan2(xs[i],zs[i]);
	}
}

bool JZCameraCloud::loadFromFile(char* filename)
{
	char readbuf[1024];
	FILE* f = fopen(filename,"r");
	if(!f)
	{
		printf("File %s not found for loading range data\n",filename);
		return false;
	}	fgets(readbuf,1024,f); // grab header string
	if (readbuf[0] != '#') {
		printf("File %s does not appear to contain a point cloud\n",filename);
		fclose(f);
		return false;
	}
	
	for(int p = 0;p<resx*resy;p++)
	{
		if( 4 != fscanf(f, "%f %f %f %f" ,xs+p,ys+p,zs+p,ls+p))
		{
			printf("not enough data reading point #%d in %s\n",p,filename);
			fclose(f);
			return false;
		}
		//attempt to gamma correct
		float f = ls[p];
		float g = 1;
		ls[p] = pow(gain*f,g)/gain;
	}
	fclose(f);
	computeAnglesForPixels();
	return true;
	
}

void JZCameraCloud::setCenterOfProjection(float x, float y, float z)
{
	cop[0] = x; cop[1] = y; cop[2] = z;
}

void JZCameraCloud::buildFlatLumaGLTexture()
{
	if(lumtextureGL == 0)
	{
		glGenTextures(1, &lumtextureGL);
	}
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, lumtextureGL);

	glPixelTransferf(GL_RED_SCALE,gain);
	glPixelTransferf(GL_GREEN_SCALE,gain);
	glPixelTransferf(GL_BLUE_SCALE,gain);
	glPixelTransferf(GL_ALPHA_SCALE,gain);
	glTexEnvf( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE );
	glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,GL_LINEAR_MIPMAP_NEAREST );
	// when texture area is large, bilinear filter the original
	glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
	
	// the texture wraps over at the edges (repeat)
	glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT );
	glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT );
	
	//JB: hack to put color in alpha for RGBA view storage
	for(int i=0;i<resx*resy;i++)
	{
		la[2*i] = ls[i];
		la[2*i+1] = ls[i];
	}
	
	
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, resx, resy, 0, GL_LUMINANCE_ALPHA, GL_FLOAT, la); // copy luma image into GL texture
	//glGenerateMipmap(GL_TEXTURE_2D);
	
	glPixelTransferf(GL_RED_SCALE,1);
	glPixelTransferf(GL_GREEN_SCALE,1);
	glPixelTransferf(GL_BLUE_SCALE,1);
	glPixelTransferf(GL_ALPHA_SCALE,1);
	
}

void JZCameraCloud::buildFlatDepthGLTexture()
{
	if(ztextureGL == 0)
	{
		glGenTextures(1, &ztextureGL);
	}
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, ztextureGL);
	
	glPixelTransferf(GL_RED_SCALE,-1.0/zmax);
	glPixelTransferf(GL_GREEN_SCALE,-1.0/zmax);
	glPixelTransferf(GL_BLUE_SCALE,-1.0/zmax);
	glPixelTransferf(GL_ALPHA_SCALE,-1.0/zmax);


	
	glPixelTransferf(GL_RED_BIAS,1.0);
	glPixelTransferf(GL_GREEN_BIAS,1.0);
	glPixelTransferf(GL_BLUE_BIAS,1.0);
	glPixelTransferf(GL_ALPHA_BIAS,1.0);
    

	glTexEnvf( GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE );
	glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,GL_LINEAR_MIPMAP_NEAREST );
	// when texture area is large, bilinear filter the original
	glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
	
	// the texture wraps over at the edges (repeat)
	glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT );
	glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT );
	
	
	//JB: hack to put color in alpha for RGBA view storage
	
	for(int i=0;i<resx*resy;i++)
	{
		la[2*i] = zs[i];
		la[2*i+1] = zs[i];
	}
	
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, resx, resy, 0, GL_LUMINANCE_ALPHA, GL_FLOAT, la); // copy luma image into GL texture
	//glGenerateMipmap(GL_TEXTURE_2D);
	
	glPixelTransferf(GL_RED_SCALE,1);
	glPixelTransferf(GL_GREEN_SCALE,1);
	glPixelTransferf(GL_BLUE_SCALE,1);
	glPixelTransferf(GL_ALPHA_SCALE,1);
	
	glPixelTransferf(GL_RED_BIAS,0);
	glPixelTransferf(GL_GREEN_BIAS,0);
	glPixelTransferf(GL_BLUE_BIAS,0);
	glPixelTransferf(GL_ALPHA_BIAS,0);
	
}
