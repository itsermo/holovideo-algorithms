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
// WJPNOTE: Configure view renderer here.
renderConf::renderConf()
{
	doubleCamera = 0;
	recenter = 1;

	screenX = 383;
	screenY = 144; //TODO: see also RIP.h nLines (DUPE)
    // in this case, this is the dim of projection plane.
	holoX = 7.54;
	holoY = 5.65;
	
    //	eyeZ = 590.0;
    //	cameraPlaneX = 381.0;
    //	viewsX = 142;
    eyeZ = 59.60;
    cameraPlaneX = 38.30;
    //viewsX = 140;
	viewsX = 128;
	farClip = 5000;
	nearClip = 1000.0;
}

renderConf::~renderConf()
{
}

void renderConf::init()
{
	halfCameraPlaneX = cameraPlaneX*0.5;
	cameraPlaneXInc = cameraPlaneX/viewsX;

	glClearColor(0.0, 0.0, 0.0, 0.0);
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_NORMALIZE);
	glDisable(GL_CULL_FACE);
    	glLightModeli(GL_LIGHT_MODEL_TWO_SIDE, GL_TRUE);
    	glShadeModel(GL_SMOOTH);
    	glEnable(GL_LIGHTING);
        glEnable(GL_POLYGON_SMOOTH);
        glHint(GL_POLYGON_SMOOTH_HINT, GL_NICEST);
}

void renderConf::placeCamera(int camx, int state)
{
	float halfHoloX = holoX/(float) 2.0;
	float halfHoloY = holoY/(float) 2.0;

	float camPosX = -halfCameraPlaneX + cameraPlaneXInc*(camx+0.5f);

	float fL, fR; //frustum left X, frustum right X
	float fB, fT; //frustum bottom Y, frustum top Y

	if(recenter)
	{
        // this defines fL on image plane.
        // shouldn't we define it on the near plane?
		fL = -halfHoloX - camPosX;
		fR = fL + holoX;
	}
	else
	{
		fL = -halfHoloX;
		fR = halfHoloX;
	}

    // Frustum top and bottom.
	fB = -halfHoloY;
	fT = halfHoloY;

	if(doubleCamera)
        {
            if(state == NORMAL)
                {
                    glMatrixMode(GL_PROJECTION);
                    glLoadIdentity();
                    glFrustum(fL/nearClip, fR/nearClip, fB/nearClip, fT/nearClip, eyeZ/nearClip, farClip);

                    glMatrixMode(GL_MODELVIEW);
                    glLoadIdentity();
                    gluLookAt(camPosX, 0.0, eyeZ/nearClip, camPosX, 0.0, 0.0, 0.0, 1.0, 0.0);
                }
            else //state == PSEUDO
                {
                    glMatrixMode(GL_PROJECTION);

                    glLoadIdentity();
                    glFrustum(fL/nearClip, fR/nearClip, fB/nearClip, fT/nearClip, eyeZ/nearClip, farClip);

                    glMatrixMode(GL_MODELVIEW);
                    glLoadIdentity();
                    // eyex eyey eyez, centerx centery, centerz, upx, upy, upz
                    gluLookAt(camPosX, 0.0, -eyeZ/nearClip, camPosX, 0.0, 0.0, 0.0, -1.0, 0.0);
		}
	}
	else // regular old shearing and recentering camera.
        {
            glMatrixMode(GL_PROJECTION);
            glLoadIdentity();
            glFrustum(fL/nearClip, fR/nearClip, fB/nearClip, fT/nearClip, eyeZ/nearClip, farClip);

            glMatrixMode(GL_MODELVIEW);
            glLoadIdentity();
            gluLookAt(camPosX, 0.0, eyeZ, camPosX, 0.0, 0.0, 0.0, 1.0, 0.0);
        }
}

void renderConf::activate(int camx, int state, holoConf *conf)
{
	if(doubleCamera)

	{
		if(state == NORMAL)
		{
			glDepthFunc(GL_LESS);
			glClearDepth(1.0);
			glCullFace(GL_BACK);
            //WJP 12/13/04 commented out line below to match stereogram code.
			//glClear(GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
			placeCamera(camx, NORMAL);

			conf->lighting->activate(NORMAL);
			conf->models->activate(NORMAL);
		}
		else if(state == PSEUDO)
		{
			glDepthFunc(GL_GREATER);
			glClearDepth(0.0);
			glCullFace(GL_FRONT);
            //WJP 12/13/04 commented out line below to match stereogram code.
			//glClear(GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);		
			placeCamera(camx, PSEUDO);

			conf->lighting->activate(PSEUDO);
			conf->models->activate(PSEUDO);
		}
	}
	else //single camera:
	{
		glDepthFunc(GL_LESS);
		glClearDepth(1.0);
		glCullFace(GL_BACK);	
            //WJP 12/13/04 commented out line below to match stereogram code.
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT);
		placeCamera(camx, NORMAL);
		conf->lighting->activate(NORMAL);
		conf->models->activate(NORMAL);
	}
}
