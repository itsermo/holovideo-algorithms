/*
 *  shapes.cpp
 *  Holorendering
 *
 *  Created by HoloVideo Bove Lab on 3/11/08.
 *  Copyright 2008 __MyCompanyName__. All rights reserved.
 *
 */

#include "shapes.h"

/*
 *  shapes.c
 *  Holorendering
 *
 *  Created by HoloVideo Bove Lab on 3/11/08.
 *  Copyright 2008 __MyCompanyName__. All rights reserved.
 *
 */

#ifdef WIN32
#include <Windows.h>
#endif

#ifdef __APPLE__
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#include <GL/glu.h>
#endif
#include "shapes.h"

/*
** Just a textured cube
*/
void		Cube (void)
{

float mat[4];
  mat[0] = 0.25;
  mat[1] = 0.25;
  mat[2] = 0.25;
  mat[3] = 1.0;
  glMaterialfv(GL_FRONT, GL_AMBIENT, mat);
  mat[0] = 0.5;
  mat[1] = 0;
  mat[2] = 0;
  glMaterialfv(GL_FRONT, GL_DIFFUSE, mat);
  mat[0] = 0.7;
  mat[1] = 0.66;
  mat[2] = 0.66;
  glMaterialfv(GL_FRONT, GL_SPECULAR, mat);
  glMaterialf(GL_FRONT, GL_SHININESS, 0.25 * 128.0);
 glMaterialf(GL_FRONT, GL_SHININESS, 0.25 * 128.0);
 
 glEnable(GL_NORMALIZE);
    	glBegin(GL_QUADS);						// Draw A Quad
// Front Face
		glNormal3f( 0.0f, 0.0f, 1.0f);					// Normal Pointing Towards Viewer
		glTexCoord2f(0.0f, 0.0f); glVertex3f(-1.0f, -1.0f,  1.0f);	// Bottom Left Of The Texture and Quad
		glTexCoord2f(1.0f, 0.0f); glVertex3f( 1.0f, -1.0f,  1.0f);	// Bottom Right Of The Texture and Quad
		glTexCoord2f(1.0f, 1.0f); glVertex3f( 1.0f,  1.0f,  1.0f);	// Top Right Of The Texture and Quad
		glTexCoord2f(0.0f, 1.0f); glVertex3f(-1.0f,  1.0f,  1.0f);	// Top Left Of The Texture and Quad
		// Back Face
		glNormal3f( 0.0f, 0.0f,-1.0f);					// Normal Pointing Away From Viewer
		glTexCoord2f(1.0f, 0.0f); glVertex3f(-1.0f, -1.0f, -1.0f);	// Bottom Right Of The Texture and Quad
		glTexCoord2f(1.0f, 1.0f); glVertex3f(-1.0f,  1.0f, -1.0f);	// Top Right Of The Texture and Quad
		glTexCoord2f(0.0f, 1.0f); glVertex3f( 1.0f,  1.0f, -1.0f);	// Top Left Of The Texture and Quad
		glTexCoord2f(0.0f, 0.0f); glVertex3f( 1.0f, -1.0f, -1.0f);	// Bottom Left Of The Texture and Quad
		// Top Face
		glNormal3f( 0.0f, 1.0f, 0.0f);					// Normal Pointing Up
		glTexCoord2f(0.0f, 1.0f); glVertex3f(-1.0f,  1.0f, -1.0f);	// Top Left Of The Texture and Quad
		glTexCoord2f(0.0f, 0.0f); glVertex3f(-1.0f,  1.0f,  1.0f);	// Bottom Left Of The Texture and Quad
		glTexCoord2f(1.0f, 0.0f); glVertex3f( 1.0f,  1.0f,  1.0f);	// Bottom Right Of The Texture and Quad
		glTexCoord2f(1.0f, 1.0f); glVertex3f( 1.0f,  1.0f, -1.0f);	// Top Right Of The Texture and Quad
		// Bottom Face
		glNormal3f( 0.0f,-1.0f, 0.0f);					// Normal Pointing Down
		glTexCoord2f(1.0f, 1.0f); glVertex3f(-1.0f, -1.0f, -1.0f);	// Top Right Of The Texture and Quad
		glTexCoord2f(0.0f, 1.0f); glVertex3f( 1.0f, -1.0f, -1.0f);	// Top Left Of The Texture and Quad
		glTexCoord2f(0.0f, 0.0f); glVertex3f( 1.0f, -1.0f,  1.0f);	// Bottom Left Of The Texture and Quad
		glTexCoord2f(1.0f, 0.0f); glVertex3f(-1.0f, -1.0f,  1.0f);	// Bottom Right Of The Texture and Quad
		// Right face
		glNormal3f( 1.0f, 0.0f, 0.0f);					// Normal Pointing Right
		glTexCoord2f(1.0f, 0.0f); glVertex3f( 1.0f, -1.0f, -1.0f);	// Bottom Right Of The Texture and Quad
		glTexCoord2f(1.0f, 1.0f); glVertex3f( 1.0f,  1.0f, -1.0f);	// Top Right Of The Texture and Quad
		glTexCoord2f(0.0f, 1.0f); glVertex3f( 1.0f,  1.0f,  1.0f);	// Top Left Of The Texture and Quad
		glTexCoord2f(0.0f, 0.0f); glVertex3f( 1.0f, -1.0f,  1.0f);	// Bottom Left Of The Texture and Quad
		// Left Face
		glNormal3f(-1.0f, 0.0f, 0.0f);					// Normal Pointing Left
		glTexCoord2f(0.0f, 0.0f); glVertex3f(-1.0f, -1.0f, -1.0f);	// Bottom Left Of The Texture and Quad
		glTexCoord2f(1.0f, 0.0f); glVertex3f(-1.0f, -1.0f,  1.0f);	// Bottom Right Of The Texture and Quad
		glTexCoord2f(1.0f, 1.0f); glVertex3f(-1.0f,  1.0f,  1.0f);	// Top Right Of The Texture and Quad
		glTexCoord2f(0.0f, 1.0f); glVertex3f(-1.0f,  1.0f, -1.0f);	// Top Left Of The Texture and Quad
	glEnd();						// Done Drawing The Quad}
}

void		pyramid (void)
{
glEnable(GL_NORMALIZE);

    	glBegin(GL_TRIANGLES);						// Draw A Triangle
glNormal3f( 0.0f, 1.0f, 1.0f);					
		glVertex3f( 0.0f, 1.0f, 0.0f);			// Top Of Triangle (Front)
		glVertex3f(-1.0f,-1.0f, 1.0f);			// Left Of Triangle (Front)
		glVertex3f( 1.0f,-1.0f, 1.0f);			// Right Of Triangle (Front)
glNormal3f( 1.0f, 1.0f, 0.0f);						
		glVertex3f( 0.0f, 1.0f, 0.0f);			// Top Of Triangle (Right)
		glVertex3f( 1.0f,-1.0f, 1.0f);			// Left Of Triangle (Right)
		glVertex3f( 1.0f,-1.0f, -1.0f);			// Right Of Triangle (Right)
glNormal3f( 0.0f, 1.0f, -1.0f);					
		glVertex3f( 0.0f, 1.0f, 0.0f);			// Top Of Triangle (Back)
		glVertex3f( 1.0f,-1.0f, -1.0f);			// Left Of Triangle (Back)
		glVertex3f(-1.0f,-1.0f, -1.0f);			// Right Of Triangle (Back)
glNormal3f( -1.0f, 1.0f, 0.0f);						
		glVertex3f( 0.0f, 1.0f, 0.0f);			// Top Of Triangle (Left)
		glVertex3f(-1.0f,-1.0f,-1.0f);			// Left Of Triangle (Left)
		glVertex3f(-1.0f,-1.0f, 1.0f);			// Right Of Triangle (Left)
	glEnd();						// Done Drawing The Pyramid
}

