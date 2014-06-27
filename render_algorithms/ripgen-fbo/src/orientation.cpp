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
// orientation

orientation::orientation()
{
	magnification = 1.0;
	scale[0] = scale[1] = scale[2] = 1.0;
	rotate[0] = rotate[1] = rotate[2] = 0.0;
	translate[0] = translate[1] = translate[2] = 0.0;
}

orientation::~orientation() {}

void orientation::activate()
{
	glTranslatef(translate[0], translate[1], translate[2]);
	glScalef(magnification*scale[0], magnification*scale[1], magnification*scale[2]);

	glRotatef(rotate[2], 0.0, 0.0, 1.0);
	glRotatef(rotate[1], 0.0, 1.0, 0.0);


	glRotatef(rotate[0], 1.0, 0.0, 0.0);
}

void orientation::inverseFixedScaleActivate()
{
	glRotatef(-rotate[0], 1.0, 0.0, 0.0);
	glRotatef(-rotate[1], 0.0, 1.0, 0.0);
	glRotatef(-rotate[2], 0.0, 0.0, 1.0);
	
	glTranslatef(-translate[0], -translate[1], -translate[2]);
}
