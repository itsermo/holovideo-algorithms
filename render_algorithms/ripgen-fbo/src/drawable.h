#ifndef __DRAWABLE_H
#define __DRAWABLE_H

#include "texturecoord.h"
struct drawable
{
	virtual void draw(textureCoordsConf *texCoords) = 0;
};

#endif
