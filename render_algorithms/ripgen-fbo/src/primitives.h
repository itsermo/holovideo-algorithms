#ifndef _PRIMITIVES_H
#define _PRIMITIVES_H

struct primitiveSquare : drawable
{
	void draw(textureCoordsConf *texCoords);
};

struct primitiveTriangle : drawable
{
	void draw(textureCoordsConf *texCoords);
};

struct primitiveCube : drawable
{
	void draw(textureCoordsConf *texCoords);
};

struct primitiveSphere : drawable
{
	void draw(textureCoordsConf *texCoords);
};

#endif
