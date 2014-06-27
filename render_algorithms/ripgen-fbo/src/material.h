#ifndef _MATERIAL_H
#define _MATERIAL_H

#include "texture.h"

struct materialConf
{
	float ambient[4];
	float diffuse[4];
	float specular[4];
	float shininess;
	float emission[4];
	texturesValConf *textures;
	int transparency;
	
	materialConf();
	~materialConf();
	
	void activate(int state, textureCoordsConf *texCoords);
	void deactivate(textureCoordsConf *texCoords);
	void init(texturesConf *texs);
};

struct materialsConf
{
	int nMaterials;
	materialConf **materials;

	materialsConf();
	~materialsConf();
	void setNMaterials(int n);
	void init(texturesConf *texs);
};

#endif
