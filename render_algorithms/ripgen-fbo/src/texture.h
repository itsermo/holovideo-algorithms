#ifndef _TEXTURE_H
#define _TEXTURE_H

#define TEX_NORMAL 0
#define TEX_CUBEMAP 1

#define TEX_RIGHT 0
#define TEX_LEFT 1
#define TEX_TOP 2
#define TEX_BOTTOM 3
#define TEX_FRONT 4
#define TEX_BACK 5

struct textureCoordsConf;

struct textureConf
{
	int texW, texH;
	unsigned int *texData;
	unsigned int *cmTexData[6];
	unsigned int texName;
	char fName[512];
	int mode;
	char cmFName[6][512];
	
	textureConf();
	~textureConf();
	void init(char *path);
	void activate();
	void deactivate();
	void _importTiff(char *path, char *fname, unsigned int *&texdata);
};

struct texturesConf
{
	int nTextures;
	textureConf **textures;
	
	texturesConf();
	~texturesConf();
	void setNTextures(int n);
	void init(char *path);
	void activate();
};

struct texturesValConf
{
	int ntextures;
	textureConf **textures;
	int *texturesIndex;

	texturesValConf();
	~texturesValConf();
	void init(texturesConf *texs);	
	void setNTextures(int _n);
	void activate(textureCoordsConf *texCoords);
	void deactivate(textureCoordsConf *texCoords);
};

#endif
