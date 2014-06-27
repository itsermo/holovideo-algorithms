#ifndef _TEXTURECOORD_H
#define _TEXTURECOORD_H

#define TEX_COORD_GENERATE 0
#define TEX_COORD_FILE 1
#define TEX_COORD_CUBEMAP_GENERATE 2
#define TEX_COORD_CYLIN 3
#define TEX_COORD_SPHERE 4
#define TEX_COORD_EMBED 5

#define TEX_COORD_TCD 0

struct textureCoordConf
{
	int mode;
	int format;

	float texS[4], texT[4];
	int xTile, yTile;
	int space;
	int combination;
	
	char fName[512];
	
	float *verts;
	int nVerts;
	
	orientation *coordOrient;
	float tVert[3];
	float ang0;
	float nWraps;
	float yBottom;
	float yTop;

	textureCoordConf();
	~textureCoordConf();
	void init(char *path);
	void activate();
	void deactivate();
	float *vertCoords(int v, float *vert, float *embCoord);
	int useCoords();
	
	void _readTCDFile(char *path);
	float *_cylindricalTextureCoords(float *vert);
};

struct textureCoordsConf
{
	int nTextureCoords;
	textureCoordConf **texCoords;
	
	textureCoordsConf();
	~textureCoordsConf();
	void setNTextureCoords(int n);
	void init(char *path);
	void activateVert(int v, float *vert, float *embCoord);
};

#endif
