#ifndef _TRIANGLES_H
#define _TRIANGLES_H

struct triangleList : virtual drawable
{
	int n;
	int useTexCoords;
	float *verts;
	float *norms;
	float *tCoords;

	triangleList(int _n = 0);
	triangleList(char *fname, int _useTexCoords);
	triangleList(char *fname, int _useTexCoords, int type);
	~triangleList();
	
	void setNTriangles(int _n, int copy = 0);
	void draw(textureCoordsConf *texCoords);
	void setTriangle(int tri, float *v1, float *v2, float *v3, float *n1, float *n2, float *n3, float *t1, float *t2, float *t3);
	void readAliasTrifile(char *fname);
	void readRawTrifile(char* fname);
};

#endif
