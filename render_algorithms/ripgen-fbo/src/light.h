#ifndef _LIGHT_H
#define _LIGHT_H

struct lightConf
{
	float position[4];
	float direction[4];
	float rDirection[4];
	float ambient[4];
	float diffuse[4];
	float specular[4];
	float spotExponent;
	float spotCutoff;
	float constantAttenuation;
	float linearAttenuation;
	float quadraticAttenuation;
	GLenum lNum;

	lightConf();
	~lightConf();
	
	void init(GLenum lNum);
	void activate(int state);
};

struct lightingConf
{
	int nLights;
	float globalAmbient[4];
	lightConf **lights;

	lightingConf();
	~lightingConf();
	void setNLights(int n);

	void init();
	void activate(int state);
};

#endif
