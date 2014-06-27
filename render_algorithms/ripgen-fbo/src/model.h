#ifndef _MODEL_H
#define _MODEL_H

struct modelConf
{
	objectsConf *objects;
	orientation *orient;
	char path[512];
	
	modelConf();
	~modelConf();
	void init(materialsConf *mats);
	void activate(int state);
};

struct modelsConf
{
	modelConf **models;
	int nModels;
	orientation *orient;
	
	modelsConf();
	~modelsConf();
	void setNModels(int n);
	void init(materialsConf *mats);
	void activate(int state);
};

#endif
