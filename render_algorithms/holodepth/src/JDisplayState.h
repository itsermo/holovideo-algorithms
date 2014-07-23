/*
 * JDisplayState.h
 *
 *  Created on: May 10, 2010
 *      Author: holo
 */

#ifdef REMOTEQT_GUI

#ifndef JDISPLAYSTATE_H_
#define JDISPLAYSTATE_H_

#define ALL_STATE_KEY 3247 //arbitrary unique key

#define JSHAREDSTATE_DEBUG_VARS 20

#define SLAVE_COUNT 3 //number of processes we need to oversee (currently == number of GPUs)

#include "JHolovideoDisplay.h"

//overall state of scene to be rendered
struct JDisplayState //these are variables intended to be asserted by the GUI/sequencer/controller
{
	float xpos;
	float ypos;
	float zpos;
	char filename1[255];
	char filename2[255];
	float xrot;
	float yrot;
	float zrot;
	float gain;
	int rendermode1;
	int rendermode2;
	float flatdepth1;
	float flatdepth2;
	int viewmask;
	//
	int shaderMode;
	int spareI2;
	int spareI3;
	float scale;
	float sparef2;
	float sparef3;
    //
    float debug[JSHAREDSTATE_DEBUG_VARS]; //extra variables for debugging
};

#define ALL_STATUS_KEY 64617 //arbitrary unique key


struct JDisplayStatus
{
	long long lastframeTime[SLAVE_COUNT];
	long long lastStatusTime[SLAVE_COUNT];
	float lastFPS[SLAVE_COUNT];
	int pid[SLAVE_COUNT]; //process id
	char statusMessage[SLAVE_COUNT][1024];
};

#define ALL_GPU_MEMCONFIG_KEY_BASE 41064 //arbitrary unique key

//layout of CUDA pointers in global address space
struct JGPUMemconfig
{
	void* pointXsHost;
	void* pointZsHost;
	void* pointLsHost;
};

#define ALL_SLICE_MEM_KEY_BASE 41942 //arbitrary unique key

struct JGPUSliceData
{
	int sizeX; //total size of slice data
	int sizeZ; //total size of slice data
	int sizeL; //total size of slice data
	int keyX; //key to use to fetch slice data
	int keyZ; //key to use to fetch slice data
	int keyL; //key to use to fetch slice data

};



#endif /* JDISPLAYSTATE_H_ */

#endif