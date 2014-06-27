/*
 * JPCLSharedmemAdaptor.h
 *
 *  Created on: Jun 17, 2013
 *      Author: holo
 */

#ifndef JPCLSHAREDMEMADAPTOR_H_
#define JPCLSHAREDMEMADAPTOR_H_


#include "JSharedMemory.h"
#include "JDisplayState.h"
#include "JHolovideoDisplay.h"
#include "JSceneSlice.h"
#include "JKinectFrame.h"
//#include "JZspaceSequencer.h"

#ifdef __CDT_PARSER__ //help Eclipse scanner find our out-of-project headers
#include "../../ripgen-fbo/src/JSharedMemory.h"
#include "../../ripgen-fbo/src/JDisplayState.h"
#include "../../ripgen-fbo/src/JHolovideoDisplay.h"
#include "../../ripgen-fbo/src/JSceneSlice.h"
#include "../../ripgen-fbo/src/JKinectFrame.h"
#endif


#include <string>

#include <vector>

#define ONE_COLOR_PACKED // use XYZL instead of XYZRGB


//#define PCL_POINT_TYPE pcl::PointXYZI
#define PCL_POINT_TYPE pcl::PointXYZRGB


#define PCL_GRABBER_CALLBACK cloud_cb_

#ifdef ONE_COLOR_PACKED
#define K_FRAME_TYPE JKinectFrameXYZL
#define K_FRAME_KEY KINECT_SHMEM_XYZL_KEY
#define K_FRAME_KEY_2 KINECT_SHMEM_XYZL_KEY_2
#else
#define K_FRAME_TYPE JKinectFrameXYZRGB
#define K_FRAME_KEY KINECT_SHMEM_XYZRGB_KEY
#define K_FRAME_KEY_2 KINECT_SHMEM_XYZRGB_KEY_2
#endif
class JPCLSharedmemAdaptor
{
public:
	JPCLSharedmemAdaptor();
//	void setupGPU(); //needs unified address space, may not work even then
	void preloadCrossAndCircle();
	void showCircleAndCrossAt(float ox, float oy, float oz, float tx, float ty, float tz);

	void setupSharedForGPU(); //more foolproof
	bool loadCloudFromFile(std::string filename);
	bool loadCloudFromFile(std::string filename, float applyScale, float circleCullRadius, float colorCullMax);

	bool loadCloudFromFile(std::string filename, float applyScale, float circleCullRadius, float colorCullMax, float zoffset);
//	void sliceScene();
	void sliceSceneToShmem(float zoffset=0);
	std::string printCloudPreviewString(int n, int skip);
	//std::string getSequenceStatusString();
	int loadedCloudSize();
	//bool setupExperiment(std::string iniDirname,std::string logname, int subject, int run);
	//void endExperiment();
//	void slicesToGPU();
	void createFixationCrossCloud(float radiusMeters, float zMeters);
	//void updateExperiment(char keypress);
	//void updateStylus(float x, float y, float z);
	void setupTweaksForRenderer(int renderer);
	void setPointingMode(bool interactive);
	virtual ~JPCLSharedmemAdaptor();
private:

	JSharedMemory *sharedMem;

	JSharedMemory *displayState;


	JSharedMemory **gpuConfigSharedMem;

	JHolovideoDisplay *displayCfg;

	//JZspaceSequencer *sequencer;

	void* pointXsDevice; //array of points
	void* pointZsDevice;
	void* pointLsDevice;

	//pointers to our shared memory
	float** pointXsShare; //array of points
	float** pointZsShare;
	char** pointLsShare;

	float stylusx;
	float stylusy;
	float stylusz;

	std::vector <struct JGPUMemconfig> memoryConfigs;

	//std::vector <JSceneSlice> sceneA;
	//std::vector <JSceneSlice> sceneB;

	std::string sceneOnDisplay;
	std::string scenePreloaded;
	bool attemptPreload;


	//store patterns for cross and circle for quick access
	bool pointingMode; //are we showing circle and cross instead of loaded cloud?
	float* crossxyzl;
	int crosscount;

	float* circlexyzl;
	int circlecount;

};

#endif /* JPCLSHAREDMEMADAPTOR_H_ */
