/*
 * JPCLSharedmemAdaptor.cpp
 * This class manages sequencing, loading and preparing point clouds. No QT here for simplicity. QT stuff should be in GUI only.
 *  Created on: Jun 17, 2013
 *      Author: Barabas
 */


#include "JPCLSharedmemAdaptor.h"

#include <pcl/visualization/cloud_viewer.h>
#include <pcl/io/openni_grabber.h>


#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#ifdef __CDT_PARSER__ //help Eclipse scanner find our out-of-project headers
#include "../../ripgen-fbo/src/JSharedMemory.h"
#include "../../ripgen-fbo/src/JDisplayState.h"
#include "../../ripgen-fbo/src/JHolovideoDisplay.h"
#include "../../ripgen-fbo/src/JSceneSlice.h"
#include "../../ripgen-fbo/src/JKinectFrame.h"
#endif


// CUDA runtime includes
#include <cuda_runtime_api.h>

// CUDA utilities and system includes
#include <helper_cuda.h>

#include <algorithm>

//TODO: this comes from uploader.cu. Should eventually create a header file for this.
//void prepareCUDA();


#define MAX_POINTS_PER_SCANLINE 16

JPCLSharedmemAdaptor::JPCLSharedmemAdaptor()
	:stylusx(0)
	,stylusy(0)
	,stylusz(0)
	,attemptPreload(true)
	,crossxyzl(NULL)
	,circlexyzl(NULL)
	,pointingMode(false)
{
int kinectNum = 0; //TODO: can eventually support multi-scenes input with this tool too.

	if (kinectNum == 0) {
		sharedMem = new JSharedMemory(sizeof(K_FRAME_TYPE),K_FRAME_KEY);
	} else {
		sharedMem = new JSharedMemory(sizeof(K_FRAME_TYPE),K_FRAME_KEY_2);
	}

	 displayState = new JSharedMemory(sizeof(JDisplayState), ALL_STATE_KEY);

	//displayCfg = new JMarkIIDisplay();
	displayCfg = new JHolovideoDisplay();
	*displayCfg = JHolovideoDisplay::newMarkIIDisplay();
	displayCfg->printInfo();


	//sequencer = NULL;

	setupSharedForGPU();


	preloadCrossAndCircle();

	std::cout << std::flush;


}


void JPCLSharedmemAdaptor::preloadCrossAndCircle() {

	this->loadCloudFromFile("../clouds/cross.pcd",0.02,99999.0, 0.0, 0.0);

	K_FRAME_TYPE* f = (K_FRAME_TYPE*)sharedMem->getptr();
	crosscount = f->count;

	if(crossxyzl) {
		free(crossxyzl);
	}
	int cloudsize = sizeof(float)*crosscount*4;
	crossxyzl = (float*) malloc(cloudsize);
	memcpy(crossxyzl, f->xyzl, cloudsize);



	this->loadCloudFromFile("../clouds/circle.pcd",0.1,99999.0, 0.0, 0.0);

	f = (K_FRAME_TYPE*)sharedMem->getptr();
	circlecount = f->count;

	if(circlexyzl) {
		free(circlexyzl);
	}
	cloudsize = sizeof(float)*circlecount*4;
	circlexyzl = (float*) malloc(cloudsize);
	memcpy(circlexyzl, f->xyzl, cloudsize);

}

void JPCLSharedmemAdaptor::showCircleAndCrossAt(float ox, float oy, float oz, float tx, float ty, float tz) {

	K_FRAME_TYPE* f = (K_FRAME_TYPE*)sharedMem->getptr();

	//do circle
	for(int i=0;i<circlecount;i++) {

			f->xyzl[i*4] = circlexyzl[i*4] + ox;
			f->xyzl[i*4+1] = circlexyzl[i*4+1] + oy;
			f->xyzl[i*4+2] = circlexyzl[i*4+2] + oz;
			f->xyzl[i*4+3] = circlexyzl[i*4+3];
	}

	//do cross
	for(int i=0;i<crosscount;i++) {

			int j = i + circlecount;
			f->xyzl[j*4] = crossxyzl[i*4] + tx;
			f->xyzl[j*4+1] = crossxyzl[i*4+1] + ty;
			f->xyzl[j*4+2] = crossxyzl[i*4+2] + tz;
			f->xyzl[j*4+3] = crossxyzl[i*4+3];
	}


	f->count = circlecount + crosscount;
	this->sliceSceneToShmem();
}


void JPCLSharedmemAdaptor::setupTweaksForRenderer(int renderer) {
	if(renderer == 2) {
		//if we want to use wafel renderer, or are debugging
		this->attemptPreload = true;
	} else {
		this->attemptPreload = false;
	}

}


void JPCLSharedmemAdaptor::setPointingMode(bool interactive) {
	this->pointingMode = interactive;
}

bool JPCLSharedmemAdaptor::loadCloudFromFile(std::string filename, float applyScale, float circleCullRadius, float colorCullMax)
{
	loadCloudFromFile(filename, applyScale, circleCullRadius, colorCullMax, /*-0.01*/0.0); //scale 1, cull everything outside circle with radius of 30 mm, drop points 10 or darker.
}


bool JPCLSharedmemAdaptor::loadCloudFromFile(std::string filename)
{
	loadCloudFromFile(filename, 1.0, 0.03, 10, /*-0.01*/0.0); //scale 1, cull everything outside circle with radius of 30 mm, drop points 10 or darker.
}
//applyScale is a scale factor applied to points in the file before any clipping
//circleCullRadius allows stripping out points that don't fall inside some radius in x-y
//colorCullMax is brightest color (out of 255) we want to consider black (set to -1 to skip black removal)
//z offset is distance in z to translate everything. (avoid artifacts at z=0)
//Following this call, scene is in xyzl buffer in meter coordinates.
bool JPCLSharedmemAdaptor::loadCloudFromFile(std::string filename, float applyScale, float circleCullRadius, float colorCullMax, float zoffset)
{
	bool fileHasNoLuma = false;
	pcl::PointCloud<PCL_POINT_TYPE>::Ptr cloud (new pcl::PointCloud<PCL_POINT_TYPE>());
	pcl::PointCloud<pcl::PointXYZ>::Ptr xyzcloud (new pcl::PointCloud<pcl::PointXYZ>());

    pcl::PCDReader p;
    /*
    readHeader (const std::string &file_name, sensor_msgs::PointCloud2 &cloud,
                      Eigen::Vector4f &origin, Eigen::Quaternionf &orientation, int &pcd_version,
                      int &data_type, unsigned int &data_idx, const int offset = 0);*/


    //read file header to find out what we are dealing with
    Eigen::Vector4f origin_unused;
    Eigen::Quaternionf orientation_unused;
    pcl::PCLPointCloud2  blob;
    int version_unused;
    int data_type_unused;
    unsigned int data_idx_unused;
    p.readHeader(filename, blob,origin_unused,orientation_unused, version_unused, data_type_unused, data_idx_unused );

    int filecount = 0; //count of points in file


    //look at data to see how many chans, then open file with appropriate loader
    if(blob.fields.size() < 4) {
    	fileHasNoLuma = true;
    	if (pcl::io::loadPCDFile<pcl::PointXYZ> (filename, *xyzcloud) == -1) //* load the file
    		{
    			PCL_ERROR ("Couldn't read file\n");
    			return false;
    		}
    	filecount = xyzcloud->size();
    } else {
		if (pcl::io::loadPCDFile<PCL_POINT_TYPE> (filename, *cloud) == -1) //* load the file
		{
			PCL_ERROR ("Couldn't read file\n");
			return false;
		}
		filecount = cloud->size();
    }

	K_FRAME_TYPE* f = (K_FRAME_TYPE*)sharedMem->getptr();

	f->count = filecount;

	if(filecount > KINECT_FRAME_PIX) {
		filecount = KINECT_FRAME_PIX;
		printf("limiting pcd load to %d points\n", filecount);
	}

	int kept = 0;//running count of how many points passed culling tests
	float sx, sy, sz;

	float sqr = circleCullRadius * circleCullRadius;

	if(fileHasNoLuma) {
#ifndef ONE_COLOR_PACKED
			printf("WARNING: Code not yet written for color clouds from xyz files.");
#endif

		for(int i=0;i<filecount;i++) {

			const pcl::PointXYZ &pt= xyzcloud->at(i);

			sx = pt.x*applyScale;
			sy = pt.y*applyScale;
			sz = pt.z*applyScale + zoffset;

			if(sx*sx+sy*sy <= sqr) {
				f->xyzl[kept*4] = sx;
				f->xyzl[kept*4+1] = sy;
				f->xyzl[kept*4+2] = sz;
				f->xyzl[kept*4+3] = 255; //we set color to 255 when none specified for cloud
				kept++;
			}
		}
	} else {


		for(int i=0;i<filecount;i++)
		{
			const PCL_POINT_TYPE &pt= cloud->at(i);

	#ifdef ONE_COLOR_PACKED


			sx = pt.x*applyScale;
			sy = pt.y*applyScale;
			sz = pt.z*applyScale + zoffset;

			if(sx*sx+sy*sy <= sqr && pt.g > colorCullMax) {
				f->xyzl[kept*4] = sx;
				f->xyzl[kept*4+1] = sy;
				f->xyzl[kept*4+2] = sz;
				f->xyzl[kept*4+3] = pt.g; //we set color to 255 when none specified for cloud
				kept++;
			}

	#else

			f->xyzrgb[i*6] = pt.x;
			f->xyzrgb[i*6+1] = pt.y;
			f->xyzrgb[i*6+2] = pt.z;
			f->xyzrgb[i*6+3] = pt.r; //can skip copy and duplicate g in vertex shader (monochrome display)
			f->xyzrgb[i*6+4] = pt.g;
			f->xyzrgb[i*6+5] = pt.b; //can skip copy and duplicate g in vertex shader (monochrome display)

	#endif
			//printf("%g\t%g\t%g\t%g\n", pt.x,pt.y,pt.z, (pt.r+pt.g+pt.b)/3.0);
		}
	}
	f->count = kept;
	printf("Loaded %s. total points: %d\n",filename.c_str(),filecount);
	fflush(stdout);
	return true;

}

/*

//TODO: Looks like this has to happen in same process that calls kernels.
void JPCLSharedmemAdaptor::setupGPU() {
	//allocate memory in GPU for each scanline's info
	int cards;
    checkCudaErrors(cudaGetDeviceCount(&cards));

	if(cards < displayCfg->cardCount) {
		std::cout << "Can't find enough graphics cards to use the selected configuration (see JPCLSharedMemAdapter::setupGPU\n ";
		exit(1);
	} else {
		cards = displayCfg->cardCount;
	}


	//set up shared memory for passing these pointers (to kernels managed by separate processes)
	gpuConfigSharedMem = (JSharedMemory**)malloc(cards*sizeof(JSharedMemory*));

	for(int i=0;i<cards;i++) {
		gpuConfigSharedMem[i] = new JSharedMemory(sizeof(JGPUMemconfig), ALL_GPU_MEMCONFIG_KEY_BASE + i);
	}

	memoryConfigs.resize(cards);



	int linesPerCard = displayCfg->scanlineCount/displayCfg->cardCount;

	for(int card=0;card<displayCfg->cardCount;card++) {
		//allocate buffer for each card
		cudaSetDevice(card);
		JGPUMemconfig ptrs;
		ptrs.pointXsHost = NULL;
		ptrs.pointZsHost = NULL;
		ptrs.pointLsHost = NULL;

		//Don't allocate on device. Allocate in shared memory
		checkCudaErrors(cudaHostAlloc((void**) &(ptrs.pointXsHost), sizeof(float) * MAX_POINTS_PER_SCANLINE * linesPerCard,cudaHostAllocPortable));
		checkCudaErrors(cudaHostAlloc((void**) &(ptrs.pointZsHost), sizeof(float) * MAX_POINTS_PER_SCANLINE * linesPerCard,cudaHostAllocPortable));
		checkCudaErrors(cudaHostAlloc((void**) &(ptrs.pointLsHost), sizeof(char) * MAX_POINTS_PER_SCANLINE * linesPerCard,cudaHostAllocPortable));

		//write the pointers we got into the shared memory block for this GPU
		gpuConfigSharedMem[card]->write(&ptrs);

		//our member copy of addrs
		memoryConfigs[card] = ptrs;

	}

}
*/
//re-factored to save slices in POSIX shared memory instead.

void JPCLSharedmemAdaptor::setupSharedForGPU() {
	//allocate memory in GPU for each scanline's info
	int cards;
    checkCudaErrors(cudaGetDeviceCount(&cards));

	if(cards < displayCfg->cardCount) {
		std::cout << "Can't find enough graphics cards to use the selected configuration (see JPCLSharedMemAdapter::setupGPU\n ";
		exit(1);
	} else {
		cards = displayCfg->cardCount;
	}


	//set up shared memory for passing these pointers (to kernels managed by separate processes)
	gpuConfigSharedMem = (JSharedMemory**)malloc(cards*sizeof(JSharedMemory*));

	pointXsShare = (float**)malloc(cards*sizeof(float*));
	pointZsShare = (float**)malloc(cards*sizeof(float*));
	pointLsShare = (char**)malloc(cards*sizeof(char*));


	int linesPerCard = displayCfg->scanlineCount/displayCfg->cardCount;

	int pointsPerGPU = MAX_POINTS_PER_SCANLINE * linesPerCard;

	int sliceBlockKey = ALL_SLICE_MEM_KEY_BASE;

	for(int card=0;card<cards;card++) {
		gpuConfigSharedMem[card] = new JSharedMemory(sizeof(JGPUSliceData), ALL_GPU_MEMCONFIG_KEY_BASE + card);

		JGPUSliceData slices;
		slices.sizeX = slices.sizeZ = sizeof(float)*pointsPerGPU;
		slices.sizeL = sizeof(char)*pointsPerGPU;

		slices.keyX = sliceBlockKey ++;
		slices.keyZ = sliceBlockKey ++;
		slices.keyL = sliceBlockKey ++;

		gpuConfigSharedMem[card]->write(&slices);

		//allocate memory for each slice block & keep pointers in member variables
		pointXsShare[card] = (float*)(new JSharedMemory(slices.sizeX,slices.keyX))->getptr();
		pointZsShare[card] = (float*)(new JSharedMemory(slices.sizeZ,slices.keyZ))->getptr();
		pointLsShare[card] = (char*)(new JSharedMemory(slices.sizeL,slices.keyL))->getptr();
	}
}

/*
void JPCLSharedmemAdaptor::sliceScene() { //replaced by sliceToShmem
	//go through points in shared memory
	K_FRAME_TYPE* f = (K_FRAME_TYPE*)sharedMem->getptr();

	int pointcount = f->count;

	float vzclip = -2.0*displayCfg->viewingDistanceMeters/displayCfg->displayHeightMeters;//compute viewing distance in clip coords

	int lines = displayCfg->scanlineCount;

	//dump any old scanlines and init new ones
	sceneA.clear();
	sceneA.resize(lines);

	for(int i=0;i<pointcount;i++) {
		//TODO: scene transform
		float px = f->xyzl[i*4];
		float py = f->xyzl[i*4+1];
		float pz = f->xyzl[i*4+2];
		float pl = f->xyzl[i*4+3];
		//project using vert-only projection

		//find y-coord on z=0 hologram plane by similar triangles
		float yh = (vzclip*py)/(vzclip - pz);

		//map to integer scanline
		int yl = floor((float)lines*(-yh + 1.0)/2.0);
		if(yl >= 0 && yl < lines) {
			//squirt into scanline's buffer
			sceneA[yl].pointsX.push_back(px);
			sceneA[yl].pointsZ.push_back(pz);
			sceneA[yl].pointsL.push_back(pl);
		}

	}
	//debug
	for(int i=0;i<lines;i++){
		std::cout << "line ["<< i << "] contains " << sceneA[i].pointsL.size() << " points\n"<<std::flush;
	}

	//slicesToGPU();

}
*/



//go through points in xyzl buffer, convert from meter coordinates to clip coordinates
void JPCLSharedmemAdaptor::sliceSceneToShmem(float zoffset) {

	float yscale = 2.0/displayCfg->displayHeightMeters; //multiply by this to get clip y coords
	float xscale = 2.0/displayCfg->displayWidthMeters; //multiply by this to get clip x coords
	//go through points in shared memory
	K_FRAME_TYPE* f = (K_FRAME_TYPE*)sharedMem->getptr();

	int pointcount = f->count;

	float vzclip = 2.0*displayCfg->viewingDistanceMeters/displayCfg->displayHeightMeters;//compute viewing distance in vertical clip coords

	int lines = displayCfg->scanlineCount;

	int linesPerCard = displayCfg->scanlineCount/displayCfg->cardCount;

	int pointsPerGPU = MAX_POINTS_PER_SCANLINE * linesPerCard;

	//black any old points. This creates tearing. Should sync with renderer or overwrite pixel-by-pixel
	for(int card=0;card<displayCfg->cardCount;card++) {
		//memset(pointXsShare[card],0,sizeof(float)*pointsPerGPU);
		//memset(pointZsShare[card],0,sizeof(float)*pointsPerGPU);
		//memset(pointLsShare[card],0,sizeof(char)*pointsPerGPU);
	}
	int linecounters[lines];
	memset(linecounters,0,sizeof(int)*lines);


	//insert new points onto appropriate line
	for(int i=0;i<pointcount;i++) {
		//TODO: scene transform
		float pxMeters = f->xyzl[i*4];
		float pyMeters = f->xyzl[i*4+1];
		float pzMeters = f->xyzl[i*4+2] + zoffset;
		float pl = f->xyzl[i*4+3];
		//project using vert-only projection (hologram takes care of perspective in x direction)

		float py = pyMeters*yscale;
		float pz = pzMeters*yscale;

		//find y-coord on z=0 hologram plane by similar triangles
		float yh = (vzclip*py)/(vzclip - pz);

		//map to integer scanline
		int yl = floor((float)lines*(-yh + 1.0)/2.0);
		if(yl >= 0 && yl < lines) {

			if(linecounters[yl] < MAX_POINTS_PER_SCANLINE) {
				int card = displayCfg->cardForLine[yl];
				//squirt into scanline's buffer
				int offset = MAX_POINTS_PER_SCANLINE * displayCfg->lineInCard[yl] + linecounters[yl];
				pointXsShare[card][offset] = pxMeters;
				pointZsShare[card][offset] = pzMeters;
				pointLsShare[card][offset] = pl;

				linecounters[yl]++;
			}
		}
	}
	//debug


	for(int i=0;i<lines;i++){
	//	std::cout << "line ["<< i << "] contains " << linecounters[i]<< " points\n"<<std::flush;
		//make everyting black that we didn't write to
		for(int p=linecounters[i];p<MAX_POINTS_PER_SCANLINE;p++) {
			pointLsShare[displayCfg->cardForLine[i]][MAX_POINTS_PER_SCANLINE * displayCfg->lineInCard[i] + p] = 0;
		}
	}

}



std::string JPCLSharedmemAdaptor::printCloudPreviewString(int n,int skip = 0) {
	std::stringstream sstm;

	K_FRAME_TYPE* f = (K_FRAME_TYPE*)sharedMem->getptr();

	int limit = n + skip;
	if (limit > KINECT_FRAME_PIX) limit = KINECT_FRAME_PIX;
	if(limit > f->count) limit = f->count;


	for (int i=skip;i<limit;i++) {
		sstm <<"["<<i<<"]:("<< f->xyzl[i*4] << ", " << f->xyzl[i*4+1]<<", "<< f->xyzl[i*4+2] << ") " << f->xyzl[i*4+3] << "<br>";
	}
	sstm << "<i>cloud total: " << f->count << " points</i>\n";
	return sstm.str();
}

int JPCLSharedmemAdaptor::loadedCloudSize() {
	K_FRAME_TYPE* f = (K_FRAME_TYPE*)sharedMem->getptr();
	if(!f) return 0;
	return f->count;
}

////Load an experiment sequence from a ini file
//bool JPCLSharedmemAdaptor::setupExperiment(std::string iniDirname, std::string logname, int subject, int run ) {
//	if(sequencer) {
//		delete sequencer;
//		sequencer = NULL;
//	}
//	sequencer = new JZspaceSequencer(iniDirname, logname, subject, run);
//
//	sequencer->logString("#Experiment running on Mark II\n");
//
//	std::string pointingString = "Pointing";
//	pointingMode = pointingString.compare(sequencer->getExperimentType()) == 0;
//
//	return sequencer->getTrialsLeft() > 0;
//
//}

//void JPCLSharedmemAdaptor::endExperiment() {
//	pointingMode = false;
//	if(sequencer) {
//		delete sequencer;
//		sequencer = NULL;
//	}
//}
//std::string JPCLSharedmemAdaptor::getSequenceStatusString() {
//	std::stringstream sstm;
//	if(sequencer) {
//		sstm << "Left: " << sequencer->getTrialsLeft() << ", Preload: " << sequencer->getStimToPreload() << " Up: " <<  sequencer->getStimToShow();
//	} else {
//		sstm << "No current sequence.";
//	}
//	return sstm.str();
//}

//update function for experiment (with optional keypress)
//void JPCLSharedmemAdaptor::updateExperiment(char keypress) {
//
//
//	//if (pointingMode) {
//	//	showCircleAndCrossAt(0.0,0.0,-0.02, stylusx, stylusy, stylusz);
//	//}
//	//if we aren't trying to reveal things with minimal latency,
//	//if(!attemptPreload) {
//
//
//		/*
//		//cheat -- overwrite first point and refresh scene
//		K_FRAME_TYPE* f = (K_FRAME_TYPE*)sharedMem->getptr();
//		f->xyzl[0] = stylusx;
//		f->xyzl[1] = stylusy;
//		f->xyzl[2] = stylusz;
//		f->xyzl[3] = 255;
//
//		sliceSceneToShmem();
//		*/
//	//}
//
//
//	if(sequencer) {
//		if(sequencer->getTrialsLeft() > 0) {
//			sequencer->update();
//
//			 std::string preload = sequencer->getStimToPreload();
//			 std::string show = sequencer->getStimToShow();
//
//			 float height = sequencer->getStimHeight();
//			 float scale = 1.0;
//			 float clipradius = 9999;//height/2.0;
//			 if(!attemptPreload || pointingMode) { //don't preload. just show what's in show.
//				 if (pointingMode) {
//					float sz = sequencer->getParam1();
//					 if(show.length() < 2) {
//						 sz = 1000; // hide circle
//					 }
//
//					 //for pointing experiments, set the stereogram/hologram rendering mode
//					 JDisplayState *state = (JDisplayState*)displayState->getptr();
//					 state->rendermode1 = sequencer->getParam3();
//
//				 	showCircleAndCrossAt(0.0,0.0,sz, stylusx, stylusy, stylusz);
//				 } else {
//					 float size = scale;
//					 if(show.length() < 1) {
//						 show = "/clouds/blank.pcd";
//						 size = height;
//					 } else if(show.length() < 2) {
//						 show = "/clouds/cross.pcd";
//						 size = height;
//					 }
//
//					 if(sceneOnDisplay.compare(show)) { //not showing requested image yet
//						 loadCloudFromFile(show,size, clipradius, -1);
//						 sceneOnDisplay = show;
//					 }
//				 }
//			 } else {
//				 if(show.compare(sceneOnDisplay)) { //something new to show
//					 std::cout << "want \"" << show << "\" but currently showing \"" << sceneOnDisplay <<"\"" << " with \"" << scenePreloaded << "\" preloaded." << std::endl << std::flush;
//					 if(!scenePreloaded.compare(show)) { //already loaded & ready, just needs slicing & copy to render processes
//						 std::cout << "slicing preloaded scene" << std::endl << std::flush;
//						 sliceSceneToShmem();
//						 sceneOnDisplay = show;
//					 } else if(show.length() < 1) { //we want blank screen so blank immediately
//						 if(sceneOnDisplay.length() > 0) { //if non-blank on screen, asking for blank
//							 std::cout << "load & display blank" << std::endl << std::flush;
//							 loadCloudFromFile("/clouds/blank.pcd",height, clipradius, -1);
//							 scenePreloaded = "";
//							 sliceSceneToShmem();
//							 sceneOnDisplay = show;
//						 }
//					 } else if(show.length() < 2) { //asking for "+"
//						 std::cout << "load & display +" << std::endl << std::flush;
//						 loadCloudFromFile("/clouds/cross.pcd",height, clipradius, -1);
//						 scenePreloaded = "+";
//						 sliceSceneToShmem();
//						 sceneOnDisplay = show;
//					 } else {
//						 std::cout << "scene requested without preloading in scene display JPCLSharedmemAdaptor::updateExperiment!" << std::endl << std::flush;
//						 preload = show;
//					 }
//				 }
//
//				 if(preload.compare(scenePreloaded)) { //something new to preload
//					 if(preload.length() < 2) {
//						 //std::cout << "preload blank" << std::endl << std::flush;
//						 //loadCloudFromFile("/clouds/blank.pcd",height, height/2.0, -1);
//						 //scenePreloaded = preload;
//
//					 } else {
//						 std::cout << "preload stimulus file:" << preload << std::endl << std::flush;
//						 loadCloudFromFile(preload,scale, clipradius, -1);
//						 scenePreloaded = preload; //store that we have preloaded
//					 }
//				 }
//			}
//			if(keypress) {
//				sequencer->keypress(keypress);
//				fflush(stdout); //make sure console is up to date
//			}
//		}
//	}
//
//}
//
//void JPCLSharedmemAdaptor::updateStylus(float x, float y, float z) {
//	stylusx = x;
//	stylusy = y;
//	stylusz = z;
//	if(sequencer) {
//		sequencer->pointerMoved(x,y,z);
//	}
//}

//create a set of points that form a cross in the center of the screen
void JPCLSharedmemAdaptor::createFixationCrossCloud(float radiusMeters = 0.01, float zMeters = 0)
{
	//write to point list and slice from there
	//sixteen points across at y=0 spanning +- radius meters
	//points on every scanline covering +-radius meters
}


JPCLSharedmemAdaptor::~JPCLSharedmemAdaptor()
{
//	if(sequencer) {
//		delete sequencer;
//		sequencer = NULL;
//	}
	// TODO Auto-generated destructor stub
}

