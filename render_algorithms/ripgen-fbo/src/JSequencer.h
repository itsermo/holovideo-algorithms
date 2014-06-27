#ifndef _JSEQUENCER_H_
#define _JSEQUENCER_H_

#include <vector>
#include <sys/time.h>
#include "RIP.h"
#include "holoren.h"
#include "JSharedMemory.h"
#include "orientation.h"
#include "object.h"
#include "drawable.h"
#include "model.h"

#define INFRONTKEY '2'
#define BEHINDKEY '8'

#define HOLO_N	9
#define HOLO_READY 10
#define HOLO_BLANK 11

#define ALL_STATE_KEY 6610
#define IM_A_KEY 6627
#define IM_B_KEY 6628
#define DEPTH_A_KEY 6629
#define DEPTH_B_KEY 6630
#define METHOD_KEY 6631

class JSequencer
{
	unsigned long long starttime;// = 9999999999999999999LLU;
	unsigned long long trialStartTime;
	unsigned long long stimOnTime;
	unsigned long long stimOffTime;
	unsigned long long response1Time;
	unsigned long long response2Time;
	char response1,response2;
	
	JSharedMemory* sharedDepthA;
	JSharedMemory* sharedDepthB;
	JSharedMemory* sharedImA;
	JSharedMemory* sharedImB;
	JSharedMemory* sharedMethod;
	
	unsigned long long currtime;
	int currtrial;
	int trialpart;
	char letternames[10];
	FILE* logfile;
	
	int framebuffernum;
	
	int subjectID;
	
	std::vector<char> lettersequence;
	std::vector<char> renderersequence;
	std::vector<float> distanceAsequence;
	std::vector<float> distanceBsequence;
	std::vector<char> IDResponses;
	std::vector<bool> OrderResponses;
	
	std::vector<RIPHologram*>holograms;
	
	RIPHologram *holo1, *holo2;
	//RIPHologram *ready1, *ready2;
	//bool blankstate; //true if blank screen
	
	unsigned long long getTime();
	
public:
	JSequencer(char* logname, int id, int repeat,HoloRenderParams *hrP, RIPHologram *ripP, HoloRenderParams *hrP2, RIPHologram *ripP2);
	
	//get a pseudo-randomized set of letters, spacings, and rendering methods from a text file.
	void loadSequenceFromFile(char* filename);
	
	//called every frame. Uses time, history of keypresses to advance state.
	void update();
	
	//log keypress with time, if appropriate, trigger start of next trial
	void keypress(char key);
	
	~JSequencer();
private:
	void preloadHolograms();		
	
};


//each sequence step has:

//letter
//front-back position
//reference position
//render method

//According to BS 4274-1:2003 only the letters C, D, E, F, H, K, N, P, R, U, V, and Z should be used for the testing of vision based upon equal legibility of the letters (wiki)

//SLOAN letters (from http://www.psych.nyu.edu/pelli/software.html)
//CDHKNORSVZ
//Pelli, D. G., Robson, J. G., & Wilkins, A. J. (1988) The design of a new letter chart for measuring contrast sensitivity.  Clinical Vision Sciences 2, 187-199. 
#endif

