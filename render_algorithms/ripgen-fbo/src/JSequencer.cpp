//Sequencer manages updating holovideo state for SID10 front/back experiment, sending to all instances of holovideo renderer

#include "JSequencer.h"

#define MODEL_SCALE 15.0
#define NEUTRAL_DEPTH 75.0

//define to show stimulii even while waiting for response (for photographing/demoing display)
#define DEMO_LONG_DISPLAY_STIMS 1

unsigned long long JSequencer::getTime()
{
	struct timeval tp;
	struct timezone tz;
	
	gettimeofday(&tp, &tz);
	//printf("time now is: %ld sec %ld usec \n", tp.tv_sec, tp.tv_usec);
	
	return currtime = tp.tv_sec*1000000 + tp.tv_usec;

}

void JSequencer::preloadHolograms()
{
	char configfilename[1024];
	for(int i=0;i<10;i++)
	{
		sprintf(configfilename,"/models/letters/SLOAN/%c.xml", letternames[i]);
		
		holoConf *ren = new holoConf();
		ren->config(configfilename);
		HoloRenderParams *holoRenParams = new HoloRenderParams(framebuffernum);
		RIPHologram *ripParams = new RIPHologram(holoRenParams, ren);
		ripParams->AfterGLInitialize();	
		ren->init();
		holograms.push_back(ripParams);
	}
	//#10 is ready screen
	holoConf *ren = new holoConf();
	ren->config("/models/letters/SLOAN/READY.xml");
	HoloRenderParams *holoRenParams = new HoloRenderParams(framebuffernum);
	RIPHologram *ripParams = new RIPHologram(holoRenParams, ren);
	ripParams->AfterGLInitialize();		
	ren->init();
	holograms.push_back(ripParams);
	
	//#11 is blank screen
	ren = new holoConf();
	ren->config("/models/blankframe.xml");
	holoRenParams = new HoloRenderParams(framebuffernum);
	ripParams = new RIPHologram(holoRenParams, ren);	
	ripParams->AfterGLInitialize();	
	ren->init();
	holograms.push_back(ripParams);
	
	
}

JSequencer::JSequencer(char* logname, int id, int repeat,HoloRenderParams *hrP, RIPHologram *ripP, HoloRenderParams *hrP2, RIPHologram *ripP2)
{
	framebuffernum = 0;
	if(ripP)
	{
		if(ripP->m_holorenparams)
		{
			framebuffernum = ripP->m_holorenparams->m_framebufferNumber;
		}
	}
	
	holo1 = ripP;
	holo2 = ripP2;
	
	letternames[0] = 'C';
	letternames[1] = 'D';
	letternames[2] = 'H';
	letternames[3] = 'K';
	letternames[4] = 'O';
	letternames[5] = 'R';
	letternames[6] = 'S';
	letternames[7] = 'V';
	letternames[8] = 'Z';
	letternames[9] = 'N';
	
	
	//create shared memory blocks
	sharedImA= new JSharedMemory(sizeof(int),IM_A_KEY);
	sharedImB= new JSharedMemory(sizeof(int),IM_B_KEY);
	sharedDepthA = new JSharedMemory(sizeof(float),DEPTH_A_KEY);
	sharedDepthB = new JSharedMemory(sizeof(float),DEPTH_B_KEY);
	sharedMethod = new JSharedMemory(sizeof(char),METHOD_KEY);

	float pdA = NEUTRAL_DEPTH;//128
	float pdB = NEUTRAL_DEPTH;//50
	
	int ready = 10;
	int blank = 11;
	
	char meth = 1; // flat renderer
	
	//blankstate = true;
	
	sharedDepthA->write(&pdA);
	sharedDepthB->write(&pdB);
	
	sharedImA->write(&ready);
	sharedImB->write(&blank);
	
	sharedMethod->write(&meth);
	
	
	//open log file
	char fnamebuf[1024];
	subjectID = id;
	sprintf(fnamebuf, "%s_%d_%d.log", logname, id, repeat);
	logfile = fopen(fnamebuf,"a");
	if(!logfile) 
	{
		perror("opening log");
		exit(1);
	}
	//prepend timestamp to filename?
	fprintf(logfile, "#Log file for hologram experiment ca. 2010, James Barabas barabas@alum.mit.edu\n");
	getTime();
	fprintf(logfile,"#Timestamp %Ld\n", currtime);
	fprintf(logfile, "#Subject %d, run %d\n", subjectID, repeat);
	fflush(logfile);
	//display ready image
	currtrial = 0;
	trialpart = -1;
	loadSequenceFromFile("../etc/seq5.dat");
	preloadHolograms();
	
}

void JSequencer::loadSequenceFromFile(char* filename)
{
	FILE* sequencefile;
	sequencefile = fopen(filename,"r");

	if(!sequencefile)
	{
		perror("loading sequence file");
		exit(1);
	}
	char linebuf[2048];
	float tmp[3];
	int got = 0;
	int maxtrials = 200;
	for(int i=0;i<maxtrials;i++)
	{
		got =fscanf(sequencefile,"%f %f %f\n",tmp, tmp+1, tmp+2);
		if(got<3) break;
		lettersequence.push_back((int)tmp[0]);
		distanceAsequence.push_back(NEUTRAL_DEPTH);//Position of reference letter
		distanceBsequence.push_back(tmp[1]*1.5);//Position of test letter
		renderersequence.push_back(tmp[2]-1);//Rendering method
		printf(".");
		
	}
	printf("\n");
	printf("loaded %d records\n", renderersequence.size());
	fclose(sequencefile);
	
}

void JSequencer::update()
{
	//get updates from shared memory
	float newdepthA;
	float newdepthB;
	int newImA;
	int newImB;
	char newMethod;

	bool ok;
	char r;
	//printf(".");
	//sharedMethod->getDataCopyIfUnlocked(&r); printf("contains %d\n",r);

#ifdef DEMO_LONG_DISPLAY_STIMS
	//don't display ready screen if in demo mode
	if (trialpart < 0) return;
#endif
	ok = sharedMethod->getDataCopy(&newMethod);
	if(ok && (((newMethod==1) != holo1->m_flatrender)))  //if renderer changed
	{
		holo2->m_flatrender = holo1->m_flatrender = (newMethod==1);
		if(holo1->m_flatrender)//TODO set up the multi-planar renderer
		{
				
		}
		else //TODO set up the RIP renderer
		{
			printf("method change -- rip\n");
			holo1->m_projectionPlaneDist = RIP_PROJ_PLANE;
			holo1->recomputeGeometry(holo1->m_holorenparams);
			holo2->m_projectionPlaneDist = RIP_PROJ_PLANE;
			holo2->recomputeGeometry(holo2->m_holorenparams);

		}
	}	
	
	ok=	sharedDepthA->getDataCopy(&newdepthA);
	if(ok && newdepthA != holo1->m_projectionPlaneDist)
	{
		if(holo1->m_flatrender)
		{
			holo1->m_projectionPlaneDist = newdepthA;
			holo1->recomputeGeometry(holo1->m_holorenparams);
		} 
		else
		{
			holo1->m_render->models->orient->translate[2] = newdepthA/MODEL_SCALE;
		}
	}
	
	
	ok = sharedDepthB->getDataCopy(&newdepthB);
	//printf("depth B is %f\n", newdepthB);
	if(ok && newdepthB != holo2->m_projectionPlaneDist)
	{
		if(holo2->m_flatrender)
		{
			holo2->m_projectionPlaneDist = newdepthB;
			holo2->recomputeGeometry(holo2->m_holorenparams);
		}else
		{
			holo2->m_render->models->orient->translate[2] = newdepthB/MODEL_SCALE;
		}
	}
	
	
	ok = sharedImA->getDataCopy(&newImA);
	if(ok) 
	{
		holo1->m_render = holograms[newImA]->m_render;
	}
	
	
	ok = sharedImB->getDataCopy(&newImB);
	if(ok)
	{
		holo2->m_render = holograms[newImB]->m_render;
	}
	
	
	/*
	ok = sharedBlank->getDataCopyIfUnlocked(&newblank);
	if(ok && blankstate != newblank)
	{
		if(newblank)
		{
			holo1->m_render = holograms[HOLO_BLANK]->m_render;
			holo2->m_render = holograms[HOLO_BLANK]->m_render;
		}
	}
	*/
	
	
	getTime();
	if(trialpart == 0)//setup image after response -- start computing fringes.
	{
		char renderer;
		
		float dist1;
		float dist2;
			
		renderer = renderersequence[currtrial];
		printf("next trial using method %d,renderer\n", renderer);
		if(holo1->m_flatrender != (renderer==1))
		{
			holo2->m_flatrender = holo1->m_flatrender = (renderer==1);
			if(holo1->m_flatrender)//TODO set up the multi-planar renderer
			{
				
			}else //TODO set up the RIP renderer
			{
				holo1->m_projectionPlaneDist = RIP_PROJ_PLANE;
				holo1->recomputeGeometry(holo1->m_holorenparams);
				holo2->m_projectionPlaneDist = RIP_PROJ_PLANE;
				holo2->recomputeGeometry(holo2->m_holorenparams);
			}
		}
		sharedMethod->write(&renderer); printf("wrote renderer %d\n", renderer);
		
		//char r;
		//sharedMethod->getDataCopyIfUnlocked(&r); printf("contains %d\n",r);
				
		
		dist1 = distanceAsequence[currtrial];
		dist2 = distanceBsequence[currtrial];
			
		if(dist1 != holo1->m_projectionPlaneDist)
		{
			if(holo1->m_flatrender)
			{
				holo1->m_projectionPlaneDist = dist1;
				holo1->recomputeGeometry(holo1->m_holorenparams);
			} else {
				holo1->m_render->models->orient->translate[2] = dist1/MODEL_SCALE;
			// move geometry
			}
		}
		if(dist2 != holo2->m_projectionPlaneDist)
		{
			if(holo2->m_flatrender)
			{
				holo2->m_projectionPlaneDist = dist2;
				holo2->recomputeGeometry(holo2->m_holorenparams);
			}else {
				holo2->m_render->models->orient->translate[2] = dist2/MODEL_SCALE;
				// move geometry
			}
		}
			
		sharedDepthA->write(&dist1);
		sharedDepthB->write(&dist2);
		
		trialpart++;
		printf("setup\n");
		
	} else if(trialpart == 1)//wait to display (so we can compute fringe)
	{
		if(currtime - trialStartTime > 2000000LLU) //100000 is one second after response
		{
			stimOnTime = currtime;
			trialpart++;
			
			int im = lettersequence[currtrial];
			int N = HOLO_N;
			
			
			holo1->m_render = holograms[N]->m_render;//letter N
			holo2->m_render = holograms[im]->m_render;
			
			sharedImA->write(&N);
			sharedImB->write(&im);
			printf("display\n");
			
		}
	}else if(trialpart == 2)//showing image
	{
		if(currtime - trialStartTime > 2750000LLU) //150000 is 1.5 seconds after response
		{
			stimOffTime = currtime;
			trialpart++;
			
#ifndef DEMO_LONG_DISPLAY_STIMS
			int blank = 11;
			
			holo1->m_render = holograms[blank]->m_render;
			holo2->m_render = holograms[blank]->m_render;
			
			sharedImA->write(&blank);
			sharedImB->write(&blank);
			printf("off\n");
#endif
		}		
	}
	//sharedMethod->getDataCopyIfUnlocked(&r); printf("contains %d\n",r);
}

void JSequencer::keypress(char key)
{
	char b[512];
	
	getTime();
	
	//spacebar pressed at start of experiment
	if(currtrial == 0 && trialpart == -1 && key == ' ')
	{
		trialpart = 0;
		trialStartTime = getTime();
		
		int blank = HOLO_BLANK;
		holo1->m_render = holograms[blank]->m_render;
		holo2->m_render = holograms[blank]->m_render;
			
		sharedImA->write(&blank);
		sharedImB->write(&blank);
		
	} else if(trialpart == 3)//blank, waiting for front/back response
	{
		if(key == INFRONTKEY || key == BEHINDKEY)
		{
			response1Time = currtime;
			response1 = key;
			trialpart++;
		}
		
		
	}
	else if(trialpart == 4)//blank, waiting for letter key response
	{
		if(key <= 'Z' && key >= 'A')
		{
			response2Time = currtime;
			response2 = key;

			trialStartTime = currtime;
			fprintf(logfile,"%Ld\t%Ld\t%Ld\t%c\t%f\t%Ld\t%c\t%c\n",stimOnTime, stimOffTime, response1Time, response1, distanceBsequence[currtrial], response2Time,letternames[lettersequence[currtrial]], response2);
			fflush(logfile);
			
			trialpart = 0;
			currtrial ++;
		} 		
	}
	
	//sprintf(b, "%d\t%d\t%d\t%d\n",);
}

JSequencer::~JSequencer()
{
	getTime();
	fprintf(logfile, "Normal program termination at %Ld\n", currtime);
	fclose(logfile);
}

