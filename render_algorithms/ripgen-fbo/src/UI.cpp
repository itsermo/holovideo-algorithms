#include <stdlib.h>
#include <stdio.h>
#include "setupglew.h"
#include <GL/gl.h>
#include <GL/glu.h>
#include <GL/glxew.h>
//#include <GL/glx.h> //for Nvida Framelock code
#include <GL/glut.h>

#include "JSharedMemory.h"
#include "JSequencer.h"
#include "JDisplayState.h"

#include "RIP.h"
#include "UI.h"
#include "flyRender.h"
#include "holoren.h"
#include <sys/time.h>

//headers for 3d mouse
#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/Xos.h>
#include <X11/Xatom.h>
#include <X11/keysym.h>

#include "xdrvlib.h"

#include <stdio.h>    /* for printf and NULL */
#include <stdlib.h>   /* for exit */
#include <string.h>
#include <GL/gl.h>     // The GL Header File
#include <GL/glut.h>   // The GL Utility Toolkit (Glut) Header
//#include <GL/glx.h>
//end 3d mouse headers

static HoloRenderParams *holoRenParams = NULL;
static HoloRenderParams *holoRenParams2 = NULL;

static RIPHologram *ripParams = NULL;
static RIPHologram *ripParams2 = NULL;

//static int attributeList[] = { GLX_RGBA, GLX_DOUBLEBUFFER, None };  //JB: The only use of this was if'd out of Tyler's old code.

static Display *dpy;
//static Window win;
//static XVisualInfo *vi;  //not used
//static XSetWindowAttributes swa; //not used
static GLXContext cx;

static JDisplayState statecopy;
static JSharedMemory *sharedstate;

float MASTER_GAIN = 0.7; //this is overridden by the external UI

double MagellanSensitivity = 1.0;
Display *dsp;
Window root, drw;
XEvent report;
MagellanFloatEvent MagellanEvent;


int fringeToUse = 0;

bool freeSpin = false; // should object be sipinning for demo?

static bool ismaster = false; // is this instance the one that has keyboard/mouse? For now, pick based on screen number & make sure 0 is opened last.

static JSharedMemory* sharedFilename;

static JSharedMemory* sharedDepthA;
static JSharedMemory* sharedDepthB;

static JSequencer* sequencer = NULL;

//#define FILENAME_KEY 6624
//#define DEPTH_A_KEY 6625
//#define DEPTH_B_KEY 6626


void spaceball()
{
	if(!dsp) return;
	XEvent report;

	report.type = 0;
	if(XPending(dsp) <= 0) return;
	XNextEvent( dsp, &report ); //BLOCKING. BOO. Hopefully checking for pending events above keeps this from blocking
	if (!XCheckTypedEvent(dsp, ClientMessage, &report))
	{
		//no event
		return;
	}
	
	switch( report.type )
	{
	case ClientMessage :
		switch( MagellanTranslateEvent( dsp, &report, &MagellanEvent, 1.0, 1.0 ) )
		{
		case MagellanInputMotionEvent :
			MagellanRemoveMotionEvents( dsp );
			 printf( 
			  "x=%+5.0lf y=%+5.0lf z=%+5.0lf a=%+5.0lf b=%+5.0lf c=%+5.0lf   \n",
			  MagellanEvent.MagellanData[ MagellanX ],
			 MagellanEvent.MagellanData[ MagellanY ],
			  MagellanEvent.MagellanData[ MagellanZ ],
			  MagellanEvent.MagellanData[ MagellanA ],
			  MagellanEvent.MagellanData[ MagellanB ],
			  MagellanEvent.MagellanData[ MagellanC ] );

			ripParams->m_render->motion(MagellanEvent.MagellanData[ MagellanX ], -MagellanEvent.MagellanData[ MagellanZ ]);
			ripParams->m_render->spin(MagellanEvent.MagellanData[ MagellanA ]/100.0, MagellanEvent.MagellanData[ MagellanB ]/100.0);
			// XClearWindow( drw, drw );
			// XDrawString( drw, drw, wingc, 10,40,
			//MagellanBuffer, (int)strlen(MagellanBuffer) );
			// XFlush( display );
			//tz= MagellanEvent.MagellanData[ MagellanZ];
			
			break;

	
		switch( MagellanEvent.MagellanButton )
		{
		case 5: 
			MagellanSensitivity = MagellanSensitivity <= 1.0/32.0 ? 1.0/32.0 : MagellanSensitivity/2.0; break;
		case 6: 
			MagellanSensitivity = MagellanSensitivity >= 32.0 ? 32.0 : MagellanSensitivity*2.0; break;
		}
	
		default : // another ClientMessage event 
		break;
		};
	break;
	};
}

void loadModelIfChanged(char* newname)
{
	if(strncmp(newname, ripParams2->m_render->loadedFile, 1024))
	{
		ripParams2->m_render->config(newname);
		ripParams2->m_render->init();
		//load the new configuration!
		//TODO: verify that this doesn't blow anything up.
	}
}

void display(void)
{


	struct timeval tp;
	struct timezone tz;

	//uncomment to restore spaceball rotation (flaky?)
	//spaceball();





	//JB: window gets reset somehow. Let's try fixing before we draw
	//if ((holoRenParams->m_framebufferNumber % 2) == 0) {
	//  glutPositionWindow(0,hit);
	//}
	//glutReshapeWindow(wid,hit);


#ifdef SCREEN_SIZE_DIAGNOSTICS
	//Some JB Diagnostics:
	{
	int screenHeight = glutGet(GLUT_SCREEN_HEIGHT);
	int screenWidth = glutGet(GLUT_SCREEN_WIDTH);
	int windowWidth = glutGet(GLUT_WINDOW_WIDTH);
	int windowHeight = glutGet(GLUT_WINDOW_HEIGHT);
	int windowXPos = glutGet(GLUT_WINDOW_X);
	int windowYPos = glutGet(GLUT_WINDOW_Y);
		printf("At top of display function,\n");
		printf("The glut Current Window is %d\n", glutGetWindow());
		printf("Screen is %d by %d\n", screenWidth, screenHeight);
		printf("OpenGL window is %d by %d\n", windowWidth, windowHeight);
		printf("Window is located at %d, %d\n",windowXPos, windowYPos);
	}
#endif
	
	
	
#ifdef XSCREENDUMP	
	static int framect = 0;	
	system("xwd -display localhost:0.0 -root -screen -silent | xwdtopnm 2>/dev/null | pnmtopng > ~/screendump.png");
	printf("dumped screen to screendump.png\n");
	if (framect++ == 5) exit(0);
#endif	
	
	
	
	
//**** get current state from shared memory
		
//get state from UI & update model

#ifndef IGNORE_GUI
	sharedstate->getDataCopy(&statecopy);
	
	MASTER_GAIN = statecopy.gain;

	if(ripParams)
		{
			ripParams->m_render->models->orient->rotate[0] = statecopy.xrot;
			ripParams->m_render->models->orient->rotate[1] = statecopy.yrot;
			ripParams->m_render->models->orient->rotate[2] = statecopy.zrot;

			ripParams->m_render->models->orient->translate[0] = statecopy.xpos;
			ripParams->m_render->models->orient->translate[1] = statecopy.ypos;
			ripParams->m_render->models->orient->translate[2] = statecopy.zpos;
			if(ripParams->m_flatrender != statecopy.rendermode1)
			{
				ripParams->m_flatrender = statecopy.rendermode1;
				if(ripParams->m_flatrender == 0) //switch to RIP Method
				{
					ripParams->m_projectionPlaneDist = RIP_PROJ_PLANE;
					ripParams->recomputeGeometry(ripParams->m_holorenparams);
				}
				else //switch to planar method
				{
					ripParams->m_projectionPlaneDist = statecopy.flatdepth1;
					ripParams->recomputeGeometry(ripParams->m_holorenparams);
				}
			}
			if(ripParams->m_flatrender && (statecopy.flatdepth1 != ripParams->m_projectionPlaneDist))
			{

				ripParams->m_projectionPlaneDist = statecopy.flatdepth1;
				ripParams->recomputeGeometry(ripParams->m_holorenparams);
			}
		}
	if(ripParams2)
		{

			ripParams2->m_render->models->orient->rotate[0] = statecopy.xrot;
			ripParams2->m_render->models->orient->rotate[1] = statecopy.yrot;
			ripParams2->m_render->models->orient->rotate[2] = statecopy.zrot;

			//ripParams2->m_render->models->orient->translate[0] = statecopy.xpos;
			//ripParams2->m_render->models->orient->translate[1] = statecopy.ypos;
			//ripParams2->m_render->models->orient->translate[2] = statecopy.zpos;

			if(ripParams2->m_flatrender != statecopy.rendermode2)
				{
					ripParams2->m_flatrender = statecopy.rendermode2;
					if(ripParams2->m_flatrender == 0) //switch to RIP Method
					{
						ripParams2->m_projectionPlaneDist = RIP_PROJ_PLANE;
						ripParams2->recomputeGeometry(ripParams2->m_holorenparams);
					}
					else //switch to planar method
					{
						ripParams2->m_projectionPlaneDist = statecopy.flatdepth2;
						ripParams2->recomputeGeometry(ripParams2->m_holorenparams);
					}
				}
				if(ripParams2->m_flatrender && (statecopy.flatdepth2 != ripParams2->m_projectionPlaneDist))
				{

					ripParams2->m_projectionPlaneDist = statecopy.flatdepth2;
					ripParams2->recomputeGeometry(ripParams2->m_holorenparams);
				}
		}
#endif

	/*char newname[1024];
	if(sharedFilename->getDataCopyIfUnlocked(newname))
	{
		loadModelIfChanged(newname);
	}
	float newdepthA;
	float newdepthB;
	
	sharedDepthA->getDataCopyIfUnlocked(&newdepthA);
	
	if(newdepthA != ripParams->m_projectionPlaneDist)
	{
		ripParams->m_projectionPlaneDist = newdepthA;
		ripParams->recomputeGeometry(ripParams->m_holorenparams);
	}
	
	sharedDepthB->getDataCopyIfUnlocked(&newdepthB);

	if(newdepthB != ripParams2->m_projectionPlaneDist)
	{
		ripParams2->m_projectionPlaneDist = newdepthB;
		ripParams2->recomputeGeometry(ripParams2->m_holorenparams);
	}
	*/
	
//**** render
	

	
	//black the color buffer 
    glClearColor(0.0,0.0,0.0,0.0);
    glClear (GL_COLOR_BUFFER_BIT);	
	
	sequencer->update();
	
	
	if(ripParams2 != NULL)
	{
		glPushAttrib(GL_ALL_ATTRIB_BITS);
		ripParams2->DisplayRIPHologramSingleFramebuffer(holoRenParams2);
		glPopAttrib();
	}	
	
	glPushAttrib(GL_ALL_ATTRIB_BITS);
	// calls RIPHologram::DisplayRIPHologramSingleFramebuffer (in RIP.cpp)
	ripParams->DisplayRIPHologramSingleFramebuffer(holoRenParams);
	glPopAttrib();
	
	
	glFlush();
	glutSwapBuffers();
	

	
	
	//JB: Make it move
#ifdef SPIN_OBJECT_HACK
	if(freeSpin)
	{
		if(ripParams) ripParams->m_render->spin(2, 0);
		if(ripParams2)	ripParams2->m_render->spin(-1.5,0);
		
	}
#endif
	
	
	
	
	//warp pointer back to center of screen if it isn't already there.
	int cx, cy;
	
#ifndef LOCKED_UI
	cx = glutGet(GLUT_WINDOW_WIDTH)/2;
	cy = glutGet(GLUT_WINDOW_HEIGHT)/2;
	if(ripParams->m_render->mLX != cx || ripParams->m_render->mLY != cy)	  
	{
		glutWarpPointer(cx,cy);
		ripParams->m_render->mLX = cx;
		ripParams->m_render->mLY = cy;
	}
#endif
	glutPostRedisplay(); //JB: This causes glut to continue to repeatedly execute display()

#ifdef TIMING_DIAGNOSTICS
{
gettimeofday(&tp, &tz);
printf("time now is: %ld sec %ld usec \n", tp.tv_sec, tp.tv_usec);
printf("time now is: (tp.tv_sec*100000)/100000)\n");
	}
#endif

#ifdef SCREEN_SIZE_DIAGNOSTICS

	int screenHeight = glutGet(GLUT_SCREEN_HEIGHT);
	int screenWidth = glutGet(GLUT_SCREEN_WIDTH);
	int windowWidth = glutGet(GLUT_WINDOW_WIDTH);
	int windowHeight = glutGet(GLUT_WINDOW_HEIGHT);
	int windowXPos = glutGet(GLUT_WINDOW_X);
	int windowYPos = glutGet(GLUT_WINDOW_Y);
		printf("After buffer swap,\n");
		printf("The glut Current Window is %d\n", glutGetWindow());
		printf("Screen is %d by %d\n", screenWidth, screenHeight);
		printf("OpenGL window is %d by %d\n", windowWidth, windowHeight);
		printf("Window is located at %d, %d\n",windowXPos, windowYPos);

#endif



		
		
#ifdef WRITE_FB_TO_FILE

		
	unsigned char *pic;
	unsigned char *bptr;
	FILE *fp;

	int allocfail = 0;

	printf("attempting to write framebuffer to file");
	if ((pic = (unsigned char*)malloc(screenWidth*screenHeight*3 * sizeof(unsigned char))) == NULL) {
		printf("couldn't allocate memory for framebuffer dump.\n");
		allocfail = 1;
	}
	if ( !allocfail) {
		char fname[255];
		glReadBuffer(GL_FRONT);
		bptr = pic;
		glReadPixels(0, 0, screenWidth, screenHeight, GL_RGB, GL_UNSIGNED_BYTE, pic);
		printf("saving %dx%dx3 to file...\n", screenWidth,screenHeight);

		sprintf(fname, "FBcontents%d.raw", holoRenParams->m_framebufferNumber);
		if ( (fp = fopen (fname, "w")) == NULL) {
			printf("failure opening file.\n");
			exit(0);
		}
		else {
			if (fwrite (pic, 1, 3*screenWidth*screenHeight, fp) != screenWidth*screenHeight*3) {
				printf("failure writing file.\n");
				//exit(0);
			}
			fclose (fp);
		}
		free (pic);
	}
	exit(1);	
#endif

}

//called by GLUT on mouseup and mousedown
void mouse(int button, int state, int x, int y)
{
	// calls holoConf::mouse which sets cursor state and mousebutton state
	ripParams->m_render->mouse(button, state, x, y);
}

//called by GLUT on mouse move (buttons up only)
//keep mouse at middle of screen
void passiveMotion(int x, int y)
{
	//TODO: check for recursive events? Dead zone?
#ifndef LOCKED_UI
	glutWarpPointer(holoRenParams->m_xRes/2,holoRenParams->m_yRes);
#endif
}

//called by GLUT on drag
void motion(int x, int y)
{
	// calls holoConf::motion() which appears to
	// translate the scene based on mouse movement.
	ripParams->m_render->motion(x, y);
}

void keyboard(unsigned char key, int x, int y) {
	printf("got key \"%c\" on screen %d\n", key, holoRenParams->m_framebufferNumber);
	sequencer->keypress(key);

#ifndef LOCKED_UI
	switch(key) {
		case 'q':
		case 'Q':
			exit(0);
			break;
		case 'a':
			fringeToUse ++;
			if(fringeToUse >= NUM_PRECOMPUTED_FRINGES)
				fringeToUse = NUM_PRECOMPUTED_FRINGES-1;
			printf("using fringe %d\n", fringeToUse);
			ripParams->m_fringeToUse = fringeToUse;
			glutPostRedisplay();
			break;
			
		case 'A':
			fringeToUse--;
			if(fringeToUse < 0)
				fringeToUse = 0;
			printf("using fringe %d\n", fringeToUse);
			ripParams->m_fringeToUse = fringeToUse;
			glutPostRedisplay();
			break;		
		case ' ':
			freeSpin = !freeSpin;
			glutPostRedisplay();
			break;
		case 'd':
			ripParams->m_projectionPlaneDist += 1.0;
			printf("projection plane now at %g\n", ripParams->m_projectionPlaneDist);
			ripParams->recomputeGeometry(ripParams->m_holorenparams);
			break;
		case 'D':
			ripParams->m_projectionPlaneDist -= 1.0;
			printf("projection plane now at %g\n", ripParams->m_projectionPlaneDist);
			ripParams->recomputeGeometry(ripParams->m_holorenparams);
			break;			
		case 'l' :
			//if(ismaster)
			{
				char fname[1024] = "/models/letters/SLOAN/C.xml";
				sharedFilename->write(fname);
				loadModelIfChanged(fname);
			}
			break;
		case 'k' :
			//if(ismaster)
			{
				char fname[1024] = "/models/letters/SLOAN/K.xml";
				sharedFilename->write(fname);
				loadModelIfChanged(fname);
			}
			break;
	};
	// calls holoConf::keyboard() which adds other commands:
	// + = +ztranlate
	// - = -ztranslate
	// 4 = -xtranslate
	// 6 = +xtranslate
	// 8 = -ytranslate
	// 2 = +ytranslate
	ripParams->m_render->keyboard(key, x, y);
#endif
}


//JB: for testing on desktop display:


void reshape(int w, int h) {
	glViewport(0,0,(GLsizei)w, (GLsizei)h);
	glutPostRedisplay();
}

// setup UI
// uses structs holoRenParams.
void InitGL(int &argc, char **&argv, HoloRenderParams *hrP, RIPHologram *ripP, HoloRenderParams *hrP2, RIPHologram *ripP2)
{
	holoRenParams = hrP;
	ripParams = ripP;

	holoRenParams2 = hrP2;
	ripParams2 = ripP2;
	
	printf("initializing GL\n");
	glutInit(&argc, argv);
	
	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH | GLUT_STENCIL | GLUT_ACCUM);
	
	
	//instance on framebuffer 0 gets to interact with keyboard & send events to other instances.
	if(holoRenParams->m_framebufferNumber == 0)
	{
		ismaster = true;
	}
		else
	{
		ismaster = false;
	}

	//state for slaving to separate UI
	sharedstate = new JSharedMemory(sizeof(JDisplayState),ALL_STATE_KEY);
	sharedstate->getDataCopy(&statecopy);

	/*
	sharedFilename = new JSharedMemory(1024,FILENAME_KEY);
	sharedDepthA = new JSharedMemory(sizeof(float),DEPTH_A_KEY);
	sharedDepthB = new JSharedMemory(sizeof(float),DEPTH_B_KEY);
	*/
	/*
	char f[1024] = "/models/blankframe.xml";
	sharedFilename->write(f);
	//initalize z-distance to first & second hologram.
	float pdA = 10;
	float pdB = 128;
	
	sharedDepthA->write(&pdA);
	sharedDepthB->write(&pdB);
	*/
	//TODO: get subject ID!
	


	//	glutInitDisplayMode(GLUT_DOUBLE | GLUT_RGBA | GLUT_DEPTH);
	
	
	//glutInitDisplayString("xvisual=250");//doesn't seem to work
	//glutInitDisplayString("depth=>1 rgba>1");//doesn't seem to work

	
	
	// set up window size to be (these are hardcoded in RIP.h) 2048x(2*1780)
	//(2 vertically stacked "framebuffers" worth per window)
	glutInitWindowSize(holoRenParams->m_xRes,holoRenParams->m_yRes*2);
	//glutInitWindowSize(2048,holoRenParams->m_yRes*2);
	
	// TODO: Update block comment for version 3.0.
	// Bottom line: each PC has 2 framebuffers, one window.
	// PCnumber = floor(framebufferNumber/2) has one window, with framebuffer
	// number PCnumber*2+1 on top and PCnumber*2 on bottom half of screen.
	// previously, we had one window per framebuffer, but the two windows on the
	// same video card liked to fight.

	// WJP's note from pre-3.0 version:
	// In holocomputation, we assume that hololines are apportioned
	// in logical order in FB0, FB1, FB2, FB3, FB4 and FB5, and comments
	// throughout the code reflect this. FB0 gets hololines 0,1,2,  18,19,20,...
	// and FB1 gets hololines 3,4,5,  21,22,23... and so on. And the
	// scripts: /home/holovideox/local/framebuffer1 and framebuffer2 return
	// 0 and 1 on holovideo1, 2 and 3 on holovideo2, and 4 and 5 on holovideo3
	// respectively.
	// Here's what we've learned about the way framebuffers on
	// holovideo1, holovideo2, holovideo3 map to holoLines.
	// on holovideo1, framebufferNumber 1 has the top 3 hololines.
	//                       framebufferNumber 0 has the next 3 hololines.
	// on holovideo2, framebufferNumber 3 has the next 3 hololines.
	//                       framebufferNumber 2 has the next three lines.
	// on holovideo3, framebufferNumber 5 has the next three lines, and
	//                       framebufferNumber 4 has the bottom three lines.
	// our software assumes that lines are arranged from top to bottom
	// in framebufferNumber 0 - 5. So, we swap the position of the output
	// windows for FBs (0and1) (2and3) (4and5) in UI.cpp,
	// so that FB0's output window is at (0,m_viewYRes) and FB1's is at (0,0).
	// I hope this is understandable.

	//JB removed for version 3.0
	//if((hrP->m_framebufferNumber % 2) == 1)
	//{
	glutInitWindowPosition(0, 0);
	//} else {
	// glutInitWindowPosition(0, hrP->m_yRes);
	//}

	// specifies display, keyboard, mouse, motion callacks,
	// configures window and turns OFF the cursor!
	int winnum = glutCreateWindow("HoloLoadExec"); //title is hint to window manager to not decorate window
	GLenum glewerr = glewInit();
	if(GLEW_OK != glewerr)
	{
		printf("Glew Init Failed! (%s) \n", glewGetErrorString(glewerr));
	}
	glutDisplayFunc(display);


	GLuint numgroups,numbars;
	int ret;



	#ifdef USE_HARDWARE_GENLOCK
	//Code block to turn on buffer swap syncronization using genlock hardware
	ret = glXQueryMaxSwapGroupsNV(dsp,drw,&numgroups, &numbars);
	if (!ret) printf("Couldn't query swap group info\n");
	ret = glXJoinSwapGroupNV(dsp,drw,1);//Make our drawable part of framelock group zero
	if (!ret) printf("Failed to start swap group\n");
	ret = glXBindSwapBarrierNV(dsp,1,1);//make framelock group zero part of barrier zero
	if (!ret) printf("Failed to start swap barrier\n");
	printf("System supports %d groups and %d barriers\n", numgroups, numbars);
	#endif
	
	

	#ifdef WORKSTATION_DEBUG
	glutReshapeFunc(reshape);
	#endif

	
	
	glutKeyboardFunc(keyboard);
	glutMouseFunc(mouse);
	glutMotionFunc(motion);
	glutPassiveMotionFunc(passiveMotion);
	//glutFullScreen();

	glutSetCursor(GLUT_CURSOR_NONE);
	//TODO: This will probably need to change with three wimultaneous windows...
	glutWarpPointer(glutGet(GLUT_WINDOW_WIDTH)/2, glutGet(GLUT_WINDOW_HEIGHT)/2); // move mouse to center of display

	
	
	//Set up Spaceball/SpaceNavigator support
	GLXDrawable drw = glXGetCurrentDrawable();
	dsp = glXGetCurrentDisplay();

	if ( !MagellanInit( dsp, drw ) )
	{
		fprintf( stderr, "Can't find the spaceball driver. try running:\n sudo /etc/3DxWare/daemon/3dxsrv -d usb \n" );
	}

	//testing GLX capabilities...
	
	int vid = glutGet(GLUT_WINDOW_FORMAT_ID);
	printf("Using openGL visual #%d. See visualinfo for more information\n", vid);

	
	XVisualInfo* newVisual = NULL;
	int visattrib[] = {GLX_RGBA, GLX_RED_SIZE, 64};
	newVisual = glXChooseVisual(dsp, 0, visattrib);
	
	
	sequencer = new JSequencer("HoloLayerLetter", 0, holoRenParams->m_framebufferNumber, hrP, ripP, hrP2, ripP2);
	
	
	//Some JB Diagnostics:
#ifdef SCREEN_SIZE_DIAGNOSTICS
	{
	int screenHeight = glutGet(GLUT_SCREEN_HEIGHT);
	int screenWidth = glutGet(GLUT_SCREEN_WIDTH);
	int windowWidth = glutGet(GLUT_WINDOW_WIDTH);
	int windowHeight = glutGet(GLUT_WINDOW_HEIGHT);
	int windowXPos = glutGet(GLUT_WINDOW_X);
	int windowYPos = glutGet(GLUT_WINDOW_Y);
	printf("Before starting main loop,\n");
	printf("glutCreateWindow returned window ID %d\n", winnum);
	printf("The glut Current Window is %d\n", glutGetWindow());
	printf("Screen is %d by %d\n", screenWidth, screenHeight);
	printf("OpenGL window is %d by %d\n", windowWidth, windowHeight);
	printf("Window is located at %d, %d\n",windowXPos, windowYPos);
	printf("swapped buffers: you should SEE something on the display now...\n");
	}
#endif
}

//I don't think this ever gets called (JB);
void CloseGL()
{
//    glXDestroyContext(dpy, cx);

//    XDestroyWindow(dpy, win);
//    XCloseDisplay(dpy);
}


