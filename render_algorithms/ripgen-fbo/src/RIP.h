#ifndef _RIP_H
#define _RIP_H

#include <math.h>
#include <tiffio.h>
#include "framebufferObject.h"

//JB's debug flags. define these to enable different debug modes


#define LOCKED_UI

//#define IGNORE_GUI

//new for 3.0 version

//pixel format of fringe texture
#define FRINGE_FORMAT GL_LUMINANCE32F_ARB
//#define FRINGE_FORMAT GL_LUMINANCE

//pixel format of parallax view stack texture
#define PARALLAX_STACK_FORMAT GL_LUMINANCE32F_ARB
//#define PARALLAX_STACK_FORMAT GL_RGBA16F_ARB
//#define PARALLAX_STACK_FORMAT GL_LUMINANCE

#define HOLOGRAM_FORMAT GL_RGBA32F_ARB
//#define HOLOGRAM_FORMAT GL_RGBA


#define NUM_PRECOMPUTED_FRINGES 256.0

//In testing on laptop, when rendering views into a 3D texture via framebuffer objects, framebuffer object refuses to attach 3D textures
//that have more than 128 z-slices (perhaps total texture size limit?) 
#define MAX_VIEWS_PER_FBO_TEXTURE 256


//Solos: MUST define all if solos on. This may not be complete.
//#define SOLO
//only populate texture stack for one solo hololine within each block
//#define SOLO_LINE 1

//only populate texture stack for one solo hololine block
//#define SOLO_BLOCK 3

//only populate texture stack for one framebuffer
//#define SOLO_FRAMEBUFFER 1

//#define SOLO_BLANK_COLOR 0

//make windows smaller so they fit on a pc monitor for debugging
//#define WORKSTATION_DEBUG


//print diagnostic info about screen size
//#define SCREEN_SIZE_DIAGNOSTICS

//Use genlocking hardware to sync machines (Use me if animating display)
#define USE_HARDWARE_GENLOCK

//extra printing to diagnose genlock/swap group features, also reloads genlock data every frame (See RIP.cpp)
//#define GENLOCK_DEBUG

//rotation hack: spin object to test frame rate
//#define SPIN_OBJECT_HACK

//wobble model hack (happens inside renderer so view-stack is a little skewed)
//#define MODEL_ANIMATE

//extra printing for 3d view texture allocation
//#define PRINT_3D_TEXTURE_DIAGNOSTICS


//older 2.0 flags:
//set random jitter to zero of holopoint locations when NO_RANDOM_JITTER is defined
#define NO_RANDOM_JITTER

//write out the projector fringe in RIP.cpp
//#define WRITE_PROJECTOR_FRINGE

//write whole framebuffer to file when WRITE_FB_TO_FILE is defined. Also define PRINT_3D_TEXTURE_DIAGNOSTICS.
//#define WRITE_FB_TO_FILE

//Populate hololines with a low density grid of points
//#define LOW_HOLOPOINT_DENSITY

//display fringes only
//#define UNMODULATED_FRINGES

//dump the output of the renderer to a bunch of files
//#define DUMP_ALL_VIEWS

//more if 0 code. Define as 0 to both set up and activate view texture.
//#define SKIP_VIEW_TEXTURE

//define NO_GENLOCK_FIX to omit loading genlock calibration variables (See RIP.cpp)
//#define NO_GENLOCK_FIX

#define RIP_PROJ_PLANE 4.0


//hardware genlock API not available on apple
#ifdef __APPLE__
#undef USE_HARDWARE_GENLOCK
#endif

struct HoloVideoParams
{
	static const int nLines = 144;
//TODO: See also renderer.cpp renderconf.screenY (DUPE)
	static const int lineLength = 262144;  //JB: Length of one hololine in pixels == 2^18 (legacy from cheops framebuffer size, but still use)
};

struct holoConf;
class HoloRenderParams;

class RIPHologram
{
public:
	//****************************PUBLIC MEMBERS
	holoConf *m_render;
	HoloRenderParams *m_holorenparams;
	
	bool m_textures_created; // has AfterGLInitialize been called?

	bool m_flatrender;
	
	bool m_useStaticImages;
	char m_baseStaticImageFilename[512];	
	
	int m_viewXRes; //in pixels
	int m_viewYRes; //in pixels
	int m_viewFBOTexYRes;
	int m_viewSize; //in pixels
	int m_nXViews;

	float m_hologramWidth; //in mm

	float m_projectionPlaneDist; //in mm
	float m_projectionPlaneWidth; //in mm
	float m_projectionPlaneHeight; //in mm

	int m_fringeToUse; //when we pre-render many fringes, index into this stack
	
	float m_diffractionAngle; //in degrees
	float m_parallaxAngle; //in degrees
	float m_referenceAngle; //in degrees
	float m_referenceLambda; //in nm
	float m_E0; //in ??

	float m_viewZoneWidth; //in mm
	float m_viewZoneDist; //in mm from projection plane

	float m_viewsPerHoloPixel;
      float m_samplesPerHoloPixel;
      float m_holoPixelSpacing;
      float m_projectorFringeFootprint;
      float m_samplesPerHoloPixelSpacing;
	float m_samplesPerHoloLine;
      float m_samplesPerMM;
      float m_holoPixelSpacingJitterFraction;

	GLuint m_fringeTexID;

	//GLuint m_parallaxStackTexID[2]; //2 parallax stacks per process (2 cards)
	GLuint m_parallaxStackFBOTexID; //all views. Replaces the view-per-framebuffer 3d texture m_parallaxStackTexID

	FramebufferObject* m_viewtexFBO;
	
	//for rendering hologram to texture
	GLuint m_hologramFBOTexID;
	FramebufferObject* m_hologramFBO;
	
	float m_fringeTexCoordScale;
	float m_fringeTexWidth;
	float m_fringeTexHeight;
	float m_fringeTexLastline;

	unsigned m_viewTexXRes; //in pixels (pow of 2)
	unsigned m_viewTexYRes; //in pixels (pow of 2)
	unsigned m_viewTexSize; //in pixels (pow of 2)
	unsigned m_nTexXViews; //(pow of 2), default number of view images to read from files

	float m_parallaxStackTexNXViewsScale;
	float m_parallaxStackTexViewXScale;

//	float m_parallaxStackTexViewYScale;
	float m_parallaxStackFBOTexViewYScale;

	//****************************PUBLIC METHODS
	RIPHologram(HoloRenderParams *holoRenParams, holoConf *ren);
	void AfterGLInitialize();
	void DisplayRIPHologramSingleFramebuffer(HoloRenderParams *holoRenParams);
    void blackmanWindow ( int nf, float *w, int n, int ieo );
    void kaiserWindow ( int nf, float *w, int n, int ieo,  float beta);
    float  kaiserBessel (float x);

	void setDefaultGeometry(HoloRenderParams *holoRenParams);
	void recomputeGeometry(HoloRenderParams *holoRenParams);
protected:
	//************************PROTECTED METHODS
	void InitializeRendererConfig();
	void BuildFringe();
	void BuildMultilineFringe();
	void BuildFringeSet(int count);
	void BuildParallaxStack(HoloRenderParams *holoRenParams);
	void InitializeForDisplay(HoloRenderParams *holoRenParams);
	void AllocateViewTexture(HoloRenderParams *holoRenParams);
	void DrawTexturedQuad(GLuint tex);
	void CleanupAfterDisplay();
	void DisplayHologram(HoloRenderParams *holoRenParams);
	
	void DrawHoloPixel(int reverseLine, int holoPixelIndex, int line, float startViewNumber, 
					   float endViewNumber, int holoPixelXStart, int holoPixelYStart, int holoPixelXEnd, int holoPixelYEnd, int linesPerFB, int modulationFlag);
	
	void RenderHoloPixel(int holoPixelIndex, int line, float startViewNumber, 
						 float endViewNumber, int holoPixelXStart, int holoPixelYStart, int holoPixelXEnd, int holoPixelYEnd, float startFringeTexCoord, float endFringeTexCoord, int linesPerFB);
    
	void RenderUnmodulatedHoloPixel(int holoPixelIndex, int line,
									int holoPixelXStart, int holoPixelYStart, int holoPixelXEnd, int holoPixelYEnd, float startFringeTexCoord, float endFringeTexCoord);
	
	void loadViewSet(char* basename);
	
	void drawMultilineHolopixel(bool revScan, int startXsample, int startYsample, float tx, float ty, float tz);
};



class HoloRenderParams
{
 public:
    int m_xRes;
	int m_xActive;
    int m_yRes;
    int m_yStartOffset;
    int m_nHBlankLines;
    int m_nFramebuffers;
    int m_framebufferNumber;
    int m_screenLinesPerHoloLine;
    int m_screenDataLinesPerHoloLine;
    int m_nHoloLinesPerFramebuffer;

    HoloRenderParams(int framebufferNumber)
        {
            //JB note: This is the size of a framebuffer, but ea GL window is xRes*(2*yRes) spanning 2 framebuffers

			//on quadro cards, we had 2032 x 1778 active pixels out of 2048 x 1788 pixels clocked out to display. Need to look through these to make sure they get used correctly.
			//On 200-series cards, we can display 2046 x 1786 out of 2048x1788
            //m_xRes = 2032;//2048;
            //m_yRes = 1790;//1780; // (128+50) * 8 + 344

			//m_nFramebuffers = 6;
            //m_yStartOffset = 356; //not 344?
            //m_nHBlankLines = 2;//50; //not 48?
            
			m_xRes = 2048;
			m_xActive = 2045; //VGA on Q5800 or Q6000
//			m_xActive = 2032; //DisplayPort on K5000

			m_yRes = 1757;//1744; // (128+50) * 8 + 344  //VGA on Q5800 or Q6000
//			m_yRes = 1722; //DisplayPort on K5000

            m_nFramebuffers = 6;
            m_yStartOffset =204 ;//358;//390;//344;//185;//356; //not 344? // use function generator phase to tweak. Tuned to 90 degree offset.
            m_nHBlankLines = 48;


            #ifdef WORKSTATION_DEBUG
			{
				//m_xRes = 512;
            	//m_yRes = 445;
            	//quarter size is 512 x 445
			
            	m_xRes = 1280;
            	m_yRes = 1024;
            

            	m_nFramebuffers = 6;
            	m_yStartOffset = 356; //not 344?
            	m_nHBlankLines = 50; //not 48?
			}
			#endif

			m_framebufferNumber = framebufferNumber;
			m_screenLinesPerHoloLine = (HoloVideoParams::lineLength) / (m_xRes) + m_nHBlankLines;
			printf("computed %d screen lines per hololine\n", m_screenLinesPerHoloLine);
            m_screenDataLinesPerHoloLine = m_screenLinesPerHoloLine - m_nHBlankLines;
            m_nHoloLinesPerFramebuffer = HoloVideoParams::nLines/m_nFramebuffers;
        }
};

void RIPDisplayHologram(HoloRenderParams *holoRenParams, RIPHologram *ripParams);

#endif
