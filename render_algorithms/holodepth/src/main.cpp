/* 09_vertex_lighting.c - OpenGL-based per-vertex lighting example
 using Cg program from Chapter 5 of "The Cg Tutorial" (Addison-Wesley,
 ISBN 0321194969). */

/* Requires the OpenGL Utility Toolkit (GLUT) and Cg runtime (version
 1.5 or higher). */

//for profiling, can auto-quit after set number of seconds (try 60). set to 0 to disable auto-quit
#define QUIT_AFTER_SECONDS 0

//#define USE_GLEW 0
//this disables view rendering from mesh. Not currently used.
#define DISABLE_VIEW_RENDERING 0
#define HOLOGRAM_DOWNSCALE_DEBUG 0 

#define REVERSE_VIEW_ORDER 1

#define MODEL_TO_USE 0 //0,1 or 2
#define DISABLE_SECOND_OBJECT 1
#define DRAW_WIRE_CUBE_ONLY 0
//#define ENABLE_REMOTEQT

#define DISABLE_HOLOGRAM_CREATION 0
#define WRITE_VIEWS_AND_EXIT 0
#define	WRITE_LUMA_VIEW_FILES 0
#define WRITE_HOLOGRAM_AND_EXIT 0

//enable both to use kinect. Disable both for cube-only or obj-file modes.
//#define VIEWS_FROM_CLOUD

//(M_PI/20), (M_PI/200)
#define ANGLE_THRESH (M_PI/20)
//16, 100
#define VIEWS_TO_LOAD 16

//SET TO 0 for images loaded from file
//set to 2 for kinect demos
//set to 3 to use dual-kinect setup
//set to 4 for PointCloudLibrary-based kinect driver
#define KINECT_MODE 0

//don't draw cloud samples darker than this value (out of 2048). Set to 0 to remove check. 300 for crane/arizona
#define REMOVE_PIXELS_DARKER_THAN 300
//for debugging hologram shader:
//#define SOLO_VIEW 16 //only draw one view. undefine to show all views

//DECIMATE of n means only use every n+1 points (0 means no skipping)
#define OPENNI_DECIMATE 0

#ifdef __APPLE__
	//#define WRITE_VIEWS_AND_EXIT 1
	//#define KINECT_MODE 0
	//#define VIEWS_FROM_CLOUD
	#define DRAW_WIRE_CUBE_ONLY 1
	//#define	WRITE_LUMA_VIEW_FILES 1
	#define DISABLE_VIEW_RENDERING 0
	#define DISABLE_HOLOGRAM_CREATION 0
	#define WRITE_HOLOGRAM_AND_EXIT 0
	#define HOLOGRAM_DOWNSCALE_DEBUG 0 //fit hologram onto smaller screen for debugging overall layout.
#endif

#ifdef __APPLE__
	#define IMAGE_PATH_TEMPLATE "/Users/barabas/Dropbox/James Storage/Range Images/Images/%s/capture_batch_%.4d/testCloud0000.txt"
#else
	#define IMAGE_PATH_TEMPLATE "/home/holo/Range_Images/Images/%s/capture_batch_%.4d/testCloud0000.txt"
#endif

#if KINECT_MODE == 4
#include "JVertexRender.h"
	static JVertexRender *KinectPCL;
#endif

#ifdef WIN32
#include <Windows.h>
#undef near
#undef far
#undef	BI_RGB
#undef	BI_RLE8
#undef	BI_RLE4
#undef	BI_BITFIELDS
#define M_PI       3.14159265358979323846
#endif

#ifdef __APPLE__
	#include <OpenGL/gl.h>
	#include <GLUT/glut.h>
#else
	#if USE_GLEW
		#include <GL/glew.h>
	#endif
	#include <GL/gl.h>
	#include <GL/glu.h>
	#include <GL/glut.h>
#endif

#include <stdio.h>    /* for printf and NULL */
#include <stdlib.h>   /* for exit */
#include <math.h>     /* for sqrt, sin, cos, and fabs */
#include <assert.h>   /* for assert */

#include <vector>

#include <Cg/cg.h>    /* Can't include this?  Is Cg Toolkit installed! */
#include <Cg/cgGL.h>

#include "myobj.h"
#include "transforms.h"
#include "mytexture.h"
#include "shapes.h"

#ifdef REMOTEQT_GUI
	#include "JDisplayState.h"
	#include "JSharedMemory.h"
	//For interfacing with QTRemote UI
	static JDisplayState statecopy;
	static volatile JDisplayStatus *displaystatus;

	static JSharedMemory *sharedstate; //controller to this process
	static JSharedMemory *sharedstatus; //this process to controller
#endif

#if KINECT_MODE > 0
	#include "JKinectFrame.h"
	static JSharedMemory *sharedkinect;
	static JKinectFrame *kinectframe;

#if (KINECT_MODE > 1)
	//for Kreylos Kinect code
	#include "KinectProjector.h"
	#include "GL/GLContextData.h"
	#include <GL/GLExtensionManager.h>
	#include "Vrui/Vrui.h"
	// 04/09/2011 SKJ: Add includes
	#include <Vrui/Viewer.h>
	#include <Vrui/Application.h>
	#include <Misc/FunctionCalls.h>
	#include <Misc/File.h>
	#include <Threads/TripleBuffer.h>
	#include <Geometry/OrthogonalTransformation.h>
	#include <Geometry/GeometryValueCoders.h>
	#include <GL/gl.h>
	#include <GL/GLGeometryWrappers.h>
	#include <GL/GLTransformationWrappers.h>
	KinectProjector *kprojector;
	GLContextData *projectorDataContext;
	GLExtensionManager *glextmgr;
	#if KINECT_MODE != 3
		//#define PROJECTION_CONFIG_FILENAME "/home/holo/Dropbox/Holovideo/Configuration/Kinect1/CameraCalibrationMatrices-B00361710470044B.dat"
		#define PROJECTION_CONFIG_FILENAME "/home/holo/Dropbox/Holovideo/Configuration/Kinect3/CameraCalibrationMatrices-A00362A06614047A.dat"
	#endif
	FrameBuffer *depthFrameBuffer;
	FrameBuffer *colorFrameBuffer;
	int kinectCloudHeight=480;
	int kinectCloudWidth=640;
	// 04/09/2011 SKJ: Added to handle second Kinect
	#if KINECT_MODE == 3
		static JSharedMemory *sharedkinect2;
		static JKinectFrame *kinectframe2;
		KinectProjector *kprojector2;
		GLContextData *projectorDataContext2;
		// GLExtensionManager *glextmgr2;
		#define PROJECTION_CONFIG_FILENAME_2 "/home/holo/Dropbox/Holovideo/Configuration/Kinect2/CameraCalibrationMatrices-A00364817648045A.dat"
		#define PROJECTION_CONFIG_FILENAME "/home/holo/Dropbox/Holovideo/Configuration/Kinect1/CameraCalibrationMatrices-B00361710470044B.dat"
		FrameBuffer *depthFrameBuffer2;
		FrameBuffer *colorFrameBuffer2;
		static float depthMatrixKinect1;
		static float textureMatrixKinect1;
		static float depthMatrixKinect2;
		static float textureMatrixKinect2;
		static float projectorTransform1;
		static float projectorTransform2;
		static int kinectNum;
	#endif
#endif
#endif

#ifdef VIEWS_FROM_CLOUD
#include "JZCameraCloud.h"
#endif


//#include "framebufferObject.h"
//#include "renderbuffer.h"
#include "glErrorUtil.h"

static float MasterHologramGain = 1.0f;

int ViewEnableBitmask = 0xffffffff;  //set bits to zero to skip rendering views 
bool zeroDepth = 0; //should we set depth-view to zero (for debuging)
bool zeroModulation = 0; // " " " color
float fakeZ = 0.5; //pretend all points on object is at this z-depth
float fakeModulation = 0.5; //pretend all points on object is at this z-depth
int hologramOutputDebugSwitch = 0; //send different intermediate variables to color buffer when rendering
#define hologramOutputDebugSwitches 10 //total number of debug modes in shader

/* Scalar */
GLuint DLid;
int headnumber = 0; //0,1,2 -- which instance of three (3 graphics cards)
int frame = 0, thetime, timebase = 0;
int rotateOn = 0;
int drawdepthOn = 0;

//dimensions of the framebuffer for output to video signals
int imwidth = 2048;
int imheight = 3444; //1722*2; //new DisplayPort mode
//int imheight = 3514;// 1757*2.; //old VGA mode

int MarkIIGLWindowWidth = 2032; //New DisplayPort Mode
//int MarkIIGLWindowWidth = 2045; //old VGA mode

int MarkIIGLWindowHeight = 3444; //1722*2; //This is display height for DisplayPort connected displays on K5000 cards
//int MarkIIGLWindowHeight = 3514; //1757*2; //This is display height for onboard VGA connector displays

#if WRITE_LUMA_VIEW_FILES != 1
	int numx = 512;//512;//horizontal resolution of view to render (also number of emitters per line in hologram), 200 Arizona mode
	int numy = 144;//144//48;//vertical resolution of views rendered. 48 for single-head tests. 144 for 3-head mode, 600 Arizona mode
	int tiley = 2;//2//tiling of views in view texture
	int tilex = 2;//2 
	int numrgb = 4;//4//number of views per pixel (stored in R, G, B, A)
#else
	int numx = 400;//512;//horizontal resolution of view to render (also number of emitters per line in hologram), 200 Arizona mode
	int numy = 1200;//144//48;//vertical resolution of views rendered. 48 for single-head tests. 144 for 3-head mode, 600 Arizona mode
	int tiley = 100;//2//tiling of views in view texture
	int tilex = 1;//2 
	int numrgb = 1;//4//number of views per pixel (stored in R, G, B, A)
#endif

//size of texture used for writing views
int VIEWTEX_WIDTH = numx * tilex;
int VIEWTEX_HEIGHT = numy * tiley * 2;//tiley views of numy pixels, X 2 (depth, luminance) was 256;

GLubyte *localFramebufferStore;

int numview=tilex*tiley*numrgb;
#if	WRITE_LUMA_VIEW_FILES !=1
	float HologramPlaneWidth_mm = 150.0;
	float HologramPlaneHeight_mm = 75.0;
#else //ARIZONA plane dimensions
	float HologramPlaneWidth_mm = 100.0;
	float HologramPlaneHeight_mm = 100.0;
#endif
float near = 400.0f, far = 800.0f; //mm, near and far planes for hologram volume (relative to camera)
float lx = 0, ly = -790, lz = -110; //Light location (direction?) inside volume
//int imwidth=2048;
//int imheight=1780;

//int numx=320, numy=120, tile9.coy=2, numrgb=4;
//5/int numview=64; /*numview=numx*tily*numrgb=320*2*4=2560'*/
float mag = 1.;
float fov = 30. /* /7 */;

float hogelSwitch = 1.;

//float fov=30;

GLfloat LightAmbient[] =
{ 0.5f, 0.5f, 0.5f, 1.0f }; // Ambient Light Values
GLfloat LightDiffuse[] =
{ 1.0f, 1.0f, 1.0f, 1.0f }; // Diffuse Light Values
GLfloat LightPosition[] =
{ 0.0f, 0.0f, 2.0f, 1.0f }; // Light Position

static CGcontext normalMapLightingCgContext;
static CGprofile normalMapLightingCgVertexProfile;
static CGprofile normalMapLightingCgFragmentProfile;
static CGprogram normalMapLightingCgVertexProgram, normalMapLightingCgFragmentProgram;

static CGparameter myCgVertexParam_modelUIScale;

static CGparameter myCgVertexParam_modelViewProj;
static CGparameter myCgFragmentParam_globalAmbient,
myCgFragmentParam_lightColor, myCgFragmentParam_lightPosition,
myCgFragmentParam_eyePosition, myCgFragmentParam_Ke,
myCgFragmentParam_Ka, myCgFragmentParam_Kd, myCgFragmentParam_Ks,
myCgFragmentParam_shininess, myCgFragmentParam_drawdepth, myCgFragmentParam_headnum;

static CGparameter myCgFragmentParam_hogelYes;
static CGparameter myCgFragmentParam_hologramGain;
static CGparameter myCgFragmentParam_hologramDebugSwitch;
static CGparameter myCgVertexParam_textureMatrix;
static CGparameter myCgVertexParam_depthMatrix;
static CGparameter myCgVertexParam_drawdepth;
// 04/10/2011 SKJ: Handle multiple Kinects
static CGparameter myCgVertexParam_textureMatrixSecond;
static CGparameter myCgVertexParam_depthMatrixSecond;
static CGparameter myCgVertexParam_drawdepthSecond;
static CGparameter myCgVertexParam_kinectNum;
static CGparameter myCgVertexParam_projectorTransform;
static CGparameter myCgVertexParam_projectorTransform2;

static const char *normalMapLightingProgramName = "09_vertex_lighting";
#ifdef VIEWS_FROM_CLOUD
	static const char *normalMapLightingProgramFileName = "../render_algorithms/holodepth/src/C5E1v_basicLightperFrag.cg";
	static const char *normalMapLightingVertexProgramFileName = "../render_algorithms/holodepth/src/C5E1f_basicNormMap.cg";
#else
	// 04/09/2011 SKJ: Or KINECT_MODE == 3??
	#if KINECT_MODE == 2
		static const char *normalMapLightingProgramFileName = "../src/kinect_vertex_shader.cg";
		static const char *normalMapLightingVertexProgramFileName = "../src/kinect_fragment_shader.cg";
	#else
		#if KINECT_MODE == 3
			static const char *normalMapLightingProgramFileName = "../src/multi_kinect_vertex_shader.cg";
			static const char *normalMapLightingVertexProgramFileName = "../src/multi_kinect_fragment_shader.cg";
		#else
			static const char *normalMapLightingProgramFileName = "../src/cloud_vertex_shader.cg";
			static const char *normalMapLightingVertexProgramFileName = "../src/cloud_fragment_shader.cg";
		#endif
	#endif
#endif

/* Page 111 */static const char *normalMapLightingVertexProgramName = "C5E1v_basicLight";
/* Page 85 */static const char *normalMapLightingFragmentProgramName = "C5E1f_basicLight";

//static CGcontext   myCgContext;
//static CGprofile   myCgVertexProfile,

static CGprofile myCgVertexProfile2, myCgFragmentProfile2;
static CGprogram myCgVertexProgram2, myCgFragmentProgram2;
// myCgFragmentProgram1;
static CGparameter myCgFragmentParam_decal, myCgFragmentParam_decal2,
myCgFragmentParam_decal0, myCgFragmentParam_decal1;
//myCgFragmentParam_layer0;

static const char *myProgramName2 = "Holo_myTextures",
		*myVertexProgramFileName2 = "../render_algorithms/holodepth/src/Holov_myTextures.cg",
		/* Page 83 */*myVertexProgramName2 = "Holov_myTextures",

		*myFragmentProgramFileName2 = "../render_algorithms/holodepth/src/Holof_myTextures.cg",
		/* Page 85 */*myFragmentProgramName2 = "Holof_myTextures";

GLuint allviewsfbo; //frame buffer object
GLuint depthbuffer;

//GLenum buffers[]={GL_COLOR_ATTACHMENT0_EXT,GL_COLOR_ATTACHMENT1_EXT};
GLfloat myProjectionMatrix1[16];
GLfloat myProjectionMatrix2[16];
#define MAX_CLOUDS 255

#ifdef VIEWS_FROM_CLOUD
JZCameraCloud * allClouds[MAX_CLOUDS];
int numLoadedClouds = 0;
#endif
//FramebufferObject* rectifiedfbo;

static float myGlobalAmbient[3] =
{ 0.1, 0.1, 0.1 }; /* Dim */
static float myLightColor[3] =
{ 0.95, 0.95, 0.95 }; /* White */

float ty0 = -9;
static float tz = 0, tx = 0, ty = ty0, rot = 0, rotx = 0;

GLuint texture_id[3];

GLuint meshTexID; //

GLuint tex_num = 0;

void writeViewsToFile();
void writeToFile2();

void printmatrix(float* m)
{
	printf("[");
	float*p = m;

		printf("[%g\t%g\t%g\t%g]\n",p[0],p[1],p[2],p[3]);
		printf("[%g\t%g\t%g\t%g]\n",p[4],p[5],p[6],p[7]);
		printf("[%g\t%g\t%g\t%g]\n",p[8],p[9],p[10],p[11]);
		printf("[%g\t%g\t%g\t%g]\n",p[12],p[13],p[14],p[15]);

	printf("]\n");
}

void checkErrors(void)
{
	GLenum error;

	while ((error = glGetError()) != GL_NO_ERROR)
	{

		//fprintf(stderr, "Error: %s\n", (char *) gluErrorString(error));
	}
}

#define checkForCgError(a) checkForCgErrorLine(a,__LINE__)

static void checkForCgErrorLine(const char *situation,int line = 0)
{
	CGerror error;
	const char *string = cgGetLastErrorString(&error);

	if (error != CG_NO_ERROR)
	{
		printf("line %d: %s: %s: %s\n", line, normalMapLightingProgramName, situation, string);
		if (error == CG_COMPILER_ERROR)
		{
			printf("%s\n", cgGetLastListing(normalMapLightingCgContext));
		}
		exit(1);
	}
}

static void checkForCgError2(const char *situation)
{
	CGerror error;
	const char *string = cgGetLastErrorString(&error);

	if (error != CG_NO_ERROR)
	{
		printf("%s: %s: %s\n", myProgramName2, situation, string);
		if (error == CG_COMPILER_ERROR)
		{
			printf("%s\n", cgGetLastListing(normalMapLightingCgContext));
		}
		exit(1);
	}
}

/* Forward declared GLUT callbacks registered by main. */
//static void reshape(int width, int height);
//static void display(void);
//static void keyboard(unsigned char c, int x, int y);
//static void menu(int item);
//static void requestSynchornizedSwapBuffers(void);

static void setBrassMaterial(void)
{
	const float brassEmissive[3] =
	{ 0.0, 0.0, 0.0 }, brassAmbient[3] =
	{ 0.33 * 2, 0.33 * 2, 0.33 * 2 }, brassDiffuse[3] =
	{ 0.78, 0.78, 0.78 }, brassSpecular[3] =
	{ 0.99, 0.99, 0.99 }, brassShininess = 27.8;

	cgSetParameter3fv(myCgFragmentParam_Ke, brassEmissive);
	cgSetParameter3fv(myCgFragmentParam_Ka, brassAmbient);
	cgSetParameter3fv(myCgFragmentParam_Kd, brassDiffuse);
	cgSetParameter3fv(myCgFragmentParam_Ks, brassSpecular);
	cgSetParameter1f(myCgFragmentParam_shininess, brassShininess);
}

static void setRedPlasticMaterial(void)
{
	const float redPlasticEmissive[3] =
	{ 0.0, 0.0, 0.0 }, redPlasticAmbient[3] =
	{ 0.2, 0.2, 0.2 }, redPlasticDiffuse[3] =
	{ 0.5, 0.5, 0.5 }, redPlasticSpecular[3] =
	{ 0.6, 0.6, 0.6 }, redPlasticShininess = 32.0;

	cgSetParameter3fv(myCgFragmentParam_Ke, redPlasticEmissive);
	checkForCgError("setting Ke parameter");
	cgSetParameter3fv(myCgFragmentParam_Ka, redPlasticAmbient);
	checkForCgError("setting Ka parameter");
	cgSetParameter3fv(myCgFragmentParam_Kd, redPlasticDiffuse);
	checkForCgError("setting Kd parameter");
	cgSetParameter3fv(myCgFragmentParam_Ks, redPlasticSpecular);
	checkForCgError("setting Ks parameter");
	cgSetParameter1f(myCgFragmentParam_shininess, redPlasticShininess);
	checkForCgError("setting shininess parameter");
}

static void setEmissiveLightColorOnly(void)
{
	const float zero[3] =
	{ 0.0, 0.0, 0.0 };

	cgSetParameter3fv(myCgFragmentParam_Ke, myLightColor);
	checkForCgError("setting Ke parameter");
	cgSetParameter3fv(myCgFragmentParam_Ka, zero);
	checkForCgError("setting Ka parameter");
	cgSetParameter3fv(myCgFragmentParam_Kd, zero);
	checkForCgError("setting Kd parameter");
	cgSetParameter3fv(myCgFragmentParam_Ks, zero);
	checkForCgError("setting Ks parameter");
	cgSetParameter1f(myCgFragmentParam_shininess, 0);
	checkForCgError("setting shininess parameter");
}

#ifdef VIEWS_FROM_CLOUD
//use this for drawing point clouds when we have multiple image sources/camera positions
void drawPointCloudFromZImage(JZCameraCloud **cloud, float shearAngle, float threshAngle)
{
	float scale = 100; // for converting units from meters (camera) to scene units
	float x,y,z,l;
	
	glDisable(GL_LIGHTING);
	glDisable(GL_TEXTURE_2D);
	//glPointParameter
	//glEnable(GL_POINT_SMOOTH);
	glPointSize(2.0f);
	//glPointSize(2.0f); //Arizona: use 2.0 point size, turn on smooth.
	//float attenparams[3] = {0,0,0}; //a b c	//size × 1 a + b × d + c × d 2
	//glPointParameterfv(GL_POINT_DISTANCE_ATTENUATION,attenparams);
	glBegin(GL_POINTS);	
	float dif;
	for(int c=0;c<numLoadedClouds;c++)
	{
		//if(c != numLoadedClouds/2) continue; //simple filter for all views
		
		
		for (int i=0;i<cloud[c]->resx*cloud[c]->resy;i++)
		{
			dif = fabs(cloud[c]->xangles[i] - shearAngle);
			if(dif > threshAngle) continue; //filter off-angle pixels
			#if REMOVE_PIXELS_DARKER_THAN != 0
				if(cloud[c]->ls[i] < REMOVE_PIXELS_DARKER_THAN) continue; //hack to remove dark samples (gets rid of dark noisy stuff in front of model. Makes holes in shadows on models)
			#endif
			x = (cloud[c]->xs[i]+cloud[c]->cop[0])*scale;
			y = (cloud[c]->ys[i]+cloud[c]->cop[1])*scale;
			z = (cloud[c]->zs[i]+cloud[c]->cop[2])*scale;

			l = cloud[c]->ls[i]*cloud[c]->gain;
			glColor3f(l,l,l);
			glVertex4f(x, y, z, 1.0);
		}
	}	
	glEnd();
}

#endif

#if KINECT_MODE > 0 && KINECT_MODE < 4



	void updateKinectCloud()
	{
		unsigned short* dptr = (unsigned short*)depthFrameBuffer->getBuffer();
		unsigned char* cptr = (unsigned char*)colorFrameBuffer->getBuffer();
		//copy shared buffer into FrameBuffer format (for now)
		int ystride = 1; //only render every ystride lines
		int i=0;
		for(int y=0;y<kinectCloudHeight;y+=ystride /*+ (headnumber/ystride)*/)
		{
			for(int x=0;x<kinectCloudWidth;x++)
			{
				dptr[i] = kinectframe->depth[y*kinectCloudWidth + x];
				cptr[i] = kinectframe->luma[y*kinectCloudWidth + x];
				i++;
			}
		}
		kprojector->setColorFrame(*colorFrameBuffer);
		kprojector->setDepthFrame(*depthFrameBuffer);
		// 04/09/2011 SKJ: Add updating for second Kinect
		#if KINECT_MODE == 3
			unsigned short* dptr2 = (unsigned short*)depthFrameBuffer2->getBuffer();
			unsigned char* cptr2 = (unsigned char*)colorFrameBuffer2->getBuffer();
			//copy shared buffer into FrameBuffer format (for now)
			i=0;
			for(int y=0;y<kinectCloudHeight;y+=ystride /*+ (headnumber/ystride)*/)
			{
				for(int x=0;x<kinectCloudWidth;x++)
				{
					dptr2[i] = kinectframe2->depth[y*kinectCloudWidth + x];
					cptr2[i] = kinectframe2->luma[y*kinectCloudWidth + x];
					i++;
				}
			}
			kprojector2->setColorFrame(*colorFrameBuffer2);
			kprojector2->setDepthFrame(*depthFrameBuffer2);
		#endif 
	}

	void drawPointCloudFromKinect_Calibrated()
	{
		glDisable(GL_LIGHTING);
		glEnable(GL_TEXTURE_2D);
		//glPointParameter
		//glEnable(GL_POINT_SMOOTH);
		glPointSize(2.0f);
		//float attenparams[3] = {0,0,0}; //a b c	//size × 1 a + b × d + c × d 2
		//glPointParameterfv(GL_POINT_DISTANCE_ATTENUATION,attenparams);
		
		// 04/09/2011 SKJ: This is where I can apply the matrix multiply (by projectorTransform) and draw per Kinect
		#if KINECT_MODE == 3
			/*
			Vrui::OGTransform projectorTransform;
			std::string transformFileName="/home/holo/Dropbox/Holovideo/Configuration/Kinect1/ProjectorTransform-B00361710470044B.txt";
			Misc::File transformFile(transformFileName.c_str(),"rt");
			char transform[1024];
			transformFile.gets(transform,sizeof(transform));
			projectorTransform=Misc::ValueCoder<Vrui::OGTransform>::decode(transform,transform+strlen(transform),0);
		
			Vrui::OGTransform projectorTransform2;
			std::string transformFileName2="/home/holo/Dropbox/Holovideo/Configuration/Kinect2/ProjectorTransform-A00364817648045A.txt";
			Misc::File transformFile2(transformFileName2.c_str(),"rt");
			char transform2[1024];
			transformFile2.gets(transform2,sizeof(transform2));
			projectorTransform2=Misc::ValueCoder<Vrui::OGTransform>::decode(transform2,transform2+strlen(transform2),0);
			*/
			// glPushMatrix();
			// glMultMatrix(projectorTransform);
			//cgSetMatrixParameterfr(myCgVertexParam_depthMatrix, &depthMatrixKinect1);
			//checkForCgError("ln 534");
			//cgSetMatrixParameterfr(myCgVertexParam_textureMatrix, &textureMatrixKinect1);
			//checkForCgError("ln 536");
			//cgSetMatrixParameterfr(myCgVertexParam_projectorTransform, &projectorTransform1);
			kinectNum = 0;
			cgSetParameter1f(myCgVertexParam_kinectNum, kinectNum);
		
			checkForCgError("ln 538");
			//glMatrixMode(GL_MODELVIEW);
			//glPushMatrix();
		
			//glRotatef(-45,0,1,0);
			//glMultMatrixf(*projectorTransform1);
			kprojector->draw(*projectorDataContext);
			//glPopMatrix();
			
			// glPushMatrix();
			// glMultMatrix(projectorTransform2);
			//if(check){
			
			//cgSetMatrixParameterfr(myCgVertexParam_depthMatrix, &depthMatrixKinect2);
			//checkForCgError("ln 546");
			//cgSetMatrixParameterfr(myCgVertexParam_textureMatrix, &textureMatrixKinect2);
			//checkForCgError("ln 548");
			//cgSetMatrixParameterfr(myCgVertexParam_projectorTransform, &projectorTransform2);
			kinectNum = 1;	
			cgSetParameter1f(myCgVertexParam_kinectNum, kinectNum);
			checkForCgError("ln 550");
			//glPushMatrix();
			//glRotatef(45,0,1,0);
			//glMultMatrixf(*projectorTransform2);
			kprojector2->draw(*projectorDataContext2);
			//glPopMatrix();
		//}
		#endif
		#if KINECT_MODE == 2
			kprojector->draw(*projectorDataContext);
		#endif
	}

	void drawPointCloudFromKinect()
	{
		// 04/09/2011 SKJ: Is this relevant anymore?
		
		//TODO: convert to vertex buffer
		int ystride = 8; //only render every ystride lines
		float scale = 0.07; // for converting units from pixel (camera) to scene units. Arbitrary.
		float gain = 1/256.0; // converting from char units to float
		float x,y,z,l;
		glDisable(GL_LIGHTING);
		glDisable(GL_TEXTURE_2D);
		//glPointParameter
		glEnable(GL_POINT_SMOOTH);
		glPointSize(1.0f);
		//float attenparams[3] = {0,0,0}; //a b c	//size × 1 a + b × d + c × d 2
		//glPointParameterfv(GL_POINT_DISTANCE_ATTENUATION,attenparams);
		glBegin(GL_POINTS);	
		int i = 0;
		for (int yi = 0;yi<480;yi+=ystride/*+(headnumber/ystride)*/)
		{
			for (int xi = 0;xi<640;xi++)
			{
				x = -xi*scale;
				y = -yi*scale;
				i = yi*640 + xi;
				z = kinectframe->depth[i]*scale;
				l = kinectframe->luma[i]*gain;
				//glColor3f(1,1,1);
				glColor3f(l,l,l);
				glVertex4f(x, y, z, 1.0);
			}
		}
		glEnd();
	}
#endif // KINECT_MODE > 0

void getFilenameForView( int viewnum, int modelnum, char* buff)
{	char modelname[255];
	switch (modelnum) {
		case 0:
			sprintf(modelname,"crane");
			break;
		case 1:
			sprintf(modelname,"Book");
			break;
		case 2:
			sprintf(modelname,"flower");
			break;
		default:
			printf("unsupported model number\n");
			sprintf(buff,"");
			return;
	}
	sprintf(buff,IMAGE_PATH_TEMPLATE,modelname,viewnum);
}

#ifdef VIEWS_FROM_CLOUD

void loadAllClouds()
{	
	int totalimages; 
	int firstimage;
	int modeltouse = MODEL_TO_USE;
	float baseline; // length of view set in meters
	float gain;
	float imzcenter; //distance from original camera to be placed at diffuser for hologram
	if(modeltouse==0)
	{
		//for crane
		totalimages = 100; //100
		firstimage = 0;
		baseline = 1.0;
		imzcenter = 0.5;
		gain = 1/1500.0;
	}else if(modeltouse == 2)
	{
		//for flowers
		totalimages = 64; //64
		firstimage = 0;
		baseline = 0.5;
		imzcenter = 0.5;
		gain = 1/2048.0;
	}else
	{
		modeltouse = 1; 
		totalimages = 16; //16
		firstimage = 0;
		baseline = 1.0;
		imzcenter = 0.5;
		gain = 1/2048.0;
	}
	
	char filename[2048];
	
	int viewsToLoad = VIEWS_TO_LOAD;//totalimages;
	
	for (int i = 0; i < viewsToLoad; i++)
	{
		int nextview = firstimage + (int)floor(i*totalimages/float(viewsToLoad));
		allClouds[i] = new JZCameraCloud();
		getFilenameForView(nextview,modeltouse,filename);
		printf("loading view at %s\n",filename);
		allClouds[i]->gain = gain;
		allClouds[i]->cop[0] = -i*baseline/float(viewsToLoad) + baseline/2.0;
		allClouds[i]->cop[2] = -imzcenter;
		allClouds[i]->loadFromFile(filename);
	}
	numLoadedClouds = viewsToLoad;
}

#endif

void drawDebugShape(float size)
{
	//glEnable(GL_LINE_SMOOTH);
	//glLineWidth(5.);
	//glBegin(GL_LINE_STRIP);

	glBegin(GL_TRIANGLES);

	//simple vertical line
	glVertex3f(-4.*size/2.,-size,-100);
	glVertex3f(-4.*size/2.,size,-100);
	
	//sloping horizontal line
	//glVertex3f(-HologramPlaneWidth_mm/2.0, 0.0, -100.0);
	//glVertex3f(HologramPlaneWidth_mm/2.0, 0.0, 100.0);
	
	glVertex3f(4.*size/2.,-size,100);
	glEnd();
	//glLineWidth(1.);
	//glDisable(GL_LINE_SMOOTH);
	//glutSolidCube(size*8);
	//glutWireCube(size);

}

static void drawme(float *eyePosition, float *modelMatrix_sphere,
		float *invModelMatrix_sphere, float *objSpaceLightPosition_sphere,
		float *modelMatrix_cone, float *invModelMatrix_cone,
		float *objSpaceLightPosition_cone, float h, float v, int drawdepthOn,
		float myobject, int viewnumber)
{
	// const float lightPosition[4] = { 10*mag+lx,20*mag+ly,-605*mag+lz, 1 };
	const float lightPosition[4] =
	{ 100, 100, -605, 1 };
	//const float eyePosition[4] = { 0,0,0, 1 };

	float translateMatrix[16], rotateMatrix[16], viewMatrix[16],
	modelViewMatrix[16], modelViewProjMatrix[16],
	modelViewProjMatrix2[16];
	float objSpaceEyePosition[4], objSpaceLightPosition[4];
	int j;
	buildLookAtMatrix(eyePosition[0], eyePosition[1], eyePosition[2], 0, 0,
			-425, 0, 1, 0, viewMatrix);

	/*** Render brass solid sphere ***/

	#ifndef VIEWS_FROM_CLOUD
		
		//setRedPlasticMaterial();
		setBrassMaterial();
		//setEmissiveLightColorOnly();
		
	#endif

	cgSetParameter1f(myCgFragmentParam_drawdepth, drawdepthOn);
	/* Transform world-space eye and light positions to sphere's object-space. */
	//transform

	(objSpaceEyePosition, invModelMatrix_sphere, eyePosition);
	cgSetParameter3fv(myCgFragmentParam_eyePosition, objSpaceEyePosition);

	//transform(objSpaceLightPosition, invModelMatrix_sphere, lightPosition);
	cgSetParameter3fv(myCgFragmentParam_lightPosition,
			objSpaceLightPosition_sphere);

	// 04/10/2011 SKJ: Change to KINECT_MODE > 1
	// #if KINECT_MODE == 2
	#if KINECT_MODE > 1 && KINECT_MODE < 4
		cgSetParameter1f(myCgVertexParam_drawdepth,drawdepthOn);
		//checkForCgError("ln715");
		#if KINECT_MODE == 3
			cgSetParameter1f(myCgVertexParam_drawdepthSecond,drawdepthOn);
			checkForCgError("ln718");
		#endif
		float pmatrix[16];
		float m[16];

	//JB Testing: was
	/*
		kprojector->getDepthProjTransform(pmatrix);
		//cgSetMatrixParameterfr(myCgVertexParam_modelViewProj, pmatrix);
		//cgUpdateProgramParameters(normalMapLightingCgVertexProgram);
		multMatrix(modelViewMatrix, modelMatrix_sphere, pmatrix);

		//multMatrix(m, pmatrix, modelViewMatrix);

		makeScaleMatrix(1,1,1,m);
		//multMatrix(m, pmatrix, modelViewMatrix);
		//multMatrix(modelViewProjMatrix, myProjectionMatrix1,m);

		//multMatrix(modelViewMatrix, viewMatrix, modelMatrix_sphere);
		multMatrix(modelViewProjMatrix, myProjectionMatrix1, modelViewMatrix);

	*/
		multMatrix(modelViewProjMatrix, myProjectionMatrix1, modelMatrix_sphere);
	#else
		multMatrix(modelViewMatrix, viewMatrix, modelMatrix_sphere);
		multMatrix(modelViewProjMatrix, myProjectionMatrix1, modelViewMatrix);
	#endif

	// glEnable(GL_LIGHTING);
		glEnable(GL_TEXTURE_2D);
	//glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fbo); // bind the frame buffer object
	//glViewport(h,v,64,440);

	/* Set matrix parameter with row-major matrix. */
	cgSetMatrixParameterfr(myCgVertexParam_modelViewProj, modelViewProjMatrix);
	cgUpdateProgramParameters(normalMapLightingCgVertexProgram);
	/*	 glBegin(GL_LINES);
	 {
	 //glVertex3f(0,20,40);
	 //glVertex3f(0,-20,40);
	 glVertex3f(-10,20,1);
	 glVertex3f(-10,-20,10);
	 }
	 glEnd();
	 glBegin(GL_LINES);
	 {
	 glVertex3f(0,20,10);
	 glVertex3f(0,-20,10);
	 // glVertex3f(-10,20,1);
	 // glVertex3f(-10,-20,1);
	 }
	 glEnd();

	 glBegin(GL_LINES);
	 {
	 glVertex3f(10,20,20);
	 glVertex3f(10,-20,20);
	 }
	 glEnd();*/

	/*
	if (myobject == 0)
	{
		glBlendFunc(GL_SRC_ALPHA, GL_ONE);
		glColor4f(0.5, 0.5, 0.5, 0.5f);
		glutSolidCube(8);
	}

	if (myobject == 1)
	{
		glBlendFunc(GL_SRC_ALPHA, GL_ONE);
		glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
		glutSolidCone(5, 5, 5, 5);
	}
	*/
	#if KINECT_MODE > 0
		#if KINECT_MODE == 1
			drawPointCloudFromKinect();
		#endif
		//glMatrixMode(GL_MODELVIEW);
		//glLoadIdentity();
		//glMultMatrixf(modelViewMatrix);
		//glMatrixMode(GL_PROJECTION);
		//glLoadIdentity();
		//glMultMatrixf(myProjectionMatrix1);
		// 04/09/2011 SKJ: Needs to change for KINECT_MODE == 3
		#if KINECT_MODE > 1 && KINECT_MODE < 4
			drawPointCloudFromKinect_Calibrated();
		#endif
		#if KINECT_MODE == 4
		glDisable(GL_TEXTURE_2D);
		glDisable(GL_LIGHTING);
		if(viewnumber == 0) {KinectPCL->uploadToGPU();} //TODO: check if frame is fresh, upload as soon as new one available
		glPointSize(2.0f);
		KinectPCL->draw();
		#endif
		//glutPostRedisplay();
	#else //not kinect mode:
		#if DRAW_WIRE_CUBE_ONLY == 1
			//glutWireCube(16); //draw cube
			drawDebugShape(16);
		#else //not "wire cube"
			#ifndef VIEWS_FROM_CLOUD
				glCallList(DLid); //draw rabbit from display list
			#else
				float ang =  (viewnumber/float(numview) - 0.5) * (fov * M_PI/ 180.);
				drawPointCloudFromZImage(allClouds,ang,ANGLE_THRESH);
			#endif
		#endif
	#endif
	//glColor4f(1.0f, 1.0f, 1.0f, 1.0f);
	//glutSolidCube(10);

	/* modelViewProj = projectionMatrix * modelViewMatrix */
	//  multMatrix(modelViewProjMatrix2, myProjectionMatrix1, modelViewMatrix);

}

//Get state from external gui. Update renderer state variables & request redraw if changes detected in gui state.
void refreshState(bool force = false)
{
#ifdef REMOTEQT_GUI
	//get state from external gui
	JDisplayState statecopy_old;
	memcpy(&statecopy_old, &statecopy, sizeof(JDisplayState));
	sharedstate->getDataCopy(&statecopy);
	if(force | memcmp(&statecopy,&statecopy_old,sizeof(JDisplayState)))
	{
		MasterHologramGain = statecopy.gain/5.0; //scale down gain from slider to use more of range
		cgSetParameter1f(myCgFragmentParam_hologramGain, MasterHologramGain);
		
		//ripParams->m_render->models->orient->rotate[0] = statecopy.xrot;
		rot = statecopy.yrot;
		rotx = statecopy.xrot;
		//ripParams->m_render->models->orient->rotate[2] = statecopy.zrot;
		
		tx = 10.0*statecopy.xpos;
		ty = /*ty0 +*/ 10.0*statecopy.ypos;
		tz = 40.0*statecopy.zpos;
		float sc = statecopy.scale;
		printf("New model location: (%g,\t %g,\t %g) x %g\n", tx, ty, tz, sc);
		hologramOutputDebugSwitch = statecopy.shaderMode;
		cgSetParameter1f(myCgFragmentParam_hologramDebugSwitch, hologramOutputDebugSwitch);
		cgSetParameter1f(myCgVertexParam_modelUIScale, statecopy.scale);
		zeroDepth = statecopy.rendermode1;
		zeroModulation = statecopy.rendermode2;
		fakeZ = (statecopy.flatdepth1 + 99)/198.0; //slider internal range is -99 to 99, map to 0-1
		fakeModulation = (statecopy.flatdepth2 + 99)/198.0;
		ViewEnableBitmask = statecopy.viewmask;
		printf("view bitmask = %x\n", ViewEnableBitmask);
		glutPostRedisplay(); //new state -- new stuff to draw
	}
#endif
}

//Draw text string to current buffer using GLUT
void drawString(float x, float y, char *string)
{
	int len, i;
	glRasterPos2f(x, y);
	len = (int) strlen(string);
	for (i = 0; i < len; i++)
	{
		glutBitmapCharacter(GLUT_BITMAP_HELVETICA_18, string[i]);
	}
}

void saveSingleView(int viewnum,int chan)
{
	GLenum frmt;
	switch(chan)
	{
		case -1: frmt = GL_LUMINANCE;
			break;
		case 0: frmt = GL_RED;
			break;
		case 1: frmt = GL_GREEN;
			break;
		case 2: frmt = GL_BLUE;
			break;
		case 3: frmt = GL_ALPHA;
			break;
	}
	glReadPixels(0, 0, numx, numy, frmt, GL_UNSIGNED_BYTE, localFramebufferStore);
	system("mkdir views 2> /dev/null");
	char fname[512];
	sprintf(fname, "/home/holo/Dropbox/Holovideo/Eclipse_Projects/holodepth/views/view_%04d.raw", viewnum);
	FILE *fp = fopen(fname,"w");
	if(!fp) {printf("failed to create file %s\n", fname);return;}
	fwrite(localFramebufferStore, sizeof(char), numx*numy, fp);
	fclose(fp);	
}

static void display(void)
{

	//get state from UI & update model
	refreshState();

	// 04/09/2011 SKJ: Needs to change for KINECT_MODE == 3
	#if KINECT_MODE > 1 && KINECT_MODE < 3
		updateKinectCloud(); //get fresh kinect data
	#endif
	#if KINECT_MODE == 4
		//KinectPCL->uploadToGPU(); //actually do this as we render each view to have absolute latest data (with possible tearing)
	#endif

	/* World-space positions for light and eye. */
	float eyePosition[4] =
	{ 0, 0, 0, 1 };
	//  const float lightPosition[4] = { 0, 500*mag,-500*mag, 1 };
	const float lightPosition[4] =
	{ 10 * mag + lx, 20 * mag + ly, -605 * mag + lz, 1 };
	float xpos, h, v, rgba, scale;
	int i, j;
	float myobject;

	float translateMatrix[16], rotateMatrix[16], rotateMatrix1[16],
	translateNegMatrix[16], rotateTransposeMatrix[16],
	rotateTransposeMatrix1[16], scaleMatrix[16], scaleNegMatrix[16],
	translateMatrixB[16], rotateMatrixB[16], rotateMatrix1B[16],
	translateNegMatrixB[16], rotateTransposeMatrixB[16],
	rotateTransposeMatrix1B[16], scaleMatrixB[16], scaleNegMatrixB[16],
	modelMatrix_sphere[16], invModelMatrix_sphere[16],
	modelMatrix_cone[16], invModelMatrix_cone[16],
	objSpaceLightPosition_sphere[16], objSpaceLightPosition_cone[16],
	objSpaceEyePosition_sphere[16], objSpaceEyePosition_sphereB[16],
	modelMatrix_sphereB[16], invModelMatrix_sphereB[16],
	modelMatrix_coneB[16], invModelMatrix_coneB[16],
	objSpaceLightPosition_sphereB[16], objSpaceLightPosition_coneB[16];

#if DISABLE_VIEW_RENDERING == 0 //can skip view rendering for debug of hologram layout
		glPushAttrib(GL_VIEWPORT_BIT | GL_COLOR_BUFFER_BIT);
		glEnable(GL_TEXTURE_2D);
	#ifndef VIEWS_FROM_CLOUD
		glBindTexture(GL_TEXTURE_2D, meshTexID);
	#endif

	cgGLBindProgram(normalMapLightingCgVertexProgram);
	//checkForCgError("ln943 binding vertex program lighting");
		cgGLEnableProfile(normalMapLightingCgVertexProfile);
		//checkForCgError("ln945 enabling vertex profile lighting");
		
		
		cgGLBindProgram(normalMapLightingCgFragmentProgram);
		//checkForCgError("ln949 binding vertex program lighting");
		cgGLEnableProfile(normalMapLightingCgFragmentProfile);
		//checkForCgError("ln951 enabling vertex profile lighting");
	/*for sphere find model and invModelMatrix */

	
	//TODO: recompute these only on change:
	makeRotateMatrix(rot, 0, 1, 0, rotateMatrix);
	makeRotateMatrix(-rot, 0, 1, 0, rotateTransposeMatrix);

	makeRotateMatrix(180 + rotx, 1, 0, 0, rotateMatrix1);
	makeRotateMatrix(-180 - rotx, 1, 0, 0, rotateTransposeMatrix1);

	multMatrix(rotateMatrix, rotateMatrix1, rotateMatrix);
	multMatrix(rotateTransposeMatrix, rotateTransposeMatrix1,
			rotateTransposeMatrix);

	//z is -600 + tz (z shift tz is centered halfway between near & far planes)
	makeTranslateMatrix(tx, ty, -(far + near) * 0.5 * mag + tz * mag,
			translateMatrix);
	makeTranslateMatrix(-tx, -ty, (far + near) * 0.5 * mag - tz * mag,
			translateNegMatrix);

	scale = 2;
	makeScaleMatrix(scale, scale, scale, scaleMatrix);
	makeScaleMatrix(1 / scale, 1 / scale, 1 / scale, scaleNegMatrix);

	multMatrix(modelMatrix_sphere, translateMatrix, rotateMatrix);
	multMatrix(invModelMatrix_sphere, rotateTransposeMatrix, translateNegMatrix);

	multMatrix(modelMatrix_sphere, modelMatrix_sphere, scaleMatrix);
	multMatrix(invModelMatrix_sphere, scaleNegMatrix, invModelMatrix_sphere);

	/* Transform world-space eye and light positions to sphere's object-space. */
	transform(objSpaceLightPosition_sphere, invModelMatrix_sphere,
			lightPosition);
	transform(objSpaceEyePosition_sphere, invModelMatrix_sphere, eyePosition);

	//manipulations for "second object" (mostly disabled)
	#if DISABLE_SECOND_OBJECT == 0
	{ // ?
		makeRotateMatrix(90 - 30 - rot * 2 * 0, 0, 1, 0, rotateMatrixB);
		makeRotateMatrix(-90 + 30 + rot * 2 * 0, 0, 1, 0, rotateTransposeMatrixB);

		makeRotateMatrix(180, 1, 0, 0, rotateMatrix1B);
		makeRotateMatrix(-180, 1, 0, 0, rotateTransposeMatrix1B);

		multMatrix(rotateMatrixB, rotateMatrix1B, rotateMatrixB);
		multMatrix(rotateTransposeMatrixB, rotateTransposeMatrix1B,
				rotateTransposeMatrixB);

		makeTranslateMatrix(tx * 0 + 25, ty * 0 - 5 + 10, -600.0 * mag - 70 * mag,
				translateMatrixB);
		makeTranslateMatrix(-tx * 0 - 25, -ty * 0 + 5 - 10, 600.0 * mag + 70 * mag,
				translateNegMatrixB);

		scale = 2;
		makeScaleMatrix(scale, scale, scale, scaleMatrixB);
		makeScaleMatrix(1 / scale, 1 / scale, 1 / scale, scaleNegMatrixB);

		multMatrix(modelMatrix_sphereB, translateMatrixB, rotateMatrixB);
		multMatrix(invModelMatrix_sphereB, rotateTransposeMatrixB,
				translateNegMatrixB);

		multMatrix(modelMatrix_sphereB, modelMatrix_sphereB, scaleMatrixB);
		multMatrix(invModelMatrix_sphereB, scaleNegMatrixB, invModelMatrix_sphereB);

		/* Transform world-space eye and light positions to sphere's object-space. */
		transform(objSpaceLightPosition_sphereB, invModelMatrix_sphereB,
				lightPosition);
		transform(objSpaceEyePosition_sphereB, invModelMatrix_sphereB, eyePosition);
	}
	#endif
	
	// glViewport(0,0,640,480);

	// glEnable(GL_CULL_FACE);
	i = 0;
	j = 0;

	xpos = 0;
	h = 0;
	v = 0;

	//glClearColor(0.5,0.5,0.5,0.5);//JB Hack: clear view buffer to gray for debugging
	
	glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	//glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);


	for (i = 0; i < numview; i++)
	{
		#ifdef SOLO_VIEW
			if(i != SOLO_VIEW) continue;
		#endif
		if((ViewEnableBitmask & (1<<i)) == 0)
		{
			continue;
		}
		
		glClear(GL_DEPTH_BUFFER_BIT);
		rgba = ((i / 4.) - int(i / 4.)) * 4.;
		h = int(i / (tiley * 4)) * numx;
		v = (int(i / 4.) / ((float) tiley) - int(int(i / 4.)
				/ ((float) tiley))) * numy * tiley;
		#if WRITE_LUMA_VIEW_FILES != 1
			glColorMask((rgba == 0), (rgba == 1), (rgba == 2), (rgba == 3));
		#else
			h = 0;v = 0; //for writing luma files (Arizona hack) draw all views on top of e/o in corner of viewport
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		#endif
		double q = tan((i - numview / 2.) / numview * fov / mag * M_PI
				/ 180.); //angle +/-fov/2
		#if REVERSE_VIEW_ORDER != 0
			q = -q;
		#endif
		//hologram is 150 mm wide, 75 mm high
		buildShearOrthographicMatrix2(
									  -HologramPlaneWidth_mm/2.0, 
									  HologramPlaneWidth_mm/2.0, 
									  -HologramPlaneHeight_mm/2.0, 
									  HologramPlaneHeight_mm/2.0, 
									  near * mag, 
									  far* mag, 
									  q, 
									  myProjectionMatrix1);
		
		glViewport(h, v, numx, numy);
		drawdepthOn = 0;

		//JB Disabling second object
		#if DISABLE_SECOND_OBJECT == 0
			myobject = 0;
			drawme(eyePosition, modelMatrix_sphereB, invModelMatrix_sphereB,
				   objSpaceLightPosition_sphereB, modelMatrix_coneB,
				   invModelMatrix_cone, objSpaceLightPosition_cone, h, v,
				   drawdepthOn, myobject,i);
		#endif
		myobject = 1;
		drawme(eyePosition, modelMatrix_sphere, invModelMatrix_sphere,
				objSpaceLightPosition_sphere, modelMatrix_cone,
				invModelMatrix_cone, objSpaceLightPosition_cone, h, v,
				drawdepthOn, myobject,i);

		
		//glViewport(h,v+120*tiley+16,160,120); //1024-6*160=64, (512-120*2*2)/2=16
		glViewport(h, v + numy * tiley, numx, numy); //setup viewport for depthbuffer render
		drawdepthOn = 1;
		myobject = 0;
		


		//JB Disabling second object
		#if DISABLE_SECOND_OBJECT == 0
			drawme(eyePosition, modelMatrix_sphereB, invModelMatrix_sphereB,
				   objSpaceLightPosition_sphereB, modelMatrix_coneB,
				   invModelMatrix_cone, objSpaceLightPosition_cone, h, v,
				   drawdepthOn, myobject,i);
		#endif

		myobject = 1;
		drawme(eyePosition, modelMatrix_sphere, invModelMatrix_sphere,
			   objSpaceLightPosition_sphere, modelMatrix_cone,
			   invModelMatrix_cone, objSpaceLightPosition_cone, h, v,
			   drawdepthOn, myobject,i);


		#if WRITE_LUMA_VIEW_FILES != 0
			saveSingleView(i, 0);
		#endif
	}


	cgGLDisableProfile(normalMapLightingCgVertexProfile);
	//checkForCgError("disabling vertex profile");
	cgGLDisableProfile(normalMapLightingCgFragmentProfile);
	//checkForCgError("disabling fragment profile");

	glPopAttrib();
	glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
	
	
	if(zeroDepth)
	{
		glViewport(0, numy*tiley , numx*tilex, numy*tiley); //setup viewport for covering depth views
		glColor4f(fakeZ,fakeZ,fakeZ,fakeZ);
		glDisable(GL_TEXTURE_2D);
		glDisable(GL_DEPTH_TEST);
		glDisable(GL_LIGHTING);
		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();
		glLoadIdentity();
		glMatrixMode(GL_PROJECTION);
		glPushMatrix();
		glLoadIdentity();
		
		glBegin(GL_QUADS);
		glVertex2f(-1,-1);glVertex2f(-1,1);glVertex2f(1,1);glVertex2f(1,-1);
		glEnd();
		
		glPopMatrix();
		glMatrixMode(GL_MODELVIEW);
		glPopMatrix();
		glEnable(GL_DEPTH_TEST);
		glEnable(GL_TEXTURE_2D);
	}
	
	if(zeroModulation)
	{
		glViewport(0, 0 , numx*tilex, numy*tiley); //setup viewport for covering color views
		glColor4f(fakeModulation,fakeModulation,fakeModulation,fakeModulation);
		glDisable(GL_TEXTURE_2D);
		glDisable(GL_DEPTH_TEST);
		glDisable(GL_LIGHTING);
		glMatrixMode(GL_MODELVIEW);
		glPushMatrix();
		glLoadIdentity();
		glMatrixMode(GL_PROJECTION);
		glPushMatrix();
		glLoadIdentity();
		
		glBegin(GL_QUADS);
		glVertex2f(-1,-1);glVertex2f(-1,1);glVertex2f(1,1);glVertex2f(1,-1);
		glEnd();
		
		glPopMatrix();
		glMatrixMode(GL_MODELVIEW);
		glPopMatrix();
		
		glEnable(GL_DEPTH_TEST);
		glEnable(GL_TEXTURE_2D);		
	}
	
	
	
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, texture_id[0]);
	//glFlush();
	//   glCopyTexSubImage			2D(GL_TEXTURE_2D, 0,0,0,0,0,imwidth,imheight);
				glCopyTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 0, 0, VIEWTEX_WIDTH, VIEWTEX_HEIGHT);
	//	printf("I'm here\n");
	checkErrors();
	glBindTexture(GL_TEXTURE_2D, 0);
	glDisable(GL_TEXTURE_2D);
#endif //end of disable view render
	
	#if DISABLE_HOLOGRAM_CREATION == 0
		if(hologramOutputDebugSwitch != -10)
			{
				float quadRadius = 0.5;
					
				// glViewport(0,0,imwidth,512);
				glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
				//glClear (GL_DEPTH_BUFFER_BIT);

				glMatrixMode(GL_PROJECTION);
				glLoadIdentity();
				glOrtho(-quadRadius, quadRadius, -quadRadius, quadRadius, 0, 125);
			// glOrtho(-512,512-1,-256,256,1,125);
			// gluLookAt(0,0,0,0,0,-100,0,1,0);
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();
			// glTranslatef(0.0,0.0,-100);
			//glViewport(0,0,1280,880);

			glViewport(0, 0, imwidth, imheight);
			#if HOLOGRAM_DOWNSCALE_DEBUG != 0
				glViewport(0, 0, imwidth/HOLOGRAM_DOWNSCALE_DEBUG, imheight/HOLOGRAM_DOWNSCALE_DEBUG);
			#endif
			glTranslatef(0.0, -0.25, 0.0); // JB: what does this do?
			glEnable(GL_TEXTURE_2D);
			glBindTexture(GL_TEXTURE_2D, texture_id[0]);

			glDisable(GL_LIGHTING);

				cgGLBindProgram(myCgVertexProgram2);
				// checkForCgError2("binding vertex program -fringes");
				cgGLEnableProfile(myCgVertexProfile2);
				// checkForCgError2("enabling vertex profile -fringes");

				cgGLBindProgram(myCgFragmentProgram2);
			// checkForCgError("binding fragment program");
			cgGLEnableProfile(myCgFragmentProfile2);
			// checkForCgError("enabling fragment profile");

			cgGLEnableTextureParameter(myCgFragmentParam_decal0);
			//  checkForCgError2("enable decal texture0");
			//  cgGLEnableTextureParameter(myCgFragmentParam_decal1);
			//  checkForCgError2("enable decal texture1");

			//cgUpdateProgramParameters(myCgFragmentProgram2);

			glColor3f(1., 1., 1.);
			//   glutSolidTeapot(75);
			//	glTranslatef(0.0,0.0,-100.);

			glBegin(GL_QUADS);
			glNormal3f(0.0, 0.0, 1.0);

			glTexCoord4f(0, 1, 0, 1);
			glVertex3f(-quadRadius, quadRadius, 0);

			glTexCoord4f(1, 1, 0, 1);
			glVertex3f(quadRadius, quadRadius, 0);

			glTexCoord4f(1, 0, 0, 1);
			glVertex3f(quadRadius, -quadRadius, 0);

			glTexCoord4f(0, 0, 0, 1);
			glVertex3f(-quadRadius, -quadRadius, 0);

			glEnd();
			glDisable(GL_TEXTURE_2D);
			glBindTexture(GL_TEXTURE_2D, 0);

			cgGLDisableProfile(normalMapLightingCgVertexProfile);
			//checkForCgError("disabling vertex profile");
			cgGLDisableProfile(normalMapLightingCgFragmentProfile);
			//checkForCgError("disabling fragment profile");
			
			if(hologramOutputDebugSwitch)
			{
				char st[255];
				glColor4f(1.0f, 0.0f, 0.0f, 1.0f);
				sprintf(st, "Shader debug mode %d", hologramOutputDebugSwitch);
				drawString(-0.5,-0.22, st);
			}
		}
	#endif //DISABLE_HOLOGRAM_CREATION
	
	glutSwapBuffers();
	
	#if KINECT_MODE
		glutPostRedisplay(); //keep drawing if we are getting streamed images
	#endif

	#if WRITE_LUMA_VIEW_FILES != 0
		exit(0);
	#endif
		
	#if WRITE_VIEWS_AND_EXIT == 1
		writeViewsToFile();
		exit(0);
	#endif
		
	#if WRITE_HOLOGRAM_AND_EXIT == 1
		writeToFile2();
		exit(0);
	#endif
}

static void idle(void)
{
	float fps;
	if (rotateOn == 1)
	{
		rot = rot - 5;
		rot = fmod(rot,360);
	}
	//tz=tz-.5;
	//  glutPostRedisplay();

	frame++;
	thetime = glutGet(GLUT_ELAPSED_TIME);
	//printf("idle\n");
	if (thetime - timebase > 1000)
	{
		const int len = 1024;
		char msg[len];
		//	printf("here\n");
		fps = frame * 1000.0 / (thetime - timebase);
		timebase = thetime;
		frame = 0;
		sprintf(msg,"Wafel %d render Mode: %d fps: %f\n", headnumber, (int)KINECT_MODE, fps);
		printf("%s",msg);

#ifdef REMOTEQT_GUI
		for(int i=0;i<len;i++) {
			displaystatus->statusMessage[headnumber][i] = msg[i];
		}
		//memcpy(&(displaystatus->statusMessage[headnumber][0]),msg,len);
#endif

		fflush(stdout);


	}
	#if QUIT_AFTER_SECONDS
	if (thetime > 1000 * QUIT_AFTER_SECONDS) exit(0);
	#endif
	refreshState();
}

//void ShutDown(void)
//{
//	//        glDeleteFramebuffersEXT(1, &fbo);
//	glDeleteTextures(1, &texture_id[0]);
//}

void writeViewsToFile()
{
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, texture_id[0]);
	glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA,
			GL_UNSIGNED_BYTE, localFramebufferStore);
	glDisable(GL_TEXTURE_2D);
	FILE *f;
	if ((f = fopen("views.raw", "w")) == NULL)
	{
		printf("failure opening file.\n");
		exit(0);
	}
	else
	{
		fwrite(localFramebufferStore,1,sizeof(localFramebufferStore),f);
		printf("wrote %d bytes\n",sizeof(localFramebufferStore));
		fclose(f);

	}

}

void writeToFile2(void)
{
	unsigned char *pic;
	unsigned char *bptr;
	FILE *fp;

	int allocfail = 0;
	int screenWidth = imwidth;
	int screenHeight = imheight;//1757 * 2;//1780*2;
	printf("attempting to write framebuffer to file");
	if ((pic = (unsigned char*) malloc(screenWidth * screenHeight * 3
			* sizeof(unsigned char))) == NULL)
	{
		printf("couldn't allocate memory for framebuffer dump.\n");
		allocfail = 1;
	}

	if (!allocfail)
	{
		char fname[255];
		glReadBuffer(GL_FRONT);
		bptr = pic;
		glReadPixels(0, 0, screenWidth, screenHeight, GL_RGB, GL_UNSIGNED_BYTE,
				pic);
		printf("saving %dx%dx3 to file...\n", screenWidth, screenHeight);

		if ((fp = fopen("holodump3.raw", "w")) == NULL)
		{
			printf("failure opening file.\n");
			exit(0);
		}
		else
		{
			if (fwrite(pic, 1, 3 * screenWidth * screenHeight, fp)
					!= screenWidth * screenHeight * 3)
			{
				printf("failure writing file.\n");
				//exit(0);
			}
			fclose(fp);
		}
		free(pic);

	}
}

static void keyboard(unsigned char c, int x, int y)
{
	//printf("keyboard \n");
	switch (c)
	{

	case 'z': //invert set of disabled views
		ViewEnableBitmask = ~ViewEnableBitmask;
		break;
	case 'Z': //enable all views
		ViewEnableBitmask = -1;
		break;
	case 'x': //cycle through debug modes in shader (output intermediate variables)
		hologramOutputDebugSwitch++;
		hologramOutputDebugSwitch = hologramOutputDebugSwitch % hologramOutputDebugSwitches;
		cgSetParameter1f(myCgFragmentParam_hologramDebugSwitch, hologramOutputDebugSwitch);
		break;		
	case 'X':
		hologramOutputDebugSwitch--;
		hologramOutputDebugSwitch = hologramOutputDebugSwitch % hologramOutputDebugSwitches;
		cgSetParameter1f(myCgFragmentParam_hologramDebugSwitch, hologramOutputDebugSwitch);
		break;	
	case 'j':
		tx = tx + 5;
		printf("tx %f \n", tx);
		break;
	case 'l':
		tx = tx - 5;
		printf("tx %f \n", tx);
		break;
	case 'i':
		ty = ty - 1;
		printf("ty %f \n", ty);
		break;
	case 'k':
		ty = ty + 1;
		printf("ty %f \n", ty);
		break;

	case 'w':
		tz = tz - 1;
		printf("%f \n", tz - 675);
		break;
	case 's':
		tz = tz + 1;
		printf("%f \n", tz - 675);
		break;

	case 'f':
		writeToFile2();
		break;
	case 'F':
		printf("writing view texture\n");
		writeViewsToFile();
		break;
	case 'e':
		tz = tz - 10;
		printf("%f \n", tz - 675);
		break;
	case 'd':
		tz = tz + 10;
		printf("%f \n", tz - 675);
		break;
	case 'c':
		tz = 0;
		printf("%f \n", tz - 675);
		break;
	case 'r':
		rotateOn = (rotateOn * -1) + 1;

		printf("rotate %i \n", rotateOn);
		break;
	case ' ':
		//makeViewtexFromFile();
		break;
	case ']':
		lz = lz - 10;
		printf("%f \n", lz);
		break;
	case '[':
		lz = lz + 10;
		printf("%f \n", lz);
		break;
	case '=':
		ly = ly - 10;
		printf("%f \n", ly);
		break;
	case '-':
		ly = ly + 10;
		printf("%f \n", ly);
		break;
	case ';':
		lx = lx - 10;
		printf("%f \n", lx);
		break;
	case '/':
		lx = lx + 10;
		printf("%f \n", lx);
		break;
	case '1':
		//cgSetParameter1f(myCgFragmentParam_hogelYes, 0.);
		//cgUpdateProgramParameters(myCgFragmentProgram2);
		//printf("Wafel");
		break;
	case '2':
		//cgSetParameter1f(myCgFragmentParam_hogelYes, 1.);
		//cgUpdateProgramParameters(myCgFragmentProgram2);
		//printf("Hogel");
		break;

	case 27: /* Esc key */
		/* Demonstrate proper deallocation of Cg runtime data structures.
		 Not strictly necessary if we are simply going to exit. */
		cgDestroyProgram(normalMapLightingCgVertexProgram);
		cgDestroyContext(normalMapLightingCgContext);
		// ShutDown();
		exit(0);
		break;
	}
	
	int mods =  glutGetModifiers();
	if(mods != 0)
	{
		if(c >= '0' && c <= '9') ViewEnableBitmask ^= 1<<(c-'0'+10); //toggle view enable bit for numbered view 10-19 (only 16 views used)
	} else {
		if(c >= '0' && c <= '9') ViewEnableBitmask ^= 1<<(c-'0'); //toggle view enable bit for numbered view 0-9
		
	}

	glutPostRedisplay();
}

void init(const char *filename_obj, char *filename_tex)
{
	GLfloat lightpos[] =
	{ 20.0f, 20.0f, -550.0f, 1.0f };

	// Initialize OpenGL context
	glClearColor(0.f, 0.f, 0.f, 1.0f);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
	// glShadeModel (GL_SMOOTH);


	glLightfv(GL_LIGHT0, GL_POSITION, lightpos);

	glShadeModel(GL_SMOOTH); // Enable Smooth Shading
	glClearColor(0.0f, 0.0f, 0.0f, 1.0f); // Black Background								// Depth Buffer Setup
	glEnable(GL_DEPTH_TEST); // Enables Depth Testing

	glLightfv(GL_LIGHT0, GL_AMBIENT, LightAmbient); // Setup The Ambient Light
	glLightfv(GL_LIGHT0, GL_DIFFUSE, LightDiffuse); // Setup The Diffuse Light
	glLightfv(GL_LIGHT0, GL_POSITION, LightPosition); // Position The Light
	glEnable(GL_LIGHT0); // Enable Light One
	glEnable(GL_LIGHTING);
	glEnable(GL_COLOR_MATERIAL);
	glHint(GL_PERSPECTIVE_CORRECTION_HINT, GL_NICEST);

	// Initialize OpenGL context
	glClearColor(0.f, 0.f, 0.f, 1.0f);
	// glShadeModel (GL_SMOOTH);

	glEnable(GL_DEPTH_TEST);
	glEnable(GL_LIGHTING);
	glEnable(GL_LIGHT0);

	glLightfv(GL_LIGHT0, GL_POSITION, lightpos);

	//glDrawBuffers(2,buffers);

#if	KINECT_MODE < 1

	printf("load textures\n");

	//Load BMP texture from file

	meshTexID = loadBMPTexture(filename_tex);

	printf("done \n");

	if (!meshTexID)
		exit(EXIT_FAILURE);

	// Load OBJ model file 
	if (!ReadOBJModel(filename_obj, &objfile))
		exit(EXIT_FAILURE);

	// Make display list 

	DLid = glGenLists(1);

	glNewList(DLid, GL_COMPILE);
	RenderOBJModel(&objfile);
	glEndList();
	FreeModel(&objfile);
#endif
	glDisable(GL_LIGHTING);


}

void cleanup()
{
#if KINECT_MODE < 1
	glDeleteLists(DLid, 1);

	glDeleteTextures(1, &meshTexID);
#endif
	//  glDeleteFramebuffersEXT(1,&fbo);
	// glDeleteRenderbuffersEXT(1,&depthbuffer);
	//  FreeModel (&objfile);
	delete localFramebufferStore;
}

int main(int argc, char **argv)
{

	if(argc == 2)
	{
		headnumber = atoi(argv[1]);
		if (headnumber > 2 || headnumber < 0)
			headnumber = 0;
	}

	localFramebufferStore = new GLubyte[VIEWTEX_WIDTH*VIEWTEX_HEIGHT*4];

	#ifdef REMOTEQT_GUI
		//state for slaving to separate UI
		sharedstate = new JSharedMemory(sizeof(JDisplayState),ALL_STATE_KEY);
		sharedstatus = new JSharedMemory(sizeof(JDisplayStatus),ALL_STATUS_KEY);
		displaystatus = (JDisplayStatus*)sharedstatus->getptr();
		sharedstate->getDataCopy(&statecopy);
	#endif
	
	#if DISABLE_HOLOGRAM_CREATION == 1
		glutInitWindowSize(VIEWTEX_WIDTH, VIEWTEX_HEIGHT); //170*12 (so 170 lines, not 440lines)
	#else
		#if HOLOGRAM_DOWNSCALE_DEBUG != 0
			glutInitWindowSize(imwidth/HOLOGRAM_DOWNSCALE_DEBUG, imheight/HOLOGRAM_DOWNSCALE_DEBUG); //170*12 (so 170 lines, not 440lines)
		#else
			glutInitWindowSize(MarkIIGLWindowWidth, MarkIIGLWindowHeight); //170*12 (so 170 lines, not 440lines)
		#endif
	#endif
	
	glutInitDisplayMode(GLUT_RGBA | GLUT_ALPHA | GLUT_DOUBLE | GLUT_DEPTH); //original code did not ask for alpha, resulting in no alpha on linux.
	glutInit(&argc, argv);

	glutCreateWindow(normalMapLightingProgramName);
	glutDisplayFunc(display);
	glutKeyboardFunc(keyboard);
	glutIdleFunc(idle);
	atexit(cleanup);

	#ifndef __APPLE__
		#if USE_GLEW
		GLenum err = glewInit();
		if (GLEW_OK != err)
		{
			printf("GLEW init failed\n");
		}
		#endif
	#endif

	// Setup our FBO
	//		glGenFramebuffersEXT(1, &fbo);
	//       glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, fbo);

	// Setup our depthbuffer
	//glGenRenderbuffersEXT(1, &depthbuffer);
	//glBindRenderbufferEXT(GL_FRAMEBUFFER_EXT, depthbuffer);
	//glRenderbufferStorageEXT(GL_FRAMEBUFFER_EXT, GL_DEPTH_COMPONENT, imheight,imwidth);

	char *filename_obj = "models/bunny502uvA.obj\0";
	char *filename_tex = "models/bunny502uv_normtex2.bmp\0";
	init(filename_obj, filename_tex);

	glGenTextures(2, texture_id);

	// requestSynchornizedSwapBuffers();
	glClearColor(0.0, 0.0, 0.0, 0); // Gray background. 
	glEnable(GL_DEPTH_TEST); // Hidden surface removal. 

	double q = tan(0.1);
	buildShearOrthographicMatrix2(-75. * mag, 75. * mag, -37.5 * mag, 37.5
			* mag, 450. * mag, 750. * mag, q / mag, myProjectionMatrix1);

	normalMapLightingCgContext = cgCreateContext();
	checkForCgError("creating context");
	cgGLSetDebugMode(CG_FALSE);
	//cgSetParameterSettingMode(myCgContext, CG_DEFERRED_PARAMETER_SETTING);

	normalMapLightingCgVertexProfile = cgGLGetLatestProfile(CG_GL_VERTEX);
	cgGLSetOptimalOptions(normalMapLightingCgVertexProfile);
	checkForCgError("selecting vertex profile1");

	normalMapLightingCgVertexProgram = cgCreateProgramFromFile(normalMapLightingCgContext, // Cg runtime context 
			CG_SOURCE, //Program in human-readable form 
			normalMapLightingProgramFileName, // Name of file containing program 
			normalMapLightingCgVertexProfile, // Profile: OpenGL ARB vertex program 
			normalMapLightingVertexProgramName, // Entry function name 
			NULL); // No extra commyPiler options 
	checkForCgError("creating vertex program from file");
	printf("created vertex program from file...\n");
	cgGLLoadProgram(normalMapLightingCgVertexProgram);
	checkForCgError("loading vertex program");
	printf("loaded vertex program\n");

	#define GET_PARAM(name) \
			myCgVertexParam_##name = \
			cgGetNamedParameter(normalMapLightingCgVertexProgram, #name); \
			checkForCgError("could not get " #name " parameter");

	GET_PARAM(modelViewProj);
	#if KINECT_MODE > 1 && KINECT_MODE < 4
		GET_PARAM(textureMatrix);
		checkForCgError("could not get textureMatrix vertex parameter ln 1707");
		GET_PARAM(depthMatrix);
		checkForCgError("could not get depthMatrix vertex parameter ln 1707");
		GET_PARAM(drawdepth);
		checkForCgError("could not get drawDepth vertex parameter ln 1707");
	#endif
	// 04/10/2011 SKJ: Handle second Kinect
	#if KINECT_MODE == 3
		GET_PARAM(textureMatrixSecond);
		GET_PARAM(depthMatrixSecond);
		GET_PARAM(drawdepthSecond);
		GET_PARAM(projectorTransform);
		GET_PARAM(projectorTransform2);
		GET_PARAM(kinectNum);
	#endif

	#if KINECT_MODE == 4
		GET_PARAM(modelUIScale);
		KinectPCL = new JVertexRender();
		KinectPCL->setDecimate(OPENNI_DECIMATE);
		//GET_PARAM(drawdepth);
		//checkForCgError("could not get drawDepth parameter");
	#endif
	normalMapLightingCgFragmentProfile = cgGLGetLatestProfile(CG_GL_FRAGMENT);
	cgGLSetOptimalOptions(normalMapLightingCgFragmentProfile);
	checkForCgError("selecting fragment profile");

	normalMapLightingCgFragmentProgram = cgCreateProgramFromFile(normalMapLightingCgContext, // Cg runtime context 
			CG_SOURCE, // Program in human-readable form 
			normalMapLightingVertexProgramFileName, normalMapLightingCgFragmentProfile, // Profile: latest fragment profile 
			normalMapLightingFragmentProgramName, // Entry function name 
			NULL); // No extra commyPiler options 
	checkForCgError("creating fragment program from string2");
	cgGLLoadProgram(normalMapLightingCgFragmentProgram);
	checkForCgError("loading fragment program");

	#define GET_FRAGMENT_PARAM(name) \
		myCgFragmentParam_##name = \
		cgGetNamedParameter(normalMapLightingCgFragmentProgram, #name); \
		checkForCgError("could not get " #name " parameter");

	GET_FRAGMENT_PARAM(globalAmbient);
	GET_FRAGMENT_PARAM(lightColor);
	GET_FRAGMENT_PARAM(lightPosition);
	GET_FRAGMENT_PARAM(eyePosition);
	GET_FRAGMENT_PARAM(Ke);
	GET_FRAGMENT_PARAM(Ka);
	GET_FRAGMENT_PARAM(Kd);
	GET_FRAGMENT_PARAM(Ks);
	GET_FRAGMENT_PARAM(shininess);
	GET_FRAGMENT_PARAM(drawdepth);
	GET_FRAGMENT_PARAM(headnum);

	myCgFragmentParam_decal = cgGetNamedParameter(normalMapLightingCgFragmentProgram, "decal");
	checkForCgError("getting decal parameter");
	cgGLSetTextureParameter(myCgFragmentParam_decal, meshTexID);
	checkForCgError("setting decal texture");

	// Set light source color parameters once. 
	cgSetParameter3fv(myCgFragmentParam_globalAmbient, myGlobalAmbient);
	checkForCgError("ln1750");
	cgSetParameter3fv(myCgFragmentParam_lightColor, myLightColor);
	checkForCgError("ln1753");
	
	//set up head number for rendering/loading with skipped lines
	cgSetParameter1i(myCgFragmentParam_headnum, headnumber);
	checkForCgError("setting head number parameter");
	
	//set up view texture (holds all view images. TODO: convert to 2 3d textures
	glBindTexture(GL_TEXTURE_2D, texture_id[0]);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	
	glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP);	  
	glTexParameteri (GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP);	

	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, VIEWTEX_WIDTH, VIEWTEX_HEIGHT, 0, GL_RGBA,
			GL_UNSIGNED_BYTE, NULL);

	//   glBindTexture(GL_TEXTURE_2D,texture_id[1]);

	// glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	// glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

	//   glTexImage2D(GL_TEXTURE_2D, 0,GL_RGBA,imwidth, imheight, 0, GL_RGBA,
	//GL_UNSIGNED_BYTE,myTexture2);


	// And attach it to the FBO so we can render to it
	//       glFramebufferTexture2DEXT(GL_FRAMEBUFFER_EXT, GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D,texture_id[0], 0);
	//             GLenum status = glCheckFramebufferStatusEXT(GL_FRAMEBUFFER_EXT);
	//if(status != GL_FRAMEBUFFER_COMPLETE_EXT)
	//          exit(1);

	// glBindFramebufferEXT(GL_FRAMEBUFFER_EXT, 0);        // Unbind the FBO for now

	//    buildTexture();
	//  glBindTexture(GL_TEXTURE_2D,texture_id[1]); //
	// glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	// glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	//    glTexImage2D(GL_TEXTURE_2D, 0,GL_RGBA,imheight,128, 0, GL_RGBA, GL_FLOAT, myTexture2);

	myCgVertexProfile2 = cgGLGetLatestProfile(CG_GL_VERTEX);
	cgGLSetOptimalOptions(myCgVertexProfile2);
	checkForCgError("selecting vertex profile2");

	myCgVertexProgram2 = cgCreateProgramFromFile(normalMapLightingCgContext, // Cg runtime context 
			CG_SOURCE, // Program in human-readable form 
			myVertexProgramFileName2, // Name of file containing program 
			//         "/home/holo/Quinn/holodepth/holodepth/src/Holov_myTextures.cg",
			myCgVertexProfile2, // Profile: OpenGL ARB vertex program 
			myVertexProgramName2, // Entry function name 
			//   "Holov_myTextures",
			NULL); // No extra compiler options 

	checkForCgError2("creating vertex program from file");
	cgGLLoadProgram(myCgVertexProgram2);
	checkForCgError2("loading vertex program");

	myCgFragmentProfile2 = cgGLGetLatestProfile(CG_GL_FRAGMENT);
	cgGLSetOptimalOptions(myCgFragmentProfile2);
	checkForCgError2("selecting fragment profile");

	myCgFragmentProgram2 = cgCreateProgramFromFile(normalMapLightingCgContext, // Cg runtime context 
			CG_SOURCE, // Program in human-readable form 
			myFragmentProgramFileName2, // Name of file containing program 
			myCgFragmentProfile2, // Profile: OpenGL ARB vertex program 
			myFragmentProgramName2, // Entry function name 
			NULL); // No extra compiler options 
	checkForCgError2("creating fragment program from file");
	cgGLLoadProgram(myCgFragmentProgram2);
	checkForCgError2("loading fragment program");

	#define GET_FRAGMENT_PARAM2(name) \
		myCgFragmentParam_##name = \
		cgGetNamedParameter(myCgFragmentProgram2, #name); \
		checkForCgError("could not get " #name " parameter");
	GET_FRAGMENT_PARAM2(hogelYes);
	cgSetParameter1f(myCgFragmentParam_hogelYes, 0.);
	GET_FRAGMENT_PARAM2(hologramGain);
	cgSetParameter1f(myCgFragmentParam_hologramGain, MasterHologramGain);
	GET_FRAGMENT_PARAM2(hologramDebugSwitch);
	cgSetParameter1f(myCgFragmentParam_hologramDebugSwitch, hologramOutputDebugSwitch);
	GET_FRAGMENT_PARAM2(headnum);
	cgSetParameter1i(myCgFragmentParam_headnum, headnumber); //TODO: investigate possible conflict with other shader using same uniform

	
	myCgFragmentParam_decal0 = cgGetNamedParameter(myCgFragmentProgram2,
			"decal0");
	checkForCgError2("getting decal parameter0");
	cgGLSetTextureParameter(myCgFragmentParam_decal0, texture_id[0]);
	//    cgGLSetTextureParameter(myCgFragmentParam_decal0, 0);
	checkForCgError2("setting decal 3D texture0");

	// myCgFragmentParam_decal1 =cgGetNamedParameter(myCgFragmentProgram2, "decal1");
	// checkForCgError2("getting decal parameter1");
	// cgGLSetTextureParameter(myCgFragmentParam_decal1, texture_id[1]);
	//// cgGLSetTextureParameter(myCgFragmentParam_decal1, 1);
	//checkForCgError2("setting decal 1D texture1");
		
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);


#ifdef VIEWS_FROM_CLOUD
#if KINECT_MODE < 1
	//loading of pre-rendered data
	loadAllClouds();
	//makeViewtexFromFile();
#endif
#endif
	
#ifdef VIEWS_FROM_CLOUD
	//makeViewtexFromFile();
	glutSwapBuffers();
#endif

// 04/09/2011 SKJ: Needs to change for KINECT_MODE == 3	
	
	#if KINECT_MODE > 0
		sharedkinect = new JSharedMemory(sizeof(JKinectFrame),KINECT_SHMEM_KEY);
		kinectframe = (JKinectFrame*)sharedkinect->getptr();

		char** appDefaults = 0;
		Vrui::init(argc,argv,appDefaults);
		glextmgr = new GLExtensionManager();
		GLExtensionManager::makeCurrent(glextmgr);


		#if KINECT_MODE > 1 && KINECT_MODE < 4
			depthFrameBuffer = new FrameBuffer(kinectCloudWidth,kinectCloudHeight,kinectCloudWidth*kinectCloudHeight*sizeof(unsigned short));
			colorFrameBuffer = new FrameBuffer(kinectCloudWidth,kinectCloudHeight,kinectCloudWidth*kinectCloudHeight*sizeof(unsigned char));
			#if KINECT_MODE == 3
				kprojector = new KinectProjector(PROJECTION_CONFIG_FILENAME,1);
				checkForCgError("ln1875");
			#else
				#if KINECT_MODE == 2
				kprojector = new KinectProjector(PROJECTION_CONFIG_FILENAME,0);
				checkForCgError("ln1880");
				#endif
			#endif
			projectorDataContext = new GLContextData(1000);
			kprojector->initContext(*projectorDataContext);
			checkForCgError("ln1884");
	
			float tmatrix[16];
			float tmatrix4[16];
			kprojector->getColorProjTransform(tmatrix);
			printf("Kinect 1 Texture Matrix:\n");
			printmatrix(tmatrix);
			cgSetMatrixParameterfr(myCgVertexParam_textureMatrix, tmatrix);
			checkForCgError("ln1890");
			//textureMatrixKinect1 = *tmatrix;
			kprojector->getDepthProjTransform(tmatrix);
			cgSetMatrixParameterfr(myCgVertexParam_depthMatrix, tmatrix);
			printf("Kinect 1 Depth Matrix:\n");
			printmatrix(tmatrix);
			//depthMatrixKinect1 = *tmatrix;
			checkForCgError("ln1894");
			
			

			
			// 04/09/2011 SKJ: Added to handle second Kinect
			#if KINECT_MODE == 3
	
				//kprojector->getProjTransform(tmatrix);
				//makeRotateMatrix(0, 0, 0, 0, tmatrix);
				//makeTranslateMatrix(0, 0, 0, tmatrix4);
				makeRotateMatrix(16.092324735547, -0.0672037642388, -0.98912306374287, -0.13084043275702, tmatrix);
				makeTranslateMatrix(6, 5, 0, tmatrix4);
				//makeRotateMatrix(39.092324735547, -0.0672037642388, -0.98912306374287, -0.13084043275702, tmatrix);
				//makeTranslateMatrix(-35.602902491527, -16.875347318442, 170.02844480397, tmatrix4);
				//makeRotateMatrix(25.819859391758, 0, 1, 0, tmatrix);
				//makeTranslateMatrix(-50, 0, 0, tmatrix4);
				//multMatrix(tmatrix, tmatrix, tmatrix4);
				
				//tmatrix[12] = 0;
				tmatrix[3] = tmatrix4[3];
				tmatrix[7] = tmatrix4[7];
				tmatrix[11] = tmatrix4[11];
				//transposeMatrix(tmatrix, tmatrix);
				cgSetMatrixParameterfr(myCgVertexParam_projectorTransform, tmatrix);
				printf("Kinect 1 Projection Matrix:\n");
				printmatrix(tmatrix);
				//projectorTransform1 = *tmatrix;
				// checkForCgError("ln1934");
				
				//kinectNum = 0;
				//cgSetParameter1f(myCgVertexParam_kinectNum, kinectNum);
				checkForCgError("ln1934");
	
				sharedkinect2 = new JSharedMemory(sizeof(JKinectFrame),KINECT_SHMEM_KEY_2); // KINECT_SHMEM_KEY_2 is defined in JKinectFrame.h
				kinectframe2 = (JKinectFrame*)sharedkinect2->getptr();
				depthFrameBuffer2 = new FrameBuffer(kinectCloudWidth,kinectCloudHeight,kinectCloudWidth*kinectCloudHeight*sizeof(unsigned short));
				colorFrameBuffer2 = new FrameBuffer(kinectCloudWidth,kinectCloudHeight,kinectCloudWidth*kinectCloudHeight*sizeof(unsigned char));
				// std::string transformFileName2="/home/holo/Dropbox/Holovideo/Configuration/Kinect2/ProjectorTransform-A00364817648045A.txt";
				kprojector2 = new KinectProjector(PROJECTION_CONFIG_FILENAME_2,2);
				projectorDataContext2 = new GLContextData(1000);
				checkForCgError("ln1907");
				kprojector2->initContext(*projectorDataContext2);
				checkForCgError("ln1909");	
	
				float tmatrix2[16];
				kprojector2->getColorProjTransform(tmatrix2);
				cgSetMatrixParameterfr(myCgVertexParam_textureMatrixSecond, tmatrix2);
				printf("Kinect 2 Texture Matrix:\n");
				printmatrix(tmatrix2);
				textureMatrixKinect2 = *tmatrix2; 
				checkForCgError("ln1915");
				kprojector2->getDepthProjTransform(tmatrix2);
				cgSetMatrixParameterfr(myCgVertexParam_depthMatrixSecond, tmatrix2);
				printf("Kinect 2 Depth Matrix:\n");
				printmatrix(tmatrix2);
				depthMatrixKinect2 = *tmatrix2;
				checkForCgError("ln1920");
				// 04/11/2011
				
				float tmatrix3[16];
				//kprojector2->getProjTransform(tmatrix2);
				//makeRotateMatrix(0, 0, 0, 0, tmatrix2);
				//makeTranslateMatrix(0, 0, 0, tmatrix3);
				makeRotateMatrix(26.819859391758, -0.03949171492209, 0.99856646911515, -0.036130474829529, tmatrix2);
				makeTranslateMatrix(-35, 0, 0, tmatrix3);
				//makeRotateMatrix(25.819859391758, -0.03949171492209, 0.99856646911515, -0.036130474829529, tmatrix2);
				//makeTranslateMatrix(55.962614888047, -22.307618594269, 171.3241052946, tmatrix3);
				//makeRotateMatrix(39.092324735547, -0.0672037642388, -0.98912306374287, -0.13084043275702, tmatrix2);
				//makeTranslateMatrix(50, 0, 0, tmatrix3);
				tmatrix2[3] = tmatrix3[3];
				tmatrix2[7] = tmatrix3[7];
				tmatrix2[11] = tmatrix3[11];
				//multMatrix(tmatrix2, tmatrix2, tmatrix3);
				//tmatrix2[12]=0;
				//transposeMatrix(tmatrix2, tmatrix2);
				cgSetMatrixParameterfr(myCgVertexParam_projectorTransform2, tmatrix2);
				printf("Kinect 2 Projection Matrix:\n");
				printmatrix(tmatrix2);
				//projectorTransform2 = *tmatrix2;
				checkForCgError("ln1936");
				
			#endif	
		
		#endif
	#endif
	refreshState(1);
	glutMainLoop();
	return 0;
}

//BONEYARD
/*
void makeViewtexFromFile()
{

	//build texture for rendering point clouds
	GLuint rtTexture, rtDepthTexture;
	float rtexheight = 640;
	float rtexwidth	= 480;
	glGenTextures(1, &rtTexture);
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, rtTexture);
	glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,GL_LINEAR_MIPMAP_NEAREST );
	// when texture area is large, bilinear filter the original
	glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );

	// the texture wraps over at the edges (repeat)
	glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT );
	glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT );
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, rtexwidth, rtexheight, 0, GL_RGBA, GL_FLOAT, NULL); // copy luma image into GL texture

	//build texture for depth buffer
	glGenTextures(1, &rtDepthTexture);
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, rtDepthTexture);
	glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER,GL_LINEAR_MIPMAP_NEAREST );
	// when texture area is large, bilinear filter the original
	glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR );
	glTexParameterf(GL_TEXTURE_2D, GL_DEPTH_TEXTURE_MODE, GL_INTENSITY);

	// the texture wraps over at the edges (repeat)
	glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT );
	glTexParameterf( GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT );
	glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, rtexwidth, rtexheight, 0, GL_DEPTH_COMPONENT, GL_UNSIGNED_BYTE, NULL); // copy luma image into GL texture

	//build fbo for rectified view
	rectifiedfbo = new FramebufferObject();
	rectifiedfbo->Bind();
	rectifiedfbo->AttachTexture(
							   GL_COLOR_ATTACHMENT0_EXT, //attachment point in FBO
							   GL_TEXTURE_2D, //type of attachment
							   rtTexture); //texture identifier to attach
	rectifiedfbo->AttachTexture(
								GL_DEPTH_ATTACHMENT_EXT, //attachment point in FBO
								GL_TEXTURE_2D, //type of attachment
								rtDepthTexture); //texture identifier to attach

	if(!rectifiedfbo->IsValid()) {
		printf("Created FBO, but it says it isn't valid\n");
	}
	CheckErrorsGL("END : creating fbo");


	//build fbo for texture containing all views
	GLuint texToWrite = texture_id[0];
	FramebufferObject *allviewsfbo = new FramebufferObject();

	if (!allviewsfbo)
	{
		printf("Framebuffer did not allocate.\n");
	}
	//activate the FBO
	allviewsfbo->Bind();
	//link to texture
	allviewsfbo->AttachTexture(
		GL_COLOR_ATTACHMENT0_EXT, //attachment point in FBO
		GL_TEXTURE_2D, //type of attachment
		texToWrite); //texture identifier to attach
	Renderbuffer *rb = new Renderbuffer();
	rb->Set( GL_DEPTH_COMPONENT24, VIEWTEX_WIDTH, VIEWTEX_HEIGHT );
	allviewsfbo->AttachRenderBuffer( GL_DEPTH_ATTACHMENT_EXT, rb->GetId() );
	glClearColor(0,0,0,0);
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	if(!allviewsfbo->IsValid()) {
		printf("Created FBO, but it says it isn't valid\n");
	}
	CheckErrorsGL("END : creating fbo");


	glEnable(GL_TEXTURE_2D);
	glDisable(GL_DEPTH_TEST);
	glLoadIdentity();


	for (int i = 0; i < numview; i++)
	{
		//int nextview = firstimage + (int)floor(i*totalimages/float(numview));
		//getFilenameForView(nextview,modeltouse,filename);
		//printf("loading view at %s\n",filename);
		//c.loadFromFile(filename);
		//c.buildFlatLumaGLTexture();//load and bind next luma texture


		//draw luminance map
		//glBindTexture(GL_TEXTURE_2D, c.lumtextureGL); //not needed here - already bound
		//drawQuadOverNumberedView(i,false);
		//setupDrawToNumberedView(i,false);
		rectifiedfbo->Bind();
		glDisable(GL_TEXTURE_2D);
		glEnable(GL_DEPTH_TEST);
		glClearColor(0,0,0,0);
		glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);
		glViewport(0, 0, rtexwidth, rtexwidth);
		drawPointCloudFromZImage(allClouds[i],0,M_PI);

		//render to stack of views
		//allviewsfbo->Bind();
		rectifiedfbo->Disable();
		glClear(GL_DEPTH_BUFFER_BIT); //clear depth buffer but keep accumulating color
		glDisable(GL_DEPTH_TEST);
		glEnable(GL_TEXTURE_2D);
		glBindTexture(GL_TEXTURE_2D,rtTexture);
		glGenerateMipmap(GL_TEXTURE_2D);
		drawQuadOverNumberedView(i,false);
		glBindTexture(GL_TEXTURE_2D,rtDepthTexture);
		glGenerateMipmap(GL_TEXTURE_2D);
		drawQuadOverNumberedView(i,true);
		glColorMask(1,1,1,1);
		//glDisable(GL_TEXTURE_2D);//disable texture to draw flat depth map instead
		//float falsedepth = 1.0;
		//glColor4f(falsedepth,falsedepth,falsedepth,falsedepth);

		//draw depth map
		//c.buildFlatDepthGLTexture();//load and bind next depth texture
		//drawQuadOverNumberedView(i,true);

		//glEnable(GL_TEXTURE_2D);
		//glColor4f(1.0,1.0,1.0,1.0);

	}
	glColorMask(1, 1, 1,1);


	allviewsfbo->Disable(); //back to drawing in main viewport
	glEnable(GL_DEPTH_TEST);
}

*/


/*
void setupDrawToNumberedView(int viewnum, bool isdepthview)
{

	//glClear(GL_DEPTH_BUFFER_BIT);
	int rgba = ((viewnum / 4.) - int(viewnum / 4.)) * 4.;
	int h = int(viewnum / (tiley * 4)) * numx;
	int v = (int(viewnum / 4.) / ((float) tiley) - int(int(viewnum / 4.)
													   / ((float) tiley))) * numy * tiley;
	glColorMask((rgba == 0), (rgba == 1), (rgba == 2), (rgba == 3));

	if(!isdepthview)
	{
		glViewport(h, v, numx, numy);
	} else {
		glViewport(h, v + numy * tiley, numx, numy); //setup viewport for depthbuffer render
	}

}

void drawQuadOverNumberedView(int viewnum, bool isdepthview)
{
	setupDrawToNumberedView(viewnum,isdepthview);
	float polywidth = 0.7; //arbitrary, use to correct for aspect ratio of display

	glBegin(GL_QUADS);
	glNormal3f(0.0, 0.0, 1.0);

	glTexCoord4f(0, 1, 0, 1);
	glVertex3f(-polywidth, 1, 0);

	glTexCoord4f(1, 1, 0, 1);
	glVertex3f(polywidth, 1, 0);

	glTexCoord4f(1, 0, 0, 1);
	glVertex3f(polywidth, -1, 0);

	glTexCoord4f(0, 0, 0, 1);
	glVertex3f(-polywidth, -1, 0);
	glEnd();
}
*/
