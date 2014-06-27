#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <assert.h>
#include <string.h>

//SID '10 was with master gain of 0.4
//#define MASTER_GAIN 0.7

extern float MASTER_GAIN; //now set from remote GUI
extern int displaystatic; //defined in main.cpp. =0 unless drawing prerendered views loaded from file.
#define REVERSE_VIEWS 0 // use when viewing through mirror. Hack. Uug.

//#define GL_GLEXT_PROTOTYPES
#include "setupglew.h"
#include <GL/gl.h>

#ifndef USING_GLEW
#include <GL/glext.h>
#endif

#include <GL/glu.h>
#include <GL/glx.h>
#include <GL/glut.h>

#include "RIP.h"
#include "UI.h"
#include "holoren.h"
#include "render.h"
#include <sys/time.h>

//stuff for framebuffer object support
#include "framebufferObject.h"
#include "glErrorUtil.h"
#include "renderbuffer.h"

#define DEGREES_TO_RADIANS (M_TWO_PI/360.0)

#ifdef M_PI
#undef M_PI
#endif
#ifdef M_TWO_PI
#undef M_TWO_PI
#endif
#define M_PI 3.14159265
#define M_TWO_PI (2.0*M_PI)



unsigned nextPowerOfTwo(unsigned n)
{
	int i;
	
	for(i = 0; i < 32; i++)
		if(n <= (((unsigned) 1) << i)) return ((unsigned) 1) << i;
	
	return 0;
}



RIPHologram::RIPHologram(HoloRenderParams *holoRenParams, holoConf *ren) :
			m_fringeTexID(0)
		,	m_parallaxStackFBOTexID(0)
		,	m_viewtexFBO(0)
		,	m_hologramFBOTexID(0)
		, 	m_hologramFBO(0)
		,	m_textures_created(false)
					  
{
	m_flatrender = true;
	m_render = ren;
	m_holorenparams = holoRenParams;
	setDefaultGeometry(holoRenParams);
	recomputeGeometry(holoRenParams);
	m_fringeToUse = 0;
}

void RIPHologram::setDefaultGeometry(HoloRenderParams *holoRenParams)
{
	 //Looks like hardcoding of RIP parameters happens here.
	
	
	m_useStaticImages = false;
	m_baseStaticImageFilename[0] = 0;	
	

    // resolution of the parallax views.
	m_viewXRes = 256;//383;//383; //this is rendered resolution AND count of emitters per line. SID '10 was with 128 emitters. BUG: emitter position not computed correctly for some values of this. TODO: check effect on emitter position
	if(displaystatic) m_viewXRes = 383;
	m_viewYRes = HoloVideoParams::nLines;//=144;
	m_viewSize = m_viewXRes*m_viewYRes;
    // number of parallax views.
    //    m_nXViews = 142;
    //m_nXViews = 140;
	m_nXViews = 128;
	
    // create a 3D texture.
    // 3D texture resolution must be a power of 2? choose a value that fits a view.
	m_viewTexXRes = nextPowerOfTwo(m_viewXRes);
	m_viewTexYRes = nextPowerOfTwo(holoRenParams->m_nHoloLinesPerFramebuffer);
	m_viewFBOTexYRes = nextPowerOfTwo(m_viewYRes);
	m_viewTexSize = m_viewTexXRes*m_viewTexYRes;
	m_nTexXViews = nextPowerOfTwo(m_nXViews);
	
	if (m_nTexXViews > MAX_VIEWS_PER_FBO_TEXTURE) m_nTexXViews = MAX_VIEWS_PER_FBO_TEXTURE;
	
    // scale factor to index only between texture samples that contain views.
    // and clip off the extra power-of-two blank samples tacked on at end.
	m_parallaxStackTexNXViewsScale = ((float) m_nXViews)/(float)m_nTexXViews;
	m_parallaxStackTexViewXScale = ((float) m_viewXRes)/(float)m_viewTexXRes;
//    m_parallaxStackTexViewYScale = ((float) holoRenParams->m_nHoloLinesPerFramebuffer)/(float)(m_viewTexYRes);
	m_parallaxStackFBOTexViewYScale = ((float) m_viewYRes)/(float)(m_viewFBOTexYRes);
	
    // width of the hologram in mm at diffuser
	m_hologramWidth = 150.0;//150.0;
    // distance of the projection plane from the hologram
    //    m_projectionPlaneDist = 10.0;
	m_projectionPlaneDist = 4.0;//128.0;
    // width/height of the projection plane
    // WJPNOTE: not sure about what this is, and I can't infer from
    // the values Tyeler has set. Tyeler only uses this for an assert;
    // I'm commenting out Tyeler's values, and including my
    // own values for the mm dimension of the plane of holoPoints.
    // Then, I'm using them to compute several other new parameters.
    //m_projectionPlaneWidth = 76.66;
	//m_projectionPlaneWidth = 75.4;
	m_projectionPlaneWidth = 75.4; //window down to "good part" of display?
	
	//m_projectionPlaneHeight = 57.5;
	m_projectionPlaneHeight = 56.5; //does not seem to be respected

	
	
    //    m_viewZoneWidth = 381.0;
	m_viewZoneWidth = 383.0; //yuck. 
    // WJPNOTE: I'm commenting out Tyeler's value and revising.
    //m_viewZoneDist = 600.0; 
    //    m_viewZoneDist = 590.0;
	m_viewZoneDist = 596.0;
	
	
	m_diffractionAngle = 14.0;//14
	m_referenceAngle = -15.0;//-15
	m_parallaxAngle = fabs(m_diffractionAngle) + fabs(m_referenceAngle);
	m_referenceLambda = 632.8e-6;
	m_E0 = 1.0;
	
    // WJPNOTE: Here's the logic:
    // viewsPerMMInViewzone = m_nXViews/m_viewZoneWidth;
    // holoPixelAddressSpaceInViewzone = 2.0 * m_viewzoneDist *
    //               tan(m_parallaxAngle/2.0*DEGREES_TO_RADIANS)
    // m_viewsPerHoloPixel = holoPixelAddressSpaceInViewzone * viewsPerMMInViewzone
	//
    // In our case, with m_projectionPlaneDist = 10.mm, we have
    // 9040 samples per holopixel / 80 samples/view = 113 views per holopixel.
    // we chose 80 samples/view in order to replicate a viewpixel across an
    // integer number of fringe samples.
    // Oops, now we have m_projectionPlaneDist = 4mm, we have
    // 3616 samples per holopixel. We'll still use 113 viwes per holopixel.
    // that'll replicate a viewpixel across 32 fringe samples.
    // However, now that implementing as a  texture, may not need to
    // keep this an integer number!
	
    // WJPNOTE: don't know what this is: 2622144/383? Let's see how it's used...
    // fix if necessary.
	float halfAngle = m_parallaxAngle/2.0;
	m_viewsPerHoloPixel = m_nXViews * 2.0 * m_viewZoneDist *
			tan(halfAngle*DEGREES_TO_RADIANS) / m_viewZoneWidth;
	
    // WJPNOTE: what is this? 262144/383??? worried about this one.
	//m_samplesPerHoloLine = HoloVideoParams::lineLength/m_viewXRes;
	
    // WJPNOTE: adding this.
	m_samplesPerMM = (float)HoloVideoParams::lineLength / m_hologramWidth;
	
	m_projectorFringeFootprint = 2.0 * m_projectionPlaneDist *
			tan(halfAngle*DEGREES_TO_RADIANS);
	
	m_samplesPerHoloPixel = m_projectorFringeFootprint * m_samplesPerMM;
	
    // WJPNOTE: adding these three things:
	m_holoPixelSpacing = m_projectionPlaneWidth / m_viewXRes;
	m_samplesPerHoloPixelSpacing = m_holoPixelSpacing * m_samplesPerMM;
	m_holoPixelSpacingJitterFraction = 0.1;
	InitializeRendererConfig();

}

void RIPHologram::recomputeGeometry(HoloRenderParams *holoRenParams) //compute all derived geometry parameters
{
	    //Looks like hardcoding of RIP parameters happens here.
	
	if(!m_flatrender)
	{
		m_projectionPlaneDist = RIP_PROJ_PLANE;
	}
	
	m_viewSize = m_viewXRes*m_viewYRes;
	
	m_viewTexXRes = nextPowerOfTwo(m_viewXRes);
	m_viewTexYRes = nextPowerOfTwo(holoRenParams->m_nHoloLinesPerFramebuffer);
	m_viewFBOTexYRes = nextPowerOfTwo(m_viewYRes);
	m_viewTexSize = m_viewTexXRes*m_viewTexYRes;
	m_nTexXViews = nextPowerOfTwo(m_nXViews);
	
	if (m_nTexXViews > MAX_VIEWS_PER_FBO_TEXTURE) m_nTexXViews = MAX_VIEWS_PER_FBO_TEXTURE;
	
    // scale factor to index only between texture samples that contain views.
    // and clip off the extra power-of-two blank samples tacked on at end.
	m_parallaxStackTexNXViewsScale = ((float) m_nXViews)/(float)m_nTexXViews;
	m_parallaxStackTexViewXScale = ((float) m_viewXRes)/(float)m_viewTexXRes;
//    m_parallaxStackTexViewYScale = ((float) holoRenParams->m_nHoloLinesPerFramebuffer)/(float)(m_viewTexYRes);
	m_parallaxStackFBOTexViewYScale = ((float) m_viewYRes)/(float)(m_viewFBOTexYRes);
	
 	m_parallaxAngle = fabs(m_diffractionAngle) + fabs(m_referenceAngle);
	
    // WJPNOTE: Here's the logic:
    // viewsPerMMInViewzone = m_nXViews/m_viewZoneWidth;
    // holoPixelAddressSpaceInViewzone = 2.0 * m_viewzoneDist *
    //               tan(m_parallaxAngle/2.0*DEGREES_TO_RADIANS)
    // m_viewsPerHoloPixel = holoPixelAddressSpaceInViewzone * viewsPerMMInViewzone
	//
    // In our case, with m_projectionPlaneDist = 10.mm, we have
    // 9040 samples per holopixel / 80 samples/view = 113 views per holopixel.
    // we chose 80 samples/view in order to replicate a viewpixel across an
    // integer number of fringe samples.
    // Oops, now we have m_projectionPlaneDist = 4mm, we have
    // 3616 samples per holopixel. We'll still use 113 viwes per holopixel.
    // that'll replicate a viewpixel across 32 fringe samples.
    // However, now that implementing as a  texture, may not need to
    // keep this an integer number!
	
    // WJPNOTE: don't know what this is: 2622144/383? Let's see how it's used...
    // fix if necessary.
	float halfAngle = m_parallaxAngle/2.0;
	
	m_viewsPerHoloPixel = m_nXViews * 2.0 * m_viewZoneDist *
			tan(halfAngle*DEGREES_TO_RADIANS) / m_viewZoneWidth;
	
    // WJPNOTE: what is this? 262144/383??? worried about this one.
	//m_samplesPerHoloLine = HoloVideoParams::lineLength/m_viewXRes;
	
    // WJPNOTE: adding this.
	m_samplesPerMM = (float)HoloVideoParams::lineLength / m_hologramWidth;
	
	m_projectorFringeFootprint = 2.0 * m_projectionPlaneDist *
			tan(halfAngle*DEGREES_TO_RADIANS);
	
	m_samplesPerHoloPixel = m_projectorFringeFootprint * m_samplesPerMM;
	
    // WJPNOTE: adding these three things:
	m_holoPixelSpacing = m_projectionPlaneWidth / m_viewXRes;
	m_samplesPerHoloPixelSpacing = m_holoPixelSpacing * m_samplesPerMM;
	
	InitializeRendererConfig(); // check values in renderer & assure they match
	BuildFringe(); //rebuild the projector fringe
}

const float FLOAT_EPSILON = 0.0001;
void RIPHologram::InitializeRendererConfig()
{
	// print diagnostic and abort program when expr = false.
	// careful, doesn't this only work when DEBUG is defined?
	//assert(m_render->ren->screenX == m_viewXRes);
	//assert(m_render->ren->screenY == m_viewYRes);
	//assert(m_render->ren->viewsX == m_nXViews);

	if(m_render->ren->screenX != m_viewXRes){
			m_render->ren->screenX = m_viewXRes;
			printf("Warning: Forcing x resolution to %d\n", m_viewXRes);
	}
	if(m_render->ren->screenY != m_viewYRes){
			m_render->ren->screenY = m_viewYRes;
			printf("Warning: Forcing y resolution to %d\n", m_viewYRes);
	}
	if(m_render->ren->viewsX != m_nXViews){
			m_render->ren->viewsX = m_nXViews;
			printf("Warning: Forcing view count to %d\n", m_nXViews);
	}


	assert(fabs(m_render->ren->cameraPlaneX - m_viewZoneWidth/10.0) < FLOAT_EPSILON);
}




void RIPHologram::AfterGLInitialize()
{
    //Generate one texture name, and store in the array $m_parallaxStackTex1D
    //glGenTextures(1, &m_parallaxStackTexID[0]);
	//loadViewSet("/Users/barabas/Documents/holovideo/shake/presence1VidFrames/frame");
    
	//Now generate two texture names for 2 framebuffers in one process
	//glGenTextures(2, m_parallaxStackTexID); //(sets both elements of m_par...)

	glGenTextures(1, &m_parallaxStackFBOTexID);

	//create empty 3D texture
	
	glBindTexture(GL_TEXTURE_3D, m_parallaxStackFBOTexID);
	
	//whoa, setting min_filter needed otherwise won't work with FBO.
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    

	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    GLint mode = GL_CLAMP_TO_BORDER;
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_S, mode);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_T, mode);
    glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_WRAP_R, mode);


//	glTexImage3D(GL_TEXTURE_3D, 0, GL_LUMINANCE, m_viewTexXRes,
//				 m_viewFBOTexYRes, m_nTexXViews, 0, GL_LUMINANCE,
//				 GL_UNSIGNED_BYTE, 0);

//		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, m_viewTexXRes,
//					 m_viewFBOTexYRes, 0, GL_LUMINANCE,
//					 GL_UNSIGNED_BYTE, 0);


		
//	glTexImage3D(GL_TEXTURE_3D, 0, GL_RGB, m_viewTexXRes,
//				 m_viewFBOTexYRes, m_nTexXViews, 0, GL_LUMINANCE,
//				 GL_UNSIGNED_BYTE, 0);

int tx = m_viewTexXRes;
int ty = m_viewFBOTexYRes;
int tz = m_nTexXViews;

	printf("creating texture of %d x %d x %d\n",tx,ty,tz);


	glTexImage3D(GL_TEXTURE_3D, 0, PARALLAX_STACK_FORMAT, tx,
				 ty, tz, 0, GL_RGBA,
				 GL_FLOAT, 0);


	//validate texture size

	//int tdimx = 0;
float txf,tyf,tzf;
	glGetTexLevelParameterfv(GL_TEXTURE_3D,0,GL_TEXTURE_WIDTH,&txf);
	glGetTexLevelParameterfv(GL_TEXTURE_3D,0,GL_TEXTURE_HEIGHT,&tyf);
	glGetTexLevelParameterfv(GL_TEXTURE_3D,0,GL_TEXTURE_DEPTH,&tzf);

	printf("Created texture is %g x %g x %g\n",txf,tyf,tzf);


	CheckErrorsGL("creating 3D texture ");

	if(m_useStaticImages)
	{
		loadViewSet(m_baseStaticImageFilename);		
	}
	else
	{	
      	//printf("Setting up Framebuffer Objects\n");
		m_viewtexFBO = new FramebufferObject();
        //printf("Binding FBO\n");
		m_viewtexFBO->Bind();
		
		int mipslice = 0;
		int zslice = 0;
        //printf("Attaching 3D texture to FBO\n");
		m_viewtexFBO->AttachTexture(GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_3D, m_parallaxStackFBOTexID,0,0);
	//	m_viewtexFBO->AttachTexture(GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, m_parallaxStackFBOTexID);

		m_viewtexFBO->IsValid();		

		//set up the depth buffer for this framebuffer object
		Renderbuffer* rb = new Renderbuffer(GL_DEPTH_COMPONENT24, m_viewTexXRes, m_viewFBOTexYRes);
		m_viewtexFBO->AttachRenderBuffer( GL_DEPTH_ATTACHMENT_EXT, rb->GetId());
		
		//glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);
		//glDrawBuffer(GL_NONE);
		//glReadBuffer(GL_NONE);
		
//		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		
		CheckErrorsGL("end of configuring depth FBO ");
		
		assert(m_viewtexFBO->IsValid());		
		
		FramebufferObject::Disable();

	}
	
	
	printf("Creating Hologram output framebuffer object\n");
	if(1){
		int fbxsize =  m_holorenparams->m_xRes;
		int fbysize = m_holorenparams->m_yRes;
		
		glGenTextures(1, &m_hologramFBOTexID);

	//create empty 2D texture
	
		glBindTexture(GL_TEXTURE_2D, m_hologramFBOTexID);
	
	//whoa, setting min_filter needed otherwise won't work with FBO.
		
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		GLint mode = GL_CLAMP_TO_EDGE;
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, mode);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, mode);
		
		glTexImage2D(GL_TEXTURE_2D, 0, HOLOGRAM_FORMAT, fbxsize,
					 fbysize,  0, GL_RGBA, GL_FLOAT, 0);
		
		CheckErrorsGL("end of creating texture for hologram FBO ");
		
		m_hologramFBO = new FramebufferObject();
		printf("Binding FBO\n");
		m_hologramFBO->Bind();
		
		int mipslice = 0;
		int zslice = 0;
		

		
		//printf("Attaching hologram texture to FBO\n");
		m_hologramFBO->AttachTexture(GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_2D, m_hologramFBOTexID);
		
		CheckErrorsGL("end of attaching texture for hologram FBO ");

		m_hologramFBO->IsValid();		
		//printf("Attaching hologram texture to FBO\n");

		//set up the depth buffer for this framebuffer object
		Renderbuffer* rb = new Renderbuffer(GL_DEPTH_COMPONENT24, fbxsize, fbysize);
		m_hologramFBO->AttachRenderBuffer( GL_DEPTH_ATTACHMENT_EXT, rb->GetId());
		
		// creating stencil renderbuffer. Hmm. Seems not supported.
		/*
		GLuint stencil_rb;
		glGenRenderbuffersEXT(1, &stencil_rb);
		glBindRenderbufferEXT(GL_RENDERBUFFER_EXT, stencil_rb);
		glRenderbufferStorageEXT(GL_RENDERBUFFER_EXT, GL_STENCIL_INDEX, fbxsize, fbysize);
		glFramebufferRenderbufferEXT(GL_FRAMEBUFFER_EXT, GL_STENCIL_ATTACHMENT_EXT, GL_RENDERBUFFER_EXT, stencil_rb);
		*/
		//Renderbuffer* srb = new Renderbuffer(GL_STENCIL_INDEX, fbxsize, fbysize);
		//m_hologramFBO->AttachRenderBuffer( GL_STENCIL_ATTACHMENT_EXT, srb->GetId());
		
						
		
		//glDrawBuffer(GL_COLOR_ATTACHMENT0_EXT);
		//glDrawBuffer(GL_NONE);
		//glReadBuffer(GL_NONE);
		
//		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		
		CheckErrorsGL("end of configuring hologram FBO ");
		
		assert(m_hologramFBO->IsValid());		
		
		FramebufferObject::Disable();
	}
	
#ifdef WORKSTATION_DEBUG
	printf("Press enter to begin\n");
	getchar();
#endif
	
	m_textures_created = true;
	
    BuildFringe();
//	BuildFringeSet(NUM_PRECOMPUTED_FRINGES);
//	BuildMultilineFringe();
#ifdef WORKSTATION_DEBUG
    glFlush();
    glutSwapBuffers();
#endif
}


void RIPHologram::blackmanWindow (int nf, float *w, int n, int ieo ) {
    /*
	nf = filter length in samples
	 w = window array of size n
	 n = filter half length=(nf+1)/2
	 ieo = even odd indicator--ieo=0 if nf even
	 */
	
    float tau, xi;
    int i;
	
    tau = (float)nf;
    for (i = 0; i < n; i++)
	{
		if (ieo == 0 )
			xi = (float)i - 0.5;
		w[i] = 0.42 +
			(0.5 * cos(M_TWO_PI* (xi/(tau))) +
			 (0.08 * cos(4.0*M_PI*(xi/(tau)))));
	}
	
	
}
void RIPHologram::kaiserWindow ( int nf, float *w, int n, int ieo,  float beta) {
    /*
	nf = filter length in samples
	 w = window array of size n
	 n = filter half length=(nf+1)/2
	 ieo = even odd indicator--ieo=0 if nf even
	 beta = parameter of kaiser window
	 larger beta, wider passband and smaller side lobes;
	 smaller beta, more energy across window
	 */
	float	bes, xind, xi;
	
	int	i;
	
	bes = kaiserBessel(beta);
	xind = (float)(nf-1)*(nf-1);
	
	for (i = 0; i < n; i++) {
		xi = i;
		if (ieo == 0)
			xi += 0.5;
		xi = 4. * xi * xi;
		xi = sqrt(1. - xi / xind);
		w[i] = kaiserBessel(beta * xi);
		w[i] /= bes;
	}
	return;
}



float RIPHologram::kaiserBessel (float x) {
	float	y, t, e, de, sde, xi;
	
	
	
	int i;
	
	y = x / 2.;
	t = 1.e-08;
	e = 1.;
	de = 1.;
	for (i = 1; i <= 25; i++) {
		xi = i;
		de = de * y / xi;
		sde = de * de;
		e += sde;
		if (e * t > sde)
		{
			break;
		}
	}
	return(e);
}

// Builds the projector fringe that reconstructs a holo_point.
void RIPHologram::BuildFringeSet(int fringecount)
{
	
//try creating emitters at [fringecount] depths between [m_projectionPlaneDist] and [m_projectionPlaneDist/16]
#define PROJPLANERANGE m_projectionPlaneDist
		
    printf("computing the RIP projector fringe set.\n");
    float fringeSamplesPerMM = (float)HoloVideoParams::lineLength/m_hologramWidth;
    float mmPerFringeSample = m_hologramWidth/(float)HoloVideoParams::lineLength;
    unsigned MaxFringeSampleLength = (unsigned)(m_projectionPlaneDist *
                                             (tanf(fabs(m_diffractionAngle)*DEGREES_TO_RADIANS) +
                                              tanf(fabs(m_referenceAngle)*DEGREES_TO_RADIANS)) *
                                             fringeSamplesPerMM);
    unsigned fringeTextureWidth = nextPowerOfTwo(MaxFringeSampleLength);
    unsigned fringeTextureHeight = nextPowerOfTwo(fringecount);
    unsigned fringeTextureDim = fringeTextureWidth * fringeTextureHeight;
	
	//HACK: Allocating extra memory to avoid bug in writing off end of array.
    float *fringe = new float[fringeTextureWidth*fringeTextureHeight*10]; //contains the raw fringe for making texture
	

	//HACK: Allocating extra memory to avoid bug in writing off end of array.
    unsigned char *fringeTexture = new unsigned char[fringeTextureDim*10];//potentially wider than raw fringe tx, but just as tall
	
	
	int i,j,k;
	int wankerValue;		
	
	m_fringeTexCoordScale = (float)MaxFringeSampleLength/(float)fringeTextureWidth; //-JB note: this is horizontal scale. what about vertical?
	
	for (unsigned int fringenum = 0;fringenum < fringeTextureHeight;fringenum++)
	{


		
		
		//fringe computation code liberally borrowed from WJP
		float z = m_projectionPlaneDist - ((fringenum / (1.0*fringeTextureHeight)))*PROJPLANERANGE;
		printf("fringe at depth %g \n", z);

		unsigned fringeSampleLength = (unsigned)(z *
												 (tanf(fabs(m_diffractionAngle)*DEGREES_TO_RADIANS) +
												  tanf(fabs(m_referenceAngle)*DEGREES_TO_RADIANS)) *
												 fringeSamplesPerMM);
		
		float zSquared = z*z;
		float sinThetaRef = sin((double) m_referenceAngle*DEGREES_TO_RADIANS);
		
		// xStart is the value of x in mm when we begin the interference simulation.
		float xStart;
		if (m_referenceAngle < 0.0)
			xStart = z * (float) tanf(m_referenceAngle*DEGREES_TO_RADIANS);
		else
			xStart = z * (float) tanf(m_diffractionAngle*DEGREES_TO_RADIANS);
		
		float x = xStart;
		// now dX is the mm increment each new fringe sample represents. mm/
		float dX = mmPerFringeSample;
		float propK = M_TWO_PI/m_referenceLambda;
		float dist, E, objPhase, refPhase, tau, t;
		float min = 9999.0;
		float max = -9999.0;

		// The larger we make beta, the more attenuation within window, better freq resp.
		// Chose small beta for now, to have less fringe attenutation, see if aliasing results.
		// Dial beta to get best compromise between brightness versus ringing in output.
		int winwidth = (int)((fringeSampleLength)+1) / 2;
		float w[winwidth];
		float beta = M_TWO_PI/2.0;
		int ieo = fringeSampleLength % 2;
		int countdown, countup;
		
		kaiserWindow((int)fringeSampleLength, w, winwidth, ieo, beta);
		//blackmanWindow((int)fringeSampleLength, w, winwidth, ieo);
		
		//compute fringe
		//WJPNOTE: right now random initial phase is 0.0
		for (j = 0; j < fringeSampleLength; j++, x+=dX)
		{//    glBindTexture(GL_TEXTURE_3D, m_parallaxStackTexID[0]);
			dist = (float) sqrt((x*x) + zSquared);
			E = m_E0; // don't attenuate by r^2
			objPhase = fmod(propK*dist, M_TWO_PI);
			refPhase = fmod(propK*x*sinThetaRef, M_TWO_PI);
			fringe[j + fringenum*fringeTextureWidth] = E*cos(fmod(objPhase - refPhase, M_TWO_PI));
			if(fringe[j + fringenum*fringeTextureWidth] < min) min = fringe[j + fringenum*fringeTextureWidth];
		}
		
		
		countdown = winwidth-1;
		countup = 0;
		i = winwidth - (int)fringeSampleLength;
		for (j = 0; j < fringeSampleLength; j++, i++)
		{
			// subtract off min
			fringe[j + fringenum*fringeTextureWidth] -= min;
			// window the fringe
			if ( i >= 0 ) {
				fringe[j + fringenum*fringeTextureWidth] *= w[countup];
				countup ++;
			} else {
				fringe [j + fringenum*fringeTextureWidth] *= w[countdown];
				countdown --;
			}
			// compute max
			if(fringe[j + fringenum*fringeTextureWidth] > max)
				max = fringe[j + fringenum*fringeTextureWidth];
		}
		
		printf("normalizing the projector fringe\n");
		// normalize to 8bit depth
		for(j = 0; j < fringeSampleLength; j++)
			fringeTexture[j + fringenum*fringeTextureWidth] = (unsigned char) (255*fringe[j + fringenum*fringeTextureWidth]/max);
		// fill up the rest with zeros.
		for(j = fringeSampleLength; j < fringeTextureWidth; j++)
			fringeTexture[j+ fringenum*fringeTextureWidth] = 0;
		
		//Since I'm having trouble with getting a 1D texture
		// to actually draw correctly, and Tyeler has used only
		// 2D textures in his DiffractionHolo code, I'm suspect
		// of 1D and 3D texturse. So, I'm going to replicate the
		// fringe row 0 into row 1 and make a 2D texture.
	//    k = fringeTextureWidth;
	//    for (j=0; j< fringeTextureWidth; j++, k++)
	//        fringeTexture[k]=fringeTexture[j];
		
	#ifdef WRITE_PROJECTOR_FRINGE
		// write business piece of fringe to file for inspection
		FILE *fp;
		unsigned char *bptr;
		if ( (fp = fopen ("projectorFringe.raw", "w")) == NULL) {
			printf("couldn't write fringe file.\n");
			exit(0);
		}
		else {
			bptr = fringeTexture;
			if (fwrite (bptr, 1, fringeTextureDim, fp) != fringeTextureDim) {
				printf("failure writing fringe file.\n");
				exit(0);
			}
			fclose (fp);
			printf("wrote out projector fringe.\n");
		}
	#endif
	}
    // Now represent the projector fringe as a texture.
    // Generate one texture, get an ID and bind it
    // selects this texture environment for configuration
    glActiveTextureARB(GL_TEXTURE0_ARB);
    if(!m_fringeTexID)
	{
		glGenTextures(1, &m_fringeTexID);
	}
	glBindTexture(GL_TEXTURE_2D, m_fringeTexID);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	
	/*
	glTexImage2D(GL_PROXY_TEXTURE_2D, 0, FRINGE_FORMAT,
                 fringeTextureWidth, fringeTextureHeight, 0,
                 GL_LUMINANCE, GL_UNSIGNED_BYTE, fringeTexture);
	
    glGetTexLevelParameteriv(GL_PROXY_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &wankerValue);
	*/
    //    printf("...fringeSampleLength = %d\n", fringeSampleLength);
    //    printf("...next power of two is %d\n", nextPowerOfTwo(fringeSampleLength));
    //    printf("...requested texture width was %d\n", fringeTextureWidth);
    //    printf ("...returned texture width is %d\n", wankerValue);
	
    glTexImage2D(GL_TEXTURE_2D, 0, FRINGE_FORMAT,
                 fringeTextureWidth, fringeTextureHeight, 0,
                 GL_LUMINANCE, GL_UNSIGNED_BYTE, fringeTexture);
	
	
	//HACK: Crashes on delete on second call. FIXME Must be writing off end of one of these. Probably fixed now...
    // clean up
    delete [] fringeTexture;
    delete [] fringe;
}


// Builds the projector fringe that reconstructs a holo_point.
void RIPHologram::BuildFringe()
{
	//HACK: redirect alternate ways of computing fringe
	//	BuildFringeSet(NUM_PRECOMPUTED_FRINGES);
	if(m_flatrender)
	{
		BuildMultilineFringe();
		return;
	}
	
	
	if (!m_textures_created) return;
	printf("computing the RIP projector fringe for %f mm.\n", m_projectionPlaneDist);
    float fringeSamplesPerMM = (float)HoloVideoParams::lineLength/m_hologramWidth;
    float mmPerFringeSample = m_hologramWidth/(float)HoloVideoParams::lineLength;
    unsigned fringeSampleLength = (unsigned)(abs(m_projectionPlaneDist) *
                                             (tanf(fabs(m_diffractionAngle)*DEGREES_TO_RADIANS) +
                                              tanf(fabs(m_referenceAngle)*DEGREES_TO_RADIANS)) *
                                             fringeSamplesPerMM);
	//printf("computing the RIP projector fringe of %d pixels.\n", fringeSampleLength);
    unsigned fringeTextureWidth = nextPowerOfTwo(fringeSampleLength);
    unsigned fringeTextureHeight = 2;
    unsigned fringeTextureDim = fringeTextureWidth * fringeTextureHeight;
    float *fringe = new float[fringeSampleLength*fringeTextureHeight]; //contains the raw fringe for making texture
    unsigned char *fringeTexture = new unsigned char[fringeTextureDim];//potentially wider than raw fringe tx, but just as tall
		int i,j,k;
		
		m_fringeTexCoordScale = (float)fringeSampleLength/(float)fringeTextureWidth; //-JB note: this is horizontal scale. what about vertical?
		
		//fringe computation code liberally borrowed from WJP
		float z = m_projectionPlaneDist;
		float zSquared = z*z;
		float sinThetaRef = sin((double) m_referenceAngle*DEGREES_TO_RADIANS);
		
		// xStart is the value of x in mm when we begin the interference simulation.
		float xStart;
		if (m_referenceAngle < 0.0)
			xStart = m_projectionPlaneDist * (float) tanf(m_referenceAngle*DEGREES_TO_RADIANS);
    	else
        	xStart = m_projectionPlaneDist * (float) tanf(m_diffractionAngle*DEGREES_TO_RADIANS);
	
    float x = xStart;
    // now dX is the mm increment each new fringe sample represents. mm/
    float dX = mmPerFringeSample;
	//if(xStart < 0) dX = -dX; //TODO: JB-- test this!
    float propK = M_TWO_PI/m_referenceLambda;
    float dist, E, objPhase, refPhase, tau, t;
    float min = 9999.0;
    float max = -9999.0;
    int wankerValue;
    // The larger we make beta, the more attenuation within window, better freq resp.
    // Chose small beta for now, to have less fringe attenutation, see if aliasing results.
    // Dial beta to get best compromise between brightness versus ringing in output.
    int winwidth = (int)((fringeSampleLength)+1) / 2;
    float w[winwidth];
    float beta = M_TWO_PI/2.0;
    int ieo = fringeSampleLength % 2;
    int countdown, countup;
	
    kaiserWindow((int)fringeSampleLength, w, winwidth, ieo, beta);
    //blackmanWindow((int)fringeSampleLength, w, winwidth, ieo);
	
    //compute fringe
    //WJPNOTE: right now random initial phase is 0.0
    for (j = 0; j < fringeSampleLength; j++, x+=dX)
	{
		dist = (float) sqrt((x*x) + zSquared);
		E = m_E0; // don't attenuate by r^2
		objPhase = fmod(propK*dist, M_TWO_PI);
		refPhase = fmod(propK*x*sinThetaRef, M_TWO_PI);
		fringe[j] = E*cos(fmod(objPhase - refPhase, M_TWO_PI));
		if(fringe[j] < min) min = fringe[j];
	}
	
	
    countdown = winwidth-1;
    countup = 0;
    i = winwidth - (int)fringeSampleLength;
    for (j = 0; j < fringeSampleLength; j++, i++)
	{
		// subtract off min
		fringe[j] -= min;
		// window the fringe
		if ( i >= 0 ) {
			fringe[j] *= w[countup];
			countup ++;
		} else {
			fringe [j] *= w[countdown];
			countdown --;
		}
		// compute max
		if(fringe[j] > max)
			max = fringe[j];
	}
	
    //printf("normalizing the projector fringe\n");
    // normalize to 8bit depth
    for(j = 0; j < fringeSampleLength; j++)
        fringeTexture[j] = (unsigned char) (255*fringe[j]/max);
    // fill up the rest with zeros.
    for(j = fringeSampleLength; j < fringeTextureWidth; j++)
        fringeTexture[j] = 0;
	
    //Since I'm having trouble with getting a 1D texture
    // to actually draw correctly, and Tyeler has used only
    // 2D textures in his DiffractionHolo code, I'm suspect
    // of 1D and 3D texturse. So, I'm going to replicate the
    // fringe row 0 into row 1 and make a 2D texture.
    k = fringeTextureWidth;
    for (j=0; j< fringeTextureWidth; j++, k++)
        fringeTexture[k]=fringeTexture[j];
	
#ifdef WRITE_PROJECTOR_FRINGE
    // write business piece of fringe to file for inspection
    FILE *fp;
    unsigned char *bptr;
    if ( (fp = fopen ("projectorFringe.raw", "w")) == NULL) {
        printf("couldn't write fringe file.\n");
        exit(0);
    }
    else {
        bptr = fringeTexture;
        if (fwrite (bptr, 1, fringeTextureDim, fp) != fringeTextureDim) {
            printf("failure writing fringe file.\n");
            exit(0);
        }
        fclose (fp);
        printf("wrote out projector fringe.\n");
    }
#endif
	
    // Now represent the projector fringe as a texture.
    // Generate one texture, get an ID and bind it
    // selects this texture environment for configuration
    glActiveTextureARB(GL_TEXTURE0_ARB);
	
	if(!m_fringeTexID)
	{
			glGenTextures(1, &m_fringeTexID);
	}
    glBindTexture(GL_TEXTURE_2D, m_fringeTexID);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	
	
	//HACK: truncate texture to max texture size. 
	//TODO: big textures go into multi-line format.
	if(fringeTextureWidth > 8192) fringeTextureWidth = 8192;
	
	glTexImage2D(GL_PROXY_TEXTURE_2D, 0, FRINGE_FORMAT,
                 fringeTextureWidth, fringeTextureHeight, 0,
                 GL_LUMINANCE, GL_UNSIGNED_BYTE, fringeTexture);
	
    glGetTexLevelParameteriv(GL_PROXY_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &wankerValue);
    //    printf("...fringeSampleLength = %d\n", fringeSampleLength);
    //    printf("...next power of two is %d\n", nextPowerOfTwo(fringeSampleLength));
    //    printf("...requested texture width was %d\n", fringeTextureWidth);
    //    printf ("...returned texture width is %d\n", wankerValue);
	
	glTexImage2D(GL_TEXTURE_2D, 0, FRINGE_FORMAT,
                 fringeTextureWidth, fringeTextureHeight, 0,
                 GL_LUMINANCE, GL_UNSIGNED_BYTE, fringeTexture);
	
    // clean up
    delete [] fringeTexture;
    delete [] fringe;
}
// Builds the projector fringe that reconstructs a holo_point.
void RIPHologram::BuildMultilineFringe()
{
	int maxTextureWidth = m_holorenparams->m_xRes*2;
	
	if (!m_textures_created) return;
	printf("computing the FLAT projector fringe for %f mm.\n", m_projectionPlaneDist);
	float fringeSamplesPerMM = (float)HoloVideoParams::lineLength/m_hologramWidth;
	float mmPerFringeSample = m_hologramWidth/(float)HoloVideoParams::lineLength;
	unsigned fringeSampleLength = (unsigned)(abs(m_projectionPlaneDist) *
			(tanf(fabs(m_diffractionAngle)*DEGREES_TO_RADIANS) +
			tanf(fabs(m_referenceAngle)*DEGREES_TO_RADIANS)) *
			fringeSamplesPerMM);
	//printf("computing the RIP projector fringe of %d pixels.\n", fringeSampleLength);
	
	unsigned fringeTextureWidth;
	unsigned fringeTextureHeight;
	unsigned fringeTextureDim;

	
	unsigned fringeBufferWidth = fringeSampleLength + maxTextureWidth;
	unsigned fringeBufferHeight = 1;
	unsigned fringeBufferDim = fringeBufferWidth;
	
	
	float *fringe = new float[fringeBufferWidth]; //contains the raw fringe for making texture

	int i,j,k;
		
	m_fringeTexCoordScale = (float)fringeSampleLength/(float)fringeTextureWidth; //-JB note: this is horizontal scale.
		
		//fringe computation code liberally borrowed from WJP
	float z = m_projectionPlaneDist;
	float zSquared = z*z;
	float sinThetaRef = sin((double) m_referenceAngle*DEGREES_TO_RADIANS);
		
		// xStart is the value of x in mm when we begin the interference simulation.
	float xStart;
	if (m_referenceAngle < 0.0)
		xStart = m_projectionPlaneDist * (float) tanf(m_referenceAngle*DEGREES_TO_RADIANS);
	else
		xStart = m_projectionPlaneDist * (float) tanf(m_diffractionAngle*DEGREES_TO_RADIANS);
	
	float x = xStart;
	// now dX is the mm increment each new fringe sample represents. mm/
	float dX = mmPerFringeSample;
	//if(xStart < 0) dX = -dX; //TODO: JB-- test this!
	float propK = M_TWO_PI/m_referenceLambda;
	float dist, E, objPhase, refPhase, tau, t;
	float min = 9999.0;
	float max = -9999.0;
	int wankerValue;
    // The larger we make beta, the more attenuation within window, better freq resp.
    // Chose small beta for now, to have less fringe attenutation, see if aliasing results.
    // Dial beta to get best compromise between brightness versus ringing in output.
	int winwidth = (int)((fringeSampleLength)+1) / 2;
	float w[winwidth];
	float beta = M_TWO_PI/2.0;
	int ieo = fringeSampleLength % 2;
	int countdown, countup;
	
	kaiserWindow((int)fringeSampleLength, w, winwidth, ieo, beta);
    //blackmanWindow((int)fringeSampleLength, w, winwidth, ieo);
	
    //compute fringe
    //WJPNOTE: right now random initial phase is 0.0
	for (j = 0; j < fringeSampleLength; j++, x+=dX)
	{
		dist = (float) sqrt((x*x) + zSquared);
		E = m_E0; // don't attenuate by r^2
		objPhase = fmod(propK*dist, M_TWO_PI);
		refPhase = fmod(propK*x*sinThetaRef, M_TWO_PI);
		fringe[j] = E*cos(fmod(objPhase - refPhase, M_TWO_PI));
		if(fringe[j] < min) min = fringe[j];
	}
	
	
	countdown = winwidth-1;
	countup = 0;
	i = winwidth - (int)fringeSampleLength;
	for (j = 0; j < fringeSampleLength; j++, i++)
	{
		// subtract off min
		fringe[j] -= min;
		// window the fringe
		if ( i >= 0 ) {
			fringe[j] *= w[countup];
			countup ++;
		} else {
			fringe [j] *= w[countdown];
			countdown --;
		}
		// compute max
		if(fringe[j] > max)
			max = fringe[j];
	}
	
	//printf("normalizing the projector fringe\n");
    // normalize
	for(j = 0; j < fringeSampleLength; j++)
		fringe[j] = fringe[j]/max;
    // fill up the rest with zeros.
	for(j = fringeSampleLength; j < fringeBufferWidth; j++)
		fringe[j] = 0;
	
	
	fringeTextureHeight = 1+(int)ceil(2.0*fringeSampleLength/(float)maxTextureWidth);
	fringeTextureWidth = maxTextureWidth;
	
	m_fringeTexHeight = fringeTextureHeight;
	m_fringeTexWidth = fringeTextureWidth;
	
	float* multilineFringe = new float[fringeTextureHeight*fringeTextureWidth];
		

	
	//fringe ABCDEFGHI gets put in texture as:
	
	//  AB
	//ABCD
	//CDEF
	//EFGH
	//GHI
	//I
	
	//so that any tall window of width/2 cut out from texture contains a continuous fringe
	
	
	//copy fringe in multiline format. 
	int frind = 0;
	int tind = 0;
	
	//first line
	for(int c = 0; c<fringeTextureWidth/2;c++) {multilineFringe[tind++] = 0;}//first line is half-blank
	for(int c = 0; c<fringeTextureWidth/2;c++) {multilineFringe[tind++] = fringe[frind++];}//other half has start of fringe
	//printf("wrote %d pixels on first line\n",tind);
	frind -= fringeTextureWidth/2; 
	
	//remaining lines
	for(int r=1;r<fringeTextureHeight;r++)
	{
		for (int c=0; c<fringeTextureWidth; c++)
		{
			multilineFringe[tind++] = fringe[frind++];
		}
		frind -= fringeTextureWidth/2; 
	}
	
	m_fringeTexLastline = fringeSampleLength % (fringeTextureWidth/2);
	//std::cout << "last line of texture has " << m_fringeTexLastline << " active samples\n";
	
			
	// Now represent the projector fringe as a texture.
    // Generate one texture, get an ID and bind it
    // selects this texture environment for configuration
	glActiveTextureARB(GL_TEXTURE0_ARB);
	
	if(!m_fringeTexID)
	{
		glGenTextures(1, &m_fringeTexID);
	}
	glBindTexture(GL_TEXTURE_2D, m_fringeTexID);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	
	glTexImage2D(GL_PROXY_TEXTURE_2D, 0, FRINGE_FORMAT,
				 fringeTextureWidth, fringeTextureHeight, 0,
	 GL_LUMINANCE, GL_FLOAT, multilineFringe);
	
	glGetTexLevelParameteriv(GL_PROXY_TEXTURE_2D, 0, GL_TEXTURE_WIDTH, &wankerValue);
	//printf("...fringeSampleLength = %d\n", fringeSampleLength);
	//printf("...next power of two is %d\n", nextPowerOfTwo(fringeSampleLength));
	//printf("...requested texture width was %d\n", fringeTextureWidth);
	//printf ("...returned texture width is %d\n", wankerValue);
	glGetTexLevelParameteriv(GL_PROXY_TEXTURE_2D, 0, GL_TEXTURE_HEIGHT, &wankerValue);
	//printf ("...returned texture height is %d\n", wankerValue);
	
	glTexImage2D(GL_TEXTURE_2D, 0, FRINGE_FORMAT,
				 fringeTextureWidth, fringeTextureHeight, 0,
	GL_LUMINANCE, GL_FLOAT, multilineFringe);
	
    // clean up

	delete [] fringe;
	delete [] multilineFringe;
}

void RIPHologram::DrawTexturedQuad(GLuint tex)
{
	float xmax = m_holorenparams->m_xRes;
	float ymax = 2.0*m_holorenparams->m_yRes;
	
	//Now render quad to screen
	glActiveTextureARB(GL_TEXTURE1_ARB);
	glDisable(GL_TEXTURE_3D);
	
	glActiveTextureARB(GL_TEXTURE0_ARB);
	glEnable(GL_TEXTURE_2D);
	glBindTexture(GL_TEXTURE_2D, tex);
	
	glDisable(GL_BLEND);
	
	glBegin(GL_QUADS);
	
	glTexCoord2f(0,0);glVertex2f(0,0);
	glTexCoord2f(0,1);glVertex2f(0,ymax);
	glTexCoord2f(1,1);glVertex2f(xmax,ymax);
	glTexCoord2f(1,0);glVertex2f(xmax,0);
    
	glEnd();
}

void RIPHologram::DisplayRIPHologramSingleFramebuffer(HoloRenderParams *holoRenParams)
{
	struct timeval tp;
	struct timezone tz;
	
#ifdef TIMING_DIAGNOSTICS
	gettimeofday(&tp, &tz);
	printf("time now is: %ld sec %ld usec \n", tp.tv_sec, tp.tv_usec);
#endif
	//    printf("building parallax stack\n");


	if(!m_useStaticImages)
	{
		glPushAttrib(GL_ALL_ATTRIB_BITS);
		BuildParallaxStack(holoRenParams);
		glPopAttrib();		
	} 
	//Instead of simply building the stack, would be better to render views,
	//then build 2 stacks (1 for each framebuffer)
	//Allocation of 2 stacks should happen earlier.
	
	//m_hologramFBO->Bind();
	
    //    printf("initializing for display\n");
    //This sets up GL for fringe modulation. Activates fringe and view textures
	//glPushAttrib(GL_ALL_ATTRIB_BITS);

	InitializeForDisplay(holoRenParams);	
    DisplayHologram(holoRenParams);
//	DrawTexturedQuad(m_fringeTexID);

	//CleanupAfterDisplay();
	//glPopAttrib();		

	//FramebufferObject::Disable();
	
	
	//glPushAttrib(GL_ALL_ATTRIB_BITS);
		
	//DrawTexturedQuad(m_fringeTexID);
	//DrawTexturedQuad(m_hologramFBOTexID);
	//glPopAttrib();		

	
	//    printf("cleaning up\n");
	
    //   printf("done displaying RIP hologram\n");
}

//TODO: Do allocation of 2 view textures: one for each framebuffer
void RIPHologram::AllocateViewTexture(HoloRenderParams *holoRenParams)
{
}

void RIPHologram::BuildParallaxStack(HoloRenderParams *holoRenParams)
{
    int i, j, k, whichline;
	
    // allocate space for a 3D texture, viewXres * viewYres * numViews.
/*
    static unsigned char *textureDataA = new unsigned char[m_viewTexSize*m_nTexXViews];
    static unsigned char *textureDataB = new unsigned char[m_viewTexSize*m_nTexXViews];
    static unsigned char *textureData[2] = {textureDataA,textureDataB};
*/	
    //textureData[0] = textureDataA;
    //textureData[1] = textureDataB;
	
    unsigned char *p;
    unsigned char *tmppic;
    unsigned char *b;
#ifdef DUMP_ALL_VIEWS
    char viewfile[1024];
#endif
    int wankerValue;
    int startRow;
    FILE *fp;
    int size;
    static int firstTime = 1;
    int framebufferInCPU = 0;
    // WJPNOTE: taking this multiply out of rendering loop.
    int thriceNumFramebuffers;
    thriceNumFramebuffers = holoRenParams->m_nFramebuffers*3;
	
	//m_viewtexFBO->Bind();
	
    glViewport(0, 0, m_viewXRes, m_viewYRes);
//    glReadBuffer(GL_BACK);
	
    // poly color.
    glColor4f(1.0, 1.0, 1.0, 1.0);
	
    // buffer clear color.
    glClearColor(0.0,0.0,0.0,0.0);

	m_viewtexFBO->Bind();

	
    // Render each parallax view.
    for(i = 0; i < m_nXViews; i++)
	{
		//bind the appropriate slice of the 3D texture
		int zslice = i;
		int mipslice = 0;
		if(zslice < MAX_VIEWS_PER_FBO_TEXTURE) 
		{
			m_viewtexFBO->AttachTexture(GL_COLOR_ATTACHMENT0_EXT, GL_TEXTURE_3D, m_parallaxStackFBOTexID, mipslice, zslice);
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
			//assert(m_viewtexFBO->IsValid());

		}
		int view = i;
#if REVERSE_VIEWS == 1
		view = m_nXViews - 1 - i;
#endif
		// printf("BuildParallaxStack: rendering view %d\n", i);
		// Clears buffer, places recentering camera, activates lighting and models.
		// Do we need to activate both light and models each render pass or no??
		//           printf("rendering view.%d\n", i);
		m_render->render(view); //JB Note: render(camx) is in holoren.cpp (member function of HoloConf) which
							 // in turn calls ren->activate(camx,state,conf) in render.cpp
							 // TODO: tweak renderer to only render lines we're going to use?
							 //(two passes at 1/6 vertical resolution plus vertical shift of PCnumber/6) ?

	}	


	FramebufferObject::Disable();
}


void RIPHologram::InitializeForDisplay(HoloRenderParams *holoRenParams)
{
    // not sure why viewport is 2*m_yRes,
    // but this duplicates config in Tyeler's DiffractionHolo code
    // JB: 2 windows now, so mystery 2 becomes 4
    // (nevermind. Now 2*yres is for 2 vertically stacked framebuffers on one dual-head card)
	
    glViewport(0, 0, holoRenParams->m_xRes, 2*holoRenParams->m_yRes);
	
	

	
    //set up for unlit, flat scene
    glDisable(GL_LIGHTING);
    glShadeModel(GL_FLAT);
    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);
	
    // WJPNOTE: experimenting with glBlend.
    //glEnable(GL_BLEND);
    //glBlendFunc(GL_SRC_ALPHA, GL_ONE);
    glEnable(GL_BLEND);
	glBlendEquation(GL_FUNC_ADD);
    //glBlendFunc(GL_SRC_ALPHA, GL_ONE);
	glBlendFunc(GL_ONE, GL_ONE);
	
	/*
	//glColorMaterial(GL_FRONT_AND_BACK, GL_DIFFUSE);
	//glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT);
	//glColorMaterial(GL_FRONT_AND_BACK, GL_SPECULAR);
	//glEnable(GL_COLOR_MATERIAL);
	*/
    // set up an orthographic projection for fringe rendering
    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    gluOrtho2D((double)0, (double)holoRenParams->m_xRes,
			   (double)0, 2.0*(double)holoRenParams->m_yRes);
	
	
	
	
    // make sure we aren't xforming the quad we render
    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
	
    // set up viewpixel and fringe textures to blend.
    // hopefully they are still configured correctly
    // and scene rendering hasn't redefined things.
    glActiveTextureARB(GL_TEXTURE0_ARB);
    glBindTexture(GL_TEXTURE_2D, m_fringeTexID);
    glEnable(GL_TEXTURE_2D);
    //glTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE);
	
#ifndef SKIP_VIEW_TEXTURE
    glActiveTextureARB(GL_TEXTURE1_ARB);
    glBindTexture(GL_TEXTURE_3D, m_parallaxStackFBOTexID);
    glEnable(GL_TEXTURE_3D);
    //;lglTexEnvi(GL_TEXTURE_ENV, GL_TEXTURE_ENV_MODE, GL_MODULATE); // modulate is default
#endif
}


void RIPHologram::CleanupAfterDisplay()
{
    //glEnd();
	
	glActiveTextureARB(GL_TEXTURE1_ARB);
	glDisable(GL_TEXTURE_3D);
	
    glActiveTextureARB(GL_TEXTURE0_ARB);
    glDisable(GL_TEXTURE_2D);
	
/*
	
	glColor4f(1.0,1.0,1.0,1.0);
//TEST drawing a big quad
	glColorMask(GL_TRUE, GL_TRUE, GL_TRUE, GL_TRUE);
	glBegin(GL_QUADS);

	glMultiTexCoord3fARB(GL_TEXTURE1_ARB, 0, 0, 0.5);
	glVertex2d(0.0, 0.0);

	glMultiTexCoord3fARB(GL_TEXTURE1_ARB, 1, 0, 0.5);
	glVertex2d(2000.0, 0.0);

	glMultiTexCoord3fARB(GL_TEXTURE1_ARB, 1, 1, 0.5);
	glVertex2d(2000.0, 3500.0);

	glMultiTexCoord3fARB(GL_TEXTURE1_ARB, 0, 1, 0.5);
	glVertex2d(0.0, 3500.0);

	glEnd();
// END TEST DRAWING QUAD
*/
	//glDisable(GL_COLOR_MATERIAL);


    // for next rendering loop.
    glEnable(GL_LIGHTING);
    glEnable(GL_DEPTH_TEST);
}

//assemble hologram//
void RIPHologram::DisplayHologram ( HoloRenderParams *holoRenParams )
{
	int x, revScan;
	int holoLine, holoLineTriplet, whichTripletLine;

	float startViewNumber;
	float endViewNumber;

	int frameBufferNumber;
	int frameBufferTripletNumber;
	int holoLineTripletYStartPixel;

	int holoScreenYStart;
	int holoScreenYEnd;
	int holoScreenXStart;
	int holoScreenXEnd;

	int holoPixelXStart;
	int holoPixelYStart;
	int holoPixelXEnd;
	int holoPixelYEnd;

	int yOffsetForThisFrameBuffer = 0;

	float holoLineSampleStart;
	float rawHoloLineSampleStart;
	int holoContributionSampleStart;
	int holoContributionSampleEnd;
	float randomJitter, jit;
	int tmp;
	int screenLineSpan;
	int screenLine;
	float accumScale;

	int headNumber; //which head on this card is rendering

	int activeViewTextureID;

	static int loadedGenlockOffsets = 0;
	static int genlockOffsetA = 0; //for first of pair of framebuffers on a card
	static int genlockOffsetB = 0; //for second
	static int genlockOffset = 0;
	

	int modulate = 1;
	//if(m_flatrender) 
	//{
		glActiveTextureARB(GL_TEXTURE1_ARB);
		glEnable(GL_TEXTURE_3D);
	
		glActiveTextureARB(GL_TEXTURE0_ARB);
		glEnable(GL_TEXTURE_2D);

		
	//}
	

#ifdef UNMODULATED_FRINGES
	modulate = 0;
#endif

	/* 'insurance' to make sure final accum buffer pixel has value < 1.0.
	// Hack with this to see if we need it. If OK without, then set it to 1.0
	// and simplify accumScale expression.
	*/
	float insurance = MASTER_GAIN;
	accumScale = insurance / ( m_projectorFringeFootprint / m_holoPixelSpacing );

	//HACK:
	accumScale = insurance*pow(1/ ( m_projectorFringeFootprint / m_holoPixelSpacing ),0.7);
	
	//accumScale = 0.3; //HACK: JB: hardcoding... above is dim, but avoids overmodulation

	/* Each holoLine will start on the same sample; might as well only do this once.
	// b = m_projectionPlaneDist * (float) tanf( (m_parallaxAngle/2.0) * DEGREES_TO_RADIANS);
	// c = 0.5 * (m_hologramWidth - m_projectionPlaneWidth);
	// a = c-b;
	// holoLineSampleStart = a * m_samplesPerMM;
	*/
	//this is sample within framebuffer at which hologram starts.
	rawHoloLineSampleStart = m_samplesPerMM *
	                         ( 0.5 * ( m_hologramWidth - m_projectionPlaneWidth ) -
	                           m_projectionPlaneDist *
	                           ( float ) tanf ( ( m_parallaxAngle/2.0 ) * DEGREES_TO_RADIANS ) );


	/*
	GENLOCK FIX:
	reads files containing number of samples to offset fringes.
	Each framebuffer has a file in genlockfix
	Calibration probably works until x exits.
	How it works: This block gets genlockOffsetA and genlockOffsetB from numbered files
	in genlockfix. genlockOffsetA is the first framebuffer's offset, genlockOffsetB
	is for the second framebuffer on each card.
	The offset is used to adjust holoLineSampleStart in this function.
	Disabling this code leaves genlockOffsetA and genlockOffsetB as initalized: both zero.
	*/
#ifndef NO_GENLOCK_FIX
	if ( !loadedGenlockOffsets )
	{
		FILE *calFile;
		char calFileName[512];
		int f = ( int ) floor ( ( holoRenParams->m_framebufferNumber ) /2 ) *2;
		sprintf ( calFileName, "../etc/genlockfix/%d", f );

		calFile = fopen ( calFileName,"r" );
		if ( !calFile )
		{
			printf ( "Could not find calibration file at %s\n",calFileName );
		}
		else
		{
			fscanf ( calFile,"%d",&genlockOffsetA );
#ifdef GENLOCK_DEBUG
			printf ( "Genlock offset for framebuffer %d was %d\n",f,genlockOffsetA );
#endif
			fclose ( calFile );
		}
		f = ( int ) floor ( ( holoRenParams->m_framebufferNumber ) /2 ) *2 + 1;
		sprintf ( calFileName, "../etc/genlockfix/%d", f );

		calFile = fopen ( calFileName,"r" );
		if ( !calFile )
		{
			printf ( "Could not find calibration file at %s\n",calFileName );
		}
		else
		{
			fscanf ( calFile,"%d",&genlockOffsetB );
#ifdef GENLOCK_DEBUG
			printf ( "Genlock offset for framebuffer %d was %d\n",f,genlockOffsetB );
#endif
			fclose ( calFile );
		}
#ifndef GENLOCK_DEBUG
		loadedGenlockOffsets = 1; //set flag so we don't continue to reload genlock offsets.
#endif
	}
#endif // END GENLOCK FIX


	/* We want to put new info on each of the rgb channels out of the FB.
	// Now, we have this thing holoLineTriplet, which marks the start of a 3 holoLine triplet.
	// A whichTripletLine is an individual holoLine, which belongs so a triplet (holoLineTriplet)
	// holoLineTriplet counts the number of triplets in a hologram, from 0 to (144/3)-1=47
	*/

	//this loop executes once per triplet of lines.
	for ( holoLineTriplet = 0; holoLineTriplet < HoloVideoParams::nLines/3; holoLineTriplet++ )
	{
		// frameBufferNumber is set to 0,1,2,3,4,5 -- 8 times.
		// This effectively assigns each successive hololine triplet to a different framebuffer.
		//frameBufferNumber = holoLineTriplet % holoRenParams->m_nFramebuffers;

		//which head will render this line
		headNumber = holoLineTriplet/ ( holoRenParams->m_nFramebuffers/2 ) % 2;
		//printf("rendering RGB trip %d using head %d\n", holoLineTriplet, headNumber);

		//higher numbered framebuffer of pair is below (closer to x=0) and gets 2nd triple
		if ( headNumber != 0 )
		{
			yOffsetForThisFrameBuffer = 0;
			genlockOffset = genlockOffsetA;
		}
		else
		{
			//new for RIP3.0: add vertical offset to draw fringe on other framebuffer
			yOffsetForThisFrameBuffer = holoRenParams->m_yRes; //offset by the height of one framebuffer
			genlockOffset = genlockOffsetB;
		}

		/* So frameBufferTripletNumber slowly counts from 0 to 7.
		// counting holoLine triplets inside each framebuffer. (8*3) * 6 = 144 holoLines.
		// 0/6 1/6 2/6 3/6 4/6 5/6 = 0 (3 forward, 3 reverse per framebuffer)
		// 6/6 7/6 8/6 9/6 10/6 11/6 = 1
		// 12/6...= 2
		// 18/6...= 3
		// 24/6...= 4
		// 30/6...= 5
		// 36/6...= 6
		// 42/6...= 7
		*/
		
		frameBufferTripletNumber = holoLineTriplet / holoRenParams->m_nFramebuffers;
		revScan = ( frameBufferTripletNumber % 2 ); //every other triple is a reverse scan

		//JB code for genlock fix: Offset start pixel by calibrated value from file.
		// Invert sign depending on scan direction
		holoLineSampleStart = rawHoloLineSampleStart + ( revScan*2-1 ) *genlockOffset;

		/* Takes 128 actual screen lines to represent one holoLine:
		// m_screenLinesPerHoloLine = 262144/2048=128.
		// We have m_yStartOffset blank lines at top of hologram,
		// and m_nHBlankLines after each 128 screen lines.
		// We are rendering m_yRes screenlines, each with 2048 samples on each framebuffer.
		// Let chunk = (128+m_nHBlankLines)
		// So below, holoLineTripletYStartPixel = m_yRes - [0 thru 7]*chunk
		// slowly counts down from 1024 by (128+m_nHBlankLines).
		// Its value tracks the screen lines that correspond to the beginning of
		// each holoLine triplet. All lines within a holoLine triplet are represented on
		// the SAME set of screenlines, packed into the RGB channels.
		//
		// Looks like this:
		// FB0:
		// holoLineTripletYStartPixel m_yRes:           FB0(r) = 0,     FB0(g) = 1,      FB0(b) = 2
		// holoLineTripletYStartPixel m_yRes-chunk: FB0(r) = 18,   FB0(g) = 19,    FB0(b) = 20
		//  ...
		// FB1:
		// holoLineTripletYStartPixel m_yRes:           FB1(r) = 3,     FB1(g) = 4,      FB1(b) = 5
		// holoLineTripletYStartPixel m_yRes-chunk: FB1(r) = 21,   FB1(g) = 22,    FB1(b) = 23
		//      ... and so on.
		//
		// So holoLineTripletYStartPixel is the actual SCREEN Y pixel that the triplet begins on.
		*/
		holoLineTripletYStartPixel = holoRenParams->m_yRes + yOffsetForThisFrameBuffer -
		                             holoRenParams->m_yStartOffset -
		                             frameBufferTripletNumber*holoRenParams->m_screenLinesPerHoloLine;
		// whichTripletLine encodes first, second or third line in the triplet.

		//for(whichTripletLine = 0; whichTripletLine < 3; whichTripletLine++)
		whichTripletLine = holoLineTriplet % 3;
		{
			// holoLineTriplet*3 + [0, 1 or 2];
			//holoLine = holoLineTriplet*3 + whichTripletLine;
			holoLine = holoLineTriplet*3 + holoRenParams->m_framebufferNumber/2; //framebuffernumber is 0,2, or 4 depending on which card is rendering. Used to be one process per head, now one per card.

			// Set drawing color to switch rgb FB channels.
			//-------------------------- color multiplex 3 lines------------------------------------

			//printf("rendering line %d using head %d in color %d\n", holoLine, headNumber, whichTripletLine);
			switch ( whichTripletLine )
			{
				case 0:
					//printf("RED pass: ");
					glColor4f ( accumScale, 0.0, 0.0, 1.0 );
					break;
				case 1:
					//printf("GRN pass: ");
					glColor4f ( 0.0, accumScale, 0.0, 1.0 ); //enable to turn channel on
					//glColor4f ( 0.0, 0.0, 0.0, 1.0 ); //enable to turn channel off
					break;
				case 2:
					//printf("BLU pass: ");
					glColor4f ( 0.0, 0.0, accumScale, 1.0 );
					//glColor4f ( 0.0, 0.0, 0.0, 1.0 ); //enable to turn channel off					
					break;
				default:
					printf ( "Invalid triplet\n" );
					break;
			}
			glBegin ( GL_QUADS );

			//----------------------------------- hologram space -----------------------------------
			//
			// Hologram sample at which the first modulated fringe begins in each holoLine.
			holoContributionSampleStart = ( int ) holoLineSampleStart;
			//printf ("DisplayHologram: starting hololine on sample %d\n", holoContributionSampleStart);
			// To construct the hologram line, must combine  fringe and the correct array of view pixels
			// for each holoPoint on the hologram line. Iterate through all holoPoints on a holoLine.
#ifdef LOW_HOLOPOINT_DENSITY
			m_viewXRes = 8;
			m_samplesPerHoloPixelSpacing = 137.62;
#endif
			for ( x = 0; x < m_viewXRes; x++ ) //iterate over hogels/line
			{
				// Index the parallax stack in texture memory.
				startViewNumber = ( m_nXViews - m_viewsPerHoloPixel ) * x / m_viewXRes;
				endViewNumber = startViewNumber + m_viewsPerHoloPixel;
				// printf("using view numbers %d to %d\n", startViewNumber, endViewNumber);

				
				/* WJPNOTE: ADD RANDOM JITTER to dephase holopoints.
				// figure out what chunk of samples constitutes a small fraction of the
				// interpoint spacing (set to 10%) and compute a random deviate
				// within that range. So first set jitter = uniform deviate between -1.0 and 1.0.
				// and holoContributionSampleStart = holoContributionSampleStart + jitter;
				// and holoContributionSampleEnd =  holoContributionSampleEnd + jitter;
				*/
#ifdef NO_RANDOM_JITTER

				randomJitter = 0.0;
#else
				randomJitter = ( float ) drand48();
				tmp = ( int ) ( randomJitter * 1000.0 );
				if ( ( tmp % 2 ) > 0.0 ) { randomJitter = -randomJitter; }
				randomJitter = m_samplesPerHoloPixelSpacing *
				               ( randomJitter * m_holoPixelSpacingJitterFraction );
#endif
				
				holoContributionSampleStart += ( int ) randomJitter;
				// This is the hologram sample at which a modulated fringe ends in each holoLine.
				holoContributionSampleEnd = holoContributionSampleStart + ( int ) m_samplesPerHoloPixel;
				//printf("line %d, point %d: from samples %d to %d\n", holoLine, i, holoContributionSampleStart, holoContributionSampleEnd);
				//printf("number of samples is %d\n", holoContributionSampleEnd - holoContributionSampleStart);
				//printf("samples per holopixelSpacing = %d\n", (int)(m_samplesPerHoloPixelSpacing));
				
				/*
				//----------------------------------- map to screen space --------------------------------
				//
				// Render the modulated fringe for a holopixel
				// on a holoLine, taking into account the forward and backward scanning of
				// each subsequent chunk of 18 holoLines. First three lines out of all six
				// framebuffers should scan one direction; next three lines out of all six
				// framebuffers should scan in reverse. So, accomplish this by using:
				// frameBufferTripletNumber (which counts the triplet number in a frambuffer)
				// frameBufferTripletNumber        = 0 1 2 3 4 5 6 7
				// frameBufferTripletNumber % 2 = 0 1 0 1 0 1 0 1 = revScan
				//
				// The number of screenlines that each modulated fringe spans depends
				// on the projection plane distance, the diffraction angle, the number of
				// hologram samples per mm and the X resolution of a screenline.
				// Must compute how many screenlines a holoPixel's modulated fringe spans.
				// This will also equal the number of rendering passes we must make. Each
				// rendering pass renders the entire modulated fringe on each screenline spanned,
				// clipped by the rendering window and shifted to put the samples in the right place.
				//
				// The screen x and y positions at which a modulated fringe starts and ends.
				// Depends whether we're scanning in fwd or reverse direction.
				*/

				//------------------------- do forward and reverse scans differently  ------------------------
				//
				
				if ( m_flatrender ) // for rendering flat objects with "multi-line" emitter textures
				{	 
					double tx = m_parallaxStackTexViewXScale*(float)x/(float)(m_viewXRes-1);
					double ty = m_parallaxStackFBOTexViewYScale*(float)holoLine/(float)(m_viewYRes-1) ;

					double tzmin = m_parallaxStackTexNXViewsScale*startViewNumber/(float)(m_nXViews-1);
					double tzmax = m_parallaxStackTexNXViewsScale*endViewNumber/(float)(m_nXViews-1);
					glMultiTexCoord3fARB(GL_TEXTURE1_ARB, tx, ty, 0.5/*tzmin*/);
					
					drawMultilineHolopixel(revScan, holoContributionSampleStart,holoLineTripletYStartPixel - m_fringeTexHeight/2,tx,ty,0.5/*tzmin*/);
				}
				else
				{
					if ( revScan )
					{
						// triplet is scanned in reverse direction, so must be sample-reversed.
						// MUST RECOMPUTE holoPixelYStart and holoPixelYEnd
						// holoScreenXStart, holoScreenXEnd, and holoScreenYStart, holoScreenYEnd
						// based on whether we're scanning in fwd or reverse direction.
	
						holoScreenXStart = holoRenParams->m_xRes -
										( holoContributionSampleStart ) % holoRenParams->m_xRes;
						holoScreenXEnd = holoRenParams->m_xRes -
										( holoContributionSampleEnd ) % holoRenParams->m_xRes;
						// offset from bottom of chunk of screenlines that represent a hololine
						holoScreenYStart = holoLineTripletYStartPixel -
										( holoRenParams->m_screenDataLinesPerHoloLine - 1 ) +
										( ( ( unsigned ) holoContributionSampleStart ) / holoRenParams->m_xRes ) - 1;
						holoScreenYEnd = holoLineTripletYStartPixel -
										( holoRenParams->m_screenDataLinesPerHoloLine - 1 ) +
										( ( ( unsigned ) holoContributionSampleEnd ) / holoRenParams->m_xRes ) - 1;
	
						screenLineSpan = holoScreenYEnd - holoScreenYStart + 1;
						//printf("DisplayHologram: projector fringe spans %d screenlines\n", screenLineSpan);
						// For each screenline, render the same modulated fringe.
						for ( screenLine=0; screenLine<screenLineSpan; screenLine++ )
						{
							// adjust y extent to span the current screenline.
							// start maps to top of poly, end maps to bottom.
							holoPixelYEnd = holoScreenYStart + ( screenLine );
							holoPixelYStart = holoPixelYEnd + 1;
							// adjust x extents so that appropriate piece of the
							// modulated fringe fits in rendering window.
							// start maps to poly left, stop maps to right.
							holoPixelXEnd = holoScreenXStart + ( screenLine*holoRenParams->m_xRes );
							holoPixelXStart = holoPixelXEnd - ( int ) m_samplesPerHoloPixel;
							//    printf("drawing pixel %d %d on screenline %d, with viewrange[%f-%f] \n", i, holoLine, holoPixelYStart, startViewNumber, endViewNumber);
							DrawHoloPixel ( revScan, x, holoLine,
											startViewNumber, endViewNumber,
											holoPixelXStart, holoPixelYStart,
											holoPixelXEnd, holoPixelYEnd,
											holoRenParams->m_nHoloLinesPerFramebuffer,
											modulate );
						}
					}
					else
					{
						// triplet is scanned forward
						holoScreenXStart = ( holoContributionSampleStart ) % holoRenParams->m_xRes;
						holoScreenXEnd = ( holoContributionSampleEnd ) % holoRenParams->m_xRes;
						holoScreenYStart = holoLineTripletYStartPixel -
										( ( unsigned ) holoContributionSampleStart ) / holoRenParams->m_xRes;
						holoScreenYEnd = holoLineTripletYStartPixel -
										( ( unsigned ) holoContributionSampleEnd ) / holoRenParams->m_xRes;
						//printf("line %d: scrXStart = %d scrXEnd = %d\n", holoLine, holoScreenXStart, holoScreenXEnd);
						//printf("line %d: scrYStart = %d scrYEnd = %d\n", holoLine, holoScreenYStart, holoScreenYEnd);
	
						screenLineSpan = holoScreenYStart - holoScreenYEnd + 1;
						//printf("DisplayHologram: projector fringe spans %d screenlines\n", screenLineSpan);
						for ( screenLine=0; screenLine<screenLineSpan; screenLine++ )
						{
							holoPixelYStart = holoScreenYStart - ( screenLine );
							holoPixelYEnd = holoPixelYStart - 1;
							holoPixelXStart = holoScreenXStart - ( screenLine*holoRenParams->m_xRes );
							holoPixelXEnd = holoPixelXStart + ( int ) m_samplesPerHoloPixel;
							//printf("drawing pixel %d %d on screenline %d, with viewrange[%f-%f] \n", i, holoLine, holoPixelYStart, startViewNumber, endViewNumber);
							DrawHoloPixel ( revScan, x, holoLine,
											startViewNumber, endViewNumber,
											holoPixelXStart, holoPixelYStart,
											holoPixelXEnd, holoPixelYEnd,
											holoRenParams->m_nHoloLinesPerFramebuffer,
											modulate );
						}
					}
				}
				// Need to shift the next projector fringe by the number of hologram
				// samples within the spacing between holoPoints.
				// If it's not the first projector fringe on a holoLine, it'll overlap the
				// last projector fringe; to blend, we use the accumulation buffer.
				holoContributionSampleStart += ( int ) m_samplesPerHoloPixelSpacing;

			} // end for each holoPoint on a holoLine
			glEnd();
		} // end for each line in a triplet of holoLines
	} // end for each triplet of holoLines.
}

/*
* Draw an emitter using a "multiline" basis fringe that spans several framebuffer rows.
* revScan 		-	boolean indicating scan direction.  Is false for first block of lines?
* startXsample	-	x-position in framebuffer of start of fringe
* startYsample	-	y-position in framebuffer of start of fringe
* tx, ty, tz	-	
*
* when called, TEXTURE0 should be fringe, TEXTURE1 is image we are modulating fringes by
*/
void RIPHologram::drawMultilineHolopixel(bool revScan, int startXsample, int startYsample, float tx, float ty, float tz)
{
	int holoScreenXStart; 
	int holoScreenYStart;
	int holoScreenXEnd;
	int holoScreenYEnd;
	
	float t0,t1;
	if(!revScan)
	
	{
		holoScreenXStart = startXsample % m_holorenparams->m_xRes; //pixel index into hololine
		holoScreenYStart = startYsample - floor( startXsample / m_holorenparams->m_xRes);
		
		glMultiTexCoord2fARB(GL_TEXTURE0_ARB, 0, 0);
		glMultiTexCoord3fARB(GL_TEXTURE1_ARB, tx, ty, tz);
		glVertex2d(holoScreenXStart + m_fringeTexWidth/2, holoScreenYStart - m_fringeTexHeight/2);
	
    // mapping two textures onto geometry at xmin, ymin
		glMultiTexCoord2fARB(GL_TEXTURE0_ARB, 0, 1);
		//glMultiTexCoord3fARB(GL_TEXTURE1_ARB, tx, ty, tz);
		glVertex2d(holoScreenXStart + m_fringeTexWidth/2, holoScreenYStart + m_fringeTexHeight/2);
	
    // mapping two textures onto geometry at xmax, ymin
		glMultiTexCoord2fARB(GL_TEXTURE0_ARB, 1, 1);
		//glMultiTexCoord3fARB(GL_TEXTURE1_ARB, tx, ty, tz);
		glVertex2d(holoScreenXStart - m_fringeTexWidth/2, holoScreenYStart + m_fringeTexHeight/2);
	
    // mapping two textures onto geometry at xmax, ymax
		glMultiTexCoord2fARB(GL_TEXTURE0_ARB, 1, 0);
		//glMultiTexCoord3fARB(GL_TEXTURE1_ARB, tx, ty, tz);
		glVertex2d(holoScreenXStart - m_fringeTexWidth/2, holoScreenYStart - m_fringeTexHeight/2);
	}
	else
	{
		holoScreenXStart = m_holorenparams->m_xRes - startXsample % m_holorenparams->m_xRes;
		holoScreenYStart = startYsample - floor( startXsample  / m_holorenparams->m_xRes);
	
		glMultiTexCoord2fARB(GL_TEXTURE0_ARB, 1, 1);
		glMultiTexCoord3fARB(GL_TEXTURE1_ARB, 1-tx, ty, tz);
		glVertex2d(holoScreenXStart + m_fringeTexWidth/2, holoScreenYStart - m_fringeTexHeight/2);
		
		// mapping two textures onto geometry at xmin, ymin
		glMultiTexCoord2fARB(GL_TEXTURE0_ARB, 1, 0);
		glVertex2d(holoScreenXStart + m_fringeTexWidth/2, holoScreenYStart + m_fringeTexHeight/2);
		
		// mapping two textures onto geometry at xmax, ymin
		glMultiTexCoord2fARB(GL_TEXTURE0_ARB, 0, 0);
		glVertex2d(holoScreenXStart - m_fringeTexWidth/2, holoScreenYStart + m_fringeTexHeight/2);
		
		// mapping two textures onto geometry at xmax, ymax
		glMultiTexCoord2fARB(GL_TEXTURE0_ARB, 0, 1);
		glVertex2d(holoScreenXStart - m_fringeTexWidth/2, holoScreenYStart - m_fringeTexHeight/2);
	}
}


void RIPHologram::DrawHoloPixel(int reverseLine, int holoPixelIndex, int line,
                                float startViewNumber, float endViewNumber,
                                int holoPixelXStart, int holoPixelYStart,
                                int holoPixelXEnd, int holoPixelYEnd,
                                int linesPerFB, int modulationFlag)
{
	if (modulationFlag) {
        //normal line direction
        if(!reverseLine)
		{
			RenderHoloPixel(holoPixelIndex, line, startViewNumber, endViewNumber,
							holoPixelXStart, holoPixelYStart, holoPixelXEnd, holoPixelYEnd,
							0.0, m_fringeTexCoordScale, linesPerFB);
		}
        else //reverse line direction
		{
			RenderHoloPixel(holoPixelIndex, line, startViewNumber, endViewNumber,
							holoPixelXEnd, holoPixelYStart, holoPixelXStart, holoPixelYEnd,
							0.0, m_fringeTexCoordScale, linesPerFB);
		}
    } else {
        //normal line direction
        if(!reverseLine)
		{
			RenderUnmodulatedHoloPixel(holoPixelIndex, line, 
									   holoPixelXStart, holoPixelYStart, holoPixelXEnd, holoPixelYEnd,
									   0.0, m_fringeTexCoordScale);
		}
        else //reverse line direction: want this line to be left-right reversed?
		{
			RenderUnmodulatedHoloPixel(holoPixelIndex, line, 
									   holoPixelXEnd, holoPixelYStart, holoPixelXStart, holoPixelYEnd,
									   0.0, m_fringeTexCoordScale);
		}
    }
	
}




void RIPHologram::RenderHoloPixel(int holoPixelIndex, int line,
                                  float startViewNumber, float endViewNumber,
                                  int holoPixelXStart, int holoPixelYStart,
                                  int holoPixelXEnd, int holoPixelYEnd,
                                  float startFringeTexCoord, float endFringeTexCoord,
                                  int linesPerFB)
{
	
    GLfloat tx, ty, tzmin, tzmax;
    int nonlinearTexLineIndex;
	GLfloat fringeRowStart = m_fringeToUse/NUM_PRECOMPUTED_FRINGES;
	GLfloat fringeRowEnd = m_fringeToUse/NUM_PRECOMPUTED_FRINGES;
	
//    nonlinearTexLineIndex = ((line/18)*3) + (line%3); //finds the hololine's row in 3D texture (which only stores data for lines used by this framebuffer) 
    tx = m_parallaxStackTexViewXScale*(float)holoPixelIndex/(float)(m_viewXRes-1);
//    ty = m_parallaxStackTexViewYScale*(float)nonlinearTexLineIndex/(float)(linesPerFB-1) ;
    ty = m_parallaxStackFBOTexViewYScale*(float)line/(float)(m_viewYRes-1) ;

    tzmin = m_parallaxStackTexNXViewsScale*startViewNumber/(float)(m_nXViews-1);
    tzmax = m_parallaxStackTexNXViewsScale*endViewNumber/(float)(m_nXViews-1);
	
    //    if (holoPixelIndex == 0)
    //        printf("line: %d index: %d  texcoord: %f  \n", line, nonlinearTexLineIndex, ty);
	
    // mapping two textures onto geometry at xmin, ymax
	glMultiTexCoord2fARB(GL_TEXTURE0_ARB, startFringeTexCoord, fringeRowStart);
    glMultiTexCoord3fARB(GL_TEXTURE1_ARB, tx, ty, tzmin);
    glVertex2d((GLdouble)holoPixelXStart, (GLdouble)holoPixelYStart);
	
    // mapping two textures onto geometry at xmin, ymin
	glMultiTexCoord2fARB(GL_TEXTURE0_ARB, startFringeTexCoord, fringeRowEnd);
    glMultiTexCoord3fARB(GL_TEXTURE1_ARB, tx, ty, tzmin);
    glVertex2d((GLdouble)holoPixelXStart, (GLdouble)holoPixelYEnd);
	
    // mapping two textures onto geometry at xmax, ymin
	glMultiTexCoord2fARB(GL_TEXTURE0_ARB, endFringeTexCoord, fringeRowEnd);
    glMultiTexCoord3fARB(GL_TEXTURE1_ARB, tx, ty, tzmax);
    glVertex2d((GLdouble)holoPixelXEnd, (GLdouble)holoPixelYEnd);
	
    // mapping two textures onto geometry at xmax, ymax
	glMultiTexCoord2fARB(GL_TEXTURE0_ARB, endFringeTexCoord, fringeRowStart);
    glMultiTexCoord3fARB(GL_TEXTURE1_ARB, tx, ty, tzmax);
    glVertex2d((GLdouble)holoPixelXEnd, (GLdouble)holoPixelYStart);
    //    printf("drew holopixel [%d %d] on screenline %d, with viewrange[%f-%f] \n\n", holoPixelIndex, line, holoPixelYStart, startViewNumber, endViewNumber);
}



void RIPHologram::RenderUnmodulatedHoloPixel(int holoPixelIndex, int line,
                                             int holoPixelXStart, int holoPixelYStart,
                                             int holoPixelXEnd, int holoPixelYEnd,
                                             float startFringeTexCoord, float endFringeTexCoord) {
	
	GLfloat fringeRowStart = m_fringeToUse/NUM_PRECOMPUTED_FRINGES;
	GLfloat fringeRowEnd = m_fringeToUse/NUM_PRECOMPUTED_FRINGES;
												 
												 
    // mapping two textures onto geometry at xmin, ymax
	glMultiTexCoord2fARB(GL_TEXTURE0_ARB, startFringeTexCoord, fringeRowStart);
    glVertex2d((GLdouble)holoPixelXStart, (GLdouble)holoPixelYStart);
	
    // mapping two textures onto geometry at xmin, ymin
	glMultiTexCoord2fARB(GL_TEXTURE0_ARB, startFringeTexCoord, fringeRowEnd);
    glVertex2d((GLdouble)holoPixelXStart, (GLdouble)holoPixelYEnd);
	
    // mapping two textures onto geometry at xmax, ymin
	glMultiTexCoord2fARB(GL_TEXTURE0_ARB, endFringeTexCoord, fringeRowEnd);
    glVertex2d((GLdouble)holoPixelXEnd, (GLdouble)holoPixelYEnd);
	
    // mapping two textures onto geometry at xmax, ymax
	glMultiTexCoord2fARB(GL_TEXTURE0_ARB, endFringeTexCoord, fringeRowStart);
    glVertex2d((GLdouble)holoPixelXEnd, (GLdouble)holoPixelYStart);
	
}

void RIPHologram::loadViewSet(char* basename)
{
	char imname[1024];
	char fName[1] = {'\0'};
	int imnum = 1;
	int done = 0;
	unsigned int * stackTextureData;
	unsigned int * tempTexture;
	
	stackTextureData = (unsigned int *) malloc(m_viewTexXRes*m_viewFBOTexYRes*m_nTexXViews*sizeof(uint32));
	tempTexture = (unsigned int *) malloc(m_viewTexXRes*m_viewFBOTexYRes*sizeof(uint32));

	memset(stackTextureData,0,m_viewTexXRes*m_viewFBOTexYRes*m_nTexXViews);
	
	for(imnum = 1; imnum <= m_nXViews; imnum++) 
	{
		TIFF *tif;

		
		sprintf(imname, "%s%03d.tif",basename, imnum);
		
		printf("loading texture file in tiff format: %s\n", imname);
		tif = TIFFOpen(imname, "r");
		if(!tif)
		{	printf("Failed to open image file stack\n");
			return;
			
		}
		int texW, texH;
		TIFFGetField(tif, TIFFTAG_IMAGEWIDTH, &texW);
		TIFFGetField(tif, TIFFTAG_IMAGELENGTH, &texH);

		tempTexture = (unsigned int *) malloc(texW*texH*sizeof(uint32));

		
		if(!TIFFReadRGBAImage(tif, texW, texH, (uint32 *) tempTexture /*+ ((imnum-1)*m_viewTexXRes*m_viewTexYRes)*/, 0))

//		if(!TIFFRead(tif, texW, texH, (uint32 *) stackTextureData /*+ ((imnum-1)*m_viewTexXRes*m_viewTexYRes)*/, 0))

		{
			printf("invalid tiff file for texture data: %s", imname);

		}
		
		//copy temp texture into lower left corner of appropriate 3D texture slice
		
		for (int h = 0;h<texH;h++)
		{
			for(int w = 0;w<texW;w++)
			{
//regular
//    			stackTextureData[h*m_viewTexXRes+w+ ((imnum-1)*m_viewTexXRes*m_viewFBOTexYRes)] =  tempTexture[h*texW+w];
//inverted
    			stackTextureData[(texH-h)*m_viewTexXRes+w+ ((imnum-1)*m_viewTexXRes*m_viewFBOTexYRes)] =  tempTexture[h*texW+w];
				
			}
		}
		
		
		free(tempTexture);
		TIFFClose(tif);	
		
	} 
	GLuint texName;
	
	glPixelStorei(GL_UNPACK_ALIGNMENT, 1);
//	glGenTextures(1, &texName);	
	glBindTexture(GL_TEXTURE_3D, m_parallaxStackFBOTexID);
	
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_3D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	
//	glTexImage3D(GL_TEXTURE_3D, 0, GL_RGBA, m_viewTexXRes,
//				 m_viewTexYRes, m_nTexXViews, 0, GL_RGBA,
//				 GL_UNSIGNED_BYTE, stackTextureData);
	
	glTexImage3D(GL_TEXTURE_3D, 0, PARALLAX_STACK_FORMAT, m_viewTexXRes,
				 m_viewFBOTexYRes, m_nTexXViews, 0, GL_RGBA,
				 GL_UNSIGNED_BYTE, stackTextureData);
}

