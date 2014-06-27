#include <stdlib.h>
#include <string.h>
#include <string>
#include <stdio.h>
#include "setupglew.h"
#include <GL/glut.h>

//#include <glh_extensions.h>
#include "RIP.h"
#include "UI.h"
#include "holoren.h"

int displaystatic = 0;

using namespace std;

int main(int argc, char **argv)
{
	char configfile[512] = "models/ripcube128.xml";
	char configfile2[512] = "models/ripcube128.xml";
	int framebuffernum = 1;
	bool twoholograms = false;

	if ((argc == 4) && (string(argv[2], strlen(argv[2])).find("-i")
			!= string::npos))
	{
		displaystatic = 1;
		framebuffernum = atoi(argv[1]);
		printf("using prerendered hologram\n");
	}
	else if (argc == 3)
	{
		strcpy(configfile, argv[2]);
		framebuffernum = atoi(argv[1]);

	}
	else if (argc == 4)
	{
		printf("displaying two holograms overlaid\n");
		strcpy(configfile, argv[2]);
		strcpy(configfile2, argv[3]);
		framebuffernum = atoi(argv[1]);
		twoholograms = true;
	}
	else
	{
		printf("usage:\n");
		printf("%s frameBufferNumber renderConfig.xml\n", argv[0]);
		printf("%s frameBufferNumber -i basefilename_of_tif_image_stack\n",
				argv[0]);
#ifdef WORKSTATION_DEBUG
		printf("running with 1 models/chirpcube.xml\n");
#else
		exit(0);
#endif
	}

	if (twoholograms)
	{
		// create "ren" which has init, mouse keyboard, motion and render methods.
		// these methods are defined in holoren.cpp (why not holoConf.cpp?, ahwell...)
		// "ren" contains configurations for: render, lighting, materials, tex and models.
		printf(
				"creating new holo, render, lighting, materials, model and texture configs\n");
		holoConf *ren = new holoConf();
		holoConf *ren2 = new holoConf();
		// and this parses an XML file to set up those various configurations.
		printf("parsing XML file %s\n", configfile);
		ren->config(configfile);

		printf("parsing second XML file %s\n", configfile2);
		ren2->config(configfile2);

		// looks like we create a fresh set of holorender params and a rip holo for every machine.
		// HoloRenderParams is defined in RIP.h and includes holo dimensions and num FBs.
		// contains hardcoded parameter values.
		HoloRenderParams *holoRenParams = new HoloRenderParams(framebuffernum);
		HoloRenderParams *holoRenParams2 = new HoloRenderParams(framebuffernum);

		// RIPHologram is also defined in RIP.h, and includes hologram geometry, specs,
		// and methods to compute fringe, render and display hologram.
		// contains hardcoded parameter values.
		RIPHologram *ripParams = new RIPHologram(holoRenParams, ren);
		RIPHologram *ripParams2 = new RIPHologram(holoRenParams2, ren2);

		ripParams->m_flatrender = false;
		ripParams2->m_flatrender = false;

		ripParams->recomputeGeometry(holoRenParams);
		ripParams2->recomputeGeometry(holoRenParams2);

		// set up glut and gl -- display window stuff. Defined in UI.cpp.
		// Specifies display, keyboard, mouse, motion callbacks.
		// Display callback calls ripParams->DisplayRIPHologramSingleFramebuffer(holoRenParams)
		InitGL(argc, argv, holoRenParams, ripParams, holoRenParams2, ripParams2);

		// build the projector_fringe and generate textxure (See RIP.cpp)
		ripParams->AfterGLInitialize();
		ripParams2->AfterGLInitialize();
		// initialize the render...
		printf("Initalizing the renderer\n");
		ren->init(); //See HoloConf::init() in holoren.cpp
		ren2->init(); //See HoloConf::init() in holoren.cpp

		// renders, displays, and waits for callback. Keyboard Q or q to quit.

		glutMainLoop();
	}
	else
	{

		// create "ren" which has init, mouse keyboard, motion and render methods.
		// these methods are defined in holoren.cpp (why not holoConf.cpp?, ahwell...)
		// "ren" contains configurations for: render, lighting, materials, tex and models.
		printf(
				"creating new holo, render, lighting, materials, model and texture configs\n");
		holoConf *ren = new holoConf();
		// and this parses an XML file to set up those various configurations.
		printf("parsing XML file %s\n", configfile);
		ren->_parseConfigFile(configfile);

		// looks like we create a fresh set of holorender params and a rip holo for every machine.
		// HoloRenderParams is defined in RIP.h and includes holo dimensions and num FBs.
		// contains hardcoded parameter values.
		HoloRenderParams *holoRenParams = new HoloRenderParams(framebuffernum);
		// RIPHologram is also defined in RIP.h, and includes hologram geometry, specs,
		// and methods to compute fringe, render and display hologram.
		// contains hardcoded parameter values.
		RIPHologram *ripParams = new RIPHologram(holoRenParams, ren);
		if (displaystatic)
		{
			ripParams->m_useStaticImages = true;
			strcpy(ripParams->m_baseStaticImageFilename, argv[3]);
		}
		ripParams->m_flatrender = false;

		// set up glut and gl -- display window stuff. Defined in UI.cpp.
		// Specifies display, keyboard, mouse, motion callbacks.
		// Display callback calls ripParams->DisplayRIPHologramSingleFramebuffer(holoRenParams)
		InitGL(argc, argv, holoRenParams, ripParams);

		// build the projector_fringe and generate textxure (See RIP.cpp)
		ripParams->AfterGLInitialize();
		// initialize the render...
		printf("Initalizing the renderer\n");
		ren->init(); //See HoloConf::init() in holoren.cpp
		// renders, displays, and waits for callback. Keyboard Q or q to quit.

		//Some JB Diagnostics:
		int screenHeight = glutGet(GLUT_SCREEN_HEIGHT);
		int screenWidth = glutGet(GLUT_SCREEN_WIDTH);
		int windowWidth = glutGet(GLUT_WINDOW_WIDTH);
		int windowHeight = glutGet(GLUT_WINDOW_HEIGHT);
		int windowXPos = glutGet(GLUT_WINDOW_X);
		int windowYPos = glutGet(GLUT_WINDOW_Y);
		printf("JUST Before starting main loop,\n");
		printf("Screen is %d by %d\n", screenWidth, screenHeight);
		printf("OpenGL window is %d by %d\n", windowWidth, windowHeight);
		printf("Window is located at %d, %d\n", windowXPos, windowYPos);

		glutMainLoop();
	}
	return 0;
}
