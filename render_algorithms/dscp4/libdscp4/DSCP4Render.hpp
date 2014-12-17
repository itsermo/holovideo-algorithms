#pragma once

#ifdef WIN32
#include <Windows.h>
#endif

#include <GL/gl.h>
#include <string>
#include <atomic>
#include <mutex>
#include <condition_variable>

#define DSCP4_DEFAULT_VOXEL_SIZE 5
#define DSCP4_XINERAMA_ENABLED true

namespace dscp4
{
	class DSCP4Render
	{
	public:
		
		enum SIMPLE_OBJECT_TYPE{
			SIMPLE_OBJECT_TYPE_SPHERE = 0,
			SIMPLE_OBJECT_TYPE_CUBE = 1,
			SIMPLE_OBJECT_TYPE_PYRAMID = 2
		};

		DSCP4Render();
		DSCP4Render(int voxelSize, bool xineramaEnabled = false);
		~DSCP4Render();
		bool init();
		void deinit();

		void addSimpleObject();

		void* getContext();

		void display(void);
		void idle(void);
		void keyboard(unsigned char c, int x, int y);
		void cleanup(void);
		void mouse(int button, int state, int x, int y);
		void mouseMotion(int x, int y);
		void reshape(int width, int height);

	private:

		// GL and GLUT related functions
		static void glutDisplay();
		static void glutIdle();
		static void glutKeyboard(unsigned char c, int x, int y);
		static void glutCleanup();
		static void glutMouse(int button, int state, int x, int y);
		static void glutMouseMotion(int x, int y);
		static void glutReshape(int width, int height);

		void glutInitLoop();
		void glCheckErrors();

		void drawPointCloud();
		void drawObjects();

		void drawBackgroundGrid(GLfloat width, GLfloat height, GLfloat depth);
		void drawSphere(GLfloat x, GLfloat y, GLfloat z, GLfloat radius);

		std::mutex localCloudMutex_;

		std::atomic<bool> haveNewRemoteCloud_;


		bool isInit_;
		bool firstInit_;

		std::mutex hasInitMutex_;
		std::condition_variable hasInitCV_;

		float voxelSize_;
		int numHeads_;
		int currentHead_;

		int mouseLeftButton_;
		int mouseMiddleButton_;
		int mouseRightButton_;
		int mouseDownX_;
		int mouseDownY_;

		bool isFullScreen_;
		int windowWidth_, windowHeight_;
		int prevWindowWidth_, prevWindowHeight_;
		int windowX_, windowY_;
		int prevWindowX_, prevWindowY_;

		float viewPhi_, viewTheta_, viewDepth_;

		bool xineramaEnabled_;

	};

	static DSCP4Render *gCurrentDSCP4Instance = nullptr;

}
