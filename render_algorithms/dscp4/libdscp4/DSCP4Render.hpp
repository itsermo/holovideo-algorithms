#pragma once

#ifdef WIN32
#include <Windows.h>
#endif

#include <GL/gl.h>
#include <SDL2/SDL.h>
#include <string>
#include <atomic>
#include <mutex>
#include <condition_variable>

#ifdef DSCP4_HAVE_LOG4CXX
#include <log4cxx/logger.h>
#endif

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
		DSCP4Render(float voxelSize, bool xineramaEnabled = false);
		~DSCP4Render();
		bool init();
		void deinit();

		void addSimpleObject(SIMPLE_OBJECT_TYPE object);

		void* getContext();

	private:

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

#ifdef DSCP4_HAVE_LOG4CXX
		log4cxx::LoggerPtr logger_ = log4cxx::Logger::getLogger("edu.mit.media.obmg.holosuite.codec.h264");
#endif

		SDL_Window *window_;
		SDL_GLContext glContext_;

		

	};

	static DSCP4Render *gCurrentDSCP4Instance = nullptr;

}
