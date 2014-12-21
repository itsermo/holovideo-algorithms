#pragma once

#ifdef DSCP4_HAVE_LOG4CXX
#include <log4cxx/logger.h>
#endif

#ifdef WIN32
#include <Windows.h>
#endif

#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glu.h>
#include <SDL2/SDL.h>

#include <string>
#include <atomic>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <map>

#include "dscp4_defs.h"  // Some standard definitions for vector data types
#include "GLSLShader.h"  // Shader class to make it easy to create GLSL shader objects
#include "Miniball.hpp"  // For calculating the bounding sphere of 3D mesh


#define DSCP4_DEFAULT_VOXEL_SIZE 5
#define DSCP4_XINERAMA_ENABLED true

namespace dscp4
{
	class DSCP4Render
	{
	public:
		
		enum SIMPLE_OBJECT_TYPE{
			SIMPLE_OBJECT_TYPE_SPHERE = 1,
			SIMPLE_OBJECT_TYPE_CUBE = 2,
			SIMPLE_OBJECT_TYPE_PYRAMID = 3
		};

		DSCP4Render();
		DSCP4Render(float voxelSize, bool xineramaEnabled = false);
		~DSCP4Render();
		bool init();
		void deinit();

		void addSimpleObject(SIMPLE_OBJECT_TYPE object, float *center, float size);
		
		// Finds the bounding sphere of a mesh, centers the mesh and scales it down or up to radius == 1.0
		void addMesh(const char *id, int numVertices, float *vertices, float *colors, unsigned int numVertexDimensions = 3, unsigned int numColorChannels = 4);
		void removeMesh(const char *id);
		//void addMesh(const char* id, int numVertices, float *vertices);
		//Expects PCL point cloud data type, or array of struct { float x,y,z,w; uchar r,g,b,a; }
		//You can simply pass the pointer to the PCL::PointCloudPtr->Data[] array
		void addPointCloud(const char *id, float *points, int numPoints, bool hasColorData = true);
		void removePointCloud(const char *id) { this->removeMesh(id); }

		void* getContext();

	private:

		bool initHead(SDL_Window*& window, SDL_GLContext& glContext, int thisHeadNum);
		void deinitHead(SDL_Window*& window, SDL_GLContext& glContext, int thisHeadNum);

		void renderLoop();

		void glCheckErrors();

		void drawPointCloud();
		void drawMesh(const mesh_t& mesh);
		void drawObjects();
		void drawCube();

		void drawBackgroundGrid(GLfloat width, GLfloat height, GLfloat depth);
		void drawSphere(GLfloat x, GLfloat y, GLfloat z, GLfloat radius);

		std::mutex localCloudMutex_;
		std::mutex meshMutex_;

		std::atomic<bool> haveNewRemoteCloud_;

		bool isInit_;

		std::mutex isInitMutex_;
		std::condition_variable isInitCV_;

		std::atomic<bool> shouldRender_;
		std::thread renderThread_;

		std::mutex glContextMutex_;
		std::condition_variable glContextCV_;

		float voxelSize_;
		int numHeads_;
		int currentHead_;

		int mouseLeftButton_;
		int mouseMiddleButton_;
		int mouseRightButton_;
		int mouseDownX_;
		int mouseDownY_;

		bool isFullScreen_;
		int *windowWidth_, *windowHeight_;

		int prevWindowWidth_, prevWindowHeight_;
		int windowX_, windowY_;
		int prevWindowX_, prevWindowY_;

		float viewPhi_, viewTheta_, viewDepth_;

		bool xineramaEnabled_;

		SDL_Window **windows_;
		SDL_GLContext *glContexts_;

		std::map<std::string, mesh_t> meshes_;

		GLSLShader diffuseLightShader_;

#ifdef DSCP4_HAVE_LOG4CXX
		log4cxx::LoggerPtr logger_ = log4cxx::Logger::getLogger("edu.mit.media.obmg.dscp4.lib.render");
#endif

	};

	static DSCP4Render *gCurrentDSCP4Instance = nullptr;

}
