#pragma once

#ifdef DSCP4_HAVE_LOG4CXX
#include <log4cxx/logger.h>
#include <log4cxx/patternlayout.h>
#include <log4cxx/consoleappender.h>
#include <log4cxx/basicconfigurator.h>
#else
#define LOG4CXX_TRACE(logger, expression)    
#define LOG4CXX_DEBUG(logger, expression)    
#define LOG4CXX_INFO(logger, expression)   
#define LOG4CXX_WARN(logger, expression)    
#define LOG4CXX_ERROR(logger, expression)    
#define LOG4CXX_FATAL(logger, expression) 
#endif

#ifdef WIN32
#include <Windows.h>
#endif

#include <glm/glm.hpp>

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
#include "vsShaderLib.h"  // Shader class to make it easy to create GLSL shader objects
#include "Miniball.hpp"  // For calculating the bounding sphere of 3D mesh

#include <boost/filesystem.hpp>

#define DSCP4_DEFAULT_VOXEL_SIZE 5.0f
#define DSCP4_XINERAMA_ENABLED true
#define DSCP4_LIGHTING_SHADER_VERTEX_FILENAME "pointlight.vert"
#define DSCP4_LIGHTING_SHADER_FRAGMENT_FILENAME "pointlight.frag"
#define DSCP4_AUTO_SCALE_ENABLED true
#define DSCP4_LIGHTING_SHADE_MODEL DSCP4_SHADE_MODEL_FLAT
#define DSCP4_RENDER_MODE_DEFAULT DSCP4_RENDER_MODE_HOLOVIDEO_FRINGE

namespace dscp4
{
	class DSCP4Render
	{
	public:
		
		DSCP4Render();
		DSCP4Render(const char* shadersPath, const char* lightingShaderVertexFileName, const char* lightingShaderFragmentFileName);
		~DSCP4Render();
		bool init();
		void deinit();

		void addSimpleObject(simple_object_t object, float *center, float size);
		
		// Finds the bounding sphere of a mesh, centers the mesh and scales it down or up to radius == 1.0
		void addMesh(const char *id, int numVertices, float *vertices, float * normals = nullptr, float *colors = nullptr, unsigned int numVertexDimensions = 3, unsigned int numColorChannels = 4);
		void removeMesh(const char *id);
		//void addMesh(const char* id, int numVertices, float *vertices);
		//Expects PCL point cloud data type, or array of struct { float x,y,z,w; uchar r,g,b,a; }
		//You can simply pass the pointer to the PCL::PointCloudPtr->Data[] array
		void addPointCloud(const char *id, float *points, int numPoints, bool hasColorData = true);
		void removePointCloud(const char *id) { this->removeMesh(id); }

		void setRenderMode(render_mode_t renderMode) { renderMode_ = renderMode; }
		void setShadingModel(shade_model_t shadeModel) { shadeModel_ = shadeModel; }
		void setAutoScaleEnabled(bool autoScaleEnabled) { autoScaleEnabled_ = autoScaleEnabled; }

		void translateMesh(std::string meshID, float x, float y, float z);
		void rotateMesh(std::string meshID, float angle, float x, float y, float z);
		void scaleMesh(std::string meshID, float x, float y, float z);

		void* getContext(); 

	private:

		bool initWindow(SDL_Window*& window, SDL_GLContext& glContext, int thisWindowNum);
		void deinitWindow(SDL_Window*& window, SDL_GLContext& glContext, int thisWindowNum);


		bool initLightingShader(int which);
		void deinitLightingShader(int which);

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

		int numWindows_;
		int currentWindow_;

		bool isFullScreen_;
		int *windowWidth_, *windowHeight_;

		SDL_Window **windows_;
		SDL_GLContext *glContexts_;

		std::map<std::string, mesh_t> meshes_;

		boost::filesystem::path shadersPath_;
		std::string lightingShaderVertexFileName_;
		std::string lightingShaderFragmentFileName_;

		VSShaderLib* lightingShader_;

		float rotateAngleX_;
		float rotateAngleY_;
		float rotateIncrement_;
		bool rotateOn_;

		shade_model_t shadeModel_;
		bool autoScaleEnabled_;
		render_mode_t renderMode_;

#ifdef DSCP4_HAVE_LOG4CXX
		log4cxx::LoggerPtr logger_ = log4cxx::Logger::getLogger("edu.mit.media.obmg.holovideo.dscp4.lib.renderer");
#endif

	};

	static DSCP4Render *gCurrentDSCP4Instance = nullptr;

}
