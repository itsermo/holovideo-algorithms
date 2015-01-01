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

#define DSCP4_RENDER_DEFAULT_ZNEAR 0.01f
#define DSCP4_RENDER_DEFAULT_ZFAR 5.0f

namespace dscp4
{

	struct Camera
	{
		glm::vec3 eye;
		glm::vec3 center;
		glm::vec3 up;
	};

	struct Lighting
	{
		glm::vec4 position;
		glm::vec4 ambientColor;
		glm::vec4 diffuseColor;
		glm::vec4 specularColor;
		glm::vec4 globalAmbientColor;
	};

	class DSCP4Render
	{
	public:

		DSCP4Render();
		DSCP4Render(render_options_t renderOptions,
			algorithm_options_t algorithmOptions,
			display_options_t displayOptions,
			unsigned int verbosity);
		~DSCP4Render();

		bool init();
		void deinit();

		// Finds the bounding sphere of a mesh, centers the mesh and scales it down or up to radius == 1.0
		void addMesh(const char *id, int numVertices, float *vertices, float * normals = nullptr, float *colors = nullptr, unsigned int numVertexDimensions = 3, unsigned int numColorChannels = 4);
		void removeMesh(const char *id);

		//Expects PCL point cloud data type, or array of struct { float x,y,z,w; uchar r,g,b,a; }
		//You can simply pass the pointer to the PCL::PointCloudPtr->Data[] array
		void addPointCloud(const char *id, float *points, int numPoints, float pointSize, bool hasColorData = true);
		void removePointCloud(const char *id) { this->removeMesh(id); }

		void setRenderMode(render_mode_t renderMode) { renderOptions_.render_mode = renderMode; }
		void setShadingModel(shader_model_t shadeModel) { renderOptions_.shader_model = shadeModel; }
		void setAutoScaleEnabled(bool autoScaleEnabled) { renderOptions_.auto_scale_enabled = autoScaleEnabled; }

		void translateMesh(std::string meshID, float x, float y, float z);
		void rotateMesh(std::string meshID, float angle, float x, float y, float z);
		void scaleMesh(std::string meshID, float x, float y, float z);

		bool getSpinOn() { return spinOn_; }
		void setSpinOn(bool spinOn) { spinOn_ = spinOn; Update(); }
		void setRotateIncrement(float rotateIncrement) { rotateIncrement_ = rotateIncrement; }
		float getRotateIncrement() { return rotateIncrement_; }

		float getRotateViewAngleX() { return rotateAngleX_; }
		float getRotateViewAngleY() { return rotateAngleY_; }
		void setRotateViewAngleX(float angleX) { rotateAngleX_ = angleX; cameraChanged_ = true; }
		void setRotateViewAngleY(float angleY) { rotateAngleY_ = angleY; cameraChanged_ = true; }

		Lighting getLighting() { std::lock_guard<std::mutex> lg(lightingMutex_); return lighting_; }
		Camera getCameraView() { std::lock_guard<std::mutex> lg(cameraMutex_);  return camera_; }

		void setCameraView(Camera cameraView) { std::lock_guard<std::mutex> lg(cameraMutex_);  camera_ = cameraView; cameraChanged_ = true; }
		void setLighting(Lighting lighting) { std::lock_guard<std::mutex> lg(lightingMutex_);  lighting_ = lighting; lightingChanged_ = true; }

		// Force frame to redraw
		void Update() { updateFrameCV_.notify_all(); }

	private:

		// for testing
		void drawCube();

		static glm::mat4 buildOrthoXPerspYProjMat(
			float left,
			float right,
			float bottom,
			float top,
			float zNear,
			float zFar,
			float q
			);

		bool isRunning() { return isInit_; }

		bool initWindow(SDL_Window*& window, SDL_GLContext& glContext, int thisWindowNum);
		void deinitWindow(SDL_Window*& window, SDL_GLContext& glContext, int thisWindowNum);

		bool initLightingShader(int which);
		void deinitLightingShader(int which);

		void renderLoop();			// The general rendering loop
		void drawForViewing();
		void drawForStereogram(); // Generates and renders stereograms
		void drawForFringe(int which);     // Renders the fringe pattern from stereograms

		void glCheckErrors();

		void drawPointCloud();
		void drawAllMeshes();
		void drawMesh(mesh_t& mesh);
		void drawObjects();

		// Renderer projection settings
		float zNear_;
		float zFar_;

		std::mutex meshMutex_;
		std::mutex lightingMutex_;
		std::mutex cameraMutex_;

		std::atomic<bool> meshChanged_;
		std::atomic<bool> cameraChanged_;
		std::atomic<bool> lightingChanged_;

		std::atomic<bool> isInit_;

		std::mutex isInitMutex_;
		std::condition_variable isInitCV_;

		std::atomic<bool> shouldRender_;
		std::thread renderThread_;
		std::mutex updateFrameMutex_;
		std::condition_variable updateFrameCV_;

		std::mutex glContextMutex_;
		std::condition_variable glContextCV_;

		int numWindows_;
		int currentWindow_;

		bool isFullScreen_;
		int *windowWidth_, *windowHeight_;

		SDL_Window **windows_;
		SDL_GLContext *glContexts_;
		static int inputStateChanged(void* userdata, SDL_Event* event);

		std::map<std::string, mesh_t> meshes_;

		VSShaderLib* lightingShader_;

		std::atomic<float> rotateAngleX_;
		std::atomic<float> rotateAngleY_;
		std::atomic<float> rotateIncrement_;
		std::atomic<bool> spinOn_;

		render_options_t renderOptions_;
		algorithm_options_t algorithmOptions_;
		display_options_t displayOptions_;

		glm::mat4 projectionMatrix_;
		glm::mat4 viewMatrix_;
		glm::mat4 modelMatrix_;

		Camera camera_;
		Lighting lighting_;

#ifdef DSCP4_HAVE_LOG4CXX
		log4cxx::LoggerPtr logger_ = log4cxx::Logger::getLogger("edu.mit.media.obmg.holovideo.dscp4.lib.renderer");
#endif

		const float DEG_TO_RAD = (float)M_PI / 180.0f; // just multiply this constant to degrees to get radians

#ifdef DSCP4_ENABLE_TRACE_LOG
		// function for measuring time elapsed when executing a function
		template<typename F, typename ...Args>
		static typename std::chrono::milliseconds::rep measureTime(F func, Args&&... args)
		{
			auto start = std::chrono::system_clock::now();

			// Now call the function with all the parameters you need.
			func(std::forward<Args>(args)...);

			auto duration = std::chrono::duration_cast<std::chrono::milliseconds>
				(std::chrono::system_clock::now() - start);

			return duration.count();
		};
#endif

	};
}
