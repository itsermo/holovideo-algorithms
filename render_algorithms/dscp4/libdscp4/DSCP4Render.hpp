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

//glew not necessary in apple
#ifndef __APPLE__
#include <GL/glew.h>
#include <GL/gl.h>
#include <GL/glu.h>
#else

#include <OpenGL/gl.h>
#include <OpenGL/gl3.h>
#endif

#ifdef DSCP4_HAVE_CUDA
#include <kernels/dscp4-fringe-cuda.h>
#endif

#ifdef DSCP4_HAVE_OPENCL
#include <kernels/dscp4-fringe-opencl.h>
#endif

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

#define DSCP4_RENDER_DEFAULT_ZNEAR 0.00001f
#define DSCP4_RENDER_DEFAULT_ZFAR 2.25f

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

	enum DrawMode
	{
		DSCP4_DRAW_MODE_COLOR,
		DSCP4_DRAW_MODE_DEPTH
	};

	class DSCP4Render
	{
	public:

		DSCP4Render();
		DSCP4Render(render_options_t *renderOptions,
			algorithm_options_t *algorithmOptions,
			display_options_t displayOptions,
			unsigned int verbosity, void * logAppender = nullptr);
		~DSCP4Render();

		bool init();
		void deinit();

		// Finds the bounding sphere of a mesh, centers the mesh and scales it down or up to radius == 1.0
		void addMesh(const char *id, int numIndecies, int numVertices, float *vertices, float * normals = nullptr, float *colors = nullptr, unsigned int numVertexDimensions = 3, unsigned int numColorChannels = 4);
		void removeMesh(const char *id);

		void addPointCloud(const char *id, unsigned int numPoints, void * cloudData);

		void setRenderMode(render_mode_t renderMode) { renderOptions_->render_mode = renderMode; }
		void setShadingModel(shader_model_t shadeModel) { renderOptions_->shader_model = shadeModel; }
		void setAutoScaleEnabled(bool autoScaleEnabled) { renderOptions_->auto_scale_enabled = autoScaleEnabled; }

		void translateMesh(std::string meshID, float x, float y, float z);
		void rotateMesh(std::string meshID, float angle, float x, float y, float z);
		void scaleMesh(std::string meshID, float x, float y, float z);

		bool getSpinOn() { return spinOn_; }
		void setSpinOn(bool spinOn) { spinOn_ = spinOn; Update(); }
		void setRotateIncrement(float rotateIncrement) { rotateIncrement_ = rotateIncrement; }
		float getRotateIncrement() { return rotateIncrement_; }

		float getRotateViewAngleX() { return rotateAngleX_; }
		float getRotateViewAngleY() { return rotateAngleY_; }
		float getRotateViewAngleZ() { return rotateAngleZ_; }
		void setRotateViewAngleX(float angleX) { rotateAngleX_ = angleX; cameraChanged_ = true; }
		void setRotateViewAngleY(float angleY) { rotateAngleY_ = angleY; cameraChanged_ = true; }
		void setRotateViewAngleZ(float angleZ) { rotateAngleZ_ = angleZ; cameraChanged_ = true; }

		Lighting getLighting() { std::lock_guard<std::mutex> lg(lightingMutex_); return lighting_; }
		Camera getCameraView() { std::lock_guard<std::mutex> lg(cameraMutex_);  return camera_; }

		void setCameraView(Camera cameraView) { std::lock_guard<std::mutex> lg(cameraMutex_);  camera_ = cameraView; cameraChanged_ = true; }
		void setLighting(Lighting lighting) { std::lock_guard<std::mutex> lg(lightingMutex_);  lighting_ = lighting; lightingChanged_ = true; }

		// For viewing, or stereogram mode, shows either color or depth to the window
		void setDrawMode(DrawMode drawMode) { drawMode_ = drawMode; }
		DrawMode getDrawMode() { return drawMode_; }

		// Force frame to redraw
		void Update() { cameraChanged_ = true; }

		bool isFullScreen() { return isFullScreen_.load(); }
		void setFullScreen(bool fullscreen);

		void setEventCallback(dscp4_event_cb_t eventCallback, void * parent = 0) { parentCallback_ = parent; eventCallback_ = eventCallback; }

		void saveScreenshot() { shouldSaveScreenshot_ = true; }

	private:

#ifdef DSCP4_HAVE_PNG
		void saveScreenshotPNG();
#endif

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

		// Initializes textures for storing generated view color
		// and depth (for viewing render mode)
		void initViewingTextures();
		void deinitViewingTextures();

		// Initializes textures (and PBOs--if necessary) for storing
		// stereogram views (for stereogram and holovideo render modes)
		void initStereogramTextures();
		void deinitStereogramTextures();

		// Initializes textures (and PBOs--if necessary) for holovideo fringe output
		void initFringeTextures();
		void deinitFringeTextures();

		// calculates the algorithm cached parameters for
		// generating stereogram views and fringe pattern computation
		void updateAlgorithmOptionsCache();

		// The general rendering loop, this function is run for all render modes
		// sets up the renderer and calls the appropriate draw() function
		// which corresponds to the selected rendermode
		void renderLoop();			
		
		// Generates a view to an FBO for model viewing mode
		void generateView();

		// Generates an N-view stereogram to a single FBO
		void generateStereogram();

		// Generates a single view of the model with aspect ratio from algorithm options
		void drawForViewing();

		// Generates stereograms as a texture, and then calls drawStereogramTexture()
		void drawForStereogram();

		// Generates stereograms and displays each view in each window
		void drawForAerialDisplay();
		
		// Generates stereograms, computes hologram, and then calls drawFringeTextures()
		void drawForFringe();     

		// Draws the model viewing texture to the GL back buffer
		// This is the last step for model viewing mode
		void drawViewingTexture();

		// Draws the stereogram texture to the GL back buffer
		// This is the last step in stereogram viewing mode
		void drawStereogramTexture();

		// Copies the the fringe pattern FBO
		// to each texture (texture per window)
		// (this is the final step, after compute hologram)
		void drawFringeTextures();

		// Copies the stereogram data to a PBO
		// This is done after generating views, meant for passing
		// to CUDA/OpenCL kernel for processing
		void copyStereogramDepthToPBO();

		// Will init/deinit OpenCL/CUDA
		void initComputeMethod();
		void deinitComputeMethod();
		
		// Uses CUDA/OpenCL to compute the hologram
		void computeHologram();

		// Drawing mesh/point cloud functions
		void drawPointCloud();
		void drawAllMeshes();
		void drawMesh(mesh_t& mesh);
		void drawObjects();

		std::atomic<bool> isFullScreen_;

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

		unsigned int numWindows_;

		unsigned int *windowWidth_, *windowHeight_;

		SDL_Window **windows_;
		SDL_GLContext *glContexts_;

		// Handle the input on window (keyboard/mouse)
		int inputStateChanged(SDL_Event* event);

		std::map<std::string, mesh_t> meshes_;

		VSShaderLib* lightingShader_;

		std::atomic<float> rotateAngleX_;
		std::atomic<float> rotateAngleY_;
		std::atomic<float> rotateAngleZ_;
		std::atomic<float> rotateIncrement_;
		std::atomic<bool> spinOn_;

		render_options_t *renderOptions_;

		dscp4_fringe_context_t fringeContext_;
		void * computeContext_;

		glm::mat4 projectionMatrix_;
		glm::mat4 viewMatrix_;
		glm::mat4 modelMatrix_;

		Camera camera_;
		Lighting lighting_;
		DrawMode drawMode_;

		void * parentCallback_;
		dscp4_event_cb_t eventCallback_;
		frame_data_t renderPreviewData_;
		unsigned char * renderPreviewBuffer_;

		std::atomic<bool> shouldSaveScreenshot_;

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
