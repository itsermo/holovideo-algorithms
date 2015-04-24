#include "DSCP4Render.hpp"

#include <sstream>
#include <iostream>
#include <iomanip>

#ifdef __LINUX__
#include <stdlib.h>
#endif

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#ifdef DSCP4_HAVE_PNG
#define png_infopp_NULL (png_infopp)NULL
#define int_p_NULL (int*)NULL
#include <boost/gil/gil_all.hpp>
#include <boost/gil/extension/io/png_io.hpp>
#endif

// This checks for a true condition, prints the error message, cleans up and returns false
#define CHECK_SDL_RC(rc_condition, what)				\
	if (rc_condition)									\
		{												\
			LOG4CXX_ERROR(logger_, what)				\
			LOG4CXX_ERROR(logger_, SDL_GetError())		\
			deinit();									\
			return false;								\
		}												\

#define CHECK_GLEW_RC(rc_condition, what)				\
if (rc_condition)										\
		{												\
		LOG4CXX_ERROR(logger_, what)					\
		LOG4CXX_ERROR(logger_, glewGetErrorString())	\
		}

#define CHECK_GL_RC(what)								\
if (glGetError() != GL_NO_ERROR)						\
		{												\
		LOG4CXX_ERROR(logger_, what)					\
		LOG4CXX_ERROR(logger_, glewGetErrorString())	\
		}												\

using namespace dscp4;

DSCP4Render::DSCP4Render() :
		DSCP4Render(new render_options_t {
						DSCP4_DEFAULT_RENDER_SHADERS_PATH,
						DSCP4_DEFAULT_RENDER_KERNELS_PATH,
						DSCP4_DEFAULT_RENDER_SHADER_FILENAME_PREFIX,
						DSCP4_DEFAULT_RENDER_RENDER_MODE,
						DSCP4_DEFAULT_RENDER_SHADER_MODEL,
						DSCP4_DEFAULT_RENDER_LIGHT_POS_X,
						DSCP4_DEFAULT_RENDER_LIGHT_POS_Y,
						DSCP4_DEFAULT_RENDER_LIGHT_POS_Z,
						DSCP4_DEFAULT_RENDER_AUTOSCALE_ENABLED },
					new algorithm_options_t {
						DSCP4_DEFAULT_ALGORITHM_NUM_VIEWS_X,
						DSCP4_DEFAULT_ALGORITHM_NUM_VIEWS_Y,
						DSCP4_DEFAULT_ALGORITHM_NUM_WAFELS,
						DSCP4_DEFAULT_ALGORITHM_NUM_SCANLINES,
						DSCP4_DEFAULT_ALGORITHM_FOV_X,
						DSCP4_DEFAULT_ALGORITHM_FOV_Y,
						DSCP4_DEFAULT_ALGORITHM_REF_BEAM_ANGLE,
						DSCP4_DEFAULT_ALGORITHM_TEMP_UPCONVERT_R,
						DSCP4_DEFAULT_ALGORITHM_TEMP_UPCONVERT_G,
						DSCP4_DEFAULT_ALGORITHM_TEMP_UPCONVERT_B,
						DSCP4_DEFAULT_ALGORITHM_WAVELENGTH_R,
						DSCP4_DEFAULT_ALGORITHM_WAVELENGTH_G,
						DSCP4_DEFAULT_ALGORITHM_WAVELENGTH_B,
						DSCP4_DEFAULT_ALGORITHM_GAIN_R,
						DSCP4_DEFAULT_ALGORITHM_GAIN_G,
						DSCP4_DEFAULT_ALGORITHM_GAIN_B,
						DSCP4_DEFAULT_ALGORITHM_Z_NEAR,
						DSCP4_DEFAULT_ALGORITHM_Z_FAR,
						DSCP4_DEFAULT_ALGORITHM_COMPUTE_METHOD,
						DSCP4_DEFAULT_ALGORITHM_OPENCL_KERNEL_FILENAME,
						{ DSCP4_DEFAULT_ALGORITHM_OPENCL_WORKSIZE_X ,
						DSCP4_DEFAULT_ALGORITHM_OPENCL_WORKSIZE_Y},
						{ DSCP4_DEFAULT_ALGORITHM_CUDA_BLOCK_DIM_X,
						DSCP4_DEFAULT_ALGORITHM_CUDA_BLOCK_DIM_Y },
						algorithm_cache_t() },
					    display_options_t {
						DSCP4_DEFAULT_DISPLAY_NAME,
						DSCP4_DEFAULT_DISPLAY_X11_ENV_VAR,
						DSCP4_DEFAULT_DISPLAY_NUM_HEADS,
						DSCP4_DEFAULT_DISPLAY_NUM_HEADS_PER_GPU,
						DSCP4_DEFAULT_DISPLAY_HEAD_RES_X,
						DSCP4_DEFAULT_DISPLAY_HEAD_RES_Y,
						DSCP4_DEFAULT_DISPLAY_HEAD_RES_X_SPEC,
						DSCP4_DEFAULT_DISPLAY_HEAD_RES_Y_SPEC,
						DSCP4_DEFAULT_DISPLAY_NUM_AOM_CHANNELS,
						DSCP4_DEFAULT_DISPLAY_NUM_SAMPLES_PER_HOLOLINE,
						DSCP4_DEFAULT_DISPLAY_HOLOGRAM_PLANE_WIDTH
						},
						DSCP4_DEFAULT_LOG_VERBOSITY, nullptr)
{
	
}

DSCP4Render::DSCP4Render(render_options_t *renderOptions,
	algorithm_options_t *algorithmOptions,
	display_options_t displayOptions,
	unsigned int verbosity,
	void * logAppender
	) :
	windows_(nullptr),
	glContexts_(nullptr),
	shouldRender_(false),
	isInit_(false),
	windowWidth_(nullptr),
	windowHeight_(nullptr),
	numWindows_(0),
	rotateAngleX_(0),
	rotateAngleY_(0),
	rotateAngleZ_(0),
	rotateIncrement_(1.0f),
	spinOn_(false),
	renderOptions_(renderOptions),
	lightingShader_(nullptr),
	projectionMatrix_(),
	viewMatrix_(),
	modelMatrix_(),
	camera_(),
	lighting_(),
	cameraChanged_(false),
	lightingChanged_(false),
	meshChanged_(false),
	isFullScreen_(false),
	drawMode_(DSCP4_DRAW_MODE_COLOR),
	renderPreviewBuffer_(nullptr),
	eventCallback_(nullptr),
	parentCallback_(nullptr),
	shouldSaveScreenshot_(false),
	fringeContext_({ algorithmOptions, displayOptions, nullptr, 0, 0, 0, 0, 0, 0, 0, nullptr })
{
#ifdef DSCP4_HAVE_LOG4CXX
	
	if (!logAppender)
	{
		log4cxx::BasicConfigurator::resetConfiguration();

#ifdef WIN32
		log4cxx::PatternLayoutPtr logLayoutPtr = new log4cxx::PatternLayout(L"%-5p %m%n");
#else
		log4cxx::PatternLayoutPtr logLayoutPtr = new log4cxx::PatternLayout("%-5p %m%n");
#endif

		log4cxx::ConsoleAppenderPtr logAppenderPtr = new log4cxx::ConsoleAppender(logLayoutPtr);
		log4cxx::BasicConfigurator::configure(logAppenderPtr);
	}

	switch (verbosity)
	{
	case 0:
		logger_->setLevel(log4cxx::Level::getFatal());
		break;
	case 1:
		logger_->setLevel(log4cxx::Level::getWarn());
		break;
	case 2:
		logger_->setLevel(log4cxx::Level::getError());
		break;
	case 3:
		logger_->setLevel(log4cxx::Level::getInfo());
		break;
	case 4:
		logger_->setLevel(log4cxx::Level::getDebug());
		break;
	case 5:
		logger_->setLevel(log4cxx::Level::getTrace());
		break;
	case 6:
		logger_->setLevel(log4cxx::Level::getAll());
		break;
	default:
		LOG4CXX_ERROR(logger_, "Invalid verbosity setting: " << verbosity)
		break;
	}

#endif

	if (renderOptions->shaders_path == nullptr)
	{
		LOG4CXX_WARN(logger_, "No shader path location specified, using current working path: " << boost::filesystem::current_path().string())
			renderOptions_->shaders_path = (char*)boost::filesystem::current_path().string().c_str();
	}

}

DSCP4Render::~DSCP4Render()
{
	delete fringeContext_.algorithm_options;
	delete renderOptions_;
}

bool DSCP4Render::init()
{
	LOG4CXX_INFO(logger_, "Initializing DSCP4...")

#if defined(__linux__) || defined(DSCP4_HAVE_X11)
	LOG4CXX_INFO(logger_, "Using X11 display " << fringeContext_.display_options.x11_env_var)
	int ret = setenv("DISPLAY", fringeContext_.display_options.x11_env_var, 1);
	if(ret !=0)
	{
		LOG4CXX_ERROR(logger_,"Could not set the display environment variable for X11")
	}
#endif

	LOG4CXX_INFO(logger_, "Initializing SDL with video subsystem")
	CHECK_SDL_RC(SDL_Init(SDL_INIT_VIDEO) < 0, "Could not initialize SDL")


	switch (renderOptions_->render_mode)
	{
	case DSCP4_RENDER_MODE_MODEL_VIEWING:
		numWindows_ = 1;
		break;
	case DSCP4_RENDER_MODE_STEREOGRAM_VIEWING:
		numWindows_ = 1;
		break;
	case DSCP4_RENDER_MODE_AERIAL_DISPLAY:
#ifdef _DEBUG
		numWindows_ = 6;
#else
		numWindows_ = SDL_GetNumVideoDisplays();
#endif
		break;
	case DSCP4_RENDER_MODE_HOLOVIDEO_FRINGE:
		numWindows_ = SDL_GetNumVideoDisplays();
		if (numWindows_ != fringeContext_.display_options.num_heads / 2)
		{
			LOG4CXX_ERROR(logger_, "The X11 setup is not correct, you do not have 2 heads per GPU window")
			
			//for debugging, open up multiple windows
			LOG4CXX_WARN(logger_, "Opening up the right amount of windows for debugging algorithm")
			numWindows_ = fringeContext_.display_options.num_heads / 2;
		}
		break;
	default:
		numWindows_ = SDL_GetNumVideoDisplays();
		break;
	}

	LOG4CXX_INFO(logger_, "Number of windows: " << numWindows_)

	SDL_GL_SetAttribute(SDL_GL_ACCELERATED_VISUAL, 1);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1);
	SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
	SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);
    SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_COMPATIBILITY);

	SDL_GL_SetSwapInterval(1);

	windows_ = new SDL_Window*[numWindows_];
	glContexts_ = new SDL_GLContext[numWindows_];
	windowWidth_ = new unsigned int[numWindows_];
	windowHeight_ = new unsigned int[numWindows_];

	updateAlgorithmOptionsCache();

	std::unique_lock<std::mutex> initLock(isInitMutex_);
	shouldRender_ = true;
	renderThread_ = std::thread(std::bind(&DSCP4Render::renderLoop, this));

	isInitCV_.wait(initLock);

	initLock.unlock();
		
	return true;
}

bool DSCP4Render::initWindow(SDL_Window*& window, SDL_GLContext& glContext, int thisWindowNum)
{
	LOG4CXX_DEBUG(logger_, "Initializing SDL for Window " << thisWindowNum)
	SDL_Rect bounds = { 0 };

	// This will get the resolution of the primary window if everything else fails
	// Useful for opening up 3 windows during hologram mode to test everything
	if (SDL_GetDisplayBounds(thisWindowNum, &bounds) == -1)
		SDL_GetDisplayBounds(0, &bounds);
#ifdef _DEBUG
	static int x = bounds.x;
	static int y = bounds.y;
#else
	int x = bounds.x;
	int y = bounds.y;
#endif
	windowWidth_[thisWindowNum] = bounds.w;
	windowHeight_[thisWindowNum] = bounds.h;

	int flags = SDL_WINDOW_OPENGL;

	switch (renderOptions_->render_mode)
	{
	case DSCP4_RENDER_MODE_MODEL_VIEWING:
		windowWidth_[thisWindowNum] = fringeContext_.algorithm_options->num_wafels_per_scanline;
		windowHeight_[thisWindowNum] = fringeContext_.algorithm_options->num_scanlines;
		x = abs((int)(bounds.w - windowWidth_[thisWindowNum])) / 2;
		y = abs((int)(bounds.h - windowHeight_[thisWindowNum])) / 2;
		break;
	case DSCP4_RENDER_MODE_STEREOGRAM_VIEWING:
		windowWidth_[thisWindowNum] *= 0.5f;
		windowHeight_[thisWindowNum] = windowWidth_[thisWindowNum] * (float)fringeContext_.algorithm_options->cache.stereogram_res_y / (float)fringeContext_.algorithm_options->cache.stereogram_res_x;
		x = abs((int)(bounds.w - windowWidth_[thisWindowNum])) / 2;
		y = abs((int)(bounds.h - windowHeight_[thisWindowNum])) / 2;
		//LOG4CXX_DEBUG(logger_, "Creating SDL OpenGL Window " << thisWindowNum << ": " << windowWidth_[thisWindowNum] << "x" << windowHeight_[thisWindowNum] << " @ " << "{" << bounds.x + 80 << "," << bounds.y + 80 << "}")
		//window = SDL_CreateWindow(("dscp4-" + std::to_string(thisWindowNum)).c_str(), bounds.x + 80, bounds.y + 80, windowWidth_[thisWindowNum], windowHeight_[thisWindowNum], SDL_WINDOW_OPENGL);
		break;
	case DSCP4_RENDER_MODE_AERIAL_DISPLAY:
#ifdef _DEBUG
		x += windowHeight_[thisWindowNum] * 0.03f;
		y += windowWidth_[thisWindowNum] * 0.03f;
		windowHeight_[thisWindowNum] *= 0.8f;
		windowWidth_[thisWindowNum] *= 0.8f;
		break;
#else
		SDL_ShowCursor(SDL_DISABLE);
		flags |= SDL_WINDOW_BORDERLESS;
		isFullScreen_ = true;
#endif
	case DSCP4_RENDER_MODE_HOLOVIDEO_FRINGE:
#ifdef _DEBUG

		windowWidth_[thisWindowNum] *= 0.3f;
		windowHeight_[thisWindowNum] = windowWidth_[thisWindowNum] * (float)fringeContext_.algorithm_options->cache.fringe_buffer_res_y / (float)fringeContext_.algorithm_options->cache.fringe_buffer_res_x;
		x += (bounds.w/3 - windowWidth_[thisWindowNum]);
		y += windowHeight_[thisWindowNum] * 0.08f;
#else
		SDL_ShowCursor(SDL_DISABLE);
		flags |= SDL_WINDOW_BORDERLESS;
		isFullScreen_ = true;
#endif
		break;
	default:
		break;
	}

	LOG4CXX_DEBUG(logger_, "Creating SDL OpenGL Window " << thisWindowNum << ": " << windowWidth_[thisWindowNum] << "x" << windowHeight_[thisWindowNum] << " @ " << "{" << x << "," << y << "}")
	window = SDL_CreateWindow(("edu.mit.media.obmg.dscp4-" + std::to_string(thisWindowNum)).c_str(), x, y, windowWidth_[thisWindowNum], windowHeight_[thisWindowNum], flags);
	
	CHECK_SDL_RC(window == nullptr, "Could not create SDL window");

	LOG4CXX_DEBUG(logger_, "Creating GL Context from SDL window " << thisWindowNum)
	
	SDL_GL_SetAttribute(SDL_GL_SHARE_WITH_CURRENT_CONTEXT, 1);

	glContext = SDL_GL_CreateContext(window);

#ifndef __APPLE__
	LOG4CXX_DEBUG(logger_, "Initializing GLEW")
	GLenum err = glewInit();
	if (err != GLEW_OK)
	{
		LOG4CXX_ERROR(logger_, "Could not initialize GLEW: " << glewGetString(err))
	}
#endif

	SDL_GL_MakeCurrent(window, glContext);

	while (glGetError() != GL_NO_ERROR)
	{

	}

	LOG4CXX_DEBUG(logger_, "Turning on VSYNC")
	SDL_GL_SetSwapInterval(1);

	glViewport(0, 0, windowWidth_[thisWindowNum], windowHeight_[thisWindowNum]);

	// Set a black background
	// Intel GPU bug does not like 0.0f for all values
	glClearColor(0.000001f, 0.f, 0.f, 0.f); // Black Background
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	SDL_GL_SwapWindow(window);

	return true;
}

void DSCP4Render::deinitWindow(SDL_Window*& window, SDL_GLContext& glContext, int thisWindowNum)
{
	LOG4CXX_DEBUG(logger_, "Deinitializing SDL for Window " << thisWindowNum)

	if (glContext)
	{
		LOG4CXX_DEBUG(logger_, "Destroying GL Context " << thisWindowNum << "...")
		SDL_GL_DeleteContext(glContext);
		glContext = nullptr;
	}

	if (window)
	{
		LOG4CXX_DEBUG(logger_, "Destroying SDL Window " << thisWindowNum << "...")
		SDL_DestroyWindow(window);
		window = nullptr;
	}

}

bool DSCP4Render::initLightingShader(int which)
{
	lightingShader_[which].init();
	lightingShader_[which].loadShader(VSShaderLib::VERTEX_SHADER,
		(boost::filesystem::path(renderOptions_->shaders_path) /
		boost::filesystem::path(std::string((const char*)renderOptions_->shader_filename_prefix).append(".vert"))).string()
		);

	lightingShader_[which].loadShader(VSShaderLib::FRAGMENT_SHADER,
		(boost::filesystem::path(renderOptions_->shaders_path) /
		boost::filesystem::path(std::string((const char*)renderOptions_->shader_filename_prefix).append(".frag"))).string()
		);

	lightingShader_[which].setProgramOutput(0, "outputF");
	lightingShader_[which].setVertexAttribName(VSShaderLib::VERTEX_COORD_ATTRIB, "position");
	lightingShader_[which].setVertexAttribName(VSShaderLib::NORMAL_ATTRIB, "normal");
	lightingShader_[which].setVertexAttribName(VSShaderLib::TEXTURE_COORD_ATTRIB, "texCoord");
	lightingShader_[which].prepareProgram();

	lightingShader_[which].setUniform("texUnit", 0);
	float f3 = 0.90f;
	lightingShader_[which].setBlockUniform("Lights", "l_spotCutOff", &f3);
	return lightingShader_[which].isProgramValid();
}

void DSCP4Render::deinitLightingShader(int which)
{

}

// this is just here for testing
void DSCP4Render::drawCube()
{
	/*
	* EXERCISE:
	* Replace this awful mess with vertex
	* arrays and a call to glDrawElements.
	*
	* EXERCISE:
	* After completing the above, change
	* it to use compiled vertex arrays.
	*
	* EXERCISE:
	* Verify my windings are correct here ;).
	*/
	static GLfloat v0[] = { -1.0f, -1.0f, 1.0f };
	static GLfloat v1[] = { 1.0f, -1.0f, 1.0f };
	static GLfloat v2[] = { 1.0f, 1.0f, 1.0f };
	static GLfloat v3[] = { -1.0f, 1.0f, 1.0f };
	static GLfloat v4[] = { -1.0f, -1.0f, -1.0f };
	static GLfloat v5[] = { 1.0f, -1.0f, -1.0f };
	static GLfloat v6[] = { 1.0f, 1.0f, -1.0f };
	static GLfloat v7[] = { -1.0f, 1.0f, -1.0f };
	static GLubyte red[] = { 255, 0, 0, 255 };
	static GLubyte green[] = { 0, 255, 0, 255 };
	static GLubyte blue[] = { 0, 0, 255, 255 };
	static GLubyte white[] = { 255, 255, 255, 255 };
	static GLubyte yellow[] = { 0, 255, 255, 255 };
	static GLubyte black[] = { 0, 0, 0, 255 };
	static GLubyte orange[] = { 255, 255, 0, 255 };
	static GLubyte purple[] = { 255, 0, 255, 0 };

	glEnable(GL_COLOR_MATERIAL);
	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	glEnable(GL_NORMALIZE);

	/* Send our triangle data to the pipeline. */
	glBegin(GL_TRIANGLES);

	glColor4ubv(red);
	glVertex3fv(v0);
	glColor4ubv(green);
	glVertex3fv(v1);
	glColor4ubv(blue);
	glVertex3fv(v2);

	glColor4ubv(red);
	glVertex3fv(v0);
	glColor4ubv(blue);
	glVertex3fv(v2);
	glColor4ubv(white);
	glVertex3fv(v3);

	glColor4ubv(green);
	glVertex3fv(v1);
	glColor4ubv(black);
	glVertex3fv(v5);
	glColor4ubv(orange);
	glVertex3fv(v6);

	glColor4ubv(green);
	glVertex3fv(v1);
	glColor4ubv(orange);
	glVertex3fv(v6);
	glColor4ubv(blue);
	glVertex3fv(v2);

	glColor4ubv(black);
	glVertex3fv(v5);
	glColor4ubv(yellow);
	glVertex3fv(v4);
	glColor4ubv(purple);
	glVertex3fv(v7);

	glColor4ubv(black);
	glVertex3fv(v5);
	glColor4ubv(purple);
	glVertex3fv(v7);
	glColor4ubv(orange);
	glVertex3fv(v6);

	glColor4ubv(yellow);
	glVertex3fv(v4);
	glColor4ubv(red);
	glVertex3fv(v0);
	glColor4ubv(white);
	glVertex3fv(v3);

	glColor4ubv(yellow);
	glVertex3fv(v4);
	glColor4ubv(white);
	glVertex3fv(v3);
	glColor4ubv(purple);
	glVertex3fv(v7);

	glColor4ubv(white);
	glVertex3fv(v3);
	glColor4ubv(blue);
	glVertex3fv(v2);
	glColor4ubv(orange);
	glVertex3fv(v6);

	glColor4ubv(white);
	glVertex3fv(v3);
	glColor4ubv(orange);
	glVertex3fv(v6);
	glColor4ubv(purple);
	glVertex3fv(v7);

	glColor4ubv(green);
	glVertex3fv(v1);
	glColor4ubv(red);
	glVertex3fv(v0);
	glColor4ubv(yellow);
	glVertex3fv(v4);

	glColor4ubv(green);
	glVertex3fv(v1);
	glColor4ubv(yellow);
	glVertex3fv(v4);
	glColor4ubv(black);
	glVertex3fv(v5);

	glEnd();

	glDisable(GL_NORMALIZE);
	glDisable(GL_COLOR_MATERIAL);
}

void DSCP4Render::renderLoop()
{
	std::unique_lock<std::mutex> initLock(isInitMutex_);

#ifdef DSCP4_ENABLE_TRACE_LOG
	long long renderFrameDuration = 0;
#endif

	float ratio = 0.f;
	float q = 0.f; //offset for rendering stereograms
	SDL_Event event = { 0 };

	//lightingShader_ = new VSShaderLib[numWindows_];

	camera_.eye = glm::vec3(0, 0, (renderOptions_->render_mode == DSCP4_RENDER_MODE_MODEL_VIEWING) || (renderOptions_->render_mode == DSCP4_RENDER_MODE_AERIAL_DISPLAY) ? 4.0f : .5f);
	camera_.center = glm::vec3(0, 0, 0);
	camera_.up = glm::vec3(0, 1, 0);

	
	lighting_.ambientColor = glm::vec4(0.2f, 0.2f, 0.2f, 1.f);
	lighting_.diffuseColor = glm::vec4(1.0f, 1.0f, 1.0f, 1.f);
	lighting_.specularColor = glm::vec4(1.f, 1.f, 1.f, 1.f);
	lighting_.globalAmbientColor = glm::vec4(0.f, 0.f, 0.f, 1.f);

	// Init windows, for stereogram and model viewing
	// this is only 1 window, for aerial it is
	// number of displays, and for holovideo it is number of GPUs
	for (unsigned int i = numWindows_; i > 0; i--)
	{
		initWindow(windows_[i-1], glContexts_[i-1], i-1);

		// Add ambient and diffuse lighting to every scene
		glLightfv(GL_LIGHT0, GL_AMBIENT, glm::value_ptr(lighting_.ambientColor));
		glLightfv(GL_LIGHT0, GL_DIFFUSE, glm::value_ptr(lighting_.diffuseColor));

		glLightModelfv(GL_AMBIENT_AND_DIFFUSE, glm::value_ptr(lighting_.globalAmbientColor));
	}

	SDL_GL_MakeCurrent(windows_[0], glContexts_[0]);

	switch (renderOptions_->render_mode)
	{
	case DSCP4_RENDER_MODE_MODEL_VIEWING:
		initViewingTextures();
		break;
	case DSCP4_RENDER_MODE_STEREOGRAM_VIEWING:
		initStereogramTextures();
		break;
	case DSCP4_RENDER_MODE_HOLOVIDEO_FRINGE:
		initStereogramTextures();
		initFringeTextures();
		initComputeMethod();
		break;
	default:
		break;
	}

	// For capturing mouse and keyboard events
	//SDL_AddEventWatch(DSCP4Render::inputStateChanged, this);

	// Sanity check, i'm pretty sure you don't want different resolutions
	bool resAreDifferent = false;
	for (unsigned int i = 1; i < numWindows_; i++)
	{
		if (windowWidth_[i] != windowWidth_[i - 1] || windowHeight_[i] != windowHeight_[i - 1])
			resAreDifferent = true;
	}

	if (resAreDifferent)
	{
		LOG4CXX_WARN(logger_, "Multiple displays with different resolutions. You're on your own...")
	}

	isInit_ = true;

	initLock.unlock();
	isInitCV_.notify_all();

	if (eventCallback_)
	{
		renderPreviewBuffer_ = new unsigned char[fringeContext_.algorithm_options->num_wafels_per_scanline * fringeContext_.algorithm_options->num_scanlines * 4];
		renderPreviewData_.x_res = fringeContext_.algorithm_options->num_wafels_per_scanline;
		renderPreviewData_.y_res = fringeContext_.algorithm_options->num_scanlines;
		renderPreviewData_.buffer = renderPreviewBuffer_;
	}

	while (shouldRender_)
	{

		lighting_.position = glm::vec4(renderOptions_->light_pos_x, renderOptions_->light_pos_y, renderOptions_->light_pos_z, 1.f);

		SDL_Event event = { 0 };

		std::unique_lock<std::mutex> updateFrameLock(updateFrameMutex_);
		if (!(meshChanged_ || cameraChanged_ || lightingChanged_ || spinOn_))
		{
			if (std::cv_status::timeout == updateFrameCV_.wait_for(updateFrameLock, std::chrono::milliseconds(1)))
				goto poll;
		}

#ifdef DSCP4_ENABLE_TRACE_LOG
		renderFrameDuration = measureTime<>([&](){
#endif

			// Increments rotation if spinOn_ is true
			// Otherwise rotates by rotateAngle_
			rotateAngleY_ = spinOn_.load() == true ?
				rotateAngleY_ > 359.f ?
				0.f : rotateAngleY_ + rotateIncrement_ : rotateAngleY_.load();

			switch (renderOptions_->render_mode)
			{
			case DSCP4_RENDER_MODE_MODEL_VIEWING:
			{
				drawForViewing();
			}
				break;
			case DSCP4_RENDER_MODE_STEREOGRAM_VIEWING:
			{
				drawForStereogram();
			}
				break;
			case DSCP4_RENDER_MODE_AERIAL_DISPLAY:
			{
		#ifdef DSCP4_ENABLE_TRACE_LOG

				auto drawAerialDuration = measureTime<>(std::bind(&DSCP4Render::drawForAerialDisplay, this));
				LOG4CXX_TRACE(logger_, "Generating " << numWindows_ << " views took " << drawAerialDuration << " ms (" << 1.f / drawAerialDuration * 1000 << " fps)")
		#else
				drawForAerialDisplay();

		#endif

			}
			break;

			case DSCP4_RENDER_MODE_HOLOVIDEO_FRINGE:
			{
				drawForFringe();
			}
				break;
			default:
				break;
			}


			for (unsigned int i = 0; i < numWindows_; i++)
			{
				SDL_GL_MakeCurrent(windows_[i], glContexts_[i]);
				SDL_GL_SwapWindow(windows_[i]);
			}

#ifdef DSCP4_ENABLE_TRACE_LOG
		});

		LOG4CXX_TRACE(logger_, "Rendering the frame took " << renderFrameDuration << " ms (" << 1.f / renderFrameDuration * 1000 << " fps)");
#endif

        if (eventCallback_ && renderOptions_->render_mode == DSCP4_RENDER_MODE_HOLOVIDEO_FRINGE)
		{
#ifdef DSCP4_ENABLE_TRACE_LOG
			renderPreviewData_.render_fps = 1.f / renderFrameDuration * 1000.f;
#endif
			eventCallback_(DSCP4_CALLBACK_TYPE_NEW_FRAME, parentCallback_, &renderPreviewData_);
		}

	poll:
		if(SDL_PollEvent(&event))
        {
			inputStateChanged(&event);
        }

		if (shouldSaveScreenshot_)
		{
#ifdef DSCP4_HAVE_PNG
			saveScreenshotPNG();
#endif
			shouldSaveScreenshot_ = false;
		}

		updateAlgorithmOptionsCache();
	}

	initLock.lock();

	std::unique_lock<std::mutex> meshLock(meshMutex_);
	for (auto it = meshes_.begin(); it != meshes_.end(); it++)
		glDeleteBuffers(3, &it->second.info.gl_vertex_buf_id);
	meshLock.unlock();

	SDL_GL_MakeCurrent(windows_[0], glContexts_[0]);

	switch (renderOptions_->render_mode)
	{
	case DSCP4_RENDER_MODE_MODEL_VIEWING:
		deinitViewingTextures();
		break;
	case DSCP4_RENDER_MODE_STEREOGRAM_VIEWING:
		deinitStereogramTextures();
		break;
	case DSCP4_RENDER_MODE_HOLOVIDEO_FRINGE:
		deinitComputeMethod();
		deinitFringeTextures();
		deinitStereogramTextures();
		break;
	default:
		break;
	}

	for (unsigned int i = 0; i < numWindows_; i++)
	{
		SDL_GL_MakeCurrent(windows_[i], glContexts_[i]);
		deinitWindow(windows_[i], glContexts_[i], i);
	}

	if (lightingShader_)
	{
		delete[] lightingShader_;
		lightingShader_ = nullptr;
	}

	if (renderPreviewBuffer_)
	{
		delete[] renderPreviewBuffer_;
		renderPreviewBuffer_ = nullptr;
	}

	initLock.unlock();
	isInitCV_.notify_all();

	isInit_ = false;

	if (eventCallback_)
		eventCallback_(DSCP4_CALLBACK_TYPE_STOPPED, parentCallback_, nullptr);
}

// Builds stereogram views and lays them out in a NxN grid
// Therefore number of views MUST be N^2 value (e.g. 16 views, 4x4 tiles)
void DSCP4Render::generateStereogram()
{
	// X and Y resolution for each tile, or stereogram view
	const int tile_x_res = fringeContext_.algorithm_options->num_wafels_per_scanline;
	const int tile_y_res = fringeContext_.algorithm_options->num_scanlines;

	// The grid dimension
	const int tile_dim = static_cast<unsigned int>(sqrt(fringeContext_.algorithm_options->num_views_x));

	// Draw to the stereogram FBO, instead of back buffer
	glBindFramebuffer(GL_FRAMEBUFFER, fringeContext_.stereogram_gl_fbo);
	glDrawBuffer(GL_COLOR_ATTACHMENT0);

	// Intel GPU bug, 0.0f has residual colors from previous frame
	//glClearColor(0.000001f, 0.f, 0.f, 0.0f);
	glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

	std::lock_guard<std::mutex> lgc(cameraMutex_);
	std::lock_guard<std::mutex> lgl(lightingMutex_);
	std::lock_guard<std::mutex> lgm(meshMutex_);

	for (unsigned int i = 0; i < fringeContext_.algorithm_options->num_views_x; i++)
	{
		glViewport(tile_x_res*(i%tile_dim), tile_y_res*(i / tile_dim), tile_x_res, tile_y_res);

		glMatrixMode(GL_PROJECTION);

		const float ratio = static_cast<float>(tile_x_res) / static_cast<float>(tile_y_res);
		const float q = (i - fringeContext_.algorithm_options->num_views_x * 0.5f) / static_cast<float>(fringeContext_.algorithm_options->num_views_x) * fringeContext_.algorithm_options->fov_y * DEG_TO_RAD;

		projectionMatrix_ = buildOrthoXPerspYProjMat(-ratio, ratio, -1.0f, 1.0f, fringeContext_.algorithm_options->z_near, fringeContext_.algorithm_options->z_far, q);

		glLoadMatrixf(glm::value_ptr(projectionMatrix_));

		if (renderOptions_->shader_model != DSCP4_SHADER_MODEL_OFF)
			glEnable(GL_LIGHTING);

		glShadeModel(renderOptions_->shader_model == DSCP4_SHADER_MODEL_SMOOTH ? GL_SMOOTH : GL_FLAT);

		glMatrixMode(GL_MODELVIEW);

		viewMatrix_ = glm::mat4() *
			glm::lookAt(
			camera_.eye,
			camera_.center,
			camera_.up
			);

		glLoadMatrixf(glm::value_ptr(viewMatrix_));

		glLightfv(GL_LIGHT0, GL_POSITION, glm::value_ptr(lighting_.position));

		// Rotate the scene
		viewMatrix_ = glm::rotate(viewMatrix_, rotateAngleX_ * DEG_TO_RAD, glm::vec3(1.0f, 0.0f, 0.0f));
		viewMatrix_ = glm::rotate(viewMatrix_, rotateAngleY_ * DEG_TO_RAD, glm::vec3(0.0f, 1.0f, 0.0f));
		viewMatrix_ = glm::rotate(viewMatrix_, rotateAngleZ_ * DEG_TO_RAD, glm::vec3(0.0f, 0.0f, 1.0f));

		glLoadMatrixf(glm::value_ptr(viewMatrix_));

		glEnable(GL_DEPTH_TEST);
		glEnable(GL_LIGHT0);

		drawAllMeshes();

		glDisable(GL_LIGHT0);
		glDisable(GL_DEPTH_TEST);

		if (renderOptions_->shader_model != DSCP4_SHADER_MODEL_OFF)
			glDisable(GL_LIGHTING);
	}

	cameraChanged_ = false;
	meshChanged_ = false;
	lightingChanged_ = false;

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glDrawBuffer(GL_BACK);
}

void DSCP4Render::generateView()
{
	glBindFramebuffer(GL_FRAMEBUFFER, fringeContext_.view_gl_fbo);
	glDrawBuffer(GL_COLOR_ATTACHMENT0);

	const float ratio = (float)fringeContext_.algorithm_options->num_wafels_per_scanline / (float)fringeContext_.algorithm_options->num_scanlines;
	{
		std::lock_guard<std::mutex> lgc(cameraMutex_);
		glMatrixMode(GL_PROJECTION);

		projectionMatrix_ = glm::mat4();
		projectionMatrix_ *= glm::perspective(fringeContext_.algorithm_options->fov_y * DEG_TO_RAD, ratio, 3.f, 5.f);

		glLoadMatrixf(glm::value_ptr(projectionMatrix_));

		if (renderOptions_->shader_model != DSCP4_SHADER_MODEL_OFF)
			glEnable(GL_LIGHTING);

		glShadeModel(renderOptions_->shader_model == DSCP4_SHADER_MODEL_SMOOTH ? GL_SMOOTH : GL_FLAT);

		/* Clear the color and depth buffers. */
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		/* We don't want to modify the projection matrix. */
		glMatrixMode(GL_MODELVIEW);

		viewMatrix_ = glm::mat4() * glm::lookAt(
			camera_.eye,
			camera_.center,
			camera_.up
			);

		glLoadMatrixf(glm::value_ptr(viewMatrix_));
		{
			std::lock_guard<std::mutex> lgl(lightingMutex_);
			glLightfv(GL_LIGHT0, GL_POSITION, glm::value_ptr(lighting_.position));
			lightingChanged_ = false;
		}
		// Rotate the scene
		viewMatrix_ = glm::rotate(viewMatrix_, rotateAngleX_ * DEG_TO_RAD, glm::vec3(1.0f, 0.0f, 0.0f));
		viewMatrix_ = glm::rotate(viewMatrix_, rotateAngleY_ * DEG_TO_RAD, glm::vec3(0.0f, 1.0f, 0.0f));
		viewMatrix_ = glm::rotate(viewMatrix_, rotateAngleZ_ * DEG_TO_RAD, glm::vec3(0.0f, 0.0f, 1.0f));

		glLoadMatrixf(glm::value_ptr(viewMatrix_));
		cameraChanged_ = false;
	}
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_LIGHT0);

	{
		std::lock_guard<std::mutex> meshLock(meshMutex_);
		drawAllMeshes();
		meshChanged_ = false;
	}

	glDisable(GL_LIGHT0);
	glDisable(GL_DEPTH_TEST);

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glDrawBuffer(GL_BACK);
}

void DSCP4Render::drawForViewing()
{
#ifdef DSCP4_ENABLE_TRACE_LOG
	auto duration = measureTime<>(std::bind(&DSCP4Render::generateView, this));
	LOG4CXX_TRACE(logger_, "Generating a single view took " << duration << " ms (" << 1.f / duration * 1000 << " fps)")
#else
	generateView();
#endif

#ifdef DSCP4_ENABLE_TRACE_LOG
	duration = measureTime<>(std::bind(&DSCP4Render::drawViewingTexture, this));
	LOG4CXX_TRACE(logger_, "Drawing a single view took " << duration << " ms (" << 1.f / duration * 1000 << " fps)")
#else
	drawViewingTexture();
#endif
	
}

void DSCP4Render::drawForStereogram()
{
#ifdef DSCP4_ENABLE_TRACE_LOG
	auto duration = measureTime<>(std::bind(&DSCP4Render::generateStereogram, this));
	LOG4CXX_TRACE(logger_, "Generating " << fringeContext_.algorithm_options->num_views_x << " stereogram views in total took " << duration << " ms (" << 1.f / duration * 1000 << " fps)")
#else
	generateStereogram();
#endif

#ifdef DSCP4_ENABLE_TRACE_LOG
	duration = measureTime<>(std::bind(&DSCP4Render::drawStereogramTexture, this));
	LOG4CXX_TRACE(logger_, "Drawing " << numWindows_ << " stereogram texture took " << duration << " ms (" << 1.f / duration * 1000 << " fps)")
#else
	drawStereogramTexture();
#endif
}

void DSCP4Render::drawForAerialDisplay()
{
	std::lock_guard<std::mutex> lgc(cameraMutex_);
	std::lock_guard<std::mutex> lgl(lightingMutex_);
	std::lock_guard<std::mutex> lgm(meshMutex_);

	for (unsigned int i = 0; i < numWindows_; i++)
	{
		SDL_GL_MakeCurrent(windows_[i], glContexts_[i]);

		glMatrixMode(GL_PROJECTION);

		const float ratio = (float)windowWidth_[i] / (float)windowHeight_[i];
		
		projectionMatrix_ = glm::mat4();
		projectionMatrix_ *= glm::perspective(fringeContext_.algorithm_options->fov_y * DEG_TO_RAD, ratio, 3.0f, 5.f);

		glLoadMatrixf(glm::value_ptr(projectionMatrix_));

		if (renderOptions_->shader_model != DSCP4_SHADER_MODEL_OFF)
			glEnable(GL_LIGHTING);

		glShadeModel(renderOptions_->shader_model == DSCP4_SHADER_MODEL_SMOOTH ? GL_SMOOTH : GL_FLAT);

		/* Clear the color and depth buffers. */
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		/* We don't want to modify the projection matrix. */
		glMatrixMode(GL_MODELVIEW);

		viewMatrix_ = glm::mat4() * glm::lookAt(
			camera_.eye,
			camera_.center,
			camera_.up
			);

		glLoadMatrixf(glm::value_ptr(viewMatrix_));

		glLightfv(GL_LIGHT0, GL_POSITION, glm::value_ptr(lighting_.position));

		// move the viewpoint for each iteration
		// this is for generating the next view
		//viewMatrix_ = glm::translate(viewMatrix_, glm::vec3(i*0.5f, 0.0f, 0.0f));

		// Rotate the scene
		viewMatrix_ = glm::rotate(viewMatrix_, rotateAngleX_ * DEG_TO_RAD, glm::vec3(1.0f, 0.0f, 0.0f));
		viewMatrix_ = glm::rotate(viewMatrix_, (rotateAngleY_ + i*10.0f) * DEG_TO_RAD, glm::vec3(0.0f, 1.0f, 0.0f));

		glLoadMatrixf(glm::value_ptr(viewMatrix_));

		glEnable(GL_DEPTH_TEST);
		glEnable(GL_LIGHT0);

		drawAllMeshes();

		glDisable(GL_LIGHT0);
		glDisable(GL_DEPTH_TEST);
	}

	cameraChanged_ = false;
	meshChanged_ = false;
	lightingChanged_ = false;
}

void DSCP4Render::drawForFringe()
{
	SDL_GL_MakeCurrent(windows_[0], glContexts_[0]);

#ifdef DSCP4_ENABLE_TRACE_LOG
	auto duration = measureTime<>(std::bind(&DSCP4Render::generateStereogram, this));
	LOG4CXX_TRACE(logger_, "Generating " << fringeContext_.algorithm_options->num_views_x << " stereogram views in total took " << duration << " ms (" << 1.f / duration * 1000 << " fps)")
#else
	generateStereogram();
#endif

	if (eventCallback_)
	{
		glBindFramebuffer(GL_READ_FRAMEBUFFER, fringeContext_.stereogram_gl_fbo);
		glReadPixels(
			fringeContext_.algorithm_options->num_wafels_per_scanline * (fringeContext_.algorithm_options->cache.stereogram_num_tiles_x - 1),
			fringeContext_.algorithm_options->num_scanlines * (fringeContext_.algorithm_options->cache.stereogram_num_tiles_y/2 - 1),
			fringeContext_.algorithm_options->num_wafels_per_scanline,
			fringeContext_.algorithm_options->num_scanlines,
			GL_RGBA, GL_UNSIGNED_BYTE, renderPreviewBuffer_);
		glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
	}

#ifdef DSCP4_ENABLE_TRACE_LOG
	duration = measureTime<>(std::bind(&DSCP4Render::copyStereogramDepthToPBO, this));
	LOG4CXX_TRACE(logger_, "Copying stereogram " << fringeContext_.algorithm_options->num_views_x << " views to PBOs took " << duration << " ms (" << 1.f / duration * 1000 << " fps)")
#else
	copyStereogramDepthToPBO();
#endif

#ifdef DSCP4_ENABLE_TRACE_LOG
	duration = measureTime<>(std::bind(&DSCP4Render::computeHologram, this));
	LOG4CXX_TRACE(logger_, "Compute hologram fringe pattern took " << duration << " ms (" << 1.f / duration * 1000 << " fps)")
	renderPreviewData_.compute_fps = 1.f / duration * 1000.f;
#else
	computeHologram();
#endif


#ifdef DSCP4_ENABLE_TRACE_LOG
	duration = measureTime<>(std::bind(&DSCP4Render::drawFringeTextures, this));
	LOG4CXX_TRACE(logger_, "Drawing " << numWindows_ << " fringe textures took " << duration << " ms (" << 1.f / duration * 1000 << " fps)")
#else
	drawFringeTextures();
#endif

}

void DSCP4Render::deinit()
{
	LOG4CXX_INFO(logger_, "Deinitializing DSCP4...")

	LOG4CXX_DEBUG(logger_, "Waiting for render thread to stop...")

	shouldRender_ = false;

	if(renderThread_.joinable() && (std::this_thread::get_id() != renderThread_.get_id()))
		renderThread_.join();
	
	if (windows_)
	{
		delete[] windows_;
		windows_ = nullptr;
	}

	if (glContexts_)
	{
		delete[] glContexts_;
		glContexts_ = nullptr;
	}

	LOG4CXX_DEBUG(logger_, "Destroying SDL context")
	SDL_Quit();

	isInit_ = false;
}

void DSCP4Render::drawMesh(mesh_t& mesh)
{

	const int CLOUD_PT_SIZE = 32;

	// This will put the mesh in the vertex array buffer
	// If it is not in there already
	if (mesh.info.gl_vertex_buf_id == -1)
	{
		if (!mesh.info.is_pcl_cloud)
		{
			glGenBuffers(3, &mesh.info.gl_vertex_buf_id);
			glBindBuffer(GL_ARRAY_BUFFER, mesh.info.gl_vertex_buf_id);
			glBufferData(GL_ARRAY_BUFFER, mesh.info.vertex_stride * mesh.info.num_vertices, mesh.vertices, GL_STATIC_DRAW);
			if (mesh.normals)
			{
				glBindBuffer(GL_ARRAY_BUFFER, mesh.info.gl_normal_buf_id);
				glBufferData(GL_ARRAY_BUFFER, mesh.info.vertex_stride * mesh.info.num_vertices, mesh.normals, GL_STATIC_DRAW);
			}

			if (mesh.colors)
			{
				glBindBuffer(GL_ARRAY_BUFFER, mesh.info.gl_color_buf_id);
				glBufferData(GL_ARRAY_BUFFER, mesh.info.color_stride * mesh.info.num_vertices, mesh.colors, GL_STATIC_DRAW);
			}
		}
		else
		{
			glGenBuffers(3, &mesh.info.gl_vertex_buf_id);
		}
	}



	if (!mesh.info.is_pcl_cloud)
	{
		glEnable(GL_COLOR_MATERIAL);
		glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
		glEnable(GL_NORMALIZE);

		glBindBuffer(GL_ARRAY_BUFFER, mesh.info.gl_vertex_buf_id);
		glEnableClientState(GL_VERTEX_ARRAY);
		glVertexPointer(mesh.info.num_points_per_vertex, GL_FLOAT, mesh.info.vertex_stride, 0);

		if (mesh.colors)
		{
			glBindBuffer(GL_ARRAY_BUFFER, mesh.info.gl_color_buf_id);
			glEnableClientState(GL_COLOR_ARRAY);
			glColorPointer(mesh.info.num_color_channels, GL_FLOAT, mesh.info.color_stride, 0);
		}
		else
			glColor4f(0.5f, 0.5f, 0.5f, 1.0f);

		if (mesh.normals)
		{
			glBindBuffer(GL_ARRAY_BUFFER, mesh.info.gl_normal_buf_id);
			glEnableClientState(GL_NORMAL_ARRAY);
			glNormalPointer(GL_FLOAT, mesh.info.vertex_stride, 0);
		}

		GLenum faceMode = 0;
		switch (mesh.info.num_indecies)
		{
		case 1: faceMode = GL_POINTS; break;
		case 2: faceMode = GL_LINES; break;
		case 3: faceMode = GL_TRIANGLES; break;
		case 4: faceMode = GL_QUADS; break;
		default: faceMode = GL_POLYGON; break;
		}

		glDrawArrays(faceMode, 0, mesh.info.num_vertices);
		glDisableClientState(GL_VERTEX_ARRAY);
		glDisableClientState(GL_NORMAL_ARRAY);
		glDisableClientState(GL_COLOR_ARRAY);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		glDisable(GL_NORMALIZE);
		glDisable(GL_COLOR_MATERIAL);
	}
	else
	{
		glDisable(GL_LIGHTING);
		//glEnable(GL_COLOR_MATERIAL);
		//glEnable(GL_POINT_SMOOTH);
		glPointSize(6.f);

		//glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);

		glBindBuffer(GL_ARRAY_BUFFER, mesh.info.gl_vertex_buf_id);
		glBufferData(GL_ARRAY_BUFFER, mesh.info.num_vertices * CLOUD_PT_SIZE, mesh.vertices, GL_STREAM_DRAW);

		glEnableClientState(GL_VERTEX_ARRAY);
		glEnableClientState(GL_COLOR_ARRAY);

		glVertexPointer(3, GL_FLOAT, CLOUD_PT_SIZE, 0);
		glColorPointer(4, GL_UNSIGNED_BYTE, CLOUD_PT_SIZE, (void*)(sizeof(float)* 4));

		glDrawArrays(GL_POINTS, 0, mesh.info.num_vertices);

		glDisableClientState(GL_VERTEX_ARRAY);
		glDisableClientState(GL_COLOR_ARRAY);
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		glDisable(GL_NORMALIZE);
		glDisable(GL_COLOR_MATERIAL);
		glEnable(GL_LIGHTING);
		//glDisable(GL_POINT_SMOOTH);
	}


}

void DSCP4Render::drawAllMeshes()
{
	for (auto it = meshes_.begin(); it != meshes_.end(); it++)
	{
		// create the model matrix
		auto &mesh = it->second;
		auto &transform = it->second.info.transform;
		const float scaleFactor = 1.0f / sqrt(mesh.info.bounding_sphere.w); //scaling factor is 1/r, r = sqrt(radius squared)

		modelMatrix_ = glm::mat4();
		modelMatrix_ *= viewMatrix_;

		modelMatrix_ = glm::scale(modelMatrix_, glm::vec3(mesh.info.transform.scale.x, mesh.info.transform.scale.y, mesh.info.transform.scale.z));

		if (renderOptions_->auto_scale_enabled)
			modelMatrix_ = glm::scale(modelMatrix_, glm::vec3(scaleFactor, scaleFactor, scaleFactor));

		modelMatrix_ = glm::translate(modelMatrix_, glm::vec3(
			mesh.info.transform.translate.x,
			mesh.info.transform.translate.y,
			mesh.info.transform.translate.z));

		if (renderOptions_->auto_scale_enabled)
			modelMatrix_ = glm::translate(modelMatrix_, glm::vec3(
				-mesh.info.bounding_sphere.x,
				-mesh.info.bounding_sphere.y,
				-mesh.info.bounding_sphere.z));

		glLoadMatrixf(glm::value_ptr(modelMatrix_));

		//draw the actual mesh
		drawMesh(it->second);
	}
}

void DSCP4Render::addMesh(const char *id, int numIndecies, int numVertices, float *vertices, float * normals, float *colors, unsigned int numVertexDimensions, unsigned int numColorChannels)
{
	mesh_t mesh = { 0 };
	mesh.vertices = vertices;
	mesh.normals = normals;
	mesh.colors = colors;
	mesh.info.num_color_channels = numColorChannels;
	mesh.info.num_points_per_vertex = numVertexDimensions;
	mesh.info.vertex_stride = numVertexDimensions * sizeof(float);
	mesh.info.color_stride = numColorChannels * sizeof(float);
	mesh.info.num_vertices = numVertices;

	mesh.info.num_indecies = numIndecies;

	mesh.info.is_pcl_cloud = false;

	mesh.info.gl_color_buf_id = -1;
	mesh.info.gl_vertex_buf_id = -1;
	mesh.info.gl_normal_buf_id = -1;

	mesh.info.transform.scale.x = 1.f;
	mesh.info.transform.scale.y = 1.f;
	mesh.info.transform.scale.z = 1.f;

	if (renderOptions_->auto_scale_enabled)
	{
#ifdef DSCP4_ENABLE_TRACE_LOG
		auto duration = measureTime<>([&](){
#endif
			// create a 2D array for miniball algorithm
			float **ap = new float*[numVertices];
			float * pv = vertices;
			for (int i = 0; i < numVertices; ++i) {
				ap[i] = pv;
				pv += numVertexDimensions;
			}


			// miniball uses a quick method of determining the bounding sphere of all the vertices
			auto miniball3f = Miniball::Miniball<Miniball::CoordAccessor<float**, float*>>(3, (float**)ap, (float**)(ap + numVertices));
			mesh.info.bounding_sphere.x = miniball3f.center()[0];
			mesh.info.bounding_sphere.y = miniball3f.center()[1];
			mesh.info.bounding_sphere.z = miniball3f.center()[2];
			mesh.info.bounding_sphere.w = miniball3f.squared_radius();

			delete[] ap;

#ifdef DSCP4_ENABLE_TRACE_LOG
		});
		LOG4CXX_TRACE(logger_, "Getting bounding sphere for '" << id << "' took " << duration << " ms (" << 1.f / duration * 1000 << " fps)")
#endif
	}

	std::unique_lock<std::mutex> meshLock(meshMutex_);

	if (meshes_.find(id) != meshes_.end())
	{
		auto m = meshes_[id];
		if (m.info.gl_vertex_buf_id != -1)
			glDeleteBuffers(3, &m.info.gl_vertex_buf_id);
		meshes_.erase(id);
	}

	meshes_[id] = mesh;
	meshChanged_ = true;
	meshLock.unlock();
}

void DSCP4Render::addPointCloud(const char *id, unsigned int numPoints, void * cloudData)
{
	mesh_t mesh = { 0 };
	mesh.vertices = new unsigned char[numPoints * 32];
	memcpy(mesh.vertices, cloudData, numPoints * 32);
	mesh.info.is_pcl_cloud = true;
	mesh.info.num_color_channels = 4;
	mesh.info.num_points_per_vertex = 3;
	mesh.info.vertex_stride = 3 * sizeof(float);
	mesh.info.color_stride = mesh.info.num_color_channels * sizeof(char);
	mesh.info.num_vertices = numPoints;

	mesh.info.num_indecies = 3;

	mesh.info.gl_color_buf_id = -1;
	mesh.info.gl_vertex_buf_id = -1;
	mesh.info.gl_normal_buf_id = -1;

	mesh.info.transform.scale.x = 5.0f;
	mesh.info.transform.scale.y = 5.0f;
	mesh.info.transform.scale.z = -3.7f;

	mesh.info.transform.translate.z = -0.75f;

	std::unique_lock<std::mutex> meshLock(meshMutex_);
	if (meshes_.find(id) != meshes_.end())
	{
		auto m = meshes_[id];
		delete[] (unsigned char*)m.vertices;
		mesh.info.gl_vertex_buf_id = m.info.gl_vertex_buf_id;
		meshes_.erase(id);
	}

	meshes_[id] = mesh;
	meshChanged_ = true;
	meshLock.unlock();
}

void DSCP4Render::removeMesh(const char *id)
{
	std::unique_lock<std::mutex> meshLock(meshMutex_);
	auto &mesh = meshes_[id];

	meshes_.erase(id);
	meshChanged_ = true;
	meshLock.unlock();
}

void DSCP4Render::translateMesh(std::string meshID, float x, float y, float z)
{
	std::lock_guard<std::mutex> lg(meshMutex_);
	auto mesh = &meshes_[meshID];
	mesh->info.transform.translate.x = x;
	mesh->info.transform.translate.y = y;
	mesh->info.transform.translate.z = z;
}

void DSCP4Render::rotateMesh(std::string meshID, float angle, float x, float y, float z)
{
	std::lock_guard<std::mutex> lg(meshMutex_);
	auto mesh = &meshes_[meshID];
	mesh->info.transform.rotate.w = angle;
	mesh->info.transform.rotate.x = x;
	mesh->info.transform.rotate.y = y;
	mesh->info.transform.rotate.z = z;
}

void DSCP4Render::scaleMesh(std::string meshID, float x, float y, float z)
{
	std::lock_guard<std::mutex> lg(meshMutex_);
	auto mesh = &meshes_[meshID];
	mesh->info.transform.scale.x = x;
	mesh->info.transform.scale.y = y;
	mesh->info.transform.scale.z = z;
}

glm::mat4 DSCP4Render::buildOrthoXPerspYProjMat(
	float left,
	float right,
	float bottom,
	float top,
	float zNear,
	float zFar,
	float q
	)
{
	glm::mat4 shearOrtho = glm::mat4(1.0f);

	shearOrtho[0] = glm::vec4(
		2.f / (right - left),
		0.f,
		2 * q / (right - left),
		-(right + left - 2 * q * (zNear + zFar) / 2.f) / (right - left)
		);

	shearOrtho[1] = glm::vec4(
		0.f,
		2.f / (top - bottom),
		0.f,
		-(top + bottom) / (top - bottom)
		);

	shearOrtho[2] = glm::vec4(
		0.f,
		0.f,
		-2.f / (zFar - zNear),
		-(zFar + zNear) / (zFar - zNear)
		);

	shearOrtho[3] = glm::vec4(
		0.f,
		0.f,
		0.f,
		1.0f
		);

	return shearOrtho;
}

int DSCP4Render::inputStateChanged(SDL_Event* event)
{
	if (event->type == SDL_WINDOWEVENT)
	{
		if (event->window.event == SDL_WINDOWEVENT_RESTORED)
			Update();
	}
	else if (event->key.type == SDL_KEYDOWN)
	{

		if (event->key.keysym.mod == SDL_Keymod::KMOD_LSHIFT)
		{
			switch (event->key.keysym.scancode)
			{
			case  SDL_Scancode::SDL_SCANCODE_W:
				lighting_.position[1] += 0.1f;
				break;
			case SDL_Scancode::SDL_SCANCODE_S:
				lighting_.position[1] -= 0.1f;
				break;
			case  SDL_Scancode::SDL_SCANCODE_A:
				lighting_.position[0] -= 0.1f;
				break;
			case SDL_Scancode::SDL_SCANCODE_D:
				lighting_.position[0] += 0.1f;
				break;
			case  SDL_Scancode::SDL_SCANCODE_Z:
				lighting_.position[2] -= 0.1f;
				break;
			case SDL_Scancode::SDL_SCANCODE_X:
				lighting_.position[2] += 0.1f;
				break;
			case  SDL_Scancode::SDL_SCANCODE_UP:
				camera_.eye[1] += 0.1f;
				camera_.center[1] += 0.1f;
				break;
			case  SDL_Scancode::SDL_SCANCODE_DOWN:
				camera_.eye[1] -= 0.1f;
				camera_.center[1] -= 0.1f;
				break;
			case  SDL_Scancode::SDL_SCANCODE_LEFT:
				camera_.eye[0] -= 0.1f;
				camera_.center[0] -= 0.1f;
				break;
			case SDL_Scancode::SDL_SCANCODE_RIGHT:
				camera_.eye[0] += 0.1f;
				camera_.center[0] += 0.1f;
				break;
			case SDL_Scancode::SDL_SCANCODE_EQUALS:
				//zFar_ += 0.01f;
				break;
			case SDL_Scancode::SDL_SCANCODE_MINUS:
				//zFar_ -= 0.01f;
				break;
			default:
				break;
			}

			switch (event->key.keysym.scancode)
			{
			case  SDL_Scancode::SDL_SCANCODE_W:
			case SDL_Scancode::SDL_SCANCODE_S:
			case  SDL_Scancode::SDL_SCANCODE_A:
			case SDL_Scancode::SDL_SCANCODE_D:
			case  SDL_Scancode::SDL_SCANCODE_Z:
			case SDL_Scancode::SDL_SCANCODE_X:
				lightingChanged_ = true;
				break;
			default:
				break;
			}

			switch (event->key.keysym.scancode)
			{
			case  SDL_Scancode::SDL_SCANCODE_UP:
			case  SDL_Scancode::SDL_SCANCODE_DOWN:
			case  SDL_Scancode::SDL_SCANCODE_LEFT:
			case SDL_Scancode::SDL_SCANCODE_RIGHT:
				cameraChanged_ = true;
				break;
			default:
				break;
			}
		}
		else if (event->key.keysym.mod == SDL_Keymod::KMOD_LCTRL || event->key.keysym.mod == SDL_Keymod::KMOD_RCTRL)
		{
			switch (event->key.keysym.scancode)
			{

			case SDL_Scancode::SDL_SCANCODE_S:
			{
#ifdef DSCP4_HAVE_PNG

#ifdef DSCP4_ENABLE_TRACE_LOG
				LOG4CXX_INFO(logger_, "Dumping framebuffer screenshots...")
				auto duration = measureTime<>(std::bind(&DSCP4Render::saveScreenshotPNG, this));
				LOG4CXX_TRACE(logger_, "Saving screenshot(s) in total took " << duration << " ms (" << 1.f / duration * 1000 << " fps)")
#else
				saveScreenshotPNG();
#endif

#endif
			}
				break;
			
			case SDL_Scancode::SDL_SCANCODE_F:
				setFullScreen(!isFullScreen());
				break;
			default:
				break;
			}
		}
		else
		{
			switch (event->key.keysym.scancode)
			{
			case  SDL_Scancode::SDL_SCANCODE_UP:
				rotateAngleX_.store(rotateAngleX_ + 10.f);
				Update();
				break;
			case  SDL_Scancode::SDL_SCANCODE_DOWN:
				rotateAngleX_.store(rotateAngleX_ - 10.f);
				Update();
				break;
			case  SDL_Scancode::SDL_SCANCODE_LEFT:
				if (spinOn_)
					rotateIncrement_.store(rotateIncrement_ - .2f);
				else
				{
					rotateAngleY_.store(rotateAngleY_ + 10.f);
					Update();
				}
				break;
			case SDL_Scancode::SDL_SCANCODE_RIGHT:
				if (spinOn_)
					rotateIncrement_.store(rotateIncrement_ + 0.2f);
				else
				{
					rotateAngleY_.store(rotateAngleY_ - 10.f);
					Update();
				}
				break;
			case  SDL_Scancode::SDL_SCANCODE_R:
				spinOn_.store(!spinOn_);
				Update();
				break;
			case SDL_Scancode::SDL_SCANCODE_LEFTBRACKET:
				//q += 0.01f;
				break;
			case SDL_Scancode::SDL_SCANCODE_RIGHTBRACKET:
				//q -= 0.01f;
				break;
			case SDL_Scancode::SDL_SCANCODE_Q:
				shouldRender_ = false;
				break;
			case SDL_Scancode::SDL_SCANCODE_EQUALS:
				//zNear_ += 0.01f;
				break;
			case SDL_Scancode::SDL_SCANCODE_MINUS:
				//zNear_ -= 0.01f;
				break;
			case SDL_Scancode::SDL_SCANCODE_SPACE:
				if (drawMode_ == DSCP4_DRAW_MODE_COLOR)
					drawMode_ = DSCP4_DRAW_MODE_DEPTH;
				else
					drawMode_ = DSCP4_DRAW_MODE_COLOR;
				Update();
				break;
			case SDL_Scancode::SDL_SCANCODE_U:
				Update();
				break;
			default:
				break;
			}

		}
	}

	return 0;
}

void DSCP4Render::initViewingTextures()
{
	// create a new FBO for single view render mode
	// we render to a custom FBO with textures, rendering to a texture quad
	// so that we can siwitch between rendering depth/color to the window
	glGenFramebuffers(1, &fringeContext_.view_gl_fbo);

	//create depth texture
	glGenTextures(1, &fringeContext_.view_gl_fbo_depth);
	glBindTexture(GL_TEXTURE_2D, fringeContext_.view_gl_fbo_depth);
	glTexImage2D(
		GL_TEXTURE_2D,
		0,
		GL_DEPTH_COMPONENT32F,
		fringeContext_.algorithm_options->num_wafels_per_scanline,
		fringeContext_.algorithm_options->num_scanlines,
		0,
		GL_DEPTH_COMPONENT,
		GL_FLOAT,
		0
		);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	glBindTexture(GL_TEXTURE_2D, 0);

	//generate color texture
	glGenTextures(1, &fringeContext_.view_gl_fbo_color);
	glBindTexture(GL_TEXTURE_2D, fringeContext_.view_gl_fbo_color);
	glTexImage2D(
		GL_TEXTURE_2D,
		0,
		GL_RGBA8,
		fringeContext_.algorithm_options->num_wafels_per_scanline,
		fringeContext_.algorithm_options->num_scanlines,
		0,
		GL_RGBA,
		GL_FLOAT,
		0
		);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	glBindTexture(GL_TEXTURE_2D, 0);

	// Attach the depth texture and the color texture (to which depths will be output)
	glBindFramebuffer(GL_FRAMEBUFFER, fringeContext_.view_gl_fbo);
	glFramebufferTexture2D(
		GL_FRAMEBUFFER,
		GL_DEPTH_ATTACHMENT,
		GL_TEXTURE_2D,
		fringeContext_.view_gl_fbo_depth,
		0
		);
	glFramebufferTexture2D(
		GL_FRAMEBUFFER,
		GL_COLOR_ATTACHMENT0,
		GL_TEXTURE_2D,
		fringeContext_.view_gl_fbo_color,
		0
		);

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void DSCP4Render::deinitViewingTextures()
{
	glDeleteTextures(1, &fringeContext_.view_gl_fbo_color);
	glDeleteTextures(1, &fringeContext_.view_gl_fbo_depth);
	glDeleteFramebuffers(1, &fringeContext_.view_gl_fbo);
}

void DSCP4Render::initStereogramTextures()
{
	// create a new FBO for stereograms, this is required because if
	// we try to render stereograms to the normal frame-buffer, they
	// will be clipped by window size being smaller than N views
	glGenFramebuffers(1, &fringeContext_.stereogram_gl_fbo);

	//create depth texture
	glGenTextures(1, &fringeContext_.stereogram_gl_fbo_depth);
	glBindTexture(GL_TEXTURE_2D, fringeContext_.stereogram_gl_fbo_depth);
	glTexImage2D(
		GL_TEXTURE_2D,
		0,
		GL_DEPTH_COMPONENT32F,
		fringeContext_.algorithm_options->cache.stereogram_res_x,
		fringeContext_.algorithm_options->cache.stereogram_res_y,
		0,
		GL_DEPTH_COMPONENT,
		GL_FLOAT,
		0
		);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	glBindTexture(GL_TEXTURE_2D, 0);

	//generate color texture
	glGenTextures(1, &fringeContext_.stereogram_gl_fbo_color);
	glBindTexture(GL_TEXTURE_2D, fringeContext_.stereogram_gl_fbo_color);
	glTexImage2D(
		GL_TEXTURE_2D,
		0,
		GL_RGBA8,
		fringeContext_.algorithm_options->cache.stereogram_res_x,
		fringeContext_.algorithm_options->cache.stereogram_res_y,
		0,
		GL_RGBA,
		GL_FLOAT,
		0
		);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	glBindTexture(GL_TEXTURE_2D, 0);

	// Attach the depth texture and the color texture (to which depths will be output)
	glBindFramebuffer(GL_FRAMEBUFFER, fringeContext_.stereogram_gl_fbo);
	glFramebufferTexture2D(
		GL_FRAMEBUFFER,
		GL_DEPTH_ATTACHMENT,
		GL_TEXTURE_2D,
		fringeContext_.stereogram_gl_fbo_depth,
		0
		);
	glBindFramebuffer(GL_FRAMEBUFFER, fringeContext_.stereogram_gl_fbo);
	glFramebufferTexture2D(
		GL_FRAMEBUFFER,
		GL_COLOR_ATTACHMENT0,
		GL_TEXTURE_2D,
		fringeContext_.stereogram_gl_fbo_color,
		0
		);

	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	// begin generation of stereogram view buffers (these will go into CUDA kernels)
	size_t rgba_size = fringeContext_.algorithm_options->cache.stereogram_res_x * fringeContext_.algorithm_options->cache.stereogram_res_y * sizeof(GLbyte)* 4;
	size_t depth_size = fringeContext_.algorithm_options->cache.stereogram_res_x * fringeContext_.algorithm_options->cache.stereogram_res_y * sizeof(GLfloat);

	// Create a PBO to store depth data.  Every frame rendered,
	// depth buffer is copied into this PBO, which is sent to OpenCL/CUDA kernel
	glGenBuffers(1, &fringeContext_.stereogram_gl_depth_pbo_in);
	glBindBuffer(GL_ARRAY_BUFFER, fringeContext_.stereogram_gl_depth_pbo_in);
	glBufferData(GL_ARRAY_BUFFER, depth_size, NULL, GL_DYNAMIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	//end generation of stereogram view buffers
}

void DSCP4Render::deinitStereogramTextures()
{
	glDeleteBuffers(1, &fringeContext_.stereogram_gl_depth_pbo_in);

	glDeleteFramebuffers(1, &fringeContext_.stereogram_gl_fbo);
	glDeleteTextures(1, &fringeContext_.stereogram_gl_fbo_color);
	glDeleteTextures(1, &fringeContext_.stereogram_gl_fbo_depth);
}

void DSCP4Render::initFringeTextures()
{
	fringeContext_.fringe_gl_tex_out = new GLuint[numWindows_];

	// Create N-textures for outputting fringe data to the X displays
	// Whatever holographic computation is done will be written
	// To these textures and ultimately displayed on the holovideo display
	glGenTextures(fringeContext_.algorithm_options->cache.num_fringe_buffers, fringeContext_.fringe_gl_tex_out);

	char *blah = new char[fringeContext_.algorithm_options->cache.fringe_buffer_res_x * fringeContext_.algorithm_options->cache.fringe_buffer_res_y * 4];
	for (size_t i = 0; i < fringeContext_.algorithm_options->cache.fringe_buffer_res_x * fringeContext_.algorithm_options->cache.fringe_buffer_res_y * 4; i++)
	{
		blah[i] = i % 255;
	}

	for (size_t i = 0; i < fringeContext_.algorithm_options->cache.num_fringe_buffers; i++)
	{
		glBindTexture(GL_TEXTURE_2D, fringeContext_.fringe_gl_tex_out[i]);

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8,
			fringeContext_.algorithm_options->cache.fringe_buffer_res_x,
			fringeContext_.algorithm_options->cache.fringe_buffer_res_y,
			0, GL_RGBA, GL_UNSIGNED_BYTE, blah);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glBindTexture(GL_TEXTURE_2D, 0);
	}

	delete[] blah;
}

void DSCP4Render::deinitFringeTextures()
{
	glDeleteTextures(numWindows_, fringeContext_.fringe_gl_tex_out);

	delete[] fringeContext_.fringe_gl_tex_out;
}

void DSCP4Render::copyStereogramDepthToPBO()
{
	glBindFramebuffer(GL_FRAMEBUFFER, fringeContext_.stereogram_gl_fbo);
	glReadBuffer(GL_COLOR_ATTACHMENT0);

	// If we compile with OpenCL support, check to see if OpenCL supports
	// depth texture extensions.  If not, then we need to copy depth texture
	// to PBO, which is a performance hit (NVIDIA so far does not support this)
#ifdef DSCP4_HAVE_OPENCL
	if (fringeContext_.algorithm_options->compute_method == DSCP4_COMPUTE_METHOD_CUDA ||
		!((dscp4_fringe_opencl_context_t*)computeContext_)->have_cl_gl_depth_images_extension)
	{
#endif
	//copy DEPTH from stereogram views, because CUDA/OpenCL cannot access depth data directly from framebuffer
	glBindBuffer(GL_PIXEL_PACK_BUFFER, fringeContext_.stereogram_gl_depth_pbo_in);
	glReadPixels(0, 0,
		fringeContext_.algorithm_options->cache.stereogram_res_x,
		fringeContext_.algorithm_options->cache.stereogram_res_y,
		GL_DEPTH_COMPONENT, GL_FLOAT, 0);

	glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

#ifdef DSCP4_HAVE_OPENCL
	}
#endif

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glReadBuffer(GL_BACK);
	glDrawBuffer(GL_BACK);

}

void DSCP4Render::drawFringeTextures()
{
	GLfloat Vertices[] = { 0.f, 0.f, 0.f,
		static_cast<float>(fringeContext_.algorithm_options->cache.fringe_buffer_res_x), 0, 0,
		static_cast<float>(fringeContext_.algorithm_options->cache.fringe_buffer_res_x),
		static_cast<float>(fringeContext_.algorithm_options->cache.fringe_buffer_res_y), 0.f,
		0.f, static_cast<float>(fringeContext_.algorithm_options->cache.fringe_buffer_res_y), 0.f
	};

	
	GLfloat TexCoord[] = { 0, 0,
		1.f, 0.f,
		1.f, 1.f,
		0.f, 1.f,
	};

	const GLubyte indices[] = { 0, 1, 2, // first triangle (bottom left - top left - top right)
		0, 2, 3 };

	for (unsigned int i = 0; i < numWindows_; i++)
	{
		SDL_GL_MakeCurrent(windows_[i], glContexts_[numWindows_ - 1]);

		glEnable(GL_TEXTURE_2D);

		glViewport(0, 0, windowWidth_[i], windowHeight_[i]);

		glMatrixMode(GL_PROJECTION);
		projectionMatrix_ = glm::ortho(
			0.f,
			static_cast<float>(fringeContext_.algorithm_options->cache.fringe_buffer_res_x),
			0.f,
			static_cast<float>(fringeContext_.algorithm_options->cache.fringe_buffer_res_y)
			);

		glLoadMatrixf(glm::value_ptr(projectionMatrix_));

		glMatrixMode(GL_MODELVIEW);
		glLoadMatrixf(glm::value_ptr(glm::mat4()));

		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glDisable(GL_LIGHTING);

		glBindTexture(GL_TEXTURE_2D, fringeContext_.fringe_gl_tex_out[i]);

		glEnableClientState(GL_VERTEX_ARRAY);
		glVertexPointer(3, GL_FLOAT, 0, Vertices);

		glEnableClientState(GL_TEXTURE_COORD_ARRAY);
		glTexCoordPointer(2, GL_FLOAT, 0, TexCoord);

		glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_BYTE, indices);

		glDisableClientState(GL_TEXTURE_COORD_ARRAY);
		glDisableClientState(GL_VERTEX_ARRAY);

		glBindTexture(GL_TEXTURE_2D, 0);
		glDisable(GL_TEXTURE_2D);
	}
}

void DSCP4Render::drawStereogramTexture()
{
	GLfloat Vertices[] = { 0.f, 0.f, 0.f,
		static_cast<float>(windowWidth_[0]), 0, 0,
		static_cast<float>(windowWidth_[0]),
		static_cast<float>(windowHeight_[0]), 0.f,
		0.f, static_cast<float>(windowHeight_[0]), 0.f
	};

	GLfloat TexCoord[] = { 0, 0,
		1.f, 0.f,
		1.f, 1.f,
		0.f, 1.f,
	};

	const GLubyte indices[] = { 0, 1, 2, // first triangle (bottom left - top left - top right)
		0, 2, 3 };

	glEnable(GL_TEXTURE_2D);

	glViewport(0, 0, windowWidth_[0], windowHeight_[0]);

	glMatrixMode(GL_PROJECTION);
	projectionMatrix_ = glm::ortho(
		0.f,
		static_cast<float>(windowWidth_[0]),
		0.f,
		static_cast<float>(windowHeight_[0])
		);

	glLoadMatrixf(glm::value_ptr(projectionMatrix_));

	glMatrixMode(GL_MODELVIEW);
	glLoadMatrixf(glm::value_ptr(glm::mat4()));

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glDisable(GL_LIGHTING);

	if (drawMode_ == DSCP4_DRAW_MODE_DEPTH)
		glBindTexture(GL_TEXTURE_2D, fringeContext_.stereogram_gl_fbo_depth);
	else
		glBindTexture(GL_TEXTURE_2D, fringeContext_.stereogram_gl_fbo_color);

	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(3, GL_FLOAT, 0, Vertices);

	glEnableClientState(GL_TEXTURE_COORD_ARRAY);
	glTexCoordPointer(2, GL_FLOAT, 0, TexCoord);

	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_BYTE, indices);

	glDisableClientState(GL_TEXTURE_COORD_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);

	glBindTexture(GL_TEXTURE_2D, 0);
	glDisable(GL_TEXTURE_2D);
}

void DSCP4Render::drawViewingTexture()
{
	GLfloat Vertices[] = { 0.f, 0.f, 0.f,
		static_cast<float>(windowWidth_[0]), 0, 0,
		static_cast<float>(windowWidth_[0]),
		static_cast<float>(windowHeight_[0]), 0.f,
		0.f, static_cast<float>(windowHeight_[0]), 0.f
	};

	GLfloat TexCoord[] = { 0, 0,
		1.f, 0.f,
		1.f, 1.f,
		0.f, 1.f,
	};

	const GLubyte indices[] = { 0, 1, 2, // first triangle (bottom left - top left - top right)
		0, 2, 3 };

	glEnable(GL_TEXTURE_2D);

	glViewport(0, 0, windowWidth_[0], windowHeight_[0]);

	glMatrixMode(GL_PROJECTION);
	projectionMatrix_ = glm::ortho(
		0.f,
		static_cast<float>(windowWidth_[0]),
		0.f,
		static_cast<float>(windowHeight_[0])
		);

	glLoadMatrixf(glm::value_ptr(projectionMatrix_));

	glMatrixMode(GL_MODELVIEW);
	glLoadMatrixf(glm::value_ptr(glm::mat4()));

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	glDisable(GL_LIGHTING);

	if (drawMode_ == DSCP4_DRAW_MODE_DEPTH)
		glBindTexture(GL_TEXTURE_2D, fringeContext_.view_gl_fbo_depth);
	else
		glBindTexture(GL_TEXTURE_2D, fringeContext_.view_gl_fbo_color);

	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(3, GL_FLOAT, 0, Vertices);

	glEnableClientState(GL_TEXTURE_COORD_ARRAY);
	glTexCoordPointer(2, GL_FLOAT, 0, TexCoord);

	glDrawElements(GL_TRIANGLES, 6, GL_UNSIGNED_BYTE, indices);

	glDisableClientState(GL_TEXTURE_COORD_ARRAY);
	glDisableClientState(GL_VERTEX_ARRAY);

	glBindTexture(GL_TEXTURE_2D, 0);
	glDisable(GL_TEXTURE_2D);
}

void DSCP4Render::initComputeMethod()
{

	switch (fringeContext_.algorithm_options->compute_method)
	{
	case DSCP4_COMPUTE_METHOD_CUDA:
#ifdef DSCP4_HAVE_CUDA
		LOG4CXX_DEBUG(logger_, "CUDA -- Initializing CUDA context")
		computeContext_ = (dscp4_fringe_cuda_context_t*)dscp4_fringe_cuda_CreateContext(&fringeContext_);
#else
		LOG4CXX_FATAL(logger_, "CUDA selected as compute method, but this binary was not compiled with CUDA")
#endif
		break;
	case DSCP4_COMPUTE_METHOD_OPENCL:
	{
#ifdef DSCP4_HAVE_OPENCL
		LOG4CXX_DEBUG(logger_, "OpenCL -- Initializing OpenCL Context")
		SDL_GL_MakeCurrent(windows_[0], glContexts_[numWindows_ - 1]);
		fringeContext_.kernel_file_path = fringeContext_.algorithm_options->opencl_kernel_filename;
		boost::filesystem::path kernelFile(fringeContext_.kernel_file_path);
		if (!boost::filesystem::exists(kernelFile))
			kernelFile = boost::filesystem::path(renderOptions_->kernels_path) / kernelFile;
		
		std::string kernelFileStr = kernelFile.string();

		fringeContext_.kernel_file_path = kernelFileStr.c_str();


		computeContext_ = (dscp4_fringe_opencl_context_t*)dscp4_fringe_opencl_CreateContext(&fringeContext_, (int*)glContexts_[0]);
#else
		LOG4CXX_FATAL(logger_, "OpenCL selected as compute method, but this binary was not compiled with OpenCL")
#endif
	}
		break;
	default:
		LOG4CXX_ERROR(logger_, "No compute method selected, no hologram will be computed")
		break;
	}

}

void DSCP4Render::deinitComputeMethod()
{
	switch (fringeContext_.algorithm_options->compute_method)
	{
	case DSCP4_COMPUTE_METHOD_CUDA:
#ifdef DSCP4_HAVE_CUDA
		LOG4CXX_DEBUG(logger_, "CUDA -- Deinitializing CUDA context")
		dscp4_fringe_cuda_DestroyContext((dscp4_fringe_cuda_context_t**)&computeContext_);
#endif
		break;
	case DSCP4_COMPUTE_METHOD_OPENCL:
#ifdef DSCP4_HAVE_OPENCL
		LOG4CXX_DEBUG(logger_, "OpenCL -- Deinitializing OpenCL context")
		dscp4_fringe_opencl_DestroyContext((dscp4_fringe_opencl_context_t**)&computeContext_);
#endif
		break;
	default:
		break;
	}
}

void DSCP4Render::computeHologram()
{
	switch (fringeContext_.algorithm_options->compute_method)
	{
	case DSCP4_COMPUTE_METHOD_CUDA:
#ifdef DSCP4_HAVE_CUDA
		dscp4_fringe_cuda_ComputeFringe((dscp4_fringe_cuda_context_t*)computeContext_);
#endif
		break;
	case DSCP4_COMPUTE_METHOD_OPENCL:
#ifdef DSCP4_HAVE_OPENCL
		dscp4_fringe_opencl_ComputeFringe((dscp4_fringe_opencl_context_t*)computeContext_);
#endif
		break;
	default:
		break;
	}

	glFinish();
}

void DSCP4Render::updateAlgorithmOptionsCache()
{
	fringeContext_.algorithm_options->cache.num_fringe_buffers =
		fringeContext_.display_options.num_heads / fringeContext_.display_options.num_heads_per_gpu;

	// the x res and y res are set by heads arranged vertically
	// in the OS display management configurations
	fringeContext_.algorithm_options->cache.fringe_buffer_res_x =
		fringeContext_.display_options.head_res_x;

	fringeContext_.algorithm_options->cache.fringe_buffer_res_y =
		fringeContext_.display_options.head_res_y * fringeContext_.display_options.num_heads_per_gpu;

	fringeContext_.algorithm_options->cache.stereogram_num_tiles_x =
		fringeContext_.algorithm_options->cache.stereogram_num_tiles_y =
		static_cast<unsigned int>(sqrt(fringeContext_.algorithm_options->num_views_x));

	fringeContext_.algorithm_options->cache.stereogram_res_x =
		fringeContext_.algorithm_options->cache.stereogram_num_tiles_x
		* fringeContext_.algorithm_options->num_wafels_per_scanline;

	fringeContext_.algorithm_options->cache.stereogram_res_y =
		fringeContext_.algorithm_options->cache.stereogram_num_tiles_y
		* fringeContext_.algorithm_options->num_scanlines;

	fringeContext_.algorithm_options->cache.reference_beam_angle_rad =
		fringeContext_.algorithm_options->reference_beam_angle * M_PI / 180.f;

	fringeContext_.algorithm_options->cache.num_samples_per_wafel =
		fringeContext_.display_options.num_samples_per_hololine /
		fringeContext_.algorithm_options->num_wafels_per_scanline;

	fringeContext_.algorithm_options->cache.k_r =
		2 * M_PI / fringeContext_.algorithm_options->wavelength_red;

	fringeContext_.algorithm_options->cache.k_g =
		2 * M_PI / fringeContext_.algorithm_options->wavelength_green;

	fringeContext_.algorithm_options->cache.k_b =
		2 * M_PI / fringeContext_.algorithm_options->wavelength_blue;

	fringeContext_.algorithm_options->cache.upconvert_const_r =
		(double)((double)fringeContext_.display_options.num_samples_per_hololine * (double)fringeContext_.algorithm_options->temporal_upconvert_red) / 
		(double)((double)fringeContext_.display_options.pixel_clock_rate * (double)fringeContext_.display_options.hologram_plane_width);

	fringeContext_.algorithm_options->cache.upconvert_const_g =
		(double)((double)fringeContext_.display_options.num_samples_per_hololine * (double)fringeContext_.algorithm_options->temporal_upconvert_green)
		/ (double)((double)fringeContext_.display_options.pixel_clock_rate * (double)fringeContext_.display_options.hologram_plane_width);

	fringeContext_.algorithm_options->cache.upconvert_const_b =
		(double)((double)fringeContext_.display_options.num_samples_per_hololine * (double)fringeContext_.algorithm_options->temporal_upconvert_blue)
		/ (double)((double)fringeContext_.display_options.pixel_clock_rate * (double)fringeContext_.display_options.hologram_plane_width);

	fringeContext_.algorithm_options->cache.sample_pitch =
		fringeContext_.display_options.hologram_plane_width
		/ (float)fringeContext_.display_options.num_samples_per_hololine;


#ifdef DSCP4_HAVE_OPENCL
	// sets the global workgroup size X to number of wafels, if it is divisible by local size X
	// otherwise it will set a global workgroup size to be the next multiple of local size X
	// this is because global workgroup size should be as big as possible (the hologram size)
	// but also divisible by the local workgroup size
	if (fringeContext_.algorithm_options->compute_method == DSCP4_COMPUTE_METHOD_OPENCL)
	{
		fringeContext_.algorithm_options->cache.opencl_global_workgroup_size[0] = fringeContext_.algorithm_options->opencl_local_workgroup_size[0] == 0 ? fringeContext_.algorithm_options->num_wafels_per_scanline :
			fringeContext_.algorithm_options->num_wafels_per_scanline % fringeContext_.algorithm_options->opencl_local_workgroup_size[0] == 0 ?
			fringeContext_.algorithm_options->num_wafels_per_scanline :
			fringeContext_.algorithm_options->opencl_local_workgroup_size[0]
			- (fringeContext_.algorithm_options->num_wafels_per_scanline % fringeContext_.algorithm_options->opencl_local_workgroup_size[0])
			+ fringeContext_.algorithm_options->num_wafels_per_scanline;

		fringeContext_.algorithm_options->cache.opencl_global_workgroup_size[1] = fringeContext_.algorithm_options->opencl_local_workgroup_size[1] == 0 ? fringeContext_.algorithm_options->num_scanlines :
			fringeContext_.algorithm_options->num_scanlines % fringeContext_.algorithm_options->opencl_local_workgroup_size[1] == 0 ?
			fringeContext_.algorithm_options->num_scanlines :
			fringeContext_.algorithm_options->opencl_local_workgroup_size[1]
			- (fringeContext_.algorithm_options->num_scanlines % fringeContext_.algorithm_options->opencl_local_workgroup_size[1])
			+ fringeContext_.algorithm_options->num_scanlines;
	}
#endif

#ifdef DSCP4_HAVE_CUDA
	if (fringeContext_.algorithm_options->compute_method == DSCP4_COMPUTE_METHOD_CUDA)
	{
		fringeContext_.algorithm_options->cache.cuda_number_of_blocks[0] =
			fringeContext_.algorithm_options->num_wafels_per_scanline % fringeContext_.algorithm_options->cuda_block_dimensions[0] == 0 ?
			fringeContext_.algorithm_options->num_wafels_per_scanline / fringeContext_.algorithm_options->cuda_block_dimensions[0] :
			(fringeContext_.algorithm_options->num_wafels_per_scanline
			+ (fringeContext_.algorithm_options->cuda_block_dimensions[0]
			- fringeContext_.algorithm_options->num_wafels_per_scanline % fringeContext_.algorithm_options->cuda_block_dimensions[0]))
			/ fringeContext_.algorithm_options->cuda_block_dimensions[0];

		fringeContext_.algorithm_options->cache.cuda_number_of_blocks[1] =
			fringeContext_.algorithm_options->num_scanlines % fringeContext_.algorithm_options->cuda_block_dimensions[1] == 0 ?
			fringeContext_.algorithm_options->num_scanlines / fringeContext_.algorithm_options->cuda_block_dimensions[1] :
			(fringeContext_.algorithm_options->num_scanlines
			+ (fringeContext_.algorithm_options->cuda_block_dimensions[1]
			- fringeContext_.algorithm_options->num_scanlines % fringeContext_.algorithm_options->cuda_block_dimensions[1]))
			/ fringeContext_.algorithm_options->cuda_block_dimensions[1];
	}

#endif

	fringeContext_.algorithm_options->cache.z_offset = 0.f;
	fringeContext_.algorithm_options->cache.z_span = 0.5f;

}

#ifdef DSCP4_HAVE_PNG
void DSCP4Render::saveScreenshotPNG()
{
	switch (renderOptions_->render_mode)
	{
	case DSCP4_RENDER_MODE_MODEL_VIEWING:
	{
		unsigned char * colorBuf = nullptr;
		float * depthBuf = nullptr;
		unsigned short * depthBuf2 = nullptr;
		boost::gil::rgba8_view_t colorImg;
		boost::gil::gray32f_view_t depthImg;
		boost::gil::gray16_view_t depthImg2;

		colorBuf = new unsigned char[fringeContext_.algorithm_options->num_wafels_per_scanline * fringeContext_.algorithm_options->num_scanlines * 4];
		depthBuf = new float[fringeContext_.algorithm_options->num_wafels_per_scanline * fringeContext_.algorithm_options->num_scanlines];
		depthBuf2 = new unsigned short[fringeContext_.algorithm_options->num_wafels_per_scanline * fringeContext_.algorithm_options->num_scanlines];

		glBindTexture(GL_TEXTURE_2D, fringeContext_.view_gl_fbo_color);
		glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, colorBuf);
		glBindTexture(GL_TEXTURE_2D, fringeContext_.view_gl_fbo_depth);
		glGetTexImage(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, GL_FLOAT, depthBuf);
		glBindTexture(GL_TEXTURE_2D, 0);

		colorImg = boost::gil::interleaved_view(fringeContext_.algorithm_options->num_wafels_per_scanline, fringeContext_.algorithm_options->num_scanlines, (boost::gil::rgba8_pixel_t*)colorBuf, fringeContext_.algorithm_options->num_wafels_per_scanline * 4);
		depthImg = boost::gil::interleaved_view(fringeContext_.algorithm_options->num_wafels_per_scanline, fringeContext_.algorithm_options->num_scanlines, (boost::gil::gray32f_pixel_t*)depthBuf, fringeContext_.algorithm_options->num_wafels_per_scanline * 4);
		depthImg2 = boost::gil::interleaved_view(fringeContext_.algorithm_options->num_wafels_per_scanline, fringeContext_.algorithm_options->num_scanlines, (boost::gil::gray16_pixel_t*)depthBuf2, fringeContext_.algorithm_options->num_wafels_per_scanline * 2);

		boost::gil::copy_and_convert_pixels(depthImg, depthImg2);

		boost::gil::png_write_view("dscp4_model_color.png", boost::gil::flipped_up_down_view(colorImg));
		LOG4CXX_INFO(logger_, "Saved model view COLOR screenshot to " << (boost::filesystem::current_path() / "dscp4_model_color.png").string())

		boost::gil::png_write_view("dscp4_model_depth.png", boost::gil::flipped_up_down_view(depthImg2));
		LOG4CXX_INFO(logger_, "Saved model view DEPTH screenshot to " << (boost::filesystem::current_path() / "dscp4_model_depth.png").string())

		delete[] colorBuf;
		delete[] depthBuf;
		delete[] depthBuf2;
	}
		break;
	case DSCP4_RENDER_MODE_AERIAL_DISPLAY:
		break;
	case DSCP4_RENDER_MODE_STEREOGRAM_VIEWING:
	{
		unsigned char * colorBuf = nullptr;
		float * depthBuf = nullptr;
		unsigned short * depthBuf2 = nullptr;
		boost::gil::rgba8_view_t colorImg;
		boost::gil::gray32f_view_t depthImg;
		boost::gil::gray16_view_t depthImg2;

		colorBuf = new unsigned char[fringeContext_.algorithm_options->cache.stereogram_res_x * fringeContext_.algorithm_options->cache.stereogram_res_y * 4];

#ifdef DSCP4_ENABLE_TRACE_LOG
		auto duration = measureTime<>([&](){
#endif

		glBindTexture(GL_TEXTURE_2D, fringeContext_.stereogram_gl_fbo_color);
		glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, colorBuf);
		colorImg = boost::gil::interleaved_view(fringeContext_.algorithm_options->cache.stereogram_res_x, fringeContext_.algorithm_options->cache.stereogram_res_y, (boost::gil::rgba8_pixel_t*)colorBuf, fringeContext_.algorithm_options->cache.stereogram_res_x * 4);
		boost::gil::png_write_view("dscp4_stereogram_color.png", boost::gil::flipped_up_down_view(colorImg));
		LOG4CXX_INFO(logger_, "Saved stereogram view COLOR panorama screenshot to '" << (boost::filesystem::current_path() / "dscp4_stereogram_color.png").string() << "'")

#ifdef DSCP4_ENABLE_TRACE_LOG
		});
		LOG4CXX_TRACE(logger_, "Saving stereogram view COLOR panorama screenshot took " << duration << " ms (" << 1.f / duration * 1000 << " fps)")
#endif

#ifdef DSCP4_ENABLE_TRACE_LOG
		duration = measureTime<>([&](){
#endif

		depthBuf = new float[fringeContext_.algorithm_options->cache.stereogram_res_x * fringeContext_.algorithm_options->cache.stereogram_res_y];
		depthBuf2 = new unsigned short[fringeContext_.algorithm_options->cache.stereogram_res_x * fringeContext_.algorithm_options->cache.stereogram_res_y];
		glBindTexture(GL_TEXTURE_2D, fringeContext_.stereogram_gl_fbo_depth);
		glGetTexImage(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, GL_FLOAT, depthBuf);
		glBindTexture(GL_TEXTURE_2D, 0);

		depthImg = boost::gil::interleaved_view(fringeContext_.algorithm_options->cache.stereogram_res_x, fringeContext_.algorithm_options->cache.stereogram_res_y, (boost::gil::gray32f_pixel_t*)depthBuf, fringeContext_.algorithm_options->cache.stereogram_res_x * 4);
		depthImg2 = boost::gil::interleaved_view(fringeContext_.algorithm_options->cache.stereogram_res_x, fringeContext_.algorithm_options->cache.stereogram_res_y, (boost::gil::gray16_pixel_t*)depthBuf2, fringeContext_.algorithm_options->cache.stereogram_res_x * 2);
		boost::gil::copy_and_convert_pixels(depthImg, depthImg2);
		boost::gil::png_write_view("dscp4_stereogram_depth.png", boost::gil::flipped_up_down_view(depthImg2));
		
		LOG4CXX_INFO(logger_, "Saved stereogram view DEPTH panorama screenshot to '" << (boost::filesystem::current_path() / "dscp4_stereogram_depth.png").string() << "'")

#ifdef DSCP4_ENABLE_TRACE_LOG
		});
		LOG4CXX_TRACE(logger_, "Saving stereogram view DEPTH panorama screenshot took " << duration << " ms (" << 1.f / duration * 1000 << " fps)")
#endif


		for (unsigned int i = 0; i < fringeContext_.algorithm_options->num_views_x; i++)
		{

				std::stringstream depthFilenameSS;
				depthFilenameSS << "dscp4_stereogram_depth_" << std::setfill('0') << std::setw(2) << i << ".png";

				std::stringstream colorFilenameSS;
				colorFilenameSS << "dscp4_stereogram_color_" << std::setfill('0') << std::setw(2) << i << ".png";

				std::string colorFilename = colorFilenameSS.str();
				std::string depthFilename = depthFilenameSS.str();

#ifdef DSCP4_ENABLE_TRACE_LOG
				duration = measureTime<>([&](){
#endif

				auto colorSubImg = boost::gil::subimage_view(colorImg,
					fringeContext_.algorithm_options->num_wafels_per_scanline * (i%static_cast<unsigned int>(sqrt(fringeContext_.algorithm_options->num_views_x))),
					fringeContext_.algorithm_options->num_scanlines *(i / static_cast<unsigned int>(sqrt(fringeContext_.algorithm_options->num_views_x))),
					fringeContext_.algorithm_options->num_wafels_per_scanline,
					fringeContext_.algorithm_options->num_scanlines);
				boost::gil::png_write_view(colorFilename.c_str(), boost::gil::flipped_up_down_view(colorSubImg));
				LOG4CXX_INFO(logger_, "Saved stereogram view COLOR screenshot " << i + 1 << " of " << fringeContext_.algorithm_options->num_views_x << " to '" << (boost::filesystem::current_path() / colorFilename).string() << "'")

#ifdef DSCP4_ENABLE_TRACE_LOG
			});
			LOG4CXX_TRACE(logger_, "Saving stereogram view COLOR screenshot took " << duration << " ms (" << 1.f / duration * 1000 << " fps)")
#endif

#ifdef DSCP4_ENABLE_TRACE_LOG
				duration = measureTime<>([&](){
#endif
					auto depthSubImg = boost::gil::subimage_view(depthImg2,
					fringeContext_.algorithm_options->num_wafels_per_scanline * (i%static_cast<unsigned int>(sqrt(fringeContext_.algorithm_options->num_views_x))),
					fringeContext_.algorithm_options->num_scanlines *(i / static_cast<unsigned int>(sqrt(fringeContext_.algorithm_options->num_views_x))),
					fringeContext_.algorithm_options->num_wafels_per_scanline,
					fringeContext_.algorithm_options->num_scanlines);
				boost::gil::png_write_view(depthFilename.c_str(), boost::gil::flipped_up_down_view(depthSubImg));
				LOG4CXX_INFO(logger_, "Saved stereogram view DEPTH screenshot " << i + 1 << " of " << fringeContext_.algorithm_options->num_views_x << " to '" << (boost::filesystem::current_path() / depthFilename).string() << "'")
#ifdef DSCP4_ENABLE_TRACE_LOG
			});
			LOG4CXX_TRACE(logger_, "Saving stereogram view DEPTH screenshot took " << duration << " ms (" << 1.f / duration * 1000 << " fps)")
#endif
		}

		delete[] colorBuf;
		delete[] depthBuf;
		delete[] depthBuf2;
	}
		break;
	case DSCP4_RENDER_MODE_HOLOVIDEO_FRINGE:
	{
		unsigned char * fringeBuffer = nullptr;
		fringeBuffer = new unsigned char[fringeContext_.algorithm_options->cache.fringe_buffer_res_x * fringeContext_.algorithm_options->cache.fringe_buffer_res_y * 4];


		for (unsigned int i = 0; i < fringeContext_.algorithm_options->cache.num_fringe_buffers; i++)
		{
			std::stringstream fringeFilenameSS;
			fringeFilenameSS << "dscp4_fringe_pattern_" << std::setfill('0') << std::setw(2) << i << ".png";

			std::string fringeFilename = fringeFilenameSS.str();

#ifdef DSCP4_ENABLE_TRACE_LOG
			auto duration = measureTime<>([&](){
#endif

			glFinish();

			glBindTexture(GL_TEXTURE_2D, fringeContext_.fringe_gl_tex_out[i]);
			glGetTexImage(GL_TEXTURE_2D, 0, GL_RGBA, GL_UNSIGNED_BYTE, fringeBuffer);

			auto fringeBufferNoAlpha = new unsigned char[fringeContext_.algorithm_options->cache.fringe_buffer_res_x * fringeContext_.algorithm_options->cache.fringe_buffer_res_y * 3];

			auto fringeImage = boost::gil::interleaved_view(fringeContext_.algorithm_options->cache.fringe_buffer_res_x, fringeContext_.algorithm_options->cache.fringe_buffer_res_y, (boost::gil::rgba8_pixel_t*)fringeBuffer, fringeContext_.algorithm_options->cache.fringe_buffer_res_x * 4);
			
			auto fringeImageNoAlpha = boost::gil::interleaved_view(fringeContext_.algorithm_options->cache.fringe_buffer_res_x, fringeContext_.algorithm_options->cache.fringe_buffer_res_y, (boost::gil::rgb8_pixel_t*)fringeBufferNoAlpha, fringeContext_.algorithm_options->cache.fringe_buffer_res_x * 3);
			boost::gil::copy_and_convert_pixels(fringeImage, fringeImageNoAlpha);

			boost::gil::png_write_view(fringeFilename.c_str(), fringeImage);


			LOG4CXX_INFO(logger_, "Saved hologram fringe pattern buffer " << i + 1 << " of " << fringeContext_.algorithm_options->cache.num_fringe_buffers << " to '" << (boost::filesystem::current_path() / fringeFilename).string() << "'")

#ifdef DSCP4_ENABLE_TRACE_LOG
			});
			LOG4CXX_TRACE(logger_, "Saving hologram fringe pattern screenshot took " << duration << " ms (" << 1.f / duration * 1000 << " fps)")
#endif	
		}

		glBindTexture(GL_TEXTURE, 0);

		delete[] fringeBuffer;
	}
		break;
	default:
		break;
	}

}
#endif

void DSCP4Render::setFullScreen(bool fullscreen)
{
	if (fullscreen != isFullScreen_.load())
	{
		for (unsigned int w = 0; w < numWindows_; w++)
		{
			SDL_Rect bounds = { 0 };
			if (SDL_GetDisplayBounds(w, &bounds) == -1)
				SDL_GetDisplayBounds(0, &bounds);

			int x = bounds.x;
			int y = bounds.y;

			if (fullscreen)
			{
				windowWidth_[w] = bounds.w;
				windowHeight_[w] = bounds.h;
			}
			else
			{
				switch (renderOptions_->render_mode)
				{
				case DSCP4_RENDER_MODE_MODEL_VIEWING:
					windowWidth_[w] = fringeContext_.algorithm_options->num_wafels_per_scanline;
					windowHeight_[w] = fringeContext_.algorithm_options->num_scanlines;
					x = (bounds.w - windowWidth_[w]) / 2;
					y = (bounds.h - windowHeight_[w]) / 2;
					break;
				case DSCP4_RENDER_MODE_STEREOGRAM_VIEWING:
					windowWidth_[w] *= 0.8f;
					windowHeight_[w] = windowWidth_[w] * (float)fringeContext_.algorithm_options->cache.stereogram_res_y / (float)fringeContext_.algorithm_options->cache.stereogram_res_x;
					x = (bounds.w - windowWidth_[w]) / 2;
					y = (bounds.h - windowHeight_[w]) / 2;
					break;
				case DSCP4_RENDER_MODE_AERIAL_DISPLAY:
					x += windowHeight_[w] * 0.03f;
					y += windowWidth_[w] * 0.03f;
					windowHeight_[w] *= 0.8f;
					windowWidth_[w] *= 0.8f;
					break;
				case DSCP4_RENDER_MODE_HOLOVIDEO_FRINGE:
					x += windowHeight_[w] * 0.03f;
					y += windowWidth_[w] * 0.03f;
					windowHeight_[w] *= 0.8f;
					windowWidth_[w] = windowHeight_[w] * (float)fringeContext_.algorithm_options->cache.fringe_buffer_res_x / (float)fringeContext_.algorithm_options->cache.fringe_buffer_res_y;
					break;
				default:
					break;

				}


			}

			SDL_GL_MakeCurrent(windows_[w], glContexts_[w]);

			SDL_SetWindowBordered(windows_[w], fullscreen ? SDL_FALSE : SDL_TRUE);
			//SDL_SetWindowFullscreen(windows_[w], fullscreen ? SDL_WINDOW_BOR : 0);
			
			SDL_SetWindowSize(windows_[w], windowWidth_[w], windowHeight_[w]);
			SDL_SetWindowPosition(windows_[w], x, y);

			Update();

		}

		isFullScreen_.store(fullscreen);

	}
}
