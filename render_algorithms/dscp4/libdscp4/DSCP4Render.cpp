#include "DSCP4Render.hpp"

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

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
		DSCP4Render(render_options_t {
						DSCP4_DEFAULT_RENDER_SHADERS_PATH,
						DSCP4_DEFAULT_RENDER_KERNELS_PATH,
						DSCP4_DEFAULT_RENDER_SHADER_FILENAME_PREFIX,
						DSCP4_DEFAULT_RENDER_RENDER_MODE,
						DSCP4_DEFAULT_RENDER_SHADER_MODEL,
						DSCP4_DEFAULT_RENDER_LIGHT_POS_X,
						DSCP4_DEFAULT_RENDER_LIGHT_POS_Y,
						DSCP4_DEFAULT_RENDER_LIGHT_POS_Z,
						DSCP4_DEFAULT_RENDER_AUTOSCALE_ENABLED },
					algorithm_options_t {
						DSCP4_DEFAULT_ALGORITHM_NUM_VIEWS_X,
						DSCP4_DEFAULT_ALGORITHM_NUM_VIEWS_Y,
						DSCP4_DEFAULT_ALGORITHM_NUM_WAFELS,
						DSCP4_DEFAULT_ALGORITHM_NUM_SCANLINES,
						DSCP4_DEFAULT_ALGORITHM_FOV_X,
						DSCP4_DEFAULT_ALGORITHM_FOV_Y,
						DSCP4_DEFAULT_COMPUTE_METHOD },
					display_options_t {
						DSCP4_DEFAULT_DISPLAY_NAME,
						DSCP4_DEFAULT_DISPLAY_NUM_HEADS,
						DSCP4_DEFAULT_DISPLAY_NUM_HEADS_PER_GPU,
						DSCP4_DEFAULT_DISPLAY_HEAD_RES_X,
						DSCP4_DEFAULT_DISPLAY_HEAD_RES_Y},
						DSCP4_DEFAULT_LOG_VERBOSITY)
{
	
}

DSCP4Render::DSCP4Render(render_options_t renderOptions,
	algorithm_options_t algorithmOptions,
	display_options_t displayOptions,
	unsigned int verbosity
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
	rotateIncrement_(1.0f),
	spinOn_(false),
	zNear_(DSCP4_RENDER_DEFAULT_ZNEAR),
	zFar_(DSCP4_RENDER_DEFAULT_ZFAR),
	renderOptions_(renderOptions),
	isFullScreen_(false),
	lightingShader_(nullptr),
	projectionMatrix_(),
	viewMatrix_(),
	modelMatrix_(),
	camera_(),
	lighting_(),
	cameraChanged_(false),
	lightingChanged_(false),
	meshChanged_(false),
	fringeContext_({ algorithmOptions, displayOptions, nullptr, 0, 0, 0, 0, 0, 0, nullptr, nullptr })
{

#ifdef DSCP4_HAVE_LOG4CXX
	
	log4cxx::BasicConfigurator::resetConfiguration();

#ifdef WIN32
	log4cxx::PatternLayoutPtr logLayoutPtr = new log4cxx::PatternLayout(L"%-5p %m%n");
#else
	log4cxx::PatternLayoutPtr logLayoutPtr = new log4cxx::PatternLayout("%-5p %m%n");
#endif

	log4cxx::ConsoleAppenderPtr logAppenderPtr = new log4cxx::ConsoleAppender(logLayoutPtr);
	log4cxx::BasicConfigurator::configure(logAppenderPtr);

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

	if (renderOptions.shaders_path == nullptr)
	{
		LOG4CXX_WARN(logger_, "No shader path location specified, using current working path: " << boost::filesystem::current_path().string())
			renderOptions_.shaders_path = (char*)boost::filesystem::current_path().string().c_str();
	}

//#ifdef DSCP4_HAVE_CUDA
//	char * helloWorldCudaStr = dscp4_fringe_cuda_HelloWorld();
//	LOG4CXX_INFO(logger_, "CUDA--If CUDA is working, this should say 'World!', not 'Hello ': " << helloWorldCudaStr)
//#endif

}

DSCP4Render::~DSCP4Render()
{
	
}

bool DSCP4Render::init()
{
	LOG4CXX_INFO(logger_, "Initializing DSCP4...")

	LOG4CXX_INFO(logger_, "Initializing SDL with video subsystem")
	CHECK_SDL_RC(SDL_Init(SDL_INIT_VIDEO) < 0, "Could not initialize SDL")


	switch (renderOptions_.render_mode)
	{
	case DSCP4_RENDER_MODE_MODEL_VIEWING:
		numWindows_ = 1;
		break;
	case DSCP4_RENDER_MODE_STEREOGRAM_VIEWING:
		numWindows_ = 1;
		break;
	case DSCP4_RENDER_MODE_AERIAL_DISPLAY:
		numWindows_ = SDL_GetNumVideoDisplays();
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

	SDL_GL_SetSwapInterval(1);

	windows_ = new SDL_Window*[numWindows_];
	glContexts_ = new SDL_GLContext[numWindows_];
	windowWidth_ = new unsigned int[numWindows_];
	windowHeight_ = new unsigned int[numWindows_];

	std::unique_lock<std::mutex> initLock(isInitMutex_);
	shouldRender_ = true;
	renderThread_ = std::thread(std::bind(&DSCP4Render::renderLoop, this));

	isInitCV_.wait(initLock);

	initLock.unlock();
		
	return true;
}

bool DSCP4Render::initWindow(SDL_Window*& window, SDL_GLContext& glContext, int thisWindowNum)
{
	LOG4CXX_DEBUG(logger_, "Inititalizing SDL for Window " << thisWindowNum)
	SDL_Rect bounds = { 0 };

	// This will get the resolution of the primary window if everything else fails
	// Useful for opening up 3 windows during hologram mode to test everything
	if (SDL_GetDisplayBounds(thisWindowNum, &bounds) == -1)
		SDL_GetDisplayBounds(0, &bounds);

	switch (renderOptions_.render_mode)
	{
	case DSCP4_RENDER_MODE_MODEL_VIEWING:
		windowWidth_[thisWindowNum] = fringeContext_.algorithm_options.num_wafels_per_scanline;
		windowHeight_[thisWindowNum] = fringeContext_.algorithm_options.num_scanlines;
		LOG4CXX_DEBUG(logger_, "Creating SDL OpenGL Window " << thisWindowNum << ": " << windowWidth_[thisWindowNum] << "x" << windowHeight_[thisWindowNum] << " @ " << "{" << bounds.x + 80 << "," << bounds.y + 80 << "}")
		window = SDL_CreateWindow(("dscp4-" + std::to_string(thisWindowNum)).c_str(), bounds.x + 80, bounds.y + 80, windowWidth_[thisWindowNum], windowHeight_[thisWindowNum], SDL_WINDOW_OPENGL);
		break;
	case DSCP4_RENDER_MODE_STEREOGRAM_VIEWING:
		windowWidth_[thisWindowNum] = fringeContext_.algorithm_options.num_wafels_per_scanline*2;
		windowHeight_[thisWindowNum] = fringeContext_.algorithm_options.num_scanlines*2;
		LOG4CXX_DEBUG(logger_, "Creating SDL OpenGL Window " << thisWindowNum << ": " << windowWidth_[thisWindowNum] << "x" << windowHeight_[thisWindowNum] << " @ " << "{" << bounds.x + 80 << "," << bounds.y + 80 << "}")
		window = SDL_CreateWindow(("dscp4-" + std::to_string(thisWindowNum)).c_str(), bounds.x + 80, bounds.y + 80, windowWidth_[thisWindowNum], windowHeight_[thisWindowNum], SDL_WINDOW_OPENGL);
		break;
	case DSCP4_RENDER_MODE_AERIAL_DISPLAY:
	case DSCP4_RENDER_MODE_HOLOVIDEO_FRINGE:
		windowWidth_[thisWindowNum] = bounds.w;
		windowHeight_[thisWindowNum] = bounds.h;
		LOG4CXX_DEBUG(logger_, "Creating fullscreen SDL OpenGL Window " << thisWindowNum << ": " << bounds.w << "x" << bounds.h << " @ " << "{" << bounds.x << "," << bounds.y << "}")
		window = SDL_CreateWindow(("dscp4-" + std::to_string(thisWindowNum)).c_str(), bounds.x, bounds.y, bounds.w, bounds.h, SDL_WINDOW_OPENGL);
		SDL_ShowCursor(SDL_DISABLE);
		break;
	default:
		break;
	}
	
	CHECK_SDL_RC(window == nullptr, "Could not create SDL window");

	LOG4CXX_DEBUG(logger_, "Creating GL Context from SDL window " << thisWindowNum)
	
	SDL_GL_SetAttribute(SDL_GL_SHARE_WITH_CURRENT_CONTEXT, 1);

	glContext = SDL_GL_CreateContext(window);

	LOG4CXX_DEBUG(logger_, "Initializing GLEW")
	
	GLenum err = glewInit();
	if (err != GLEW_OK)
	{
		LOG4CXX_ERROR(logger_, "Could not initialize GLEW: " << glewGetString(err))
	}

	SDL_GL_MakeCurrent(window, glContext);

	LOG4CXX_DEBUG(logger_, "Turning on VSYNC")
	SDL_GL_SetSwapInterval(1);

	glViewport(0, 0, windowWidth_[thisWindowNum], windowHeight_[thisWindowNum]);

	// Set a black background
	glClearColor(0.0f, 0.0f, 0.0f, 0.0f); // Black Background
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
		(boost::filesystem::path(renderOptions_.shaders_path) /
		boost::filesystem::path(std::string((const char*)renderOptions_.shader_filename_prefix).append(".vert"))).string()
		);

	lightingShader_[which].loadShader(VSShaderLib::FRAGMENT_SHADER,
		(boost::filesystem::path(renderOptions_.shaders_path) /
		boost::filesystem::path(std::string((const char*)renderOptions_.shader_filename_prefix).append(".frag"))).string()
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
	long long duration = 0;
#endif

	float ratio = 0.f;
	float q = 0.f; //offset for rendering stereograms
	SDL_Event event = { 0 };

	//lightingShader_ = new VSShaderLib[numWindows_];

	camera_.eye = glm::vec3(0, 0, (renderOptions_.render_mode == DSCP4_RENDER_MODE_MODEL_VIEWING) || (renderOptions_.render_mode == DSCP4_RENDER_MODE_AERIAL_DISPLAY) ? 4.0f : .5f);
	camera_.center = glm::vec3(0, 0, 0);
	camera_.up = glm::vec3(0, 1, 0);

	lighting_.position = glm::vec4(renderOptions_.light_pos_x, renderOptions_.light_pos_y, renderOptions_.light_pos_z, 1.f);
	lighting_.ambientColor = glm::vec4(0.2f, 0.2f, 0.2f, 1.f);
	lighting_.diffuseColor = glm::vec4(1.0f, 1.0f, 1.0f, 1.f);
	lighting_.specularColor = glm::vec4(1.f, 1.f, 1.f, 1.f);
	lighting_.globalAmbientColor = glm::vec4(0.f, 0.f, 0.f, 1.f);

	// Both model viewing and stereogram viewing just require 1 window,
	// so we only initialize one context and window
	// These modes are only for debugging/visualizing purposes
	switch (renderOptions_.render_mode)
	{
	case DSCP4_RENDER_MODE_MODEL_VIEWING:
	case DSCP4_RENDER_MODE_STEREOGRAM_VIEWING:

		initWindow(windows_[0], glContexts_[0], 0);

		// Add ambient and diffuse lighting to the scene
		glLightfv(GL_LIGHT0, GL_AMBIENT, glm::value_ptr(lighting_.ambientColor));
		glLightfv(GL_LIGHT0, GL_DIFFUSE, glm::value_ptr(lighting_.diffuseColor));

		glLightModelfv(GL_AMBIENT_AND_DIFFUSE, glm::value_ptr(lighting_.globalAmbientColor));

		break;
	case DSCP4_RENDER_MODE_AERIAL_DISPLAY:

		for (unsigned int i = 0; i < numWindows_; i++)
		{
			initWindow(windows_[i], glContexts_[i], i);

			// Add ambient and diffuse lighting to every scene
			glLightfv(GL_LIGHT0, GL_AMBIENT, glm::value_ptr(lighting_.ambientColor));
			glLightfv(GL_LIGHT0, GL_DIFFUSE, glm::value_ptr(lighting_.diffuseColor));

			glLightModelfv(GL_AMBIENT_AND_DIFFUSE, glm::value_ptr(lighting_.globalAmbientColor));
		}

		break;
	case DSCP4_RENDER_MODE_HOLOVIDEO_FRINGE:
	{
		for (unsigned int i = 0; i < numWindows_; i++)
		{
			initWindow(windows_[i], glContexts_[i], i);

			// Add ambient and diffuse lighting to every scene
			glLightfv(GL_LIGHT0, GL_AMBIENT, glm::value_ptr(lighting_.ambientColor));
			glLightfv(GL_LIGHT0, GL_DIFFUSE, glm::value_ptr(lighting_.diffuseColor));

			glLightModelfv(GL_AMBIENT_AND_DIFFUSE, glm::value_ptr(lighting_.globalAmbientColor));
		}

		SDL_GL_MakeCurrent(windows_[0], glContexts_[0]);

		initFringeBuffers();

		initComputeMethod();
		
	}
		break;
	default:
		break;
	}

	// For capturing mouse and keyboard events
	SDL_AddEventWatch(DSCP4Render::inputStateChanged, this);

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

	while (shouldRender_)
	{
		SDL_Event event = { 0 };

		std::unique_lock<std::mutex> updateFrameLock(updateFrameMutex_);
		if (!(meshChanged_ || cameraChanged_ || lightingChanged_ || spinOn_))
		{
			if (std::cv_status::timeout == updateFrameCV_.wait_for(updateFrameLock, std::chrono::milliseconds(1)))
				goto poll;
		}

#ifdef DSCP4_ENABLE_TRACE_LOG
		duration = measureTime<>([&](){
#endif

			// Increments rotation if spinOn_ is true
			// Otherwise rotates by rotateAngle_
			rotateAngleY_ = spinOn_.load() == true ?
				rotateAngleY_ > 359.f ?
				0.f : rotateAngleY_ + rotateIncrement_ : rotateAngleY_.load();

			switch (renderOptions_.render_mode)
			{
			case DSCP4_RENDER_MODE_MODEL_VIEWING:
			{
#ifdef DSCP4_ENABLE_TRACE_LOG
				auto duration = measureTime<>(std::bind(&DSCP4Render::drawForViewing, this));
				LOG4CXX_TRACE(logger_, "Generating a single view took " << duration << " ms (" << 1.f / duration * 1000 << " fps)")
#else
				drawForViewing();
#endif
				SDL_GL_SwapWindow(windows_[0]);
			}
				break;
			case DSCP4_RENDER_MODE_STEREOGRAM_VIEWING:
			{
#ifdef DSCP4_ENABLE_TRACE_LOG
				auto duration = measureTime<>(std::bind(&DSCP4Render::drawForStereogram, this));
				LOG4CXX_TRACE(logger_, "Generating " << fringeContext_.algorithm_options.num_views_x << " views took " << duration << " ms (" << 1.f / duration * 1000 << " fps)")
#else
				drawForStereogram();
#endif
				SDL_GL_SwapWindow(windows_[0]);
			}
				break;
			case DSCP4_RENDER_MODE_AERIAL_DISPLAY:
			{
		#ifdef DSCP4_ENABLE_TRACE_LOG

				auto duration = measureTime<>(std::bind(&DSCP4Render::drawForAerialDisplay, this));
				LOG4CXX_TRACE(logger_, "Generating " << numWindows_ << " views took " << duration << " ms (" << 1.f / duration * 1000 << " fps)")
		#else
				drawForAerialDisplay();

		#endif
				for (unsigned int i = 0; i < numWindows_; i++)
				{
					SDL_GL_MakeCurrent(windows_[i], glContexts_[i]);
					SDL_GL_SwapWindow(windows_[i]);
				}
			}
				break;

			case DSCP4_RENDER_MODE_HOLOVIDEO_FRINGE:
			{
				drawForFringe();
				for (unsigned int i = 0; i < numWindows_; i++)
				{
					SDL_GL_MakeCurrent(windows_[i], glContexts_[i]);
					SDL_GL_SwapWindow(windows_[i]);
				}
			}
				break;
			default:
				break;
			}

#ifdef DSCP4_ENABLE_TRACE_LOG
		});

		LOG4CXX_TRACE(logger_, "Rendering the frame took " << duration << " ms (" << 1.f / duration * 1000 << " fps)");
#endif

	poll:
		SDL_PollEvent(&event);
	}

	initLock.lock();

	SDL_DelEventWatch(DSCP4Render::inputStateChanged, this);

	std::unique_lock<std::mutex> meshLock(meshMutex_);
	for (auto it = meshes_.begin(); it != meshes_.end(); it++)
		glDeleteBuffers(3, &it->second.info.gl_vertex_buf_id);
	meshLock.unlock();

	SDL_GL_MakeCurrent(windows_[0], glContexts_[0]);

	if (renderOptions_.render_mode == DSCP4_RENDER_MODE_HOLOVIDEO_FRINGE)
	{
		deinitComputeMethod();
		deinitFringeBuffers();
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

	initLock.unlock();
	isInitCV_.notify_all();

	isInit_ = false;
}

void DSCP4Render::drawForViewing()
{
	const float ratio = (float)windowWidth_[0] / (float)windowHeight_[0];
	{
		std::lock_guard<std::mutex> lgc(cameraMutex_);
		glMatrixMode(GL_PROJECTION);

		projectionMatrix_ = glm::mat4();
		projectionMatrix_ *= glm::perspective(fringeContext_.algorithm_options.fov_y * DEG_TO_RAD, ratio, zNear_, zFar_);


		glLoadMatrixf(glm::value_ptr(projectionMatrix_));

		if (renderOptions_.shader_model != DSCP4_SHADER_MODEL_OFF)
			glEnable(GL_LIGHTING);

		glShadeModel(renderOptions_.shader_model == DSCP4_SHADER_MODEL_SMOOTH ? GL_SMOOTH : GL_FLAT);

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
}

// Builds stereogram views and lays them out in a NxN grid
// Therefore number of views MUST be N^2 value (e.g. 16 views, 4x4 tiles)
void DSCP4Render::drawForStereogram()
{
	// X and Y resolution for each tile, or stereogram view
	// For "STEREOGRAM" render mode, we divide x and y res by 2
	// so that many views can fit in the window, just for user friendliness
	const int tileX = renderOptions_.render_mode == DSCP4_RENDER_MODE_STEREOGRAM_VIEWING ?
		fringeContext_.algorithm_options.num_wafels_per_scanline / 2 :
		fringeContext_.algorithm_options.num_wafels_per_scanline;

	const int tileY = renderOptions_.render_mode == DSCP4_RENDER_MODE_STEREOGRAM_VIEWING ?
		fringeContext_.algorithm_options.num_scanlines / 2 :
		fringeContext_.algorithm_options.num_scanlines;

	// The grid dimension
	const int tileDim = static_cast<unsigned int>(sqrt(fringeContext_.algorithm_options.num_views_x));

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	std::lock_guard<std::mutex> lgc(cameraMutex_);
	std::lock_guard<std::mutex> lgl(lightingMutex_);
	std::lock_guard<std::mutex> lgm(meshMutex_);

	for (unsigned int i = 0; i < fringeContext_.algorithm_options.num_views_x; i++)
	{
		glViewport(tileX*(i%tileDim), tileY*(i / tileDim), tileX, tileY);
		
		glMatrixMode(GL_PROJECTION);

		const float ratio = (float)windowWidth_[0] / (float)windowHeight_[0];
		const float q = (i - fringeContext_.algorithm_options.num_views_x * 0.5f) / static_cast<float>(fringeContext_.algorithm_options.num_views_x) * fringeContext_.algorithm_options.fov_y * DEG_TO_RAD;

		projectionMatrix_ = buildOrthoXPerspYProjMat(-ratio, ratio, -1.0f, 1.0f, zNear_, zFar_, q);

		glLoadMatrixf(glm::value_ptr(projectionMatrix_));

		if (renderOptions_.shader_model != DSCP4_SHADER_MODEL_OFF)
			glEnable(GL_LIGHTING);

		glShadeModel(renderOptions_.shader_model == DSCP4_SHADER_MODEL_SMOOTH ? GL_SMOOTH : GL_FLAT);

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

		glLoadMatrixf(glm::value_ptr(viewMatrix_));

		glEnable(GL_DEPTH_TEST);
		glEnable(GL_LIGHT0);

		drawAllMeshes();

		glDisable(GL_LIGHT0);
		glDisable(GL_DEPTH_TEST);

		if (renderOptions_.shader_model != DSCP4_SHADER_MODEL_OFF)
			glDisable(GL_LIGHTING);
	}

	cameraChanged_ = false;
	meshChanged_ = false;
	lightingChanged_ = false;
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
		projectionMatrix_ *= glm::perspective(fringeContext_.algorithm_options.fov_y * DEG_TO_RAD, ratio, zNear_, zFar_);

		glLoadMatrixf(glm::value_ptr(projectionMatrix_));

		if (renderOptions_.shader_model != DSCP4_SHADER_MODEL_OFF)
			glEnable(GL_LIGHTING);

		glShadeModel(renderOptions_.shader_model == DSCP4_SHADER_MODEL_SMOOTH ? GL_SMOOTH : GL_FLAT);

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
		//drawCube();

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

	glBindFramebuffer(GL_FRAMEBUFFER, fringeContext_.stereogram_gl_fbo);
	glDrawBuffer(GL_COLOR_ATTACHMENT0);

	// Intel GPU bug, 0.0f has residual colors from previous frame
	glClearColor(0.000001f, 0.f, 0.f, 1.0f);
	glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);


#ifdef DSCP4_ENABLE_TRACE_LOG
	auto duration = measureTime<>(std::bind(&DSCP4Render::drawForStereogram, this));
	LOG4CXX_TRACE(logger_, "Rendering " << fringeContext_.algorithm_options.num_views_x << " views in total took " << duration << " ms (" << 1.f / duration * 1000 << " fps)")
#else
	drawForStereogram();
#endif

	glReadBuffer(GL_COLOR_ATTACHMENT0);
	if (fringeContext_.algorithm_options.compute_method == DSCP4_COMPUTE_METHOD_CUDA)
	{
#ifdef DSCP4_ENABLE_TRACE_LOG
		duration = measureTime<>(std::bind(&DSCP4Render::copyStereogramToPBOs, this));
		LOG4CXX_TRACE(logger_, "Copying stereogram " << fringeContext_.algorithm_options.num_views_x << " views to PBOs took " << duration << " ms (" << 1.f / duration * 1000 << " fps)")
#else
		copyStereogramToPBOs();
#endif
	}

	glBindFramebuffer(GL_FRAMEBUFFER, 0);
	glReadBuffer(GL_BACK);
	glDrawBuffer(GL_BACK);


#ifdef DSCP4_ENABLE_TRACE_LOG
	duration = measureTime<>(std::bind(&DSCP4Render::computeHologram, this));
	LOG4CXX_TRACE(logger_, "Compute hologram fringe pattern took " << duration << " ms (" << 1.f / duration * 1000 << " fps)")
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
	// This will put the mesh in the vertex array buffer
	// If it is not in there already
	if (mesh.info.gl_vertex_buf_id == -1)
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

	glDrawArrays(GL_TRIANGLES, 0, mesh.info.num_vertices);
	glDisableClientState(GL_VERTEX_ARRAY);
	glDisableClientState(GL_NORMAL_ARRAY);
	glDisableClientState(GL_COLOR_ARRAY);
	glBindBuffer(GL_ARRAY_BUFFER, 0);

	glDisable(GL_NORMALIZE);
	glDisable(GL_COLOR_MATERIAL);
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
		modelMatrix_ = glm::scale(modelMatrix_, glm::vec3(scaleFactor + transform.scale.x, scaleFactor + transform.scale.y, scaleFactor + transform.scale.z));

		modelMatrix_ = glm::translate(modelMatrix_, glm::vec3(
			-mesh.info.bounding_sphere.x + transform.translate.x,
			-mesh.info.bounding_sphere.y + transform.translate.y,
			-mesh.info.bounding_sphere.z + transform.translate.z));

		glLoadMatrixf(glm::value_ptr(modelMatrix_));

		//draw the actual mesh
		drawMesh(it->second);
	}
}

void DSCP4Render::addMesh(const char *id, int numVertices, float *vertices, float * normals, float *colors, unsigned int numVertexDimensions, unsigned int numColorChannels)
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
	mesh.info.is_point_cloud = false;
	mesh.info.gl_color_buf_id = -1;
	mesh.info.gl_vertex_buf_id = -1;
	mesh.info.gl_normal_buf_id = -1;

	if (renderOptions_.auto_scale_enabled)
	{
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
	}

	std::unique_lock<std::mutex> meshLock(meshMutex_);
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

void DSCP4Render::addPointCloud(const char *id, float *points, int numPoints, float pointSize, bool hasColorData)
{
	// create a 2D array for miniball algorithm
	//float** ap = new float*[numPoints];
	//float * pv = points;
	//for (int i = 0; i<numPoints; ++i) {
	//	ap[i] = pv;
	//	pv += 4;
	//}

	// miniball uses a quick method of determining the bounding sphere of all the vertices
	//auto miniball3f = Miniball::Miniball<Miniball::CoordAccessor<float**, float*>>(3, (float**)ap, (float**)(ap + numPoints));

	mesh_t mesh = { 0 };
	mesh.vertices = points;
	mesh.colors = &points[3];
	mesh.info.num_color_channels = 4;
	mesh.info.num_points_per_vertex = 3;
	mesh.info.vertex_stride = 3 * sizeof(float)+ 4 * sizeof(char);
	mesh.info.color_stride = 3 * sizeof(float)+ 4 * sizeof(char);
	mesh.info.num_vertices = numPoints;
	//mesh.info.center_x = miniball3f.center()[0];
	//mesh.info.center_y = miniball3f.center()[1];
	//mesh.info.center_z = miniball3f.center()[2];
	//mesh.info.sq_radius = miniball3f.squared_radius();
	mesh.info.is_point_cloud = false;

	std::unique_lock<std::mutex> meshLock(meshMutex_);
	meshes_[id] = mesh;
	meshLock.unlock();

	//need to optimize this
	//for (int i = 0; i<numVertices; ++i)
	//	delete[] ap[i];
	//delete[] ap;
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

int DSCP4Render::inputStateChanged(void* userdata, SDL_Event* event)
{
	auto render = (DSCP4Render*)userdata;

	if (event->key.type == SDL_KEYDOWN)
	{

		if (event->key.keysym.mod == SDL_Keymod::KMOD_LSHIFT)
		{
			auto camera = render->getCameraView();
			auto lighting = render->getLighting();

			switch (event->key.keysym.scancode)
			{
			case  SDL_Scancode::SDL_SCANCODE_W:
				lighting.position[1] += 0.1f;
				break;
			case SDL_Scancode::SDL_SCANCODE_S:
				lighting.position[1] -= 0.1f;
				break;
			case  SDL_Scancode::SDL_SCANCODE_A:
				lighting.position[0] -= 0.1f;
				break;
			case SDL_Scancode::SDL_SCANCODE_D:
				lighting.position[0] += 0.1f;
				break;
			case  SDL_Scancode::SDL_SCANCODE_Z:
				lighting.position[2] -= 0.1f;
				break;
			case SDL_Scancode::SDL_SCANCODE_X:
				lighting.position[2] += 0.1f;
				break;
			case  SDL_Scancode::SDL_SCANCODE_UP:
				camera.eye[1] += 0.1f;
				camera.center[1] += 0.1f;
				break;
			case  SDL_Scancode::SDL_SCANCODE_DOWN:
				camera.eye[1] -= 0.1f;
				camera.center[1] -= 0.1f;
				break;
			case  SDL_Scancode::SDL_SCANCODE_LEFT:
				camera.eye[0] -= 0.1f;
				camera.center[0] -= 0.1f;
				break;
			case SDL_Scancode::SDL_SCANCODE_RIGHT:
				camera.eye[0] += 0.1f;
				camera.center[0] += 0.1f;
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
				render->setLighting(lighting);
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
				render->setCameraView(camera);
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
				render->setRotateViewAngleX(render->getRotateViewAngleX() + 10.f);
				break;
			case  SDL_Scancode::SDL_SCANCODE_DOWN:
				render->setRotateViewAngleX(render->getRotateViewAngleX() - 10.f);
				break;
			case  SDL_Scancode::SDL_SCANCODE_LEFT:
				if (render->getSpinOn())
					render->setRotateIncrement(render->getRotateIncrement() - 0.2f);
				else
					render->setRotateViewAngleY(render->getRotateViewAngleY() + 10.f);
				break;
			case SDL_Scancode::SDL_SCANCODE_RIGHT:
				if (render->getSpinOn())
					render->setRotateIncrement(render->getRotateIncrement() + 0.2f);
				else
					render->setRotateViewAngleY(render->getRotateViewAngleY() - 10.f);
				break;
			case  SDL_Scancode::SDL_SCANCODE_R:
				render->setSpinOn(!render->getSpinOn());
				break;
			case SDL_Scancode::SDL_SCANCODE_LEFTBRACKET:
				//q += 0.01f;
				break;
			case SDL_Scancode::SDL_SCANCODE_RIGHTBRACKET:
				//q -= 0.01f;
				break;
			case SDL_Scancode::SDL_SCANCODE_Q:
				render->deinit();
				break;
			case SDL_Scancode::SDL_SCANCODE_EQUALS:
				//zNear_ += 0.01f;
				break;
			case SDL_Scancode::SDL_SCANCODE_MINUS:
				//zNear_ -= 0.01f;
				break;
			default:
				break;
			}
		}
	}

	return 0;
}

void DSCP4Render::initFringeBuffers()
{
	const int stereogramWidth = fringeContext_.algorithm_options.num_wafels_per_scanline * static_cast<unsigned int>(sqrt(fringeContext_.algorithm_options.num_views_x));
	const int stereogramHeight = fringeContext_.algorithm_options.num_scanlines * static_cast<unsigned int>(sqrt(fringeContext_.algorithm_options.num_views_x));

	fringeContext_.fringe_gl_tex_out = new GLuint[numWindows_];
	fringeContext_.fringe_gl_buf_out = new GLuint[numWindows_];

	// create a new FBO for stereograms, this is required because if
	// we try to render stereograms to the normal frame-buffer, they
	// will be clipped by window size being smaller than N views
	glGenFramebuffers(1, &fringeContext_.stereogram_gl_fbo);
	//glGenTextures(1, &fringeContext_.stereogram_gl_fbo_color);
	//glGenRenderbuffers(1, &fringeContext_.stereogram_gl_fbo_color);
	//glGenRenderbuffers(1, &fringeContext_.stereogram_gl_fbo_depth);

	//create depth texture
	glGenTextures(1, &fringeContext_.stereogram_gl_fbo_depth);
	glBindTexture(GL_TEXTURE_2D, fringeContext_.stereogram_gl_fbo_depth);
	glTexImage2D(
		GL_TEXTURE_2D,
		0,
		GL_DEPTH_COMPONENT32F,
		stereogramWidth,
		stereogramHeight,
		0,
		GL_DEPTH_COMPONENT,
		GL_FLOAT,
		0
		);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
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
		stereogramWidth,
		stereogramHeight,
		0,
		GL_RGBA,
		GL_FLOAT,
		0
		);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);

	glBindTexture(GL_TEXTURE_2D, 0);

	glGenTextures(1, &fringeContext_.stereogram_gl_fbo_depth_r32f);
	glBindTexture(GL_TEXTURE_2D, fringeContext_.stereogram_gl_fbo_depth_r32f);
	glTexImage2D(
		GL_TEXTURE_2D,
		0,
		GL_R32F,
		stereogramWidth,
		stereogramHeight,
		0,
		GL_RED,
		GL_FLOAT,
		0
		);

	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
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
		GL_COLOR_ATTACHMENT1,
		GL_TEXTURE_2D,
		fringeContext_.stereogram_gl_fbo_depth_r32f,
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

	glFramebufferTexture2D(
		GL_FRAMEBUFFER,
		GL_COLOR_ATTACHMENT1,
		GL_TEXTURE_2D,
		fringeContext_.stereogram_gl_fbo_color,
		0
		);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);

	//glBindRenderbuffer(GL_RENDERBUFFER, fringeContext_.stereogram_gl_fbo_color);
	//glRenderbufferStorage(GL_RENDERBUFFER, GL_RGBA8, stereogramWidth, stereogramHeight);
	//glFramebufferRenderbuffer(GL_FRAMEBUFFER,
	//	GL_COLOR_ATTACHMENT0, GL_RENDERBUFFER, fringeContext_.stereogram_gl_fbo_color);

	//glBindRenderbuffer(GL_RENDERBUFFER, fringeContext_.stereogram_gl_fbo_depth);
	//glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH_COMPONENT32F,
	//	stereogramWidth,
	//	stereogramHeight);
	//glFramebufferRenderbuffer(GL_DRAW_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_RENDERBUFFER, fringeContext_.stereogram_gl_fbo_depth);


	if (fringeContext_.algorithm_options.compute_method == DSCP4_COMPUTE_METHOD_CUDA)
	{
		// begin generation of stereogram view buffers (these will go into CUDA kernels)
		size_t rgba_size = stereogramWidth * stereogramHeight * sizeof(GLbyte)* 4;
		size_t depth_size = stereogramWidth * stereogramHeight * sizeof(GLuint);

		// Create a PBO to store RGBA and DEPTH buffer of stereogram views
		// This will be passed to CUDA or OpenCL kernels for fringe computation
		glGenBuffers(1, &fringeContext_.stereogram_gl_rgba_buf_in);
		glGenBuffers(1, &fringeContext_.stereogram_gl_depth_buf_in);

		glBindBuffer(GL_ARRAY_BUFFER, fringeContext_.stereogram_gl_rgba_buf_in);
		glBufferData(GL_ARRAY_BUFFER, rgba_size, NULL, GL_DYNAMIC_DRAW);

		glBindBuffer(GL_ARRAY_BUFFER, fringeContext_.stereogram_gl_depth_buf_in);
		glBufferData(GL_ARRAY_BUFFER, depth_size, NULL, GL_DYNAMIC_DRAW);

		glBindBuffer(GL_ARRAY_BUFFER, 0);
		//end generation of stereogram view buffers
	}

	// Create N-textures for outputting fringe data to the X displays
	// Whatever holographic computation is done will be written
	// To these textures and ultimately displayed on the holovideo display
	glGenTextures(numWindows_, fringeContext_.fringe_gl_tex_out);

	char *blah = new char[fringeContext_.display_options.head_res_x * fringeContext_.display_options.head_res_y * 2 * 4];
	for (size_t i = 0; i < fringeContext_.display_options.head_res_x * fringeContext_.display_options.head_res_y * 2 * 4; i++)
	{
		blah[i] = i % 255;
	}

	if (fringeContext_.algorithm_options.compute_method == DSCP4_COMPUTE_METHOD_CUDA)
	{
		glGenBuffers(numWindows_, fringeContext_.fringe_gl_buf_out);

		for (unsigned int i = 0; i < numWindows_; i++)
		{
			glBindBuffer(GL_ARRAY_BUFFER, fringeContext_.fringe_gl_buf_out[i]);
			glBufferData(GL_ARRAY_BUFFER, fringeContext_.display_options.head_res_x * fringeContext_.display_options.head_res_y * 2 * sizeof(GLbyte)* 4, blah, GL_DYNAMIC_DRAW);
			glBindBuffer(GL_ARRAY_BUFFER, 0);
		}
	}

	for (size_t i = 0; i < numWindows_; i++)
	{
		glBindTexture(GL_TEXTURE_2D, fringeContext_.fringe_gl_tex_out[i]);

		glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA8,
			fringeContext_.display_options.head_res_x,
			fringeContext_.display_options.head_res_y * 2,
			0, GL_RGBA, GL_UNSIGNED_BYTE, blah);

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
		glBindTexture(GL_TEXTURE_2D, 0);
	}

	delete[] blah;

	//glBindBuffer(GL_PIXEL_UNPACK_BUFFER, 0);
}

void DSCP4Render::deinitFringeBuffers()
{
	if (fringeContext_.algorithm_options.compute_method == DSCP4_COMPUTE_METHOD_CUDA)
	{
		glDeleteBuffers(1, &fringeContext_.stereogram_gl_rgba_buf_in);
		glDeleteBuffers(1, &fringeContext_.stereogram_gl_depth_buf_in);
		glDeleteBuffers(numWindows_, fringeContext_.fringe_gl_buf_out);
	}

	glDeleteTextures(numWindows_, fringeContext_.fringe_gl_tex_out);
	glDeleteFramebuffers(1, &fringeContext_.stereogram_gl_fbo);
	glDeleteTextures(1, &fringeContext_.stereogram_gl_fbo_color);
	glDeleteTextures(1, &fringeContext_.stereogram_gl_fbo_depth);

	delete[] fringeContext_.fringe_gl_buf_out;
	delete[] fringeContext_.fringe_gl_tex_out;
}

void DSCP4Render::copyStereogramToPBOs()
{
	const int stereogramWidth = fringeContext_.algorithm_options.num_wafels_per_scanline * static_cast<unsigned int>(sqrt(fringeContext_.algorithm_options.num_views_x));
	const int stereogramHeight = fringeContext_.algorithm_options.num_scanlines * static_cast<unsigned int>(sqrt(fringeContext_.algorithm_options.num_views_x));

	//copy RGBA from stereogram views
	glBindBuffer(GL_PIXEL_PACK_BUFFER, fringeContext_.stereogram_gl_rgba_buf_in);
	glReadPixels(0, 0,
		stereogramWidth,
		stereogramHeight,
		GL_RGBA, GL_UNSIGNED_BYTE, 0);

	//copy DEPTH from stereogram views
	glBindBuffer(GL_PIXEL_PACK_BUFFER, fringeContext_.stereogram_gl_depth_buf_in);
	glReadPixels(0, 0,
		stereogramWidth,
		stereogramHeight,
		GL_DEPTH_COMPONENT, GL_FLOAT, 0);

	glBindBuffer(GL_PIXEL_PACK_BUFFER, 0);
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void DSCP4Render::drawFringeTextures()
{
	GLfloat Vertices[] = { 0.f, 0.f, 0.f,
		static_cast<float>(fringeContext_.display_options.head_res_x), 0, 0,
		static_cast<float>(fringeContext_.display_options.head_res_x),
		static_cast<float>(fringeContext_.display_options.head_res_y) * 2.f, 0.f,
		0.f, static_cast<float>(fringeContext_.display_options.head_res_y)*2.f, 0.f
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
			static_cast<float>(fringeContext_.display_options.head_res_x),
			0.f,
			static_cast<float>(fringeContext_.display_options.head_res_y)
			);

		glLoadMatrixf(glm::value_ptr(projectionMatrix_));

		glMatrixMode(GL_MODELVIEW);
		glLoadMatrixf(glm::value_ptr(glm::mat4()));

		glClearColor(1.f, 1.f, .5f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glDisable(GL_LIGHTING);

		glBindTexture(GL_TEXTURE_2D, fringeContext_.fringe_gl_tex_out[i]);

		if (fringeContext_.algorithm_options.compute_method == DSCP4_COMPUTE_METHOD_CUDA)
		{
			glBindBuffer(GL_PIXEL_UNPACK_BUFFER, fringeContext_.fringe_gl_buf_out[i]);
			
			#ifdef DSCP4_ENABLE_TRACE_LOG
					auto duration = measureTime<>([&](){
						glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
							fringeContext_.display_options.head_res_x,
							fringeContext_.display_options.head_res_y * 2,
							0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
					});
					LOG4CXX_TRACE(logger_, "Copying hologram fringe result " << i << " to texture took " << duration << " ms (" << 1.f / duration * 1000 << " fps)")
			#else
					glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA,
						fringeContext_.display_options.head_res_x,
						fringeContext_.display_options.head_res_y * 2,
						0, GL_RGBA, GL_UNSIGNED_BYTE, 0);
			#endif

			//glBindBuffer(GL_PIXEL_UNPACK_BUFFER, fringeContext_.stereogram_gl_rgba_buf_in);

			//glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 4*fringeContext_.algorithm_options.num_wafels_per_scanline, 4*fringeContext_.algorithm_options.num_scanlines, GL_RGBA, GL_UNSIGNED_BYTE,0);
		}

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

void DSCP4Render::initComputeMethod()
{

	switch (fringeContext_.algorithm_options.compute_method)
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
#ifdef DSCP4_HAVE_OPENCL
		LOG4CXX_DEBUG(logger_, "OpenCL -- Initializing OpenCL Context")
		SDL_GL_MakeCurrent(windows_[0], glContexts_[numWindows_ - 1]);
		computeContext_ = (dscp4_fringe_opencl_context_t*)dscp4_fringe_opencl_CreateContext(&fringeContext_, (int*)glContexts_[0]);
#else
		LOG4CXX_FATAL(logger_, "OpenCL selected as compute method, but this binary was not compiled with OpenCL")
#endif
		break;
	default:
		LOG4CXX_ERROR(logger_, "No compute method selected, no hologram will be computed")
		break;
	}

}

void DSCP4Render::deinitComputeMethod()
{
	switch (fringeContext_.algorithm_options.compute_method)
	{
	case DSCP4_COMPUTE_METHOD_CUDA:
#ifdef DSCP4_HAVE_CUDA
		LOG4CXX_DEBUG(logger_, "CUDA -- Deinitializing CUDA context")
		dscp4_fringe_cuda_DestroyContext((dscp4_fringe_cuda_context_t**)&computeContext_);
#endif
		break;
	case DSCP4_COMPUTE_METHOD_OPENCL:
#ifdef DSCP4_HAVE_OPENCL
		LOG4CXX_DEBUG(logger_, "OpenCL -- Deinitializing CUDA context")
		dscp4_fringe_opencl_DestroyContext((dscp4_fringe_opencl_context_t**)&computeContext_);
#endif
		break;
	default:
		break;
	}
}

void DSCP4Render::computeHologram()
{
	switch (fringeContext_.algorithm_options.compute_method)
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
}
