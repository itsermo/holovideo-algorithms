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
							DSCP4_DEFAULT_ALGORITHM_FOV_Y },
					display_options_t {
								DSCP4_DEFAULT_DISPLAY_NAME,
								DSCP4_DEFAULT_DISPLAY_NUM_HEADS,
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
algorithmOptions_(algorithmOptions),
displayOptions_(displayOptions),
isFullScreen_(false),
lightingShader_(nullptr),
currentWindow_(0),
projectionMatrix_(),
viewMatrix_(),
modelMatrix_(),
camera_(),
lighting_(),
cameraChanged_(false),
lightingChanged_(false),
meshChanged_(false)
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
}

DSCP4Render::~DSCP4Render()
{
	
}

bool DSCP4Render::init()
{
	LOG4CXX_INFO(logger_, "Initializing DSCP4...")

	LOG4CXX_INFO(logger_, "Initializing SDL with video subsystem")
	CHECK_SDL_RC(SDL_Init(SDL_INIT_VIDEO) < 0, "Could not initialize SDL")

	// If we can get the number of Windows from Xinerama
	// we can create a pixel buffer object for each Window
	// for displaying the final fringe pattern textures
	if (numWindows_ == 0)
		numWindows_ = SDL_GetNumVideoDisplays();

	LOG4CXX_INFO(logger_, "Number of displays: " << numWindows_)

	SDL_GL_SetAttribute(SDL_GL_ACCELERATED_VISUAL, 1);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1);

	SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
	SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

	SDL_GL_SetSwapInterval(1);

	windows_ = new SDL_Window*[numWindows_];
	glContexts_ = new SDL_GLContext[numWindows_];
	windowWidth_ = new int[numWindows_];
	windowHeight_ = new int[numWindows_];

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
	SDL_GetDisplayBounds(thisWindowNum, &bounds);

	switch (renderOptions_.render_mode)
	{
	case DSCP4_RENDER_MODE_MODEL_VIEWING:
		windowWidth_[thisWindowNum] = algorithmOptions_.num_wafels_per_scanline;
		windowHeight_[thisWindowNum] = algorithmOptions_.num_scanlines;
		LOG4CXX_DEBUG(logger_, "Creating SDL OpenGL Window " << thisWindowNum << ": " << windowWidth_[thisWindowNum] << "x" << windowHeight_[thisWindowNum] << " @ " << "{" << bounds.x + 80 << "," << bounds.y + 80 << "}")
		window = SDL_CreateWindow(("dscp4-" + std::to_string(thisWindowNum)).c_str(), bounds.x + 80, bounds.y + 80, windowWidth_[thisWindowNum], windowHeight_[thisWindowNum], SDL_WINDOW_OPENGL);
		break;
	case DSCP4_RENDER_MODE_STEREOGRAM_VIEWING:
		windowWidth_[thisWindowNum] = algorithmOptions_.num_wafels_per_scanline*2;
		windowHeight_[thisWindowNum] = algorithmOptions_.num_scanlines*2;
		LOG4CXX_DEBUG(logger_, "Creating SDL OpenGL Window " << thisWindowNum << ": " << windowWidth_[thisWindowNum] << "x" << windowHeight_[thisWindowNum] << " @ " << "{" << bounds.x + 80 << "," << bounds.y + 80 << "}")
		window = SDL_CreateWindow(("dscp4-" + std::to_string(thisWindowNum)).c_str(), bounds.x + 80, bounds.y + 80, windowWidth_[thisWindowNum], windowHeight_[thisWindowNum], SDL_WINDOW_OPENGL);
		break;
	case DSCP4_RENDER_MODE_HOLOVIDEO_FRINGE:
		windowWidth_[thisWindowNum] = bounds.w;
		windowHeight_[thisWindowNum] = bounds.h;
		LOG4CXX_DEBUG(logger_, "Creating SDL OpenGL Window " << thisWindowNum << ": " << bounds.w << "x" << bounds.h << " @ " << "{" << bounds.x << "," << bounds.y << "}")
		window = SDL_CreateWindow(("dscp4-" + std::to_string(thisWindowNum)).c_str(), bounds.x, bounds.y, bounds.w, bounds.h, SDL_WINDOW_OPENGL | SDL_WINDOW_BORDERLESS);
		SDL_ShowCursor(SDL_DISABLE);
		break;
	default:
		break;
	}
	
	CHECK_SDL_RC(window == nullptr, "Could not create SDL window");

	LOG4CXX_DEBUG(logger_, "Creating GL Context from SDL window " << thisWindowNum)
	glContext = SDL_GL_CreateContext(window);

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

	lightingShader_ = new VSShaderLib[numWindows_];

	camera_.eye = glm::vec3(0, 0, renderOptions_.render_mode == DSCP4_RENDER_MODE_MODEL_VIEWING ? 4.0f : .5f);
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
		SDL_GL_MakeCurrent(windows_[0], glContexts_[0]);
		glewInit();

		SDL_GL_SetSwapInterval(1);

		ratio = (float)windowWidth_[0] / (float)windowHeight_[0];

		// GLUT settings
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glClearColor(0.0f, 0.0f, 0.0f, 0.0f); // Black Background

		glLightfv(GL_LIGHT0, GL_AMBIENT, glm::value_ptr(lighting_.ambientColor));
		glLightfv(GL_LIGHT0, GL_DIFFUSE, glm::value_ptr(lighting_.diffuseColor));

		glLightModelfv(GL_AMBIENT_AND_DIFFUSE, glm::value_ptr(lighting_.globalAmbientColor));

		/* Setup our viewport. */
		glViewport(0, 0, windowWidth_[0], windowHeight_[0]);
		numWindows_ = 0;
		break;
	case DSCP4_RENDER_MODE_HOLOVIDEO_FRINGE:
		//for (int i = 0; i < numWindows_; i++)
		//{
		//initWindow(windows_[i], glContexts_[i], i);



		//bool isShader = initLightingShader(i);


		//}

		//bool resAreDifferent = false;
		//for (int i = 1; i < numWindows_; i++)
		//{
		//	if (windowWidth_[i] != windowWidth_[i-1] || windowHeight_[i] != windowHeight_[i-1])
		//		resAreDifferent = true;
		//}

		//if (resAreDifferent)
		//	LOG4CXX_WARN(logger_, "Multiple displays with different resolutions. You're on your own...")

		//init shaders
		break;
	default:
		break;
	}

	SDL_AddEventWatch(DSCP4Render::inputStateChanged, this);

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
				rotateAngleY_ > 360.0f ?
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
				LOG4CXX_TRACE(logger_, "Generating " << algorithmOptions_.num_views_x << " views took " << duration << " ms (" << 1.f / duration * 1000 << " fps)")
#else
				drawForStereogram();
#endif
				SDL_GL_SwapWindow(windows_[0]);
			}
				break;
			case DSCP4_RENDER_MODE_HOLOVIDEO_FRINGE:
				break;
			default:
				break;
			}

#ifdef DSCP4_ENABLE_TRACE_LOG
		});

		LOG4CXX_TRACE(logger_, "Rendering the frame took " << duration << " ms (" << 1.f/duration * 1000 << " fps)");
#endif

		poll:
			SDL_PollEvent(&event);


	}

	initLock.lock();

	std::unique_lock<std::mutex> meshLock(meshMutex_);
	for (auto it = meshes_.begin(); it != meshes_.end(); it++)
		glDeleteBuffers(3, &it->second.info.gl_vertex_buf_id);
	meshLock.unlock();

	if (lightingShader_)
	{
		delete[] lightingShader_;
		lightingShader_ = nullptr;
	}

	for (int i = 0; i < numWindows_; i++)
	{
		SDL_GL_MakeCurrent(windows_[i], glContexts_[i]);
		deinitWindow(windows_[i], glContexts_[i], i);
	}

	initLock.unlock();
	isInitCV_.notify_all();
}

void DSCP4Render::drawForViewing()
{
	const float ratio = (float)windowWidth_[0] / (float)windowHeight_[0];
	{
		std::lock_guard<std::mutex> lgc(cameraMutex_);
		glMatrixMode(GL_PROJECTION);

		projectionMatrix_ = glm::mat4();
		projectionMatrix_ *= glm::perspective(algorithmOptions_.fov_y * DEG_TO_RAD, ratio, zNear_, zFar_);


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

void DSCP4Render::drawForStereogram()
{
	const int tileX = algorithmOptions_.num_wafels_per_scanline / 2;
	const int tileY = algorithmOptions_.num_scanlines / 2;

	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

	std::lock_guard<std::mutex> lgc(cameraMutex_);
	std::lock_guard<std::mutex> lgl(lightingMutex_);
	std::lock_guard<std::mutex> lgm(meshMutex_);

	for (unsigned int i = 0; i < algorithmOptions_.num_views_x; i++)
	{
		glViewport(tileX*(i%4), tileY*(i/4), algorithmOptions_.num_wafels_per_scanline / 2, algorithmOptions_.num_scanlines / 2);
		
		glMatrixMode(GL_PROJECTION);

		const float ratio = (float)windowWidth_[0] / (float)windowHeight_[0];
		const float q = (i - algorithmOptions_.num_views_x * 0.5f) / static_cast<float>(algorithmOptions_.num_views_x) * 30.f * DEG_TO_RAD;

		projectionMatrix_ = buildOrthoXPerspYProjMat(-ratio, ratio, -1.0f, 1.0f, zNear_, zFar_, q);

		glLoadMatrixf(glm::value_ptr(projectionMatrix_));

		if (renderOptions_.shader_model != DSCP4_SHADER_MODEL_OFF)
			glEnable(GL_LIGHTING);

		glShadeModel(renderOptions_.shader_model == DSCP4_SHADER_MODEL_SMOOTH ? GL_SMOOTH : GL_FLAT);

		/* Clear the color and depth buffers. */
		//glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		/* We don't want to modify the projection matrix. */
		glMatrixMode(GL_MODELVIEW);

		viewMatrix_ = glm::mat4() * glm::lookAt(
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
	}

	cameraChanged_ = false;
	meshChanged_ = false;
	lightingChanged_ = false;
}


void DSCP4Render::deinit()
{
	LOG4CXX_INFO(logger_, "Deinitializing DSCP4...")

	LOG4CXX_DEBUG(logger_, "Waiting for render thread to stop...")

	shouldRender_ = false;

	if(renderThread_.joinable() && (std::this_thread::get_id() != renderThread_.get_id()))
		renderThread_.join();

	if (lightingShader_)
	{
		delete[] lightingShader_;
		lightingShader_ = nullptr;
	}

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
