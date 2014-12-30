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
							DSCP4_DEFAULT_ALGORITHM_NUM_SCANLINES },
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
rotateOn_(false),
zNear_(DSCP4_RENDER_DEFAULT_ZNEAR),
zFar_(DSCP4_RENDER_DEFAULT_ZFAR),
fovy_(DSCP4_RENDER_DEFAULT_FOVY),
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
lighting_()
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
	// we can create a pixel buffer for each Window
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
	case DSCP4_RENDER_MODE_STEREOGRAM_VIEWING:
		windowWidth_[thisWindowNum] = bounds.w / 2;
		windowHeight_[thisWindowNum] = bounds.h / 2;
		LOG4CXX_DEBUG(logger_, "Creating SDL OpenGL Window " << thisWindowNum << ": " << bounds.w / 2 << "x" << bounds.h / 2 << " @ " << "{" << bounds.x + 80 << "," << bounds.y + 80 << "}")
		window = SDL_CreateWindow(("dscp4-" + std::to_string(thisWindowNum)).c_str(), bounds.x + 80, bounds.y + 80, bounds.w / 2, bounds.h / 2, SDL_WINDOW_OPENGL);
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
	
	float q = 0.f; //offset for rendering stereograms
	SDL_Event event = { 0 };

	lightingShader_ = new VSShaderLib[numWindows_];

	camera_.eye = glm::vec3(0, 0, renderOptions_.render_mode == DSCP4_RENDER_MODE_MODEL_VIEWING ? 2.0f : 0.4f);
	camera_.center = glm::vec3(0, 0, 0);
	camera_.up = glm::vec3(0, 1, 0);

	lighting_.position = glm::vec4(renderOptions_.light_pos_x, renderOptions_.light_pos_y, renderOptions_.light_pos_z, 1.f);
	lighting_.ambientColor = glm::vec4(0.2f, 0.2f, 0.2f, 1.f);
	lighting_.diffuseColor = glm::vec4(1.0f, 1.0f, 1.0f, 1.f);
	lighting_.specularColor = glm::vec4(1.f, 1.f, 1.f, 1.f);
	lighting_.globalAmbientColor = glm::vec4(0.f, 0.f, 0.f, 1.f);


	for (int i = 0; i < numWindows_; i++)
	{
		initWindow(windows_[i], glContexts_[i], i);

		SDL_GL_MakeCurrent(windows_[i], glContexts_[i]);
		
		//glewInit();

		//bool isShader = initLightingShader(i);

		SDL_GL_SetSwapInterval(1);

		float ratio = (float)windowWidth_[i] / (float)windowHeight_[i];

		// GLUT settings
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glClearColor(0.0f, 0.0f, 0.0f, 0.0f); // Black Background

		glLightfv(GL_LIGHT0, GL_AMBIENT, glm::value_ptr(lighting_.ambientColor));
		glLightfv(GL_LIGHT0, GL_DIFFUSE, glm::value_ptr(lighting_.diffuseColor));

		glLightModelfv(GL_AMBIENT_AND_DIFFUSE, glm::value_ptr(lighting_.globalAmbientColor));

		/* Setup our viewport. */
		glViewport(0, 0, windowWidth_[i], windowHeight_[i]);
	}

	bool resAreDifferent = false;
	for (int i = 1; i < numWindows_; i++)
	{
		if (windowWidth_[i] != windowWidth_[i-1] || windowHeight_[i] != windowHeight_[i-1])
			resAreDifferent = true;
	}

	if (resAreDifferent)
		LOG4CXX_WARN(logger_, "Multiple displays with different resolutions. You're on your own...")

	//init shaders

	isInit_ = true;

	initLock.unlock();
	isInitCV_.notify_all();

	while (shouldRender_)
	{
		// Increments if rotateOn_ is true
		// Otherwise rotates by rotateAngle_
		rotateAngleY_ = rotateOn_ == true ?
			rotateAngleY_ > 360.0f ? 
			0.f : rotateAngleY_ + rotateIncrement_ : rotateAngleY_;

		switch (renderOptions_.render_mode)
		{
		case DSCP4_RENDER_MODE_MODEL_VIEWING:
			drawForViewing();
			break;
		case DSCP4_RENDER_MODE_STEREOGRAM_VIEWING:
		case DSCP4_RENDER_MODE_HOLOVIDEO_FRINGE:
			break;
		default:
			break;
		}

		for (int h = 0; h < numWindows_; h++)
		{
			SDL_GL_MakeCurrent(windows_[h], glContexts_[h]);
			SDL_GL_SwapWindow(windows_[h]);
		}

		while (SDL_PollEvent(&event)) {

			if (event.key.keysym.mod == SDL_Keymod::KMOD_LSHIFT)
			{
				switch (event.key.keysym.scancode)
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
					break;
				case  SDL_Scancode::SDL_SCANCODE_DOWN:
					break;
				case  SDL_Scancode::SDL_SCANCODE_LEFT:
					break;
				case SDL_Scancode::SDL_SCANCODE_RIGHT:
					break;
				default:
					break;
				}
			}

			if (event.key.type == SDL_KEYDOWN)
			{
				switch (event.key.keysym.scancode)
				{
				case  SDL_Scancode::SDL_SCANCODE_UP:
					rotateAngleX_ -= 10;
					break;
				case  SDL_Scancode::SDL_SCANCODE_DOWN:
					rotateAngleX_ += 10;
					break;
				case  SDL_Scancode::SDL_SCANCODE_LEFT:
					if (rotateOn_)
						rotateIncrement_ -= 0.1f;
					else
						rotateAngleY_ -= 10;
					break;
				case SDL_Scancode::SDL_SCANCODE_RIGHT:
					if (rotateOn_)
						rotateIncrement_ += 0.1f;
					else
						rotateAngleY_ += 10;
					break;
				case  SDL_Scancode::SDL_SCANCODE_R:
					rotateOn_ = !rotateOn_;
					break;
				case SDL_Scancode::SDL_SCANCODE_LEFTBRACKET:
					q += 0.01f;
					break;
				case SDL_Scancode::SDL_SCANCODE_RIGHTBRACKET:
					q -= 0.01f;
					break;
				case SDL_Scancode::SDL_SCANCODE_Q:
					shouldRender_ = false;
				default:
					break;
				}
			}
		}
		//std::this_thread::sleep_for(std::chrono::milliseconds(1));
	}

	delete[] lightingShader_;
	lightingShader_ = nullptr;

	for (int i = 0; i < numWindows_; i++)
	{
		SDL_GL_MakeCurrent(windows_[i], glContexts_[i]);
		deinitWindow(windows_[i], glContexts_[i], i);
	}

	isInit_ = false;
}

void DSCP4Render::drawForViewing()
{
	for (int i = 0; i < numWindows_; i++)
	{
		SDL_GL_MakeCurrent(windows_[i], glContexts_[i]);

		glMatrixMode(GL_PROJECTION);

		projectionMatrix_ = glm::mat4();
		projectionMatrix_ *= glm::perspective(fovy_ * DEG_TO_RAD, (float)windowWidth_[i] / (float)windowHeight_[i], zNear_, zFar_);

		glLoadMatrixf(glm::value_ptr(projectionMatrix_));

		if (renderOptions_.shader_model != DSCP4_SHADER_MODEL_OFF)
			glEnable(GL_LIGHTING);

		glShadeModel(renderOptions_.shader_model == DSCP4_SHADER_MODEL_SMOOTH ? GL_SMOOTH : GL_FLAT);

		/* Clear the color and depth buffers. */
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		/* We don't want to modify the projection matrix. */
		glMatrixMode(GL_MODELVIEW);

		// Move the camera (back away from the scene a bit)
		//viewMatrix_ = glm::mat4() * glm::lookAt(
		//	glm::vec3(0, 0, renderOptions_.render_mode == DSCP4_RENDER_MODE_MODEL_VIEWING ? 2.0f : 0.4f), //eye point, or the point where your camera is located
		//	glm::vec3(0, 0, 0), //center point, or the point where your camera is pointed toward
		//	glm::vec3(0, 1, 0)); //up vector, explains the orientation of the camera (which axis is up)

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

		std::unique_lock<std::mutex> meshLock(meshMutex_);
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
		meshLock.unlock();

		glDisable(GL_LIGHT0);
		glDisable(GL_DEPTH_TEST);
	}

}



void DSCP4Render::deinit()
{
	LOG4CXX_INFO(logger_, "Deinitializing DSCP4...")

	LOG4CXX_DEBUG(logger_, "Waiting for render thread to stop...")
	shouldRender_ = false;

	if(renderThread_.joinable())
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
}

void DSCP4Render::drawMesh(const mesh_t& mesh)
{
	glEnable(GL_COLOR_MATERIAL);
	glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE);
	glEnable(GL_NORMALIZE);

	if (mesh.colors && mesh.normals)
	{
		glEnableClientState(GL_VERTEX_ARRAY);
		glEnableClientState(GL_COLOR_ARRAY);
		glEnableClientState(GL_NORMAL_ARRAY);

		glNormalPointer(GL_FLOAT, mesh.info.vertex_stride, mesh.normals);
		glColorPointer(mesh.info.num_color_channels, GL_FLOAT, mesh.info.color_stride, mesh.colors);
		glVertexPointer(mesh.info.num_points_per_vertex, GL_FLOAT, mesh.info.vertex_stride, mesh.vertices);
		glDrawArrays(GL_TRIANGLES, 0, mesh.info.num_vertices);
	
		glDisableClientState(GL_NORMAL_ARRAY);
		glDisableClientState(GL_COLOR_ARRAY);
		glDisableClientState(GL_VERTEX_ARRAY);

	}
	else if (mesh.colors)
	{
		glEnableClientState(GL_VERTEX_ARRAY);
		glEnableClientState(GL_COLOR_ARRAY);

		glColorPointer(mesh.info.num_color_channels, GL_FLOAT, mesh.info.color_stride, mesh.colors);
		glVertexPointer(mesh.info.num_points_per_vertex, GL_FLOAT, mesh.info.vertex_stride, mesh.vertices);
		glDrawArrays(GL_TRIANGLES, 0, mesh.info.num_vertices);

		glDisableClientState(GL_COLOR_ARRAY);
		glDisableClientState(GL_VERTEX_ARRAY);
	}
	else if (mesh.normals)
	{
		glColor4f(0.5f, 0.5f, 0.5f, 1.0f);

		glEnableClientState(GL_VERTEX_ARRAY);
		glEnableClientState(GL_NORMAL_ARRAY);

		glNormalPointer(GL_FLOAT, mesh.info.vertex_stride, mesh.normals);
		glVertexPointer(mesh.info.num_points_per_vertex, GL_FLOAT, mesh.info.vertex_stride, mesh.vertices);
		glDrawArrays(GL_TRIANGLES, mesh.info.vertex_stride, mesh.info.num_vertices);

		glDisableClientState(GL_VERTEX_ARRAY);
		glDisableClientState(GL_NORMAL_ARRAY);
	}
	else
	{
		glColor4f(0.5f, 0.5f, 0.5f, 1.0f);

		glEnableClientState(GL_VERTEX_ARRAY);

		glVertexPointer(mesh.info.num_points_per_vertex, GL_FLOAT, mesh.info.vertex_stride, mesh.vertices);
		glDrawArrays(GL_TRIANGLES, mesh.info.vertex_stride, mesh.info.num_vertices);

		glDisableClientState(GL_VERTEX_ARRAY);
	}

	glDisable(GL_NORMALIZE);
	glDisable(GL_COLOR_MATERIAL);
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
	meshLock.unlock();
}

void DSCP4Render::removeMesh(const char *id)
{
	std::unique_lock<std::mutex> meshLock(meshMutex_);
	meshes_.erase(id);
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
