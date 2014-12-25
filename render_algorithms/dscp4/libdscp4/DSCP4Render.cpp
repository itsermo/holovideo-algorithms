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

DSCP4Render::DSCP4Render() : DSCP4Render(nullptr, DSCP4_LIGHTING_SHADER_VERTEX_FILENAME, DSCP4_LIGHTING_SHADER_FRAGMENT_FILENAME)
{
	
}

DSCP4Render::DSCP4Render(const char* shadersPath, const char* lightingShaderVertexFileName, const char* lightingShaderFragmentFileName) :
windows_(nullptr),
glContexts_(nullptr),
shouldRender_(false),
isInit_(false),
windowWidth_(nullptr),
windowHeight_(nullptr),
numWindows_(0),
lightingShaderFragmentFileName_(lightingShaderFragmentFileName),
lightingShaderVertexFileName_(lightingShaderVertexFileName),
rotateAngleX_(0),
rotateAngleY_(0),
rotateIncrement_(1.0f),
rotateOn_(false),
shadeModel_(DSCP4_LIGHTING_SHADE_MODEL),
autoScaleEnabled_(DSCP4_AUTO_SCALE_ENABLED),
renderMode_(DSCP4_RENDER_MODE_DEFAULT)
{
	if (shadersPath == nullptr)
	{
		LOG4CXX_WARN(logger_, "No shader path location specified, using current working path: " << boost::filesystem::current_path().string())
		shadersPath_ = boost::filesystem::current_path();
	}
	else
		shadersPath_ = boost::filesystem::path(shadersPath);

#ifdef DSCP4_HAVE_LOG4CXX
	
	log4cxx::BasicConfigurator::resetConfiguration();

#ifdef WIN32
	log4cxx::PatternLayoutPtr logLayoutPtr = new log4cxx::PatternLayout(L"%-5p %m%n");
#else
	log4cxx::PatternLayoutPtr logLayoutPtr = new log4cxx::PatternLayout("%-5p %m%n");
#endif


	log4cxx::ConsoleAppenderPtr logAppenderPtr = new log4cxx::ConsoleAppender(logLayoutPtr);
	log4cxx::BasicConfigurator::configure(logAppenderPtr);

#endif

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

	switch (renderMode_)
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
	lightingShader_[which].loadShader(VSShaderLib::VERTEX_SHADER, (shadersPath_ / lightingShaderVertexFileName_).string());
	lightingShader_[which].loadShader(VSShaderLib::FRAGMENT_SHADER, (shadersPath_ / lightingShaderFragmentFileName_).string());

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
}

void DSCP4Render::renderLoop()
{
	std::unique_lock<std::mutex> initLock(isInitMutex_);
	
	lightingShader_ = new VSShaderLib[numWindows_];

	GLfloat lightPosition[] = { -0.7f, 0.7f, 0.5f, 0.0 };
	GLfloat lightAmbientColor[] = { 0.2f, 0.2f, 0.2f, 1 };
	GLfloat lightDiffuseColor[] = { 1.0f, 1.0f, 1.0f, 1 };
	GLfloat lightSpecularColor[] = { 1, 1, 1, 1 };
	GLfloat lightGlobalAmbient[] = { 0.0f, 0.0f, 0.0f, 1 };

	for (int i = 0; i < numWindows_; i++)
	{
		initWindow(windows_[i], glContexts_[i], i);

		SDL_GL_MakeCurrent(windows_[i], glContexts_[i]);
		
		//glewInit();

		//bool isShader = initLightingShader(i);

		SDL_GL_SetSwapInterval(1);

		float ratio = (float)windowWidth_[i] / (float)windowHeight_[i];


		/* Culling. */
		//glCullFace(GL_BACK);
		//glFrontFace(GL_CCW);
		//glEnable(GL_CULL_FACE);

		/* Set the clear color. */
		//glClearColor(0, 0, 0, 0);

		// GLUT settings
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		glClearColor(0.0f, 0.0f, 0.0f, 0.0f); // Black Background

		//glShadeModel(GL_FLAT); // Enable Smooth Shading
		glLightfv(GL_LIGHT0, GL_AMBIENT, lightAmbientColor);
		glLightfv(GL_LIGHT0, GL_DIFFUSE, lightDiffuseColor);
		//glLightfv(GL_LIGHT0, GL_SPECULAR, lightSpecularColor);

		glLightModelfv(GL_AMBIENT_AND_DIFFUSE, lightGlobalAmbient);

		//glMaterialfv(GL_FRONT, GL_SPECULAR, materialSpecular);
		//glMaterialfv(GL_FRONT, GL_SHININESS, materialShininess);
		//glLightfv(GL_LIGHT0, GL_POSITION, lightPosition);

		
		//glEnable(GL_LIGHT0);
		//Takes care of occlusions for point cloud
		//glEnable(GL_DEPTH_TEST);


		/* Setup our viewport. */
		glViewport(0, 0, windowWidth_[i], windowHeight_[i]);

		/*
		* Change to the projection matrix and set
		* our viewing volume.
		*/
		glMatrixMode(GL_PROJECTION);
		glLoadIdentity();
		/*
		* EXERCISE:
		* Replace this with a call to glFrustum.
		*/
		gluPerspective(60.0f, ratio, 0.1f, 10.0f);
		
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
		SDL_Event event;
		while (SDL_PollEvent(&event)) {
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
				case  SDL_Scancode::SDL_SCANCODE_W:
					lightPosition[1] += 0.1f;
					break;
				case SDL_Scancode::SDL_SCANCODE_S:
					lightPosition[1] -= 0.1f;
					break;
				case  SDL_Scancode::SDL_SCANCODE_A:
					lightPosition[0] -= 0.1f;
					break;
				case SDL_Scancode::SDL_SCANCODE_D:
					lightPosition[0] += 0.1f;
					break;
				case  SDL_Scancode::SDL_SCANCODE_Z:
					lightPosition[2] -= 0.1f;
					break;
				case SDL_Scancode::SDL_SCANCODE_X:
					lightPosition[2] += 0.1f;
					break;
				case  SDL_Scancode::SDL_SCANCODE_R:
					rotateOn_ = !rotateOn_;
					break;
				default:
					break;
				}
			}
			// handle your event here
		}

		if (rotateOn_) {
			rotateAngleY_ += rotateIncrement_;
			if (rotateAngleY_ > 360.0f) {
				rotateAngleY_ = 0.0f;
			}

		}
		
		for (int i = 0; i < numWindows_; i++)
		{
			SDL_GL_MakeCurrent(windows_[i], glContexts_[i]);

			if (shadeModel_ != DSCP4_SHADE_MODEL_OFF)
				glEnable(GL_LIGHTING);

			glShadeModel(shadeModel_ == DSCP4_SHADE_MODEL_SMOOTH ? GL_SMOOTH : GL_FLAT);

			/* Clear the color and depth buffers. */
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

			glPushMatrix();
			glMatrixMode(GL_PROJECTION);
			glLightfv(GL_LIGHT0, GL_POSITION, lightPosition);
			glPopMatrix();

			/* We don't want to modify the projection matrix. */
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();

			/* Move down the z-axis. */
			glTranslatef(0.0, 0.0, -2.0f);

			/* Rotate. */
			glRotatef(rotateAngleX_, 1.0, 0.0, 0.0);
			glRotatef(rotateAngleY_, 0.0, 1.0, 0.0);

			glEnable(GL_DEPTH_TEST);
			glEnable(GL_LIGHT0);

			std::unique_lock<std::mutex> meshLock(meshMutex_);
			for (auto it = meshes_.begin(); it != meshes_.end(); it++)
			{
				drawMesh(it->second);
			}
			meshLock.unlock();

			glDisable(GL_LIGHT0);
			glDisable(GL_DEPTH_TEST);

		}

		for (int h = 0; h < numWindows_; h++)
		{
			SDL_GL_MakeCurrent(windows_[h], glContexts_[h]);
			SDL_GL_SwapWindow(windows_[h]);
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

}

void DSCP4Render::deinit()
{
	LOG4CXX_INFO(logger_, "Deinitializing DSCP4...")

	LOG4CXX_DEBUG(logger_, "Waiting for render thread to stop...")
	shouldRender_ = false;
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
	const float radius = sqrt(mesh.info.sq_radius);
	const float factor = 1.0f/radius;
	
	glPushMatrix();
	glScalef(factor, factor, factor);
	glTranslatef(-mesh.info.center_x, -mesh.info.center_y, -mesh.info.center_z);

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

	glPopMatrix();
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

	if (autoScaleEnabled_)
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
		mesh.info.center_x = miniball3f.center()[0];
		mesh.info.center_y = miniball3f.center()[1];
		mesh.info.center_z = miniball3f.center()[2];
		mesh.info.sq_radius = miniball3f.squared_radius();

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

void DSCP4Render::addPointCloud(const char *id, float *points, int numPoints, bool hasColorData)
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