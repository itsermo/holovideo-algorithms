#include "DSCP4Render.hpp"

// This checks for a true condition, prints the error message, cleans up and returns false
#define CHECK_SDL_RC(rc_condition, what) \
	if (rc_condition)								\
		{									\
			LOG4CXX_ERROR(logger_, what);	\
			LOG4CXX_ERROR(logger_, SDL_GetError()) \
			deinit();						\
			return false;					\
		}									\

using namespace dscp4;

DSCP4Render::DSCP4Render() : DSCP4Render(DSCP4_DEFAULT_VOXEL_SIZE, DSCP4_XINERAMA_ENABLED)
{

}

DSCP4Render::DSCP4Render(float voxelSize, bool xineramaEnabled) :
voxelSize_(voxelSize),
xineramaEnabled_(xineramaEnabled),
windows_(nullptr),
glContexts_(nullptr),
shouldRender_(false),
isInit_(false),
windowWidth_(nullptr),
windowHeight_(nullptr),
numHeads_(0),
numVertices_(0)
{

}

DSCP4Render::~DSCP4Render()
{

}

bool DSCP4Render::init()
{
	LOG4CXX_INFO(logger_, "Initializing DSCP4...")

	LOG4CXX_INFO(logger_, "Initializing SDL with video subsystem");
	CHECK_SDL_RC(SDL_Init(SDL_INIT_VIDEO) < 0, "Could not initialize SDL");

	// If we can get the number of heads from Xinerama
	// we can create a pixel buffer for each head
	// for displaying the final fringe pattern textures
	if (numHeads_ == 0)
		numHeads_ = SDL_GetNumVideoDisplays();

	LOG4CXX_INFO(logger_, "Number of displays: " << numHeads_);

	SDL_GL_SetAttribute(SDL_GL_ACCELERATED_VISUAL, 1);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1);

	SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
	SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

	SDL_GL_SetSwapInterval(1);

	windows_ = new SDL_Window*[numHeads_];
	glContexts_ = new SDL_GLContext[numHeads_];
	windowWidth_ = new int[numHeads_];
	windowHeight_ = new int[numHeads_];

	std::unique_lock<std::mutex> initLock(isInitMutex_);
	shouldRender_ = true;
	renderThread_ = std::thread(std::bind(&DSCP4Render::renderLoop, this));

	isInitCV_.wait(initLock);

	initLock.unlock();
		
	return true;
}

bool DSCP4Render::initHead(SDL_Window*& window, SDL_GLContext& glContext, int thisHeadNum)
{
	LOG4CXX_DEBUG(logger_, "Inititalizing SDL for head " << thisHeadNum);
	SDL_Rect bounds = { 0 };
	SDL_GetDisplayBounds(thisHeadNum, &bounds);


	windowWidth_[thisHeadNum] = bounds.w;
	windowHeight_[thisHeadNum] = bounds.h;

	LOG4CXX_DEBUG(logger_, "Creating SDL OpenGL window " << thisHeadNum << ": " << bounds.w << "x" << bounds.h << " @ " << "{" << bounds.x << "," << bounds.y << "}");

	window = SDL_CreateWindow(("dscp4-" + std::to_string(thisHeadNum)).c_str(), bounds.x, bounds.y, bounds.w, bounds.h, SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN);
	CHECK_SDL_RC(window == nullptr, "Could not create SDL window");

	LOG4CXX_DEBUG(logger_, "Creating GL context from SDL window " << thisHeadNum);
	glContext = SDL_GL_CreateContext(window);

	return true;
}

void DSCP4Render::deinitHead(SDL_Window*& window, SDL_GLContext& glContext, int thisHeadNum)
{
	LOG4CXX_DEBUG(logger_, "Deinitializing SDL for window " << thisHeadNum);

	if (glContext)
	{
		LOG4CXX_DEBUG(logger_, "Destroying GL context " << thisHeadNum << "...");
		SDL_GL_DeleteContext(glContext);
		glContext = nullptr;
	}

	if (window)
	{
		LOG4CXX_DEBUG(logger_, "Destroying SDL window " << thisHeadNum << "...");
		SDL_DestroyWindow(window);
		window = nullptr;
	}

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
	
	for (int i = 0; i < numHeads_; i++)
	{
		initHead(windows_[i], glContexts_[i], i);

		SDL_GL_MakeCurrent(windows_[i], glContexts_[i]);
		
		SDL_GL_SetSwapInterval(1);

		float ratio = (float)windowWidth_[i] / (float)windowHeight_[i];

		/* Our shading model--Gouraud (smooth). */
		glShadeModel(GL_SMOOTH);

		/* Culling. */
		glCullFace(GL_BACK);
		glFrontFace(GL_CCW);
		glEnable(GL_CULL_FACE);

		/* Set the clear color. */
		glClearColor(0, 0, 0, 0);

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
		gluPerspective(60.0, ratio, 0.001f, 1024.0);
	}

	bool resAreDifferent = false;
	for (int i = 1; i < numHeads_; i++)
	{
		if (windowWidth_[i] != windowWidth_[i-1] || windowHeight_[i] != windowHeight_[i-1])
			resAreDifferent = true;
	}

	if (resAreDifferent)
		LOG4CXX_WARN(logger_, "Multiple displays with different resolutions. You're on your own...");

	isInit_ = true;

	initLock.unlock();
	isInitCV_.notify_all();

	/* Our angle of rotation. */
	static float angle = 0.0f;

	while (shouldRender_)
	{
		SDL_Event event;
		while (SDL_PollEvent(&event)) {
			// handle your event here
		}

		for (int i = 0; i < numHeads_; i++)
		{
			SDL_GL_MakeCurrent(windows_[i], glContexts_[i]);

			/* Clear the color and depth buffers. */
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

			/* We don't want to modify the projection matrix. */
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();

			/* Move down the z-axis. */
			glTranslatef(0.0, -0.1, -0.5);

			/* Rotate. */
			glRotatef(angle, 0.0, 1.0, 0.0);

			if (true) {

				if (++angle > 360.0f) {
					angle = 0.0f;
				}

			}

			//drawCube();
			drawMesh();
		}

		for (int h = 0; h < numHeads_; h++)
		{
			SDL_GL_MakeCurrent(windows_[h], glContexts_[h]);
			SDL_GL_SwapWindow(windows_[h]);
		}

		//std::this_thread::sleep_for(std::chrono::milliseconds(13));
	}
}

void DSCP4Render::deinit()
{
	LOG4CXX_INFO(logger_, "Deinitializing DSCP4...");

	LOG4CXX_DEBUG(logger_, "Waiting for render thread to stop...");
	shouldRender_ = false;
	renderThread_.join();

	for (int h = 0; h < numHeads_; h++)
	{
		deinitHead(windows_[h], glContexts_[h], h);
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

	LOG4CXX_DEBUG(logger_, "Destroying SDL context");
	SDL_Quit();
}

void DSCP4Render::drawMesh()
{
	glColor4f(255, 0, 0, 255);

	glEnableClientState(GL_VERTEX_ARRAY);
	glVertexPointer(3, GL_FLOAT, 0, vertices_);

	glDrawArrays(GL_TRIANGLES, 0, numVertices_);

	glDisableClientState(GL_VERTEX_ARRAY);

	//for (int v = 0; v < numVertices_; v++)
	//{
	//	glVertex3fv(&vertices_[3 * v]);
	//}
}

void DSCP4Render::addMesh(float* vertices, int numVertices)
{
	vertices_ = vertices;
	numVertices_ = numVertices;
}
