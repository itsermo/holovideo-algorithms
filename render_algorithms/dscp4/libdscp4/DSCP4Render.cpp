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
shouldRender_(false)
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
	numHeads_ = SDL_GetNumVideoDisplays();
	LOG4CXX_INFO(logger_, "Number of displays: " << numHeads_);

	SDL_GL_SetAttribute(SDL_GL_ACCELERATED_VISUAL, 1);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1);

	SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
	SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

	windows_ = new SDL_Window*[numHeads_];
	glContexts_ = new SDL_GLContext[numHeads_];

	//for (int h = 0; h < numHeads_; h++)
	//{
		//initHead(windows_[h], glContexts_[h], h);
	//}

	shouldRender_ = true;

	auto renderThread = renderThreads_.create_thread(boost::bind(&DSCP4Render::renderLoop, this));

	//for (int h = 0; h < numHeads_; h++)
	//{
	//	auto renderThread = renderThreads_.create_thread(boost::bind(&DSCP4Render::renderLoop, this, windows_[h], glContexts_[h], h));
	//	renderThreads_.add_thread(renderThread);
	//}

	// wait for every thread to initialize (GL context + window) before proceeding

	while (true)
	{
		std::this_thread::sleep_for(std::chrono::milliseconds(1000));
	}
		
	return true;
}

bool DSCP4Render::initHead(SDL_Window*& window, SDL_GLContext& glContext, int thisHeadNum)
{
	LOG4CXX_INFO(logger_, "Inititalizing SDL for head " << thisHeadNum);
	SDL_Rect bounds = { 0 };
	SDL_GetDisplayBounds(thisHeadNum, &bounds);

	if (thisHeadNum == 0)
	{
		windowWidth_ = bounds.w;
		windowHeight_ = bounds.h;
	}

	LOG4CXX_DEBUG(logger_, "Creating SDL OpenGL window: " << bounds.w << "x" << bounds.h << " @ " << "{" << bounds.x << "," << bounds.y << "}");

	window = SDL_CreateWindow(("dscp4-" + std::to_string(thisHeadNum)).c_str(), bounds.x, bounds.y, bounds.w, bounds.h, SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN);
	CHECK_SDL_RC(window == nullptr, "Could not create SDL window");

	LOG4CXX_DEBUG(logger_, "Creating GL context from SDL window " << thisHeadNum);
	glContext = SDL_GL_CreateContext(window);

	return true;
}

void DSCP4Render::deinitHead(SDL_Window*& window, SDL_GLContext& glContext, int thisHeadNum)
{
	LOG4CXX_INFO(logger_, "Deinitializing SDL for head " << thisHeadNum);

	if (glContext)
	{
		LOG4CXX_DEBUG(logger_, "Destroying GL context...");
		SDL_GL_DeleteContext(glContext);
		glContext = nullptr;
	}

	if (window)
	{
		LOG4CXX_DEBUG(logger_, "Destroying SDL window...");
		SDL_DestroyWindow(window);
	}

}

void DSCP4Render::renderLoop()
{
	for (int i = 0; i < numHeads_; i++)
	{
		initHead(windows_[i], glContexts_[i], i);

		SDL_GL_MakeCurrent(windows_[i], glContexts_[i]);

		float ratio = (float)windowWidth_ / (float)windowHeight_;

		/* Our shading model--Gouraud (smooth). */
		glShadeModel(GL_SMOOTH);

		/* Culling. */
		glCullFace(GL_BACK);
		glFrontFace(GL_CCW);
		glEnable(GL_CULL_FACE);

		/* Set the clear color. */
		glClearColor(0, 0, 0, 0);

		/* Setup our viewport. */
		glViewport(0, 0, windowWidth_, windowHeight_);

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
		gluPerspective(60.0, ratio, 1.0, 1024.0);
	}
	/* Our angle of rotation. */
	static float angle = 0.0f;

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



	while (shouldRender_)
	{
		for (int i = 0; i < numHeads_; i++)
		{
			SDL_GL_MakeCurrent(windows_[i], glContexts_[i]);

			//SDL_GL_MakeCurrent(thisWindow, glContext);

			/* Clear the color and depth buffers. */
			glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

			/* We don't want to modify the projection matrix. */
			glMatrixMode(GL_MODELVIEW);
			glLoadIdentity();

			/* Move down the z-axis. */
			glTranslatef(0.0, 0.0, -5.0);

			/* Rotate. */
			glRotatef(angle, 0.0, 1.0, 0.0);

			if (true) {

				if (++angle > 360.0f) {
					angle = 0.0f;
				}

			}

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

			//glLock.unlock();

			
		}

		for (int h = 0; h < numHeads_; h++)
		{
			SDL_GL_MakeCurrent(windows_[h], glContexts_[h]);
			SDL_GL_SwapWindow(windows_[h]);
		}

		std::this_thread::sleep_for(std::chrono::milliseconds(13));
	}
}

void DSCP4Render::deinit()
{
	LOG4CXX_INFO(logger_, "Deinitializing DSCP4");
}
