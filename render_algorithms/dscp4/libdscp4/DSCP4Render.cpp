#include "DSCP4Render.hpp"

// This checks for a true condition, prints the error message, cleans up and returns false
#define CHECK_RC(rc_condition, what) \
	if (rc_condition)								\
		{									\
			LOG4CXX_ERROR(logger_, what);	\
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
window_(nullptr)
{

}

DSCP4Render::~DSCP4Render()
{

}

bool DSCP4Render::init()
{
	LOG4CXX_INFO(logger_, "Initializing DSCP4...")

	LOG4CXX_INFO(logger_, "Initializing SDL with video subsystem");
	CHECK_RC(SDL_Init(SDL_INIT_VIDEO) < 0, "Could not initialize SDL");

	// If we can get the number of heads from Xinerama
	// we can create a pixel buffer for each head
	// for displaying the final fringe pattern textures
	numHeads_ = SDL_GetNumVideoDisplays();
	LOG4CXX_INFO(logger_, "Number of displays: " << numHeads_);

	SDL_GL_SetAttribute(SDL_GL_ACCELERATED_VISUAL, 1);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 3);
	SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1);

	SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
	SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 32);

	LOG4CXX_DEBUG(logger_, "Creating SDL OpenGL window: " << windowWidth_ << "x" << windowHeight_);
	window_ = SDL_CreateWindow("dscp4", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, windowWidth_, windowHeight_, SDL_WINDOW_OPENGL | SDL_WINDOW_SHOWN);
	CHECK_RC(window_, "Could not create SDL window");

	LOG4CXX_DEBUG(logger_, "Creating GL context from SDL window");
	glContext_ = SDL_GL_CreateContext(window_);
	
	//enable v-sync
	//SDL_GL_SetSwapInterval(1);

	return true;
}

void DSCP4Render::deinit()
{
	LOG4CXX_INFO(logger_, "Deinitializing DSCP4");
}