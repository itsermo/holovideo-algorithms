#include "DSCP4Renderer.hpp"

using namespace dscp4;

DSCP4Render::DSCP4Render() : DSCP4Render(DSCP4_DEFAULT_VOXEL_SIZE, DSCP4_XINERAMA_ENABLED)
{

}

DSCP4Render::DSCP4Render(int voxelSize, bool xineramaEnabled) :
voxelSize_(voxelSize),
xineramaEnabled_(xineramaEnabled)
{

}

DSCP4Render::~DSCP4Render()
{

}