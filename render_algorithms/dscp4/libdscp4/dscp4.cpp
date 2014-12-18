#include "DSCP4Render.hpp"

extern "C"
{

	bool CreateRenderer()
	{

		dscp4::DSCP4Render render;

		render.init();

		return true;
	}
}