#ifndef _ORIENTATION_H
#define _ORIENTATION_H

struct orientation
{
	float magnification;
	float scale[3];
	float rotate[3];
	float translate[3];

	orientation();
	~orientation();
	
	void activate();
	void inverseFixedScaleActivate();
};

#endif
