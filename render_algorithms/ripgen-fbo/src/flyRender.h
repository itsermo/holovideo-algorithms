#ifndef _FRENDER_H
#define _FRENDER_H

class Render
{
public:
	virtual void render(int x) = 0;
	virtual void mouse(int button, int state, int x, int y) = 0;
	virtual void motion(int x, int y) = 0;
	virtual void keyboard(unsigned char key, int x, int y) = 0;
};

#endif
