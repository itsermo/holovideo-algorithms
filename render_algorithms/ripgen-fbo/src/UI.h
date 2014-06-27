#ifndef _UI_H
#define _UI_H

class HoloRenderParams;
class RIPParams;

void InitGL(int &argc, char **&argv, HoloRenderParams *hrP, RIPHologram *ripP,
			HoloRenderParams *hrP2 = NULL, RIPHologram *ripP2 = NULL);

#endif
