#ifndef _NET_H
#define _NET_H

#define MY_RPC_PROGNUM   0x29200001  /* choose this number or above */
#define MY_RPC_VERSIONNUM 1 

struct CamData
{
	int vx, vy;
};

extern bool_t xdr_CamData(XDR *xdrs, void *cdata);

#define RENDER_FRAME 1
#define KILL_SERVER 2
#define RESTART 3

#endif
