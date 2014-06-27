#if 0

#include <stdlib.h>
#include <stdio.h>
#include <math.h>

#include <rpc/rpc.h>
#include <rpc/pmap_clnt.h>
#include <signal.h>
#include <unistd.h>

#include "setupglew.h"
#include <GL/gl.h>

#include "net.h"

//#include "xmlnode.h"
#include "orientation.h"
#include "light.h"
#include "texture.h"
#include "texturecoord.h"
#include "material.h"
#include "drawable.h"
#include "primitives.h"
#include "triangles.h"
#include "object.h"
#include "model.h"
#include "render.h"
#include "holoren.h"
#include "parser.h"
#include "utils.h"

holoConf *holoRen = new holoConf;

void dispatch(svc_req *request, SVCXPRT *hdl);
int create_tcp_server(int prognum, int versionnum, void (*dispatch)(svc_req*, SVCXPRT *) );

int main(int argc, char **argv)
{	
	if(argc == 2) holoRen->config(argv[1]);
	else holoRen->config(NULL);

	holoRen->init();
		
	if(create_tcp_server(MY_RPC_PROGNUM, MY_RPC_VERSIONNUM, dispatch)) error("Cannot setup communications service","");

	printf("------------------------------------\n");
	printf("rendering service ready for requests\n");
	svc_run();
}

void dispatch(svc_req *request, SVCXPRT *hdl)
{
	switch(request->rq_proc)
	{

	case NULLPROC:
	{         
		/* ALL servers respond to this message by sending a void (no data)
		back to the client.  This acts as a ping to see if the server
		is working. */

		svc_sendreply(hdl, (xdrproc_t) xdr_void, NULL);
		return;
	}

	case RENDER_FRAME:
	{
		CamData client_data;
		int result;
	
		if(!svc_getargs(hdl, (xdrproc_t) xdr_CamData, (char *) &client_data))
		{
			svcerr_decode(hdl);
			return;
		}

//		holoRen->render(client_data.vx, client_data.vy);		
		result = 0;

		svc_sendreply(hdl, (xdrproc_t) xdr_int, (char *) &result);
      
		return;
	}
	
	case KILL_SERVER:
	{
		exit(0);
	}
	
	case RESTART:
	{
		printf("restart\n");		
		if(fork())
		{
			printf("parent\n");
			exit(0);
		}
		else
		{
			printf("child\n");
			exit(0);
		}
	}
		
	default:
		svcerr_noproc(hdl);
	}
}

int create_tcp_server(int prognum, int versionnum, void (*disp)(svc_req*, SVCXPRT *) )
{
	register SVCXPRT *tcp_hdl;

	signal(SIGPIPE, SIG_IGN);
	pmap_unset(prognum, versionnum); 

	if((tcp_hdl = svctcp_create(RPC_ANYSOCK, 0, 0)) == NULL) error("Service creation error", "");

	if(!svc_register(tcp_hdl, prognum, versionnum, disp, IPPROTO_TCP)) error("Service registration failed", "");

	return 0;
}

#endif
