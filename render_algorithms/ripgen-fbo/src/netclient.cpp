#if 0

#include <stdio.h>
#include <rpc/rpc.h>
#include <sys/time.h>
#include "net.h"

int call_render_scene(CLIENT *client, CamData *cdata);
CLIENT *create_tcp_client(char *host, unsigned long prognum, unsigned long versionnum);

timeval TimeOut = {120, 0}; /* 120 second timeout */

int main(int argc, char **argv)
{
	char *hostname;
	CLIENT *client;
	int result;

	if((argc != 3) && (argc != 4))
	{
		printf("usage:\n");
		printf("	 %s camx camy [server_host]\n", argv[0]);
		exit(1);
	}

	if(argc == 3) hostname = "localhost";
	else hostname = argv[3];

	client = create_tcp_client(hostname, MY_RPC_PROGNUM, MY_RPC_VERSIONNUM);

	CamData view;
	view.vx = atoi(argv[1]);
	view.vy = atoi(argv[2]);
	result = call_render_scene(client, &view);
	printf("rendered %d %d: %d\n", view.vx, view.vy, result);

	return 0;
}

int call_render_scene(CLIENT *client, CamData *cdata)
{
	int i;

	fprintf(stderr, "rendering (%d %d)\n", cdata->vx, cdata->vy);
	if(clnt_call(client, RENDER_FRAME, (xdrproc_t) xdr_CamData, (char *) cdata, (xdrproc_t) xdr_int, (char *) &i, TimeOut))
	{
		fprintf(stderr, "client call failed\n");
		exit(-1);
	}

	return i;
}

CLIENT *create_tcp_client(char *host, unsigned long prog, unsigned long vers)
{
	CLIENT *c;
	struct timeval tv;

	c = clnt_create(host, prog, vers, "tcp");

	if(c == NULL)
	{
		fprintf(stderr, "cannot create client, server may not be running on host \"%s\".\n", host);
		exit(-1);
	}

	clnt_control(c, CLSET_TIMEOUT, (char *) &TimeOut);
	return c;
}

#endif
