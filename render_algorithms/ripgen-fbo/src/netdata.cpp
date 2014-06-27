//#include "version.h"

#include <stdio.h>
#include <rpc/rpc.h>
#include "net.h"

bool_t xdr_CamData(XDR *xdrs, void *cdata)
{
	if(
		xdr_int(xdrs, &((CamData *) cdata)->vx) &&
		xdr_int(xdrs, &((CamData *) cdata)->vy)
	) return 1;

	return 0;
}
