/*
 * JVRPNClient.cpp
 *
 *  Created on: Jul 22, 2013
 *      Author: holo
 */

#ifdef __CDT_PARSER__
#include "/home/holo/Dropbox/Holovideo/Eclipse_Projects/cmake/RemoteQt/JVRPNClient.h"
#else
#include "JVRPNClient.h"
#endif

#include <stdlib.h>
//no callbacks to member function so doing this nonsense
int	VRPN_CALLBACK g_msg_handler(void * userdata, vrpn_HANDLERPARAM p) {
	return ((JVRPNClient*)userdata)->msg_handler(p);
}

void	VRPN_CALLBACK g_handle_tracker_update(void *userdata,  vrpn_TRACKERCB t)
{
  ((JVRPNClient*)userdata)->head_handler(t);
}

void	VRPN_CALLBACK g_handle_tracker_update_stylus(void *userdata,  vrpn_TRACKERCB t)
{
  ((JVRPNClient*)userdata)->stylus_handler(t);
}

void	VRPN_CALLBACK g_handle_button_update(void *userdata,  vrpn_BUTTONCB b)
{
  ((JVRPNClient*)userdata)->button_handler(b);
}


JVRPNClient::JVRPNClient()
	:conn(NULL)
	,tkr(NULL)
	,button0(0)
	,button1(0)
	,button2(0)
{

}


bool JVRPNClient::connectToZspace(std::string conn_name) {

	if(conn) {
		delete conn;
		conn = NULL;
	}
	// Open the connection, with a file for incoming log required for some reason.
	// (I think it's so that there is a log that we can filter by displaying it)
	system("rm vrpn_temp.deleteme");
	conn = vrpn_get_connection_by_name((char*)(conn_name.c_str()), "vrpn_temp.deleteme");
	tkr = new vrpn_Tracker_Remote("Tracker0", conn);
	btn = new vrpn_Button_Remote("Tracker0", conn);
	tkr->register_change_handler(this, g_handle_tracker_update, 0); //ask for tracker 0 only
	tkr->register_change_handler(this, g_handle_tracker_update_stylus, 1); //ask for tracker 0 only
	btn->register_change_handler(this, g_handle_button_update);

	if (conn == NULL) {
		fprintf(stderr,"ERROR: Can't get connection %s\n",conn_name.c_str());
		return false;
	}

	// Set up the callback for all message types

	//conn->register_log_filter(g_msg_handler, (void*)this);

	return true;

}


int	VRPN_CALLBACK JVRPNClient::msg_handler(vrpn_HANDLERPARAM p)
{
	const char	*sender_name = conn->sender_name(p.sender);
	const char	*type_name = conn->message_type_name(p.type);

	// We'll need to adjust the sender and type if it was
	// unknown.
	if (sender_name == NULL) { sender_name = "UNKNOWN_SENDER"; }
	if (type_name == NULL) { type_name = "UNKNOWN_TYPE"; }
	printf("Time: %ld:%ld, Sender: %s, Type %s, Length %d\n",
		static_cast<long>(p.msg_time.tv_sec),
		static_cast<long>(p.msg_time.tv_usec),
		sender_name,
		type_name,
		p.payload_len);

	return -1;	// Do not log the message
}

void JVRPNClient::head_handler(vrpn_TRACKERCB t)
{
	headx = t.pos[0];
	heady = t.pos[1];
	headz = t.pos[2];

}

void JVRPNClient::stylus_handler(vrpn_TRACKERCB t)
{
	stylusx = t.pos[0];
	stylusy = t.pos[1];
	stylusz = t.pos[2];
	//printf("Stylus: %g\t, %g\t, %g\n", t.pos[0], t.pos[1], t.pos[2]);

}

void JVRPNClient::button_handler(vrpn_BUTTONCB b)
{
	if( b.button = 0);
	button0 = b.state;
	printf("Button: %d\t, %d\n", b.button, b.state);

}


void JVRPNClient::update() {
	if(conn) {
		conn->mainloop();
	}
}




JVRPNClient::~JVRPNClient() {
	if(conn) {
		delete conn;
		conn = NULL;
	}
	// TODO Auto-generated destructor stub
}

