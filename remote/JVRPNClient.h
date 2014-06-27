/*
 * JVRPNClient.h
 *
 *  Created on: Jul 22, 2013
 *      Author: holo
 */

#ifndef JVRPNCLIENT_H_
#define JVRPNCLIENT_H_

#include <string>
#include <vrpn_Configure.h>             // for VRPN_CALLBACK
#include <vrpn_Shared.h>                // for timeval
#include <vrpn_Connection.h>            // for vrpn_HANDLERPARAM, etc
#include <vrpn_Tracker.h>
#include <vrpn_Button.h>

class JVRPNClient {
public:
	JVRPNClient();
	bool connectToZspace(std::string server);
	void update();
	virtual ~JVRPNClient();

	int	VRPN_CALLBACK msg_handler(vrpn_HANDLERPARAM p);
	void head_handler(vrpn_TRACKERCB t);
	void stylus_handler(vrpn_TRACKERCB t);
	void button_handler(vrpn_BUTTONCB t);

	float headx;
	float heady;
	float headz;

	float stylusx;
	float stylusy;
	float stylusz;

	bool button0;
	bool button1;
	bool button2;


private:
	vrpn_Connection		*conn;	// Connection pointer
	vrpn_Tracker_Remote *tkr;
	vrpn_Button_Remote *btn;



};





#endif /* JVRPNCLIENT_H_ */
