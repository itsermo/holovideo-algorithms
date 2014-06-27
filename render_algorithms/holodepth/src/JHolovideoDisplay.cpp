/*
 * JHolovideoDisplay.cpp
 *
 *  Created on: Jun 28, 2013
 *      Author: holo
 */

#include "JHolovideoDisplay.h"
#include <iostream>

JHolovideoDisplay::JHolovideoDisplay() {
	JHolovideoDisplay(144);
}

JHolovideoDisplay::JHolovideoDisplay(int scanlines = 144):
	activeWidthVGA(1),
	clockedWidthVGA(1),
	activeHeightVGA(1),
	clockedHeightVGA(1),
	cardCount(1),
	connectorsPerCard(1),
	colorsPerConnector(3),
	activeSweepsPerFrame(1),
	inactiveSweepsPerFrame(0),
	zigzag(false),
	viewingDistanceMeters(1.0),
	displayHeightMeters(0.1),
	displayWidthMeters(0.1),
	samplesPerScanline(1024)
		{
	scanlineCount = scanlines;
	cardForLine = new char[scanlines];
	colorChanForLine = new char[scanlines];
	connectorForLine = new char[scanlines];
	sweepForLine = new char[scanlines];
	directionForLine = new char[scanlines];
	sceneSlices = new void*[scanlines]; //??
	lineInCard = new int[scanlines];

	for (int i=0;i<scanlines;i++) {
		sceneSlices[i] = NULL;
	}

}

void JHolovideoDisplay::printInfo() {

	for (int i=0;i<scanlineCount;i++) {
		std::cout << "line [" << i << "]" << (int)cardForLine[i]<< " " << (int)colorChanForLine[i] << " "
				<< (int)connectorForLine[i] << " " << (int)sweepForLine[i] << " " << (int)directionForLine[i]<< "\n" << std::flush;
	}

}

JHolovideoDisplay::~JHolovideoDisplay() {
	// TODO Auto-generated destructor stub
}


JHolovideoDisplay JHolovideoDisplay::newMarkIIDisplay() {
	JHolovideoDisplay d(144);

	//VGA configuration
/*
	d.activeWidthVGA =2046;
	d.clockedWidthVGA =2048;
	d.activeHeightVGA =1757;
	d.clockedHeightVGA =1760;
*/


	//DisplayPort K5000 configuration
	d.activeWidthVGA =2032;
	d.clockedWidthVGA =2048;
	d.activeHeightVGA =1722;
	d.clockedHeightVGA =1760;


	d.cardCount=3;
	d.connectorsPerCard=2;
	d.activeSweepsPerFrame = 8;
	d.inactiveSweepsPerFrame = 2;
	d.zigzag = true;
	d.viewingDistanceMeters = 0.5;
	d.displayHeightMeters = 0.055;

	const float telescope = 275.0/2830.0;//Design is for: 275.0/2830.0; (about .0972)
	const float teO2Velocity = 617.0; //meters / sec
	const float pixelClock = 108134000.;//hz
	const float samplesPerScanline = 262144;

	d.displayWidthMeters = /*0.145347;*/ samplesPerScanline*telescope*teO2Velocity/pixelClock;
	d.samplesPerScanline = 262144;

	for (int i=0;i<d.scanlineCount;i++) {
		d.cardForLine[i] = i%d.cardCount;
		d.colorChanForLine[i] = (i/d.colorsPerConnector)%d.colorsPerConnector;
		d.connectorForLine[i] = (i/(d.colorsPerConnector*d.cardCount))%d.connectorsPerCard;
		d.sweepForLine[i] = (i*d.activeSweepsPerFrame/d.scanlineCount);
		d.lineInCard[i] = (d.connectorForLine[i]*d.activeSweepsPerFrame + d.sweepForLine[i])*d.colorsPerConnector + d.colorChanForLine[i];
		d.directionForLine[i] = d.zigzag?i%2:0;
	}

	return d;
}
/*
JMarkIIDisplay::JMarkIIDisplay() : JHolovideoDisplay(144)
{
		activeWidthVGA =2046;
		clockedWidthVGA =2048;
		activeHeightVGA =1757;
		clockedHeightVGA =1760;
		cardCount=3;
		connectorsPerCard=2;
		activeSweepsPerFrame = 8;
		inactiveSweepsPerFrame = 2;
		zigzag = true;
		viewingDistanceMeters = 1.0;
		displayHeightMeters = 0.2;
		displayWidthMeters = 0.1;
		samplesPerScanline = 262144;
		for (int i=0;i<scanlineCount;i++) {
			cardForLine[i] = i%cardCount;
			colorChanForLine[i] = (i/colorsPerConnector)%colorsPerConnector;
			connectorForLine[i] = (i/(colorsPerConnector*cardCount))%connectorsPerCard;
			sweepForLine[i] = (i*activeSweepsPerFrame/scanlineCount);
			directionForLine[i] = zigzag?i%2:0;
		}

}
*/
