/*
 * JHolovideoDisplay.h
 *
 *  Created on: Jun 28, 2013
 *      Author: holo
 */
#define R_INDEX 0
#define G_INDEX 1
#define B_INDEX 2
#define A_INDEX 4

#define N_RGB 3
#define N_RGBA 4

#define LEFT_TO_RIGHT 0
#define RIGHT_TO_LEFT 1

#ifndef JHOLOVIDEODISPLAY_H_
#define JHOLOVIDEODISPLAY_H_

class JHolovideoDisplay {
public:
	JHolovideoDisplay();
	JHolovideoDisplay(int scanlines);

	static JHolovideoDisplay newMarkIIDisplay();

	void printInfo();
	virtual ~JHolovideoDisplay();

	int scanlineCount;
	int cardCount;
	int connectorsPerCard;
	int colorsPerConnector;
	int activeWidthVGA;
	int clockedWidthVGA;
	int activeHeightVGA;
	int clockedHeightVGA;
	int activeSweepsPerFrame;
	int inactiveSweepsPerFrame;

	int samplesPerScanline;

	//float lensFtFocal; //2.830
	//float lensOutputFocal;//0.275
	//int pixelsPerHologramLine; //262144
	//float TelescopeScaleFactor;// = N[focalOutputLens/focalFTxLens];

	float viewingDistanceMeters;
	float displayWidthMeters;
	float displayHeightMeters;
	bool zigzag; // is scan direction reversed on alternate lines?
	char *cardForLine; //which GPU handles line n?
	char *colorChanForLine; //which color within a VGA cable is this scanline mapped to?
	int *lineInCard; //this is the order of the scanline in the video card's line list
	char *connectorForLine; //is this scanline handled by top (0) or bottom (1) head on this GPU?
	char *sweepForLine; //which sweep handles this line (in which block of lines does it appear)
	char *directionForLine; //is this a forward or backward scan?
	void** sceneSlices;

};

/*
class JMarkIIDisplay : public JHolovideoDisplay {
	public:
	JMarkIIDisplay();
};
*/

#endif /* JHOLOVIDEODISPLAY_H_ */
