#include <stdlib.h>
#include <stdio.h>
#include "setupglew.h"
#include <GL/gl.h>
#include <GL/glut.h>

#include "HoloSave.h"
#include "RIP.h"

void WriteNormalLineTriplet(HoloRenderParams *holoRenParams, FILE *f, unsigned char *holoData, unsigned holoMajorLineNumber);
void WriteReverseLineTriplet(HoloRenderParams *holoRenParams, FILE *f, unsigned char *holoData, unsigned holoMajorLineNumber);
void WriteNormalScreenLine(FILE *f, unsigned char *p, unsigned screenLineLength, unsigned colorOffset);
void WriteReverseScreenLine(FILE *f, unsigned char *p, unsigned screenLineLength, unsigned colorOffset);

void HoloSaveWriteHologram(HoloRenderParams *holoRenParams, char *holoSaveFileName)
{
	static unsigned char *holoData = new unsigned char[HoloVideoParams::lineLength * HoloVideoParams::nLines / holoRenParams->m_nFramebuffers];
	int holoMajorLineNumber;

	FILE *f = fopen(holoSaveFileName, "w");
	if(f == NULL) printf("error opening output file %s\n", holoSaveFileName);

	glReadBuffer(GL_FRONT);
	glReadPixels(0, holoRenParams->m_yRes-1, holoRenParams->m_xRes, holoRenParams->m_yRes, GL_RGB, GL_UNSIGNED_BYTE, holoData);

	for(holoMajorLineNumber = 0; holoMajorLineNumber < holoRenParams->m_nHoloLinesPerFramebuffer/3; holoMajorLineNumber++)
	{
		if(holoMajorLineNumber/2 == 0) WriteNormalLineTriplet(holoRenParams, f, holoData, holoMajorLineNumber);
		else WriteReverseLineTriplet(holoRenParams, f, holoData, holoMajorLineNumber);
	}

	fclose(f);
}

void WriteNormalLineTriplet(HoloRenderParams *holoRenParams, FILE *f, unsigned char *holoData, unsigned holoMajorLineNumber)
{
	int i;
	unsigned channel;
	unsigned char *p;
	unsigned char *holoLineStart = holoData + 	
		(holoRenParams->m_nHoloLinesPerFramebuffer/3 - holoMajorLineNumber)*HoloVideoParams::lineLength*3
		- holoRenParams->m_xRes*3;

	for(channel = 0; channel < 3; channel++)
	{
		p = holoLineStart;
		for(i = 0; i < holoRenParams->m_screenLinesPerHoloLine; i++)
		{			
			WriteNormalScreenLine(f, p, holoRenParams->m_xRes, channel);
			p -= holoRenParams->m_xRes*3;
		}


	}
}

void WriteReverseLineTriplet(HoloRenderParams *holoRenParams, FILE *f, unsigned char *holoData, unsigned holoMajorLineNumber)
{
	int i;
	unsigned channel;
	unsigned char *p;
	unsigned char *holoLineStart = holoData + 
		(holoRenParams->m_nHoloLinesPerFramebuffer/3 - holoMajorLineNumber - 1)*HoloVideoParams::lineLength*3;

	for(channel = 0; channel < 3; channel++)
	{
		p = holoLineStart;
		for(i = 0; i < holoRenParams->m_screenLinesPerHoloLine; i++)
		{			
			WriteReverseScreenLine(f, p, holoRenParams->m_xRes, channel);
			p += holoRenParams->m_xRes*3;
		}
	}
}

void WriteNormalScreenLine(FILE *f, unsigned char *p, unsigned screenLineLength, unsigned colorOffset)
{
	p += colorOffset;

	for(unsigned i = 0; i < screenLineLength; i++)
	{
		fwrite((void *) p, sizeof(unsigned char), 1, f);
		p += 3;
	}
}

void WriteReverseScreenLine(FILE *f, unsigned char *p, unsigned screenLineLength, unsigned colorOffset)
{
	p += colorOffset;
	p += (screenLineLength-1)*3;


	for(unsigned i = 0; i < screenLineLength; i++)
	{
		fwrite((void *) p, sizeof(unsigned char), 1, f);
		p -= 3;
	}
}
