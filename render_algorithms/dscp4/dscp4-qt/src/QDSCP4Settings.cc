#include "QDSCP4Settings.h"

QDSCP4Settings::QDSCP4Settings(QWidget *parent) : QDSCP4Settings(0, nullptr, parent)
{


}

QDSCP4Settings::QDSCP4Settings(int argc, const char **argv, QWidget *parent) : QObject(parent),
algorithmOptions_(new algorithm_options_t{}),
renderOptions_(new render_options_t{}),
argc_(argc),
argv_(argv)
{

}

void QDSCP4Settings::populateSettings()
{
	programOptions_.parseCommandLine(argc_, argv_);
	programOptions_.parseConfigFile();

	// General and input options
	this->setVerbosity((int)programOptions_.getVerbosity());
	this->setObjectFileName(QString::fromStdString(programOptions_.getFileName()));
	this->setGenerateNormals(QString::fromStdString(programOptions_.getGenerateNormals() == "smooth" ? "Smooth" : programOptions_.getGenerateNormals() == "flat" ? "Flat" : "Off"));
	bool triangulate = programOptions_.getTriangulateMesh();
	this->setTriangulateMesh(triangulate);

	this->setInstallPath(QString::fromStdString(programOptions_.getInstallPath().string()));
	this->setBinPath(QString::fromStdString(programOptions_.getBinPath().string()));
	this->setLibPath(QString::fromStdString(programOptions_.getLibPath().string()));
	this->setModelsPath(QString::fromStdString(programOptions_.getModelsPath().string()));
	this->setShadersPath(QString::fromStdString(programOptions_.getShadersPath().string()));
	this->setKernelsPath(QString::fromStdString(programOptions_.getKernelsPath().string()));

	// Render options
	this->setAutoScaleEnabled(programOptions_.getAutoscale());
	this->setShadeModel(programOptions_.getShadeModel() == "smooth" ? "Smooth" : programOptions_.getShadeModel() == "flat" ? "Flat" : "Off");
	this->setShaderFileName(QString::fromStdString(programOptions_.getShaderFileName()));
	
	auto renderMode = programOptions_.getRenderMode();
	if (renderMode == "viewing")
		this->setRenderMode(DSCP4_RENDER_MODE_MODEL_VIEWING);
	else if (renderMode == "stereogram")
		this->setRenderMode(DSCP4_RENDER_MODE_STEREOGRAM_VIEWING);
	else if (renderMode == "aerial")
		this->setRenderMode(DSCP4_RENDER_MODE_AERIAL_DISPLAY);
	else if (renderMode == "holovideo")
		this->setRenderMode(DSCP4_RENDER_MODE_HOLOVIDEO_FRINGE);

	this->setLightPosX(programOptions_.getLightPosX());
	this->setLightPosY(programOptions_.getLightPosY());
	this->setLightPosZ(programOptions_.getLightPosZ());

	// Algorithm options
	this->setNumViewsX(programOptions_.getNumViewsX());
	this->setNumViewsY(programOptions_.getNumViewsY());
	this->setNumWafelsPerScanline(programOptions_.getNumWafelsPerScanline());
	this->setNumScanlines(programOptions_.getNumScanlines());
	this->setFOVX(programOptions_.getFovX());
	this->setFOVY(programOptions_.getFovY());
	this->setZNear(programOptions_.getZNear());
	this->setZFar(programOptions_.getZFar());
	this->setComputeMethod(programOptions_.getComputeMethod() == "cuda" ? "CUDA" : "OpenCL");
	this->setComputeBlockDimX(programOptions_.getComputeMethod() == "cuda" ?
#ifdef DSCP4_HAVE_CUDA
		programOptions_.getCUDABlockDimensionX()
#else
		32 
#endif
		: programOptions_.getComputeMethod() == "opencl" ?

#ifdef DSCP4_HAVE_OPENCL
		programOptions_.getOpenCLKernelWorksizeX()
#else
		32
#endif
		: 32);

	this->setComputeBlockDimY(programOptions_.getComputeMethod() == "cuda" ?
#ifdef DSCP4_HAVE_CUDA
		programOptions_.getCUDABlockDimensionY()
#else
		32
#endif
		: programOptions_.getComputeMethod() == "opencl" ?

#ifdef DSCP4_HAVE_OPENCL
		programOptions_.getOpenCLKernelWorksizeY()
#else
		32
#endif
		: 32);

	this->setOpenCLKernelFileName(QString::fromStdString("BLAHEBEE.cl"));
	this->setRefBeamAngle_Deg((double)programOptions_.getReferenceBeamAngle());
	this->setTemporalUpconvertRed(programOptions_.getTemporalUpconvertRed());
	this->setTemporalUpconvertGreen(programOptions_.getTemporalUpconvertGreen());
	this->setTemporalUpconvertBlue(programOptions_.getTemporalUpconvertBlue());
	this->setWavelengthRed_100nm((double)(programOptions_.getWavelengthRed() * pow(10, 7)));
	this->setWavelengthGreen_100nm((double)(programOptions_.getWavelengthGreen() * pow(10, 7)));
	this->setWavelengthBlue_100nm((double)(programOptions_.getWavelengthBlue() * pow(10, 7)));

	//Display options
	this->setDisplayName(QString::fromStdString(programOptions_.getDisplayName()));
	this->setNumHeads(programOptions_.getNumHeads());
	this->setNumHeadsPerGPU(programOptions_.getNumHeadsPerGPU());
	this->setHeadResX(programOptions_.getHeadResX());
	this->setHeadResXSpec(programOptions_.getHeadResXSpec());
	this->setHeadResY(programOptions_.getHeadResY());
	this->setHeadResYSpec(programOptions_.getHeadResYSpec());
	this->setNumAOMChannels(programOptions_.getNumAOMChannels());
	this->setNumSamplesPerHololine(programOptions_.getNumSamplesPerHololine());
	this->setPixelClockRate(programOptions_.getPixelClockRate());
	this->setHologramPlaneWidth((double)programOptions_.getHologramPlaneWidth());

#ifdef DSCP4_HAVE_X11
	this->setX11EnvVar(QString::fromStdString(programOptions_.getX11DisplayEnvironmentVar()));
#endif
	
	int x = 0;
}

