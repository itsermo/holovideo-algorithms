#include "QDSCP4Settings.h"
#include <fstream>
#include <boost/property_tree/ini_parser.hpp>
#include <qfile.h>
#include <qdir.h>

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

	this->setOpenCLKernelFileName(QString::fromStdString(programOptions_.getOpenCLKernelFileName()));
	this->setRefBeamAngle_Deg((double)programOptions_.getReferenceBeamAngle());
	this->setTemporalUpconvertRed(programOptions_.getTemporalUpconvertRed());
	this->setTemporalUpconvertGreen(programOptions_.getTemporalUpconvertGreen());
	this->setTemporalUpconvertBlue(programOptions_.getTemporalUpconvertBlue());
	this->setWavelengthRed_100nm((double)(programOptions_.getWavelengthRed()) * pow(10,7));
	this->setWavelengthGreen_100nm((double)(programOptions_.getWavelengthGreen())* pow(10, 7));
	this->setWavelengthBlue_100nm((double)(programOptions_.getWavelengthBlue())* pow(10, 7));

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


	this->setX11EnvVar(QString::fromStdString(programOptions_.getX11DisplayEnvironmentVar()));

	
	int x = 0;
}

void QDSCP4Settings::saveSettings()
{
	boost::filesystem::path homePath;

#ifdef WIN32
	homePath /= getenv("HOMEDRIVE");
	homePath /= getenv("HOMEPATH");
#else
	homePath /= getenv("HOME");
#endif

	homePath /= std::string(".").append(DSCP4_PATH_PREFIX);
	
	if (!boost::filesystem::exists(homePath))
		boost::filesystem::create_directory(homePath);
	
	homePath /= DSCP4_CONF_FILENAME;

	boost::property_tree::ptree homeSettings;

	homeSettings.put("general.install_prefix", installPath_.toStdString());
	homeSettings.put("general.bin_path", binPath_.toStdString());
	homeSettings.put("general.lib_path", libPath_.toStdString());
	homeSettings.put("general.shaders_path", shadersPathStr_);
	homeSettings.put("general.models_path", modelsPath_.toStdString());
	homeSettings.put("general.kernels_path", kernelsPathStr_);
	homeSettings.put("general.verbosity", verbosity_);

	std::string renderModeStr;
	switch (renderOptions_->render_mode)
	{
	case DSCP4_RENDER_MODE_MODEL_VIEWING:
		renderModeStr = "viewing";
		break;
	case DSCP4_RENDER_MODE_STEREOGRAM_VIEWING:
		renderModeStr = "stereogram";
		break;
	case DSCP4_RENDER_MODE_AERIAL_DISPLAY:
		renderModeStr = "aerial";
		break;
	case DSCP4_RENDER_MODE_HOLOVIDEO_FRINGE:
		renderModeStr = "holovideo";
		break;
	default:
		break;
	}

	homeSettings.put("render.render_mode", renderModeStr);
	homeSettings.put("render.shader_filename", shaderFileNameStr_);
	homeSettings.put("render.light_pos_x", renderOptions_->light_pos_x);
	homeSettings.put("render.light_pos_y", renderOptions_->light_pos_y);
	homeSettings.put("render.light_pos_z", renderOptions_->light_pos_z);

	homeSettings.put("algorithm.num_views_x", algorithmOptions_->num_views_x);
	homeSettings.put("algorithm.num_views_y", algorithmOptions_->num_views_y);
	homeSettings.put("algorithm.num_wafels", algorithmOptions_->num_wafels_per_scanline);
	homeSettings.put("algorithm.num_scanlines", algorithmOptions_->num_scanlines);
	homeSettings.put("algorithm.fov_x", algorithmOptions_->fov_x);
	homeSettings.put("algorithm.fov_y", algorithmOptions_->fov_y);
	homeSettings.put("algorithm.z_near", algorithmOptions_->z_near);
	homeSettings.put("algorithm.z_far", algorithmOptions_->z_far);

	std::string computeMethodStr;
	switch (algorithmOptions_->compute_method)
	{
	case DSCP4_COMPUTE_METHOD_CUDA:
		computeMethodStr = "cuda";
		break;
	case DSCP4_COMPUTE_METHOD_OPENCL:
		computeMethodStr = "opencl";
		break;
	default:
		break;
	}

	homeSettings.put("algorithm.compute_method", computeMethodStr);
	homeSettings.put("algorithm.opencl_kernel_filename", algorithmOptions_->opencl_kernel_filename);
	homeSettings.put("algorithm.opencl_local_workgroup_size_x", algorithmOptions_->opencl_local_workgroup_size[0]);
	homeSettings.put("algorithm.opencl_local_workgroup_size_y", algorithmOptions_->opencl_local_workgroup_size[1]);
	homeSettings.put("algorithm.reference_beam_angle", algorithmOptions_->reference_beam_angle);
	homeSettings.put("algorithm.temporal_upconvert_red", algorithmOptions_->temporal_upconvert_red);
	homeSettings.put("algorithm.temporal_upconvert_green", algorithmOptions_->temporal_upconvert_green);
	homeSettings.put("algorithm.temporal_upconvert_blue", algorithmOptions_->temporal_upconvert_blue);
	homeSettings.put("algorithm.wavelength_red", algorithmOptions_->wavelength_red);
	homeSettings.put("algorithm.wavelength_green", algorithmOptions_->wavelength_green);
	homeSettings.put("algorithm.wavelength_blue", algorithmOptions_->wavelength_blue);

	homeSettings.put("display.display_name", displayOptions_.name);
	homeSettings.put("display.num_heads", displayOptions_.num_heads);
	homeSettings.put("display.num_heads_per_gpu", displayOptions_.num_heads_per_gpu);
	homeSettings.put("display.head_res_x", displayOptions_.head_res_x);
	homeSettings.put("display.head_res_y", displayOptions_.head_res_y);
	homeSettings.put("display.head_res_x_spec", displayOptions_.head_res_x_spec);
	homeSettings.put("display.head_res_y_spec", displayOptions_.head_res_y_spec);
	homeSettings.put("display.num_aom_channels", displayOptions_.num_aom_channels);
	homeSettings.put("display.num_samples_per_hololine", displayOptions_.num_samples_per_hololine);
	homeSettings.put("display.hologram_plane_width", displayOptions_.hologram_plane_width);
	homeSettings.put("display.pixel_clock_rate", displayOptions_.pixel_clock_rate);
	homeSettings.put("display.x11_display_env_arg", displayOptions_.x11_env_var);

	boost::property_tree::ini_parser::write_ini(homePath.string(), homeSettings);
}

void QDSCP4Settings::restoreDefaultSettings()
{
	boost::filesystem::path homePath;

#ifdef WIN32
	homePath /= getenv("HOMEDRIVE");
	homePath /= getenv("HOMEPATH");
#else
	homePath /= getenv("HOME");
#endif

	homePath /= std::string(".").append(DSCP4_PATH_PREFIX);
	homePath /= DSCP4_CONF_FILENAME;

	QFile::remove(QString::fromStdString(homePath.string()));

	populateSettings();
}