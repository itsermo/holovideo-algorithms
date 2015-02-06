#ifndef DSCP4_SETTINGS_H
#define DSCP4_SETTINGS_H
#include <qstring.h>
#include <qobject.h>
#include <DSCP4ProgOptions.hpp>
#include "../libdscp4/dscp4_defs.h"
#include "mainwindow.h"

namespace Ui
{
	class QDSCP4Settings;
}

class QDSCP4Settings : public QObject {
	Q_OBJECT

public:
	QDSCP4Settings(QWidget *parent = 0);
	QDSCP4Settings(int argc, const char **argv, QWidget *parent = 0);

	void populateSettings();

	algorithm_options_t* getAlgorithmOptions() { return algorithmOptions_; }
	display_options_t getDisplayOptions() { return displayOptions_; }
	render_options_t* getRenderOptions() { return renderOptions_; }

public slots:
	void setInstallPath(QString installPath) { 
		setValue<QString>(installPath, installPath_, std::bind(&QDSCP4Settings::installPathChanged, this, std::placeholders::_1)); 	}
	void setBinPath(QString binPath) { setValue<QString>(binPath, binPath_, std::bind(&QDSCP4Settings::binPathChanged, this, std::placeholders::_1)); }
	void setLibPath(QString libPath) { setValue<QString>(libPath, libPath_, std::bind(&QDSCP4Settings::libPathChanged, this, std::placeholders::_1)); }
	void setModelsPath(QString modelsPath) { setValue<QString>(modelsPath, modelsPath_, std::bind(&QDSCP4Settings::modelsPathChanged, this, std::placeholders::_1)); }
	void setShadersPath(QString shadersPath) {
		setValue<QString>(shadersPath, shadersPath_, std::bind(&QDSCP4Settings::shadersPathChanged, this, std::placeholders::_1));
			shadersPathStr_ = shadersPath_.toStdString();
			renderOptions_->shaders_path = shadersPathStr_.c_str();
		}

	void setKernelsPath(QString kernelsPath) {
			setValue<QString>(kernelsPath, kernelsPath_, std::bind(&QDSCP4Settings::kernelsPathChanged, this, std::placeholders::_1)); 
			kernelsPathStr_ = kernelsPath_.toStdString();
			renderOptions_->kernels_path = kernelsPathStr_.c_str();
		}

	void setVerbosity(int verbosity) { setValue<int>(verbosity, verbosity_, std::bind(&QDSCP4Settings::verbosityChanged, this, std::placeholders::_1)); }

	// Input options
	void setObjectFileName(QString objectFileName) { setValue<QString>(objectFileName, objectFileName_, std::bind(&QDSCP4Settings::objectFileNameChanged, this, std::placeholders::_1)); }
	void setGenerateNormals(QString generateNormals) { setValue<QString>(generateNormals, generateNormals_, std::bind(&QDSCP4Settings::generateNormalsChanged, this, std::placeholders::_1)); }
	void setTriangulateMesh(bool triangulateMesh) { setValue<bool>(triangulateMesh, triangulateMesh_, std::bind(&QDSCP4Settings::triangulateMeshChanged, this, std::placeholders::_1)); }

	// Render options
	void setAutoScaleEnabled(bool autoScaleEnabled) { setValue<bool>(autoScaleEnabled, renderOptions_->auto_scale_enabled, std::bind(&QDSCP4Settings::autoScaleEnabledChanged, this, std::placeholders::_1)); }
	void setShadeModel(QString shadeModel) { 
		setValue<QString>(shadeModel, shadeModel_, std::bind(&QDSCP4Settings::shadeModelChanged, this, std::placeholders::_1)); 
		renderOptions_->shader_model = shadeModel_ == "Flat" ? DSCP4_SHADER_MODEL_FLAT : shadeModel_ == "Smooth" ? DSCP4_SHADER_MODEL_SMOOTH : DSCP4_SHADER_MODEL_OFF;
	}
	void setShaderFileName(QString shaderFileName) {
		setValue<QString>(shaderFileName, shaderFileName_, std::bind(&QDSCP4Settings::shaderFileNameChanged, this, std::placeholders::_1)); 
		shaderFileNameStr_ = shaderFileName_.toStdString();
		renderOptions_->shader_filename_prefix = shaderFileNameStr_.c_str();
	}
	void setRenderMode(int renderMode) { setValue<render_mode_t>((render_mode_t)renderMode, renderOptions_->render_mode, std::bind(&QDSCP4Settings::renderModeChanged, this, std::placeholders::_1)); }
	void setLightPosX(double lightPosX) { setValue<float>(lightPosX, renderOptions_->light_pos_x, std::bind(&QDSCP4Settings::lightPosXChanged, this, std::placeholders::_1)); }
	void setLightPosY(double lightPosY) { setValue<float>(lightPosY, renderOptions_->light_pos_y, std::bind(&QDSCP4Settings::lightPosYChanged, this, std::placeholders::_1)); }
	void setLightPosZ(double lightPosZ) { setValue<float>(lightPosZ, renderOptions_->light_pos_z, std::bind(&QDSCP4Settings::lightPosZChanged, this, std::placeholders::_1)); }

	// Algorithm options
	void setNumViewsX(int numViewsX) { setValue<unsigned int>(numViewsX, algorithmOptions_->num_views_x, std::bind(&QDSCP4Settings::numViewsXChanged, this, std::placeholders::_1)); }
	void setNumViewsY(int numViewsY) { setValue<unsigned int>(numViewsY, algorithmOptions_->num_views_y, std::bind(&QDSCP4Settings::numViewsYChanged, this, std::placeholders::_1)); }
	void setNumWafelsPerScanline(int numWafelsPerScanline) { setValue<unsigned int>(numWafelsPerScanline, algorithmOptions_->num_wafels_per_scanline, std::bind(&QDSCP4Settings::numWafelsPerScanlineChanged, this, std::placeholders::_1)); }
	void setFOVX(double fovX){ setValue<float>(fovX, algorithmOptions_->fov_x, std::bind(&QDSCP4Settings::fovXChanged, this, std::placeholders::_1)); }
	void setFOVY(double fovY) { setValue<float>(fovY, algorithmOptions_->fov_y, std::bind(&QDSCP4Settings::fovYChanged, this, std::placeholders::_1)); }
	void setZNear(double zNear){ setValue<float>(zNear, algorithmOptions_->z_near, std::bind(&QDSCP4Settings::zNearChanged, this, std::placeholders::_1)); }
	void setZFar(double zFar) { setValue<float>(zFar, algorithmOptions_->z_far, std::bind(&QDSCP4Settings::zFarChanged, this, std::placeholders::_1)); }

	void setComputeMethod(QString computeMethod) { 
		setValue<QString>(computeMethod, computeMethod_, std::bind(&QDSCP4Settings::computeMethodChanged, this, std::placeholders::_1));
		algorithmOptions_->compute_method = computeMethod_ == "OpenCL" ? DSCP4_COMPUTE_METHOD_OPENCL : DSCP4_COMPUTE_METHOD_CUDA;
		setComputeBlockDimX(computeMethod_ == "OpenCL" ? (int)programOptions_.getOpenCLKernelWorksizeX() : (int)programOptions_.getCUDABlockDimensionX());
		setComputeBlockDimY(computeMethod_ == "OpenCL" ? (int)programOptions_.getOpenCLKernelWorksizeY() : (int)programOptions_.getCUDABlockDimensionY());
	}
	void setComputeBlockDimX(int computeBlockDimX) {
		setValue<unsigned int>(computeBlockDimX, computeBlockDimX_, std::bind(&QDSCP4Settings::computeBlockDimXChanged, this, std::placeholders::_1));
		switch (algorithmOptions_->compute_method)
		{
		case DSCP4_COMPUTE_METHOD_OPENCL:
			algorithmOptions_->opencl_local_workgroup_size[0] = computeBlockDimX_;
			break;

		case DSCP4_COMPUTE_METHOD_CUDA:
			algorithmOptions_->cuda_block_dimensions[0] = computeBlockDimX_;
			break;
		default:
			break;
		}
	}
	void setComputeBlockDimY(int computeBlockDimY) { 
		setValue<unsigned int>(computeBlockDimY, computeBlockDimY_, std::bind(&QDSCP4Settings::computeBlockDimYChanged, this, std::placeholders::_1));
		switch (algorithmOptions_->compute_method)
		{
		case DSCP4_COMPUTE_METHOD_OPENCL:
			algorithmOptions_->opencl_local_workgroup_size[1] = computeBlockDimY_;
			break;

		case DSCP4_COMPUTE_METHOD_CUDA:
			algorithmOptions_->cuda_block_dimensions[1] = computeBlockDimY_;
			break;
		default:
			break;
		}
	}
	void setOpenCLKernelFileName(QString openCLKernelFileName) { 
		setValue<QString>(openCLKernelFileName, openCLKernelFileName_, std::bind(&QDSCP4Settings::openCLKernelFileNameChanged, this, std::placeholders::_1));
		openCLKernelFileNameStr_ = openCLKernelFileName_.toStdString();
		algorithmOptions_->opencl_kernel_filename = openCLKernelFileNameStr_.c_str();
	}

	void setRefBeamAngle_Deg(double refBeamAngle_Deg) { setValue<float>(refBeamAngle_Deg, algorithmOptions_->reference_beam_angle, std::bind(&QDSCP4Settings::refBeamAngle_DegChanged, this, std::placeholders::_1)); }
	void setTemporalUpconvertRed(int temporalUpconvertRed)  { setValue<unsigned int>(temporalUpconvertRed, algorithmOptions_->temporal_upconvert_red, std::bind(&QDSCP4Settings::temporalUpconvertRedChanged, this, std::placeholders::_1)); }
	void setTemporalUpconvertGreen(int temporalUpconvertGreen) { setValue<unsigned int>(temporalUpconvertGreen, algorithmOptions_->temporal_upconvert_green, std::bind(&QDSCP4Settings::temporalUpconvertGreenChanged, this, std::placeholders::_1)); }
	void setTemporalUpconvertBlue(int temporalUpconvertBlue) { setValue<unsigned int>(temporalUpconvertBlue, algorithmOptions_->temporal_upconvert_blue, std::bind(&QDSCP4Settings::temporalUpconvertBlueChanged, this, std::placeholders::_1)); }
	void setWavelengthRed_100nm(double wavelengthRed_100nm)  { 
		setValue<double>(wavelengthRed_100nm, wavelengthRed_100nm_, std::bind(&QDSCP4Settings::wavelengthRed_100nmChanged, this, std::placeholders::_1));
		algorithmOptions_->wavelength_red = (float)(wavelengthRed_100nm_ / pow(10, 7));
	}
	void setWavelengthGreen_100nm(double wavelengthGreen_100nm)  { 
		setValue<double>(wavelengthGreen_100nm, wavelengthGreen_100nm_, std::bind(&QDSCP4Settings::wavelengthGreen_100nmChanged, this, std::placeholders::_1));
		algorithmOptions_->wavelength_green = (float)(wavelengthGreen_100nm_ / pow(10, 7));

	}
	void setWavelengthBlue_100nm(double wavelengthBlue_100nm) { 
		setValue<double>(wavelengthBlue_100nm, wavelengthBlue_100nm_, std::bind(&QDSCP4Settings::wavelengthBlue_100nmChanged, this, std::placeholders::_1));
		algorithmOptions_->wavelength_blue = (float)(wavelengthBlue_100nm_ / pow(10, 7));
	}

	// Display options
	void setDisplayName(QString displayName) { 
		setValue<QString>(displayName, displayName_, std::bind(&QDSCP4Settings::displayNameChanged, this, std::placeholders::_1)); 
		displayNameStr_ = displayName_.toStdString();
		displayOptions_.name = displayNameStr_.c_str();
	}

	void setX11EnvVar(QString x11EnvVar) {
		setValue<QString>(x11EnvVar, x11EnvVar_, std::bind(&QDSCP4Settings::x11EnvVarChanged, this, std::placeholders::_1));
		x11EnvVarStr_ = x11EnvVar_.toStdString();
		displayOptions_.x11_env_var = x11EnvVarStr_.c_str();
	}

	void setNumHeads(int numHeads) { setValue<unsigned int>(numHeads, displayOptions_.num_heads, std::bind(&QDSCP4Settings::numHeadsChanged, this, std::placeholders::_1)); }
	void setNumHeadsPerGPU(int numHeadsPerGPU) { setValue<unsigned int>(numHeadsPerGPU, displayOptions_.num_heads_per_gpu, std::bind(&QDSCP4Settings::numHeadsPerGPUChanged, this, std::placeholders::_1)); }
	void setHeadResX(int headResX) { setValue<unsigned int>(headResX, displayOptions_.head_res_x, std::bind(&QDSCP4Settings::headResXChanged, this, std::placeholders::_1)); }
	void setHeadResY(int headResY) { setValue<unsigned int>(headResY, displayOptions_.head_res_y, std::bind(&QDSCP4Settings::headResYChanged, this, std::placeholders::_1)); }
	void setHeadResXSpec(int headResXSpec) { setValue<unsigned int>(headResXSpec, displayOptions_.head_res_x_spec, std::bind(&QDSCP4Settings::headResXSpecChanged, this, std::placeholders::_1)); }
	void setHeadResYSpec(int headResYSpec) { setValue<unsigned int>(headResYSpec, displayOptions_.head_res_y_spec, std::bind(&QDSCP4Settings::headResYSpecChanged, this, std::placeholders::_1)); }
	void setNumAOMChannels(int numAOMChannels) { setValue<unsigned int>(numAOMChannels, displayOptions_.num_aom_channels, std::bind(&QDSCP4Settings::numAOMChannelsChanged, this, std::placeholders::_1)); }
	void setNumSamplesPerHololine(int numSamplesPerHololine) { setValue<unsigned int>(numSamplesPerHololine, displayOptions_.num_samples_per_hololine, std::bind(&QDSCP4Settings::numSamplesPerHololineChanged, this, std::placeholders::_1)); }
	void setPixelClockRate(int pixelClockRate) { setValue<unsigned int>(pixelClockRate, displayOptions_.pixel_clock_rate, std::bind(&QDSCP4Settings::pixelClockRateChanged, this, std::placeholders::_1)); }
	void setHologramPlaneWidth(double hologramPlaneWidth) { setValue<float>(hologramPlaneWidth, displayOptions_.hologram_plane_width, std::bind(&QDSCP4Settings::hologramPlaneWidthChanged, this, std::placeholders::_1)); }
	void setNumScanlines(int numScanlines) { setValue<unsigned int>(numScanlines, algorithmOptions_->num_scanlines, std::bind(&QDSCP4Settings::numScanlinesChanged, this, std::placeholders::_1)); }

	void saveSettings();
	void restoreDefaultSettings();

signals:
	void installPathChanged(QString newInstallPath);
	void binPathChanged(QString newBinPath);
	void libPathChanged(QString newLibPath);
	void modelsPathChanged(QString newModelsPath);
	void shadersPathChanged(QString newShadersPath);
	void kernelsPathChanged(QString newKernelsPath);
	void verbosityChanged(int verbosity);

	// Input options
	void objectFileNameChanged(QString newObjectFileName);
	void generateNormalsChanged(QString newGenerateNormals);
	void triangulateMeshChanged(bool newTriangulateMesh);

	// Render options
	void autoScaleEnabledChanged(bool newAutoScaleEnabled);
	void shadeModelChanged(QString newShadeModel);
	void shaderFileNameChanged(QString newShaderFileName);
	void renderModeChanged(int renderMode);
	void lightPosXChanged(double newLightPosX);
	void lightPosYChanged(double newLightPosY);
	void lightPosZChanged(double newLightPosZ);

	// Algorithm options
	void numViewsXChanged(int newNumViewsX);
	void numViewsYChanged(int newNumViewsY);
	void numWafelsPerScanlineChanged(int newNumWafelsPerScanline);
	void fovXChanged(double newFOVX);
	void fovYChanged(double newFOVY);
	void zNearChanged(double newZNear);
	void zFarChanged(double newZFar);
	void computeMethodChanged(QString newComputeMethod);
	void computeBlockDimXChanged(int newComputeBlockDimX);
	void computeBlockDimYChanged(int newComputeBlockDimY);
	void openCLKernelFileNameChanged(QString newOpenCLKernelFileName);
	void refBeamAngle_DegChanged(double newRefBeamAngle_Deg);
	void temporalUpconvertRedChanged(int newTemporalUpconvertRed);
	void temporalUpconvertGreenChanged(int newTemporalUpconvertGreen);
	void temporalUpconvertBlueChanged(int newTemporalUpconvertBlue);
	void wavelengthRed_100nmChanged(double newWavelengthRed_100nm);
	void wavelengthGreen_100nmChanged(double newWavelengthGreen_100nm);
	void wavelengthBlue_100nmChanged(double newWavelengthBlue_100nm);

	// Display options
	void displayNameChanged(QString newDisplayName);
	void x11EnvVarChanged(QString newX11EnvVar);
	void numHeadsChanged(int newNumHeads);
	void numHeadsPerGPUChanged(int newNumHeadsPerGPU);
	void headResXChanged(int newHeadResX);
	void headResYChanged(int newHeadResY);
	void headResXSpecChanged(int newHeadResXSpec);
	void headResYSpecChanged(int newHeadResYSpec);
	void numAOMChannelsChanged(int newNumAOMChannels);
	void pixelClockRateChanged(int newPixelClockRate);
	void numScanlinesChanged(int newNumScanlines);
	void hologramPlaneWidthChanged(double newHologramPlaneWidth);
	void numSamplesPerHololineChanged(int newNumSamplesPerHololine);


private:

	template<typename T>
	void setValue(T newValue, T& where, std::function<void(T)> valueChanged)
	{
		if (where != newValue)
		{
			where = newValue;
			emit valueChanged(newValue);
		}
		
		//MainWindow* parentWindow = (MainWindow*)this->parent();
		//parentWindow->forceRedraw();
	}

	DSCP4ProgramOptions programOptions_;

	render_options_t *renderOptions_;
	algorithm_options_t *algorithmOptions_;
	display_options_t displayOptions_;

	int argc_;
	const char ** argv_;

	QString installPath_;
	QString binPath_;
	QString libPath_;
	QString modelsPath_;
	QString shadersPath_;
	std::string shadersPathStr_;

	QString kernelsPath_;
	std::string kernelsPathStr_;

	QString objectFileName_;
	QString generateNormals_;
	QString shadeModel_;
	QString shaderFileName_;
	std::string shaderFileNameStr_;


	QString computeMethod_;
	QString openCLKernelFileName_;
	std::string openCLKernelFileNameStr_;

	QString displayName_;
	std::string displayNameStr_;

	QString x11EnvVar_;
	std::string x11EnvVarStr_;

	unsigned int computeBlockDimX_;
	unsigned int computeBlockDimY_;

	// backing store for wavelength
	double wavelengthRed_100nm_;
	double wavelengthBlue_100nm_;
	double wavelengthGreen_100nm_;

	int verbosity_;

	bool triangulateMesh_;

};


#endif
