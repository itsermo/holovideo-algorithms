#ifndef DSCP4_SETTINGS_H
#define DSCP4_SETTINGS_H

#include <QObject.h>
#include <QString.h>

#include <DSCP4ProgOptions.hpp>
#include "../libdscp4/dscp4_defs.h"

namespace Ui
{
	class QDSCP4Settings;
}

class QDSCP4Settings : public QObject {
	Q_OBJECT

public:
	QDSCP4Settings();
	QDSCP4Settings(int argc, const char **argv);

	void restoreDefaultSettings();
	void populateSettings();

public slots:
	void setInstallPath(QString installPath) { setValue<QString>(installPath, installPath_, std::bind(&QDSCP4Settings::installPathChanged, this, std::placeholders::_1)); }
	void setBinPath(QString binPath) { setValue<QString>(binPath, binPath_, std::bind(&QDSCP4Settings::binPathChanged, this, std::placeholders::_1)); }
	void setLibPath(QString libPath) { setValue<QString>(libPath, libPath_, std::bind(&QDSCP4Settings::libPathChanged, this, std::placeholders::_1)); }
	void setModelsPath(QString modelsPath) { setValue<QString>(modelsPath, modelsPath_, std::bind(&QDSCP4Settings::modelsPathChanged, this, std::placeholders::_1)); }
	void setShadersPath(QString shadersPath) {
		setValue<QString>(shadersPath, shadersPath_, std::bind(&QDSCP4Settings::shadersPathChanged, this, std::placeholders::_1));
			renderOptions_->shaders_path = shadersPath_.toUtf8().constData();
		}

	void setKernelsPath(QString kernelsPath) {
			setValue<QString>(kernelsPath, kernelsPath_, std::bind(&QDSCP4Settings::kernelsPathChanged, this, std::placeholders::_1)); 
			renderOptions_->kernels_path = kernelsPath_.toUtf8().constData();
		}

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
		renderOptions_->shader_filename_prefix = shaderFileName_.toUtf8().constData();
	}

	void setLightPosX(float lightPosX) { setValue<float>(lightPosX, renderOptions_->light_pos_x, std::bind(&QDSCP4Settings::lightPosXChanged, this, std::placeholders::_1)); }
	void setLightPosY(float lightPosY) { setValue<float>(lightPosY, renderOptions_->light_pos_y, std::bind(&QDSCP4Settings::lightPosYChanged, this, std::placeholders::_1)); }
	void setLightPosZ(float lightPosZ) { setValue<float>(lightPosZ, renderOptions_->light_pos_z, std::bind(&QDSCP4Settings::lightPosZChanged, this, std::placeholders::_1)); }

	// Algorithm options
	void setNumViewsX(unsigned int numViewsX) { setValue<unsigned int>(numViewsX, algorithmOptions_->num_views_x, std::bind(&QDSCP4Settings::numViewsXChanged, this, std::placeholders::_1)); }
	void setNumViewsY(unsigned int numViewsY) { setValue<unsigned int>(numViewsY, algorithmOptions_->num_views_y, std::bind(&QDSCP4Settings::numViewsYChanged, this, std::placeholders::_1)); }
	void setNumWafelsPerScanline(unsigned int numWafelsPerScanline) { setValue<unsigned int>(numWafelsPerScanline, algorithmOptions_->num_wafels_per_scanline, std::bind(&QDSCP4Settings::numWafelsPerScanlineChanged, this, std::placeholders::_1)); }
	void setFOVX(float fovX){ setValue<float>(fovX, algorithmOptions_->fov_x, std::bind(&QDSCP4Settings::fovXChanged, this, std::placeholders::_1)); }
	void setFOVY(float fovY) { setValue<float>(fovY, algorithmOptions_->fov_y, std::bind(&QDSCP4Settings::fovYChanged, this, std::placeholders::_1)); }
	void setComputeMethod(QString computeMethod) { 
		setValue<QString>(computeMethod, computeMethod_, std::bind(&QDSCP4Settings::computeMethodChanged, this, std::placeholders::_1));
		algorithmOptions_->compute_method = computeMethod_ == "OpenCL" ? DSCP4_COMPUTE_METHOD_OPENCL : DSCP4_COMPUTE_METHOD_CUDA;
	}

	void setComputeBlockDimX(unsigned int computeBlockDimX) {
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

	void setComputeBlockDimY(unsigned int computeBlockDimY) { 
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
		algorithmOptions_->opencl_kernel_filename = openCLKernelFileName_.toUtf8().constData();
	}
	void setRefBeamAngle_Deg(float refBeamAngle_Deg) { setValue<float>(refBeamAngle_Deg, algorithmOptions_->reference_beam_angle, std::bind(&QDSCP4Settings::refBeamAngle_DegChanged, this, std::placeholders::_1)); }
	void setTemporalUpconvertRed(unsigned int temporalUpconvertRed)  { setValue<unsigned int>(temporalUpconvertRed, algorithmOptions_->temporal_upconvert_red, std::bind(&QDSCP4Settings::temporalUpconvertRedChanged, this, std::placeholders::_1)); }
	void setTemporalUpconvertGreen(unsigned int temporalUpconvertGreen) { setValue<unsigned int>(temporalUpconvertGreen, algorithmOptions_->temporal_upconvert_green, std::bind(&QDSCP4Settings::temporalUpconvertGreenChanged, this, std::placeholders::_1)); }
	void setTemporalUpconvertBlue(unsigned int temporalUpconvertBlue) { setValue<unsigned int>(temporalUpconvertBlue, algorithmOptions_->temporal_upconvert_blue, std::bind(&QDSCP4Settings::temporalUpconvertBlueChanged, this, std::placeholders::_1)); }
	void setWavelengthRed_100nm(float wavelengthRed_100nm)  { setValue<float>(wavelengthRed_100nm, algorithmOptions_->wavelength_red * powf(10,7), std::bind(&QDSCP4Settings::wavelengthRed_100nmChanged, this, std::placeholders::_1)); }
	void setWavelengthGreen_100nm(float wavelengthGreen_100nm)  { setValue<float>(wavelengthGreen_100nm, algorithmOptions_->wavelength_green * powf(10, 7), std::bind(&QDSCP4Settings::wavelengthGreen_100nmChanged, this, std::placeholders::_1)); }
	void setWavelengthBlue_100nm(float wavelengthBlue_100nm) { setValue<float>(wavelengthBlue_100nm, algorithmOptions_->wavelength_blue * powf(10, 7), std::bind(&QDSCP4Settings::wavelengthBlue_100nmChanged, this, std::placeholders::_1)); }

	// Display options
	void setDisplayName(QString displayName) { 
		setValue<QString>(displayName, displayName_, std::bind(&QDSCP4Settings::displayNameChanged, this, std::placeholders::_1)); 
		displayOptions_.name = displayName_.toUtf8().constData();
	}
	void setNumHeads(unsigned int numHeads) { setValue<unsigned int>(numHeads, displayOptions_.num_heads, std::bind(&QDSCP4Settings::numHeadsChanged, this, std::placeholders::_1)); }
	void setNumHeadsPerGPU(unsigned int numHeadsPerGPU) { setValue<unsigned int>(numHeadsPerGPU, displayOptions_.num_heads_per_gpu, std::bind(&QDSCP4Settings::numHeadsPerGPUChanged, this, std::placeholders::_1)); }
	void setHeadResX(unsigned int headResX) { setValue<unsigned int>(headResX, displayOptions_.head_res_x, std::bind(&QDSCP4Settings::headResXChanged, this, std::placeholders::_1)); }
	void setHeadResY(unsigned int headResY) { setValue<unsigned int>(headResY, displayOptions_.head_res_y, std::bind(&QDSCP4Settings::headResYChanged, this, std::placeholders::_1)); }
	void setHeadResXSpec(unsigned int headResXSpec) { setValue<unsigned int>(headResXSpec, displayOptions_.head_res_x_spec, std::bind(&QDSCP4Settings::headResXSpecChanged, this, std::placeholders::_1)); }
	void setHeadResYSpec(unsigned int headResYSpec) { setValue<unsigned int>(headResYSpec, displayOptions_.head_res_y_spec, std::bind(&QDSCP4Settings::headResYSpecChanged, this, std::placeholders::_1)); }
	void setNumAOMChannels(unsigned int numAOMChannels) { setValue<unsigned int>(numAOMChannels, displayOptions_.num_aom_channels, std::bind(&QDSCP4Settings::numAOMChannelsChanged, this, std::placeholders::_1)); }
	void setNumSamplesPerHololine(unsigned int numSamplesPerHololine) { setValue<unsigned int>(numSamplesPerHololine, displayOptions_.num_samples_per_hololine, std::bind(&QDSCP4Settings::numSamplesPerHololineChanged, this, std::placeholders::_1)); }
	void setPixelClockRate(unsigned int pixelClockRate) { setValue<unsigned int>(pixelClockRate, displayOptions_.pixel_clock_rate, std::bind(&QDSCP4Settings::pixelClockRateChanged, this, std::placeholders::_1)); }
	void setHologramPlaneWidth(float hologramPlaneWidth) { setValue<float>(hologramPlaneWidth, displayOptions_.hologram_plane_width, std::bind(&QDSCP4Settings::hologramPlaneWidthChanged, this, std::placeholders::_1)); }
	void setNumScanlines(unsigned int numScanlines) { setValue<unsigned int>(numScanlines, algorithmOptions_->num_scanlines, std::bind(&QDSCP4Settings::numScanlinesChanged, this, std::placeholders::_1)); }

signals:
	void installPathChanged(QString newInstallPath);
	void binPathChanged(QString newBinPath);
	void libPathChanged(QString newLibPath);
	void modelsPathChanged(QString newModelsPath);
	void shadersPathChanged(QString newShadersPath);
	void kernelsPathChanged(QString newKernelsPath);

	// Input options
	void objectFileNameChanged(QString newObjectFileName);
	void generateNormalsChanged(QString newGenerateNormals);
	void triangulateMeshChanged(bool newTriangulateMesh);

	// Render options
	void autoScaleEnabledChanged(bool newAutoScaleEnabled);
	void shadeModelChanged(QString newShadeModel);
	void shaderFileNameChanged(QString newShaderFileName);

	void lightPosXChanged(float newLightPosX);
	void lightPosYChanged(float newLightPosY);
	void lightPosZChanged(float newLightPosZ);

	// Algorithm options
	void numViewsXChanged(unsigned int newNumViewsX);
	void numViewsYChanged(unsigned int newNumViewsY);
	void numWafelsPerScanlineChanged(unsigned int newNumWafelsPerScanline);
	void fovXChanged(float newFOVX);
	void fovYChanged(float newFOVY);
	void computeMethodChanged(QString newComputeMethod);
	void computeBlockDimXChanged(unsigned int newComputeBlockDimX);
	void computeBlockDimYChanged(unsigned int newComputeBlockDimY);
	void openCLKernelFileNameChanged(QString newOpenCLKernelFileName);
	void refBeamAngle_DegChanged(float newRefBeamAngle_Deg);
	void temporalUpconvertRedChanged(unsigned int newTemporalUpconvertRed);
	void temporalUpconvertGreenChanged(unsigned int newTemporalUpconvertGreen);
	void temporalUpconvertBlueChanged(unsigned int newTemporalUpconvertBlue);
	void wavelengthRed_100nmChanged(float newWavelengthRed_100nm);
	void wavelengthGreen_100nmChanged(float newWavelengthGreen_100nm);
	void wavelengthBlue_100nmChanged(float newWavelengthBlue_100nm);

	// Display options
	void displayNameChanged(QString newDisplayName);
	void numHeadsChanged(unsigned int newNumHeads);
	void numHeadsPerGPUChanged(unsigned int newNumHeadsPerGPU);
	void headResXChanged(unsigned int newHeadResX);
	void headResYChanged(unsigned int newHeadResY);
	void headResXSpecChanged(unsigned int newHeadResXSpec);
	void headResYSpecChanged(unsigned int newHeadResYSpec);
	void numAOMChannelsChanged(unsigned int newNumAOMChannels);
	void numSamplesPerHololineChanged(unsigned int newNumSamplesPerHololine);
	void pixelClockRateChanged(unsigned int newPixelClockRate);
	void hologramPlaneWidthChanged(float newHologramPlaneWidth);
	void numScanlinesChanged(unsigned int newNumScanlines);

private:

	template<typename T>
	void setValue(T newValue, T where, std::function<void(T)> valueChanged)
	{
		if (where != newValue)
		{
			where = newValue;
			emit valueChanged(newValue);
		}
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
	QString kernelsPath_;

	QString objectFileName_;
	QString generateNormals_;
	QString shadeModel_;
	QString shaderFileName_;

	QString computeMethod_;
	QString openCLKernelFileName_;

	QString displayName_;

	unsigned int computeBlockDimX_;
	unsigned int computeBlockDimY_;

	bool triangulateMesh_;

};


#endif