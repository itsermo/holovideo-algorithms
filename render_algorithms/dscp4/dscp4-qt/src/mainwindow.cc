#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
: MainWindow(nullptr, parent)
{

}

MainWindow::MainWindow(QDSCP4Settings* settings, QWidget *parent) : QMainWindow(parent), ui(new Ui::MainWindow), settings_(settings)
{
	ui->setupUi(this);

	// General/Input options
	QObject::connect(settings_, SIGNAL(objectFileNameChanged(QString)), ui->inputFileComboBox, SLOT(setCurrentText(QString)));
	QObject::connect(ui->inputFileComboBox, SIGNAL(currentTextChanged(QString)), settings_, SLOT(setObjectFileName(QString)));
	QObject::connect(settings_, SIGNAL(generateNormalsChanged(QString)), ui->generateNormalsComboBox, SLOT(setCurrentText(QString)));
	QObject::connect(ui->generateNormalsComboBox, SIGNAL(currentTextChanged(QString)), settings_, SLOT(setGenerateNormals(QString)));
	QObject::connect(settings_, SIGNAL(triangulateMeshChanged(bool)), ui->triangulateMeshCheckBox, SLOT(setChecked(bool)));
	QObject::connect(ui->triangulateMeshCheckBox, SIGNAL(toggled(bool)), settings_, SLOT(setTriangulateMesh(bool)));
	QObject::connect(settings_, SIGNAL(installPathChanged(QString)), ui->installPathLineEdit, SLOT(setText(QString)));
	QObject::connect(ui->installPathLineEdit, SIGNAL(textChanged(QString)), settings_, SLOT(setInstallPath(QString)));
	QObject::connect(settings_, SIGNAL(binPathChanged(QString)), ui->binPathLineEdit, SLOT(setText(QString)));
	QObject::connect(ui->binPathLineEdit, SIGNAL(textChanged(QString)), settings_, SLOT(setBinPath(QString)));
	QObject::connect(settings_, SIGNAL(libPathChanged(QString)), ui->libPathLineEdit, SLOT(setText(QString)));
	QObject::connect(ui->libPathLineEdit, SIGNAL(textChanged(QString)), settings_, SLOT(setLibPath(QString)));
	QObject::connect(settings_, SIGNAL(modelsPathChanged(QString)), ui->modelsPathLineEdit, SLOT(setText(QString)));
	QObject::connect(ui->modelsPathLineEdit, SIGNAL(textChanged(QString)), settings_, SLOT(setModelsPath(QString)));
	QObject::connect(settings_, SIGNAL(shadersPathChanged(QString)), ui->shadersPathLineEdit, SLOT(setText(QString)));
	QObject::connect(ui->shadersPathLineEdit, SIGNAL(textChanged(QString)), settings_, SLOT(setShadersPath(QString)));
	QObject::connect(settings_, SIGNAL(kernelsPathChanged(QString)), ui->kernelsPathLineEdit, SLOT(setText(QString)));
	QObject::connect(ui->kernelsPathLineEdit, SIGNAL(textChanged(QString)), settings_, SLOT(setKernelsPath(QString)));
	QObject::connect(settings_, SIGNAL(verbosityChanged(int)), ui->verbosityComboBox, SLOT(setCurrentIndex(int)));
	QObject::connect(ui->verbosityComboBox, SIGNAL(currentIndexChanged(int)), settings_, SLOT(setVerbosity(int)));

	// Render options
	QObject::connect(settings_, SIGNAL(autoScaleEnabledChanged(bool)), ui->autoscaleModelCheckBox, SLOT(setChecked(bool)));
	QObject::connect(ui->autoscaleModelCheckBox, SIGNAL(toggled(bool)), settings_, SLOT(setAutoScaleEnabled(bool)));
	QObject::connect(settings_, SIGNAL(shadeModelChanged(QString)), ui->shaderModelComboBox, SLOT(setCurrentText(QString)));
	QObject::connect(ui->shaderModelComboBox, SIGNAL(currentTextChanged(QString)), settings_, SLOT(setShadeModel(QString)));
	QObject::connect(settings_, SIGNAL(shaderFileNameChanged(QString)), ui->shaderFileNameLineEdit, SLOT(setText(QString)));
	QObject::connect(ui->shaderFileNameLineEdit, SIGNAL(textChanged(QString)), settings_, SLOT(setShaderFileName(QString)));

	QObject::connect(settings_, SIGNAL(lightPosXChanged(double)), ui->lightPositionXDoubleSpinBox, SLOT(setValue(double)));
	QObject::connect(ui->lightPositionXDoubleSpinBox, SIGNAL(valueChanged(double)), settings_, SLOT(setLightPosX(double)));
	QObject::connect(settings_, SIGNAL(lightPosYChanged(double)), ui->lightPositionYDoubleSpinBox, SLOT(setValue(double)));
	QObject::connect(ui->lightPositionYDoubleSpinBox, SIGNAL(valueChanged(double)), settings_, SLOT(setLightPosY(double)));
	QObject::connect(settings_, SIGNAL(lightPosZChanged(double)), ui->lightPositionZDoubleSpinBox, SLOT(setValue(double)));
	QObject::connect(ui->lightPositionZDoubleSpinBox, SIGNAL(valueChanged(double)), settings_, SLOT(setLightPosZ(double)));

	// Algorithm options
	QObject::connect(settings_, SIGNAL(numViewsXChanged(int)), ui->xViewsSpinBox, SLOT(setValue(int)));
	QObject::connect(ui->xViewsSpinBox, SIGNAL(toggled(int)), settings_, SLOT(setNumViewsX(int)));
	QObject::connect(settings_, SIGNAL(numViewsYChanged(int)), ui->yViewsSpinBox, SLOT(setValue(int)));
	QObject::connect(ui->yViewsSpinBox, SIGNAL(toggled(int)), settings_, SLOT(setNumViewsY(int)));
	QObject::connect(settings_, SIGNAL(numWafelsPerScanlineChanged(int)), ui->numWafelsSpinBox, SLOT(setValue(int)));
	QObject::connect(ui->numWafelsSpinBox, SIGNAL(toggled(int)), settings_, SLOT(setNumWafelsPerScanline(int)));
	QObject::connect(settings_, SIGNAL(fovXChanged(double)), ui->xFOVDoubleSpinBox, SLOT(setValue(double)));
	QObject::connect(ui->xFOVDoubleSpinBox, SIGNAL(valueChanged(double)), settings_, SLOT(setFOVX(double)));
	QObject::connect(settings_, SIGNAL(fovYChanged(double)), ui->yFOVDoubleSpinBox, SLOT(setValue(double)));
	QObject::connect(ui->yFOVDoubleSpinBox, SIGNAL(valueChanged(double)), settings_, SLOT(setFOVY(double)));
	QObject::connect(settings_, SIGNAL(computeMethodChanged(QString)), ui->computeMethodComboBox, SLOT(setCurrentText(QString)));
	QObject::connect(ui->computeMethodComboBox, SIGNAL(currentTextChanged(QString)), settings_, SLOT(setComputeMethod(QString)));
	QObject::connect(settings_, SIGNAL(openCLKernelFileNameChanged(QString)), ui->openclKernelFileComboBox, SLOT(setCurrentText(QString)));
	QObject::connect(ui->openclKernelFileComboBox, SIGNAL(currentTextChanged(QString)), settings_, SLOT(setOpenCLKernelFileName(QString)));
	QObject::connect(settings_, SIGNAL(computeBlockDimXChanged(int)), ui->computeXBlockSizeSpinBox, SLOT(setValue(int)));
	QObject::connect(ui->computeXBlockSizeSpinBox, SIGNAL(valueChanged(int)), settings_, SLOT(setComputeBlockDimX(int)));
	QObject::connect(settings_, SIGNAL(computeBlockDimYChanged(int)), ui->computeYBlockSizeSpinBox, SLOT(setValue(int)));
	QObject::connect(ui->computeYBlockSizeSpinBox, SIGNAL(valueChanged(int)), settings_, SLOT(setComputeBlockDimY(int)));
	QObject::connect(settings_, SIGNAL(refBeamAngle_DegChanged(double)), ui->refBeamAngleDoubleSpinBox, SLOT(setValue(double)));
	QObject::connect(ui->refBeamAngleDoubleSpinBox, SIGNAL(valueChanged(double)), settings_, SLOT(setRefBeamAngle_Deg(double)));
	QObject::connect(settings_, SIGNAL(temporalUpconvertRedChanged(int)), ui->redUpconvertConstSpinbox, SLOT(setValue(int)));
	QObject::connect(ui->redUpconvertConstSpinbox, SIGNAL(valueChanged(int)), settings_, SLOT(setTemporalUpconvertRed(int)));
	QObject::connect(settings_, SIGNAL(temporalUpconvertGreenChanged(int)), ui->greenUpconvertConstSpinbox, SLOT(setValue(int)));
	QObject::connect(ui->greenUpconvertConstSpinbox, SIGNAL(valueChanged(int)), settings_, SLOT(setTemporalUpconvertGreen(int)));
	QObject::connect(settings_, SIGNAL(temporalUpconvertBlueChanged(int)), ui->blueUpconvertConstSpinbox, SLOT(setValue(int)));
	QObject::connect(ui->blueUpconvertConstSpinbox, SIGNAL(valueChanged(int)), settings_, SLOT(setTemporalUpconvertBlue(int)));
	QObject::connect(settings_, SIGNAL(wavelengthRed_100nmChanged(double)), ui->redWavelengthDoubleSpinBox, SLOT(setValue(double)));
	QObject::connect(ui->redWavelengthDoubleSpinBox, SIGNAL(valueChanged(double)), settings_, SLOT(setWavelengthRed_100nm(double)));
	QObject::connect(settings_, SIGNAL(wavelengthGreen_100nmChanged(double)), ui->greenWavelengthDoubleSpinBox, SLOT(setValue(double)));
	QObject::connect(ui->greenWavelengthDoubleSpinBox, SIGNAL(valueChanged(double)), settings_, SLOT(setWavelengthGreen_100nm(double)));
	QObject::connect(settings_, SIGNAL(wavelengthBlue_100nmChanged(double)), ui->blueWavelengthDoubleSpinBox, SLOT(setValue(double)));
	QObject::connect(ui->blueWavelengthDoubleSpinBox, SIGNAL(valueChanged(double)), settings_, SLOT(setWavelengthBlue_100nm(double)));

	// Display options
	QObject::connect(settings_, SIGNAL(displayNameChanged(QString)), ui->friendlyNameLineEdit, SLOT(setText(QString)));
	QObject::connect(ui->friendlyNameLineEdit, SIGNAL(textChanged(QString)), settings_, SLOT(setDisplayName(QString)));

	QObject::connect(settings_, SIGNAL(x11EnvVarChanged(QString)), ui->x11DisplayEnvLineEdit, SLOT(setText(QString)));
	QObject::connect(ui->x11DisplayEnvLineEdit, SIGNAL(textChanged(QString)), settings_, SLOT(setX11EnvVar(QString)));

	QObject::connect(settings_, SIGNAL(numHeadsChanged(int)), ui->numHeadsSpinBox, SLOT(setValue(int)));
	QObject::connect(ui->numHeadsSpinBox, SIGNAL(valueChanged(int)), settings_, SLOT(setNumHeads(int)));
	QObject::connect(settings_, SIGNAL(numHeadsPerGPUChanged(int)), ui->numHeadsPerGPUSpinBox, SLOT(setValue(int)));
	QObject::connect(ui->numHeadsPerGPUSpinBox, SIGNAL(valueChanged(int)), settings_, SLOT(setNumHeadsPerGPU(int)));
	QObject::connect(settings_, SIGNAL(headResXChanged(int)), ui->xHeadResSpinBox, SLOT(setValue(int)));
	QObject::connect(ui->xHeadResSpinBox, SIGNAL(valueChanged(int)), settings_, SLOT(setHeadResX(int)));
	QObject::connect(settings_, SIGNAL(headResXSpecChanged(int)), ui->xHeadResSpecSpinBox, SLOT(setValue(int)));
	QObject::connect(ui->xHeadResSpecSpinBox, SIGNAL(valueChanged(int)), settings_, SLOT(setHeadResXSpec(int)));
	QObject::connect(settings_, SIGNAL(headResYChanged(int)), ui->yHeadResSpinBox, SLOT(setValue(int)));
	QObject::connect(ui->yHeadResSpinBox, SIGNAL(valueChanged(int)), settings_, SLOT(setHeadResY(int)));
	QObject::connect(settings_, SIGNAL(headResYSpecChanged(int)), ui->yHeadResSpecSpinBox, SLOT(setValue(int)));
	QObject::connect(ui->yHeadResSpecSpinBox, SIGNAL(valueChanged(int)), settings_, SLOT(setHeadResYSpec(int)));
	QObject::connect(settings_, SIGNAL(pixelClockRateChanged(int)), ui->clockRateSpinBox, SLOT(setValue(int)));
	QObject::connect(ui->clockRateSpinBox, SIGNAL(valueChanged(int)), settings_, SLOT(setPixelClockRate(int)));
	QObject::connect(settings_, SIGNAL(numScanlinesChanged(int)), ui->numScanlinesSpinBox, SLOT(setValue(int)));
	QObject::connect(ui->numScanlinesSpinBox, SIGNAL(valueChanged(int)), settings_, SLOT(setNumScanlines(int)));
	QObject::connect(settings_, SIGNAL(numAOMChannelsChanged(int)), ui->numAOMChannelsSpinBox, SLOT(setValue(int)));
	QObject::connect(ui->numAOMChannelsSpinBox, SIGNAL(valueChanged(int)), settings_, SLOT(setNumAOMChannels(int)));
	QObject::connect(settings_, SIGNAL(hologramPlaneWidthChanged(double)), ui->planeWidthDoubleSpinBox, SLOT(setValue(double)));
	QObject::connect(ui->planeWidthDoubleSpinBox, SIGNAL(valueChanged(double)), settings_, SLOT(setHologramPlaneWidth(double)));
	QObject::connect(settings_, SIGNAL(numSamplesPerHololineChanged(int)), ui->numSamplesPerHololineSpinBox, SLOT(setValue(int)));
	QObject::connect(ui->numSamplesPerHololineSpinBox, SIGNAL(valueChanged(int)), settings_, SLOT(setNumSamplesPerHololine(int)));



	settings->populateSettings();
}


MainWindow::~MainWindow()
{

}
