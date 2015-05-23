#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "QLogAppender.h"

#include <boost/filesystem.hpp>
#include <qfiledialog.h>
#include <boost/tokenizer.hpp>

#include <dscp4.h>

#include <qmessagebox.h>
#include <qtimer.h>

MainWindow::MainWindow(QWidget *parent)
: MainWindow(0, nullptr, parent)
{

}

MainWindow::MainWindow(int argc, const char ** argv, QWidget *parent) : 
QMainWindow(parent), 
ui(new Ui::MainWindow), 
algorithmContext_(nullptr), 
x11Process_(nullptr),
nvidiaSettingsProcess_(nullptr),
x11vncProcess_(nullptr),
haveNewFrame_(false),
failedInit_(false)
{

	settings_ = new QDSCP4Settings(argc, argv, this);

#ifdef DSCP4_HAVE_LOG4CXX

	logger_ = log4cxx::Logger::getLogger("edu.mit.media.obmg.holovideo.dscp4");

#ifdef WIN32
	log4cxx::PatternLayoutPtr logLayoutPtr = new log4cxx::PatternLayout(L"%-5p\t%m%n");
	log4cxx::helpers::ObjectPtrT<log4cxx::QLogAppender> logAppenderPtr = new log4cxx::QLogAppender(logLayoutPtr);
	logAppenderPtr->setName(log4cxx::LogString(L"dscp4-qt"));
#else
	log4cxx::PatternLayoutPtr logLayoutPtr = new log4cxx::PatternLayout("%-5p\t%m%n");
	log4cxx::helpers::ObjectPtrT<log4cxx::QLogAppender> logAppenderPtr = new log4cxx::QLogAppender((log4cxx::helpers::ObjectPtrT<log4cxx::Layout>)logLayoutPtr);
	logAppenderPtr->setName(log4cxx::LogString("dscp4-qt"));
#endif
	logger_->addAppender(logAppenderPtr);
	//log4cxx::BasicConfigurator::configure(logAppenderPtr);
#endif

	ui->setupUi(this);

	QObject::connect(this->ui->saveSettingsButton, SIGNAL(clicked()), settings_, SLOT(saveSettings()));
	QObject::connect(this->ui->restoreDefaultSettingsButton, SIGNAL(clicked()), settings_, SLOT(restoreDefaultSettings()));

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
	QObject::connect(settings_, SIGNAL(shaderFileNameChanged(QString)), ui->shaderFileNameComboBox, SLOT(setCurrentText(QString)));
	QObject::connect(ui->shaderFileNameComboBox, SIGNAL(currentTextChanged(QString)), settings_, SLOT(setShaderFileName(QString)));
	QObject::connect(settings_, SIGNAL(renderModeChanged(int)), ui->renderModeComboBox, SLOT(setCurrentIndex(int)));
	QObject::connect(ui->renderModeComboBox, SIGNAL(currentIndexChanged(int)), settings_, SLOT(setRenderMode(int)));
	QObject::connect(settings_, SIGNAL(lightPosXChanged(double)), ui->lightPositionXDoubleSpinBox, SLOT(setValue(double)));
	QObject::connect(ui->lightPositionXDoubleSpinBox, SIGNAL(valueChanged(double)), settings_, SLOT(setLightPosX(double)));
	QObject::connect(settings_, SIGNAL(lightPosYChanged(double)), ui->lightPositionYDoubleSpinBox, SLOT(setValue(double)));
	QObject::connect(ui->lightPositionYDoubleSpinBox, SIGNAL(valueChanged(double)), settings_, SLOT(setLightPosY(double)));
	QObject::connect(settings_, SIGNAL(lightPosZChanged(double)), ui->lightPositionZDoubleSpinBox, SLOT(setValue(double)));
	QObject::connect(ui->lightPositionZDoubleSpinBox, SIGNAL(valueChanged(double)), settings_, SLOT(setLightPosZ(double)));

	// Algorithm options
	QObject::connect(settings_, SIGNAL(numViewsXChanged(int)), ui->xViewsSpinBox, SLOT(setValue(int)));
	QObject::connect(ui->xViewsSpinBox, SIGNAL(valueChanged(int)), settings_, SLOT(setNumViewsX(int)));
	QObject::connect(settings_, SIGNAL(numViewsYChanged(int)), ui->yViewsSpinBox, SLOT(setValue(int)));
	QObject::connect(ui->yViewsSpinBox, SIGNAL(valueChanged(int)), settings_, SLOT(setNumViewsY(int)));
	QObject::connect(settings_, SIGNAL(numWafelsPerScanlineChanged(int)), ui->numWafelsSpinBox, SLOT(setValue(int)));
	QObject::connect(ui->numWafelsSpinBox, SIGNAL(valueChanged(int)), settings_, SLOT(setNumWafelsPerScanline(int)));
	QObject::connect(settings_, SIGNAL(fovXChanged(double)), ui->xFOVDoubleSpinBox, SLOT(setValue(double)));
	QObject::connect(ui->xFOVDoubleSpinBox, SIGNAL(valueChanged(double)), settings_, SLOT(setFOVX(double)));
	QObject::connect(settings_, SIGNAL(fovYChanged(double)), ui->yFOVDoubleSpinBox, SLOT(setValue(double)));
	QObject::connect(ui->yFOVDoubleSpinBox, SIGNAL(valueChanged(double)), settings_, SLOT(setFOVY(double)));
	QObject::connect(settings_, SIGNAL(zNearChanged(double)), ui->zNearAlgorithmDoubleSpinBox, SLOT(setValue(double)));
	QObject::connect(ui->zNearAlgorithmDoubleSpinBox, SIGNAL(valueChanged(double)), settings_, SLOT(setZNear(double)));
	QObject::connect(settings_, SIGNAL(zFarChanged(double)), ui->zFarAlgorithmDoubleSpinBox, SLOT(setValue(double)));
	QObject::connect(ui->zFarAlgorithmDoubleSpinBox, SIGNAL(valueChanged(double)), settings_, SLOT(setZFar(double)));
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
	QObject::connect(ui->redColorGainHorizontalSlider, SIGNAL(valueChanged(int)), settings_, SLOT(setRedGain(int)));
	QObject::connect(settings_, SIGNAL(redGainChangedInt(int)), ui->redColorGainHorizontalSlider, SLOT(setValue(int)));
	QObject::connect(ui->greenColorGainHorizontalSlider, SIGNAL(valueChanged(int)), settings_, SLOT(setGreenGain(int)));
	QObject::connect(settings_, SIGNAL(greenGainChangedInt(int)), ui->greenColorGainHorizontalSlider, SLOT(setValue(int)));
	QObject::connect(ui->blueColorGainHorizontalSlider, SIGNAL(valueChanged(int)), settings_, SLOT(setBlueGain(int)));
	QObject::connect(settings_, SIGNAL(blueGainChangedInt(int)), ui->blueColorGainHorizontalSlider, SLOT(setValue(int)));

	if (!settings_->populateSettings())
	{
		this->failedInit_ = true;
		return;
	}

	populateModelFiles();
	populateKernelFiles();
	populateShaderFiles();

	QObject::connect(ui->modelsPathLineEdit, SIGNAL(textChanged(QString)), this, SLOT(populateModelFiles()));
	QObject::connect(ui->kernelsPathLineEdit, SIGNAL(textChanged(QString)), this, SLOT(populateKernelFiles()));
	QObject::connect(ui->shadersPathLineEdit, SIGNAL(textChanged(QString)), this, SLOT(populateShaderFiles()));

	QObject::connect(ui->inputFileToolButton, SIGNAL(clicked()), this, SLOT(browseAndSetInputModelFile()));
	QObject::connect(ui->openclKernelFileToolButton, SIGNAL(clicked()), this, SLOT(browseAndSetOpenCLKernelFile()));
	QObject::connect(ui->shaderFileNameToolButton, SIGNAL(clicked()), this, SLOT(browseAndSetShaderFileName()));

	QObject::connect(ui->installPathToolButton, SIGNAL(clicked()), this, SLOT(browseAndSetInstallPath()));
	QObject::connect(ui->binPathToolButton, SIGNAL(clicked()), this, SLOT(browseAndSetBinPath()));
	QObject::connect(ui->modelPathToolButton, SIGNAL(clicked()), this, SLOT(browseAndSetModelsPath()));
	QObject::connect(ui->libPathToolButton, SIGNAL(clicked()), this, SLOT(browseAndSetLibPath()));
	QObject::connect(ui->shadersPathToolButton, SIGNAL(clicked()), this, SLOT(browseAndSetShadersPath()));
	QObject::connect(ui->kernelsPathToolButton, SIGNAL(clicked()), this, SLOT(browseAndSetKernelsPath()));

	// Log
	QObject::connect(ui->clearLogButton, SIGNAL(clicked()), this, SLOT(clearLog()));
#ifdef DSCP4_HAVE_LOG4CXX
	QObject::connect(logAppenderPtr, SIGNAL(gotNewLogMessage(QString)), ui->dscp4LogTextEdit, SLOT(append(QString)));
	QObject::connect(logAppenderPtr, SIGNAL(gotNewLogMessage(QString)), this, SLOT(handleLogMessage(QString)));
#endif

	//Controls
	QObject::connect(ui->dscp4ToggleButton, SIGNAL(clicked()), this, SLOT(startDSCP4()));
	QObject::connect(ui->x11ToggleButton, SIGNAL(clicked()), this, SLOT(startX11()));
	QObject::connect(ui->x11vncToggleButton, SIGNAL(clicked()), this, SLOT(startX11Vnc()));
	QObject::connect(ui->nvidiaSettingsToggleButton, SIGNAL(clicked()), this, SLOT(startNVIDIASettings()));
	QObject::connect(ui->saveScreenshotButton, SIGNAL(clicked()), this, SLOT(dumpFramebufferToPNG()));
	QObject::connect(ui->forceRedrawButton, SIGNAL(clicked()), this, SLOT(forceRedraw()));
	QObject::connect(ui->spinModelCheckBox, SIGNAL(toggled(bool)), this, SLOT(setSpinOn(bool)));
	QObject::connect(ui->xTranslateHorizontalSlider, SIGNAL(valueChanged(int)), this, SLOT(translateModelX(int)));
	QObject::connect(ui->yTranslateHorizontalSlider, SIGNAL(valueChanged(int)), this, SLOT(translateModelY(int)));
	QObject::connect(ui->zTranslateHorizontalSlider, SIGNAL(valueChanged(int)), this, SLOT(translateModelZ(int)));
	QObject::connect(ui->xRotateHorizontalSlider, SIGNAL(valueChanged(int)), this, SLOT(rotateModelX(int)));
	QObject::connect(ui->yRotateHorizontalSlider, SIGNAL(valueChanged(int)), this, SLOT(rotateModelY(int)));
	QObject::connect(ui->zRotateHorizontalSlider, SIGNAL(valueChanged(int)), this, SLOT(rotateModelZ(int)));
	QObject::connect(ui->zOffsetHorizontalSlider, SIGNAL(valueChanged(int)), this, SLOT(setPlaneZOffset(int)));

	QObject::connect(ui->redColorGainHorizontalSlider, SIGNAL(valueChanged(int)), this, SLOT(forceRedraw()));
	QObject::connect(ui->greenColorGainHorizontalSlider, SIGNAL(valueChanged(int)), this, SLOT(forceRedraw()));
	QObject::connect(ui->blueColorGainHorizontalSlider, SIGNAL(valueChanged(int)), this, SLOT(forceRedraw()));

	LOG4CXX_INFO(logger_, "Logger initialized")

	// These UI items will be disabled while DSCP4 is running
	unchangeables_ <<
	ui->generalSettingsGroupBox <<
	ui->renderModeComboBox <<
	ui->shaderFileNameComboBox <<
	ui->shaderFileNameToolButton <<
	ui->displaySettingsGroupBox <<
	ui->computeMethodComboBox <<
	ui->inputGroupBox <<
	ui->displayGroupBox <<
	ui->renderGroupBox <<
	ui->x11ToggleButton <<
	ui->numWafelsSpinBox <<
	ui->xViewsSpinBox <<
	ui->yViewsSpinBox;

	// These UI items will be enabled while DSCP4 is running
	dscp4Controls_ << ui->renderPreviewGroupBox << ui->controlGroupBox;

	disableControlsUI();

	ui->tabWidget->setCurrentIndex(0);
	ui->renderFPSCounter->setStyleSheet("QLCDNumber { background: black; color: #0F0; text: #0F0;}");
	ui->computeFPSCounter->setStyleSheet("QLCDNumber { background: black; color: #0F0; text: #0F0;}");

	x11Process_ = new QProcess(this);
	QObject::connect(x11Process_, SIGNAL(readyReadStandardError()), this, SLOT(logX11()));
	QObject::connect(x11Process_, SIGNAL(error(QProcess::ProcessError)), this, SLOT(handleX11ProcessError(QProcess::ProcessError)));
	QObject::connect(x11Process_, SIGNAL(stateChanged(QProcess::ProcessState)), this, SLOT(handleX11ProcessStateChanged(QProcess::ProcessState)));

	nvidiaSettingsProcess_ = new QProcess(this);
	QObject::connect(nvidiaSettingsProcess_, SIGNAL(readyReadStandardError()), this, SLOT(logNVIDIASettings()));
	QObject::connect(nvidiaSettingsProcess_, SIGNAL(error(QProcess::ProcessError)), this, SLOT(handleNVIDIASettingsProcessError(QProcess::ProcessError)));
	QObject::connect(nvidiaSettingsProcess_, SIGNAL(stateChanged(QProcess::ProcessState)), this, SLOT(handleNVIDIASettingsProcessStateChanged(QProcess::ProcessState)));

	x11vncProcess_ = new QProcess(this);
	QObject::connect(x11vncProcess_, SIGNAL(readyReadStandardError()), this, SLOT(logX11Vnc()));
	QObject::connect(x11vncProcess_, SIGNAL(error(QProcess::ProcessError)), this, SLOT(handleX11VncProcessError(QProcess::ProcessError)));
	QObject::connect(x11vncProcess_, SIGNAL(stateChanged(QProcess::ProcessState)), this, SLOT(handleX11VncProcessStateChanged(QProcess::ProcessState)));
}


MainWindow::~MainWindow()
{
	delete settings_;

	if (algorithmContext_)
		stopDSCP4();

	QObject::disconnect(nvidiaSettingsProcess_, SIGNAL(error(QProcess::ProcessError)), this, SLOT(handleNVIDIASettingsProcessError(QProcess::ProcessError)));
	QObject::disconnect(nvidiaSettingsProcess_, SIGNAL(stateChanged(QProcess::ProcessState)), this, SLOT(handleNVIDIASettingsProcessStateChanged(QProcess::ProcessState)));
	QObject::disconnect(x11vncProcess_, SIGNAL(error(QProcess::ProcessError)), this, SLOT(handleX11VncProcessError(QProcess::ProcessError)));
	QObject::disconnect(x11vncProcess_, SIGNAL(stateChanged(QProcess::ProcessState)), this, SLOT(handleX11VncProcessStateChanged(QProcess::ProcessState)));
	QObject::disconnect(x11Process_, SIGNAL(error(QProcess::ProcessError)), this, SLOT(handleX11ProcessError(QProcess::ProcessError)));
	QObject::disconnect(x11Process_, SIGNAL(stateChanged(QProcess::ProcessState)), this, SLOT(handleX11ProcessStateChanged(QProcess::ProcessState)));

	if (nvidiaSettingsProcess_)
	{
		nvidiaSettingsProcess_->kill();
		delete nvidiaSettingsProcess_;
	}

	if (x11vncProcess_)
	{
		x11vncProcess_->kill();
		delete x11vncProcess_;
	}

	if (x11Process_)
	{
		x11Process_->kill();
		delete x11Process_;
	}

}


void MainWindow::populateModelFiles()
{
	boost::filesystem::path modelsPath(ui->modelsPathLineEdit->text().toStdString());
	boost::filesystem::directory_iterator end_iter;

	QString initialSelection = ui->inputFileComboBox->currentText();
	std::string supportedExtStr;
	assetImporter_.GetExtensionList(supportedExtStr);

	QString supportedExtension = QString::fromStdString(supportedExtStr);
	QStringList extList = supportedExtension.remove("*").split(";");

	ui->inputFileComboBox->clear();

	if (boost::filesystem::exists(modelsPath) && boost::filesystem::is_directory(modelsPath))
	{
		for (boost::filesystem::directory_iterator dir_iter(modelsPath); dir_iter != end_iter; ++dir_iter)
		{
			if (boost::filesystem::is_regular_file(dir_iter->status()))
			{
				for (const auto&t : extList)
				{
					if (dir_iter->path().extension().string() == t.toLower().toStdString())
						ui->inputFileComboBox->addItem(QString::fromStdString(dir_iter->path().filename().string()));
				}
			}
		}
	}

	if (ui->inputFileComboBox->findText(initialSelection) != -1)
		ui->inputFileComboBox->setCurrentIndex(ui->inputFileComboBox->findText(initialSelection));

}

void MainWindow::populateKernelFiles()
{
	boost::filesystem::path kernelsPath(ui->kernelsPathLineEdit->text().toStdString());
	boost::filesystem::directory_iterator end_iter;

	QString initialSelection = ui->openclKernelFileComboBox->currentText();

	ui->openclKernelFileComboBox->clear();
	
	if (boost::filesystem::exists(kernelsPath) && boost::filesystem::is_directory(kernelsPath))
	{
		for (boost::filesystem::directory_iterator dir_iter(kernelsPath); dir_iter != end_iter; ++dir_iter)
		{
			if (boost::filesystem::is_regular_file(dir_iter->status()))
			{
				auto extStr = dir_iter->path().extension().string();
				std::transform(extStr.begin(), extStr.end(), extStr.begin(), ::tolower);
				if (extStr == ".cl")
					ui->openclKernelFileComboBox->addItem(QString::fromStdString(dir_iter->path().filename().string()));
			}
		}
	}

	if (ui->openclKernelFileComboBox->findText(initialSelection) != -1)
		ui->openclKernelFileComboBox->setCurrentIndex(ui->openclKernelFileComboBox->findText(initialSelection));
}

void MainWindow::populateShaderFiles()
{
	boost::filesystem::path shadersPath(ui->shadersPathLineEdit->text().toStdString());
	boost::filesystem::directory_iterator end_iter;

	QString initialSelection = ui->shaderFileNameComboBox->currentText();

	ui->shaderFileNameComboBox->clear();

	if (boost::filesystem::exists(shadersPath) && boost::filesystem::is_directory(shadersPath))
	{
		for (boost::filesystem::directory_iterator dir_iter(shadersPath); dir_iter != end_iter; ++dir_iter)
		{
			if (boost::filesystem::is_regular_file(dir_iter->status()))
			{
				auto extStr = dir_iter->path().extension().string();
				std::transform(extStr.begin(), extStr.end(), extStr.begin(), ::tolower);
				if (extStr == ".frag")
					ui->shaderFileNameComboBox->addItem(QString::fromStdString(dir_iter->path().filename().stem().string()));
			}
		}
	}

	if (ui->shaderFileNameComboBox->findText(initialSelection) != -1)
		ui->shaderFileNameComboBox->setCurrentIndex(ui->shaderFileNameComboBox->findText(initialSelection));
}

QString MainWindow::browseDir()
{
	QFileDialog dialog;
	dialog.setFileMode(QFileDialog::Directory);
	dialog.setOption(QFileDialog::ShowDirsOnly);

	return dialog.getExistingDirectory(this);
}

QString MainWindow::browseFile(const char * title, QString currentDir, const char * filter)
{
	return QFileDialog::getOpenFileName(this,
		tr(title), currentDir, tr(filter));
}

void MainWindow::browseAndSetInputModelFile()
{
	std::string supportedExtensions;
	assetImporter_.GetExtensionList(supportedExtensions);

	QString result = browseFile("3D Object Files", "", supportedExtensions.c_str());

	if (!result.isNull())
	{
		ui->inputFileComboBox->addItem(result);
		ui->inputFileComboBox->setCurrentIndex(ui->inputFileComboBox->findText(result));
	}
}

void MainWindow::browseAndSetOpenCLKernelFile()
{
	QString result = browseFile("OpenCL Kernel Files", "", "OpenCL Code (*.cl)");

	if (!result.isNull())
	{
		ui->openclKernelFileComboBox->addItem(result);
		ui->openclKernelFileComboBox->setCurrentIndex(ui->openclKernelFileComboBox->findText(result));
	}
}

void MainWindow::browseAndSetShaderFileName()
{
	QString result = browseFile("OpenGL Shader File", "", "OpenGL GLSL Code (*.frag;*.vert)");
	
	if (!result.isNull())
	{
		auto name = result.split(".");
		ui->shaderFileNameComboBox->addItem(name[0]);
		ui->shaderFileNameComboBox->setCurrentIndex(ui->shaderFileNameComboBox->findText(name[0]));
	}
}

void MainWindow::browseAndSetInstallPath()
{
	QString dir = browseDir();
	if (!dir.isNull())
		ui->installPathLineEdit->setText(dir);
}

void MainWindow::browseAndSetBinPath()
{
	QString dir = browseDir();
	if (!dir.isNull())
		ui->binPathLineEdit->setText(dir);
}

void MainWindow::browseAndSetModelsPath()
{
	QString dir = browseDir();
	if (!dir.isNull())
		ui->modelsPathLineEdit->setText(dir);
}

void MainWindow::browseAndSetLibPath()
{
	QString dir = browseDir();
	if (!dir.isNull())
		ui->libPathLineEdit->setText(dir);
}

void MainWindow::browseAndSetShadersPath()
{
	QString dir = browseDir();
	if (!dir.isNull())
		ui->shadersPathLineEdit->setText(dir);
}

void MainWindow::browseAndSetKernelsPath()
{
	QString dir = browseDir();
	if (!dir.isNull())
		ui->kernelsPathLineEdit->setText(dir);
}

void MainWindow::startDSCP4()
{
	LOG4CXX_INFO(logger_, "Starting DSCP4 algorithm...")

	ui->dscp4ToggleButton->setEnabled(false);

	disableUnchangeableUI();

	unsigned int aiFlags = 0;

	if (ui->generateNormalsComboBox->currentText() == "Smooth")
		aiFlags |= aiProcess_GenSmoothNormals;
	else if (ui->generateNormalsComboBox->currentText() == "Flat")
		aiFlags |= aiProcess_GenNormals;

	if (ui->triangulateMeshCheckBox->isChecked())
		aiFlags |= aiProcess_Triangulate;

	boost::filesystem::path modelFile(ui->inputFileComboBox->currentText().toStdString());
	if (!boost::filesystem::exists(modelFile))
		modelFile = boost::filesystem::path(ui->modelsPathLineEdit->text().toStdString()) / modelFile;

	LOG4CXX_INFO(logger_, "Loading 3D object file \'" << modelFile.string() << "\'...")

	objectScene_ = assetImporter_.ReadFile(modelFile.string(), aiFlags);

	if (objectScene_ == nullptr || !objectScene_->HasMeshes())
	{
		LOG4CXX_FATAL(logger_, "3D object file does not appear to have any meshes")
		assetImporter_.FreeScene();
		enableUnchangeableUI();
		return;
	}

	LOG4CXX_DEBUG(logger_, "Starting DSCP4 lib")

	auto algorithmOptions = settings_->getAlgorithmOptions();
	auto displayOptions = settings_->getDisplayOptions();
	auto renderOptions = settings_->getRenderOptions();
	int logLevel = ui->verbosityComboBox->currentIndex();

	LOG4CXX_INFO(logger_, "Creating DSCP4 context")

	auto logAppenders = logger_->getAllAppenders();
	auto logAppender = logAppenders[0];

	//algorithmContext_ = dscp4_CreateContextDefault();
	algorithmContext_ = dscp4_CreateContext(renderOptions, algorithmOptions, displayOptions, logLevel, logAppender);

	dscp4_SetEventCallback(algorithmContext_, MainWindow::dscp4RenderEvent, this);

	if (!dscp4_InitRenderer(algorithmContext_))
	{
		LOG4CXX_FATAL(logger_, "Could not initialize DSCP4 lib")
		dscp4_DestroyContext(&algorithmContext_);
		assetImporter_.FreeScene();
		enableUnchangeableUI();
		ui->dscp4ToggleButton->setEnabled(true);
		return;
	}

	renderPreviewScene_ = new QGraphicsScene(this);
	renderPreviewTimer_ = new QTimer(this);
	connect(renderPreviewTimer_, SIGNAL(timeout()), this, SLOT(setRenderPreview()));
	renderPreviewTimer_->start(33);

	for (unsigned int m = 0; m < objectScene_->mNumMeshes; m++)
	{
		// if it has faces, treat as mesh, otherwise as point cloud
		if (objectScene_->mMeshes[m]->HasFaces())
		{
			std::string meshID;
			meshID += std::string("Mesh ") += std::to_string(m);
			LOG4CXX_INFO(logger_, "Found " << meshID << " from 3D object file '" << modelFile.string() << "'")
			LOG4CXX_INFO(logger_, meshID << " has " << objectScene_->mMeshes[m]->mNumVertices << " vertices")
			if (objectScene_->mMeshes[m]->HasNormals())
			{
				LOG4CXX_INFO(logger_, meshID << " has normals")
			}
			else
			{
				LOG4CXX_WARN(logger_, meshID << " does not have normals, lighting effects will look fucked up")
			}

			LOG4CXX_INFO(logger_, meshID << " has " << objectScene_->mMeshes[m]->mNumFaces << " faces")

			if (objectScene_->mMeshes[m]->HasVertexColors(0))
			{
				LOG4CXX_INFO(logger_, meshID << " has vertex colors")
					dscp4_AddMesh(algorithmContext_, meshID.c_str(), objectScene_->mMeshes[m]->mFaces[0].mNumIndices, objectScene_->mMeshes[m]->mNumVertices, (float*)objectScene_->mMeshes[m]->mVertices, (float*)objectScene_->mMeshes[m]->mNormals, (float*)objectScene_->mMeshes[m]->mColors[0]);
			}
			else
			{
				LOG4CXX_WARN(logger_, meshID << " does not have vertex colors--it may look dull")
					dscp4_AddMesh(algorithmContext_, meshID.c_str(), objectScene_->mMeshes[m]->mFaces[0].mNumIndices, objectScene_->mMeshes[m]->mNumVertices, (float*)objectScene_->mMeshes[m]->mVertices, (float*)objectScene_->mMeshes[m]->mNormals);
			}

		}
		else
		{
			LOG4CXX_DEBUG(logger_, "Found mesh " << m << " with no faces.  Treating vertecies as point cloud")
		}
	}

	enableControlsUI();

	QObject::disconnect(ui->dscp4ToggleButton, SIGNAL(clicked()), this, SLOT(startDSCP4()));
	QObject::connect(ui->dscp4ToggleButton, SIGNAL(clicked()), this, SLOT(stopDSCP4()));
	ui->dscp4ToggleButton->setText("Stop DSCP4");
	ui->dscp4ToggleButton->setEnabled(true);
}

void MainWindow::dscp4RenderEvent(callback_type_t evt, void * parent, void * userData)
{
	if (evt == DSCP4_CALLBACK_TYPE_NEW_FRAME)
	{
		((MainWindow*)parent)->pushNewRenderPreviewFrame(*(frame_data_t*)userData);
	}
}

void MainWindow::pushNewRenderPreviewFrame(frame_data_t & frameData)
{
	std::unique_lock<std::mutex> dataLock(renderPreviewDataMutex_);
	frameData_ = frameData;
	haveNewFrame_ = true;
	dataLock.unlock();
	haveNewFrameCV_.notify_all();
}

void MainWindow::stopDSCP4()
{
	renderPreviewTimer_->stop();
	ui->spinModelCheckBox->setChecked(false);
	disableControlsUI();

	dscp4_DeinitRenderer(algorithmContext_);
	dscp4_DestroyContext(&algorithmContext_);

	assetImporter_.FreeScene();

	enableUnchangeableUI();

	delete renderPreviewScene_;
	renderPreviewScene_ = nullptr;

	haveNewFrame_ = false;

	ui->renderFPSCounter->display(0);
	ui->computeFPSCounter->display(0);

	QObject::connect(ui->dscp4ToggleButton, SIGNAL(clicked()), this, SLOT(startDSCP4()));
	QObject::disconnect(ui->dscp4ToggleButton, SIGNAL(clicked()), this, SLOT(stopDSCP4()));
	ui->dscp4ToggleButton->setText("Start DSCP4");
}

void MainWindow::startX11()
{
	x11Command_ = "X ";
	x11Command_.append(ui->x11DisplayEnvLineEdit->text()); // set the display env. variable
	x11Command_.append(" -br"); // set black background
	x11Command_.append(" +xinerama"); //enable xinerama
	x11Process_->start(x11Command_);
}

void MainWindow::startNVIDIASettings()
{
	nvidiaSettingsCommand_ = "nvidia-settings -V all -c ";
	nvidiaSettingsCommand_.append(ui->x11DisplayEnvLineEdit->text());
	nvidiaSettingsProcess_->start(nvidiaSettingsCommand_);
}


void MainWindow::startX11Vnc()
{
	x11vncCommand_ = "x11vnc -display ";
	x11vncCommand_.append(ui->x11DisplayEnvLineEdit->text()); // set the display env. variable
	x11vncProcess_->start(x11vncCommand_);
}

void MainWindow::stopX11()
{
	x11Process_->terminate();
}

void MainWindow::stopX11Vnc()
{
	x11vncProcess_->terminate();
}

void MainWindow::stopNVIDIASettings()
{
	nvidiaSettingsProcess_->terminate();
}

void MainWindow::dumpFramebufferToPNG()
{
	dscp4_SaveFrameBufferToPNG(algorithmContext_);
}

void MainWindow::forceRedraw()
{
	if (algorithmContext_)
		dscp4_ForceRedraw(algorithmContext_);
}

void MainWindow::translateModelX(int x)
{
	camera_t cam = { 0 };
	dscp4_GetCameraView(algorithmContext_, &cam);

	cam.eye.x = x * 0.001f;
	cam.center.x = x * 0.001f;

	dscp4_SetCameraView(algorithmContext_, cam);
}

void MainWindow::translateModelY(int y)
{
	camera_t cam = { 0 };
	dscp4_GetCameraView(algorithmContext_, &cam);

	cam.eye.y = y * 0.001f;
	cam.center.y = y * 0.001f;

	dscp4_SetCameraView(algorithmContext_, cam);
}

void MainWindow::translateModelZ(int z)
{
	camera_t cam = { 0 };
	dscp4_GetCameraView(algorithmContext_, &cam);

	cam.eye.z = 0.5 + z * 0.001f;
	cam.center.z = z * 0.001f;

	dscp4_SetCameraView(algorithmContext_, cam);
}

void MainWindow::rotateModelX(int x)
{
	dscp4_SetRotateViewAngleX(algorithmContext_, 180.f*x*0.001);
}

void MainWindow::rotateModelY(int y)
{
	dscp4_SetRotateViewAngleY(algorithmContext_, 180.f*y*0.001);
}

void MainWindow::rotateModelZ(int z)
{
	dscp4_SetRotateViewAngleZ(algorithmContext_, 180.0f*z*0.001);
}

void MainWindow::setPlaneZOffset(int zOffset)
{
	dscp4_SetPlaneZOffset(algorithmContext_, zOffset * 0.001);
}

void MainWindow::setSpinOn(bool spinOn)
{
	if (spinOn)
	{
		ui->yRotateHorizontalSlider->setValue(0);
		ui->yRotateHorizontalSlider->setDisabled(true);
	}
	else
		ui->yRotateHorizontalSlider->setDisabled(false);

	dscp4_SetSpinOn(algorithmContext_, spinOn);
	forceRedraw();
}

void MainWindow::enableUnchangeableUI()
{
	for (QWidget* var : unchangeables_)
	{
		var->setEnabled(true);
	}
}

void MainWindow::disableUnchangeableUI()
{
	for (QWidget* var : unchangeables_)
	{
		var->setEnabled(false);
	}
}

void MainWindow::enableControlsUI()
{
	for (QWidget* var : dscp4Controls_)
	{
		var->setEnabled(true);
	}
}

void MainWindow::disableControlsUI()
{
	ui->xTranslateHorizontalSlider->setValue(0);
	ui->yTranslateHorizontalSlider->setValue(0);
	ui->zTranslateHorizontalSlider->setValue(0);
	ui->xRotateHorizontalSlider->setValue(0);
	ui->yRotateHorizontalSlider->setValue(0);
	ui->zRotateHorizontalSlider->setValue(0);
	ui->redColorGainHorizontalSlider->setValue(100);
	ui->greenColorGainHorizontalSlider->setValue(100);
	ui->blueColorGainHorizontalSlider->setValue(100);
	ui->spinModelCheckBox->setChecked(false);
	for (QWidget* var : dscp4Controls_)
	{
		var->setEnabled(false);
	}
}

void MainWindow::setRenderPreview()
{
	if (haveNewFrame_)
	{
		std::unique_lock<std::mutex> updateFrameLock(renderPreviewDataMutex_);
		QImage theImage((unsigned char *)frameData_.buffer, ui->numWafelsSpinBox->value(), ui->numScanlinesSpinBox->value(), QImage::Format_RGBA8888);
		renderPreviewImage_ = QPixmap::fromImage(theImage.mirrored(false, true));
		renderPreviewScene_->clear();
		renderPreviewScene_->addPixmap(renderPreviewImage_.scaled(ui->renderPreviewGraphicsView->width()-4, ui->renderPreviewGraphicsView->height()-4, Qt::IgnoreAspectRatio, Qt::SmoothTransformation));
		renderPreviewScene_->setSceneRect(QRect(0, 0, ui->renderPreviewGraphicsView->width()-4, ui->renderPreviewGraphicsView->height()-4));
		this->ui->renderPreviewGraphicsView->setScene(renderPreviewScene_);
		this->ui->computeFPSCounter->display(frameData_.compute_fps);
		this->ui->renderFPSCounter->display(frameData_.render_fps);
		haveNewFrame_ = false;
		updateFrameLock.unlock();
	}
}


void MainWindow::logX11()
{
	ui->x11LogTextEdit->insertPlainText(x11Process_->readAllStandardError());
}

void MainWindow::logNVIDIASettings()
{
	ui->nvidiaSettingsLogTextEdit->insertPlainText(nvidiaSettingsProcess_->readAllStandardError());
}

void MainWindow::logX11Vnc()
{
	ui->x11vncLogTextEdit->insertPlainText(x11vncProcess_->readAllStandardError());
}

void MainWindow::clearLog()
{
	switch (ui->whichLogTabWidget->currentIndex())
	{
	case 0:
		ui->dscp4LogTextEdit->clear();
		break;
	case 1:
		ui->x11LogTextEdit->clear();
		break;
	case 2:
		ui->nvidiaSettingsLogTextEdit->clear();
		break;
	default:
		break;
	}
}

void MainWindow::handleLogMessage(QString logMessage)
{
	QMessageBox messageBox;
	
	if (logMessage.contains("ERROR"))
	{
		QMessageBox::critical(this, "Error in libdscp4", logMessage.replace("ERROR\t", ""), QMessageBox::StandardButton::Ok);
	}
	else if (logMessage.contains("WARN"))
	{
		QMessageBox::warning(this, "Warning in libdscp4", logMessage.replace("WARN \t", ""), QMessageBox::StandardButton::Ok);
	}
}

void MainWindow::handleX11ProcessError(QProcess::ProcessError processError)
{
	switch (processError)
	{
		case QProcess::ProcessError::FailedToStart:
		{
			QString message("Please check your X11 settings in /etc/X11/xorg.conf.\n");
			message.append("Launch command used: \"");
			message.append(x11Command_);
			message.append("\"");
			QMessageBox::critical(this, "Error Launching X11", message, QMessageBox::Ok);
			break;
		}
		default:
			QMessageBox::critical(this, "Error in X11 Process", "X11 process has crashed or stopped responding.");
			break;
	}
}

void MainWindow::handleX11ProcessStateChanged(QProcess::ProcessState processState)
{
	switch (processState)
	{
	case QProcess::ProcessState::Running:
	{
		QObject::disconnect(ui->x11ToggleButton, SIGNAL(clicked()), this, SLOT(startX11()));
		QObject::connect(ui->x11ToggleButton, SIGNAL(clicked()), this, SLOT(stopX11()));
		ui->x11ToggleButton->setText("Stop X-Windows");
		ui->nvidiaSettingsToggleButton->setEnabled(true);
		ui->x11vncToggleButton->setEnabled(true);

		break;
	}
	case QProcess::ProcessState::NotRunning:
	{
		QObject::disconnect(ui->x11ToggleButton, SIGNAL(clicked()), this, SLOT(stopX11()));

		if (x11vncProcess_)
		{
			x11vncProcess_->terminate();
		}

		if (nvidiaSettingsProcess_)
		{
			nvidiaSettingsProcess_->terminate();
		}

		QObject::connect(ui->x11ToggleButton, SIGNAL(clicked()), this, SLOT(startX11()));
		ui->x11ToggleButton->setText("Start X-Windows");

		ui->nvidiaSettingsToggleButton->setEnabled(false);
		ui->x11vncToggleButton->setEnabled(false);
	}
	default:
		break;
	}
}

void MainWindow::handleNVIDIASettingsProcessError(QProcess::ProcessError processError)
{
	switch (processError)
	{
	case QProcess::ProcessError::FailedToStart:
	{
		QString message("Please check your NVIDIA driver install\n");
		message.append("Launch command used: \"");
		message.append(nvidiaSettingsCommand_);
		message.append("\"");
		QMessageBox::critical(this, "Error Launching NVIDIA Settings", message, QMessageBox::Ok);
		break;
	}
	default:
		QMessageBox::critical(this, "Error in NVIDIA Settings Process", "NVIDIA Settings process has crashed or stopped responding.");
		break;
	}
}

void MainWindow::handleNVIDIASettingsProcessStateChanged(QProcess::ProcessState processState)
{
	switch (processState)
	{
	case QProcess::ProcessState::Running:
	{
		QObject::disconnect(ui->nvidiaSettingsToggleButton, SIGNAL(clicked()), this, SLOT(startNVIDIASettings()));
		QObject::connect(ui->nvidiaSettingsToggleButton, SIGNAL(clicked()), this, SLOT(stopNVIDIASettings()));
		ui->nvidiaSettingsToggleButton->setText("Stop NVIDIA Settings");
		break;
	}
	case QProcess::ProcessState::NotRunning:
	{
		QObject::connect(ui->nvidiaSettingsToggleButton, SIGNAL(clicked()), this, SLOT(startNVIDIASettings()));
		QObject::disconnect(ui->nvidiaSettingsToggleButton, SIGNAL(clicked()), this, SLOT(stopNVIDIASettings()));
		ui->nvidiaSettingsToggleButton->setText("Start NVIDIA Settings");
		break;
	}
	default:
		break;
	}
}

void MainWindow::handleX11VncProcessError(QProcess::ProcessError processError)
{
	switch (processError)
	{
	case QProcess::ProcessError::FailedToStart:
	{
		QString message("Please check that x11vnc is installed on your system");
		message.append("Launch command used: \"");
		message.append(x11vncCommand_);
		message.append("\"");
		QMessageBox::critical(this, "Error Launching x11vnc", message, QMessageBox::Ok);
		break;
	}
	default:
		QMessageBox::critical(this, "Error in x11vnc Process", "x11vnc process has crashed or stopped responding.");
		break;
	}
}

void MainWindow::handleX11VncProcessStateChanged(QProcess::ProcessState processState)
{
	switch (processState)
	{
	case QProcess::ProcessState::Running:
	{
			QObject::disconnect(ui->x11vncToggleButton, SIGNAL(clicked()), this, SLOT(startX11Vnc()));
			QObject::connect(ui->x11vncToggleButton, SIGNAL(clicked()), this, SLOT(stopX11Vnc()));
			ui->x11vncToggleButton->setText("Stop X-VNC");
			break;
	}
	case QProcess::ProcessState::NotRunning:
	{
			QObject::disconnect(ui->x11vncToggleButton, SIGNAL(clicked()), this, SLOT(stopX11Vnc()));
			QObject::connect(ui->x11vncToggleButton, SIGNAL(clicked()), this, SLOT(startX11Vnc()));
			ui->x11vncToggleButton->setText("Start X-VNC");
			break;
	}
	default:
		break;
	}
}