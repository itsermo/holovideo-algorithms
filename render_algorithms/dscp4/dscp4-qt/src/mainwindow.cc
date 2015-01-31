#include "mainwindow.h"
#include "ui_mainwindow.h"
#include "QLogAppender.h"

#include <boost/filesystem.hpp>
#include <qfiledialog.h>
#include <boost/tokenizer.hpp>

MainWindow::MainWindow(QWidget *parent)
: MainWindow(nullptr, parent)
{

}

MainWindow::MainWindow(QDSCP4Settings* settings, QWidget *parent) : QMainWindow(parent), ui(new Ui::MainWindow), settings_(settings)
{

#ifdef DSCP4_HAVE_LOG4CXX

	logger_ = log4cxx::Logger::getLogger("edu.mit.media.obmg.holovideo.dscp4");

#ifdef WIN32
	log4cxx::PatternLayoutPtr logLayoutPtr = new log4cxx::PatternLayout(L"%-5p\t%m%n");
#else
	log4cxx::PatternLayoutPtr logLayoutPtr = new log4cxx::PatternLayout("%-5p\t%m%n");
#endif
	log4cxx::helpers::ObjectPtrT<log4cxx::QLogAppender> logAppenderPtr = new log4cxx::QLogAppender(logLayoutPtr);
	log4cxx::BasicConfigurator::configure(logAppenderPtr);
#endif

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
	QObject::connect(ui->clearLogButton, SIGNAL(clicked()), ui->logTextEdit, SLOT(clear()));

#ifdef DSCP4_HAVE_LOG4CXX
	QObject::connect(logAppenderPtr, SIGNAL(gotNewLogMessage(QString)), ui->logTextEdit, SLOT(append(QString)));
#endif

	LOG4CXX_INFO(logger_, "Logger initialized")

	ui->tabWidget->setCurrentIndex(0);
}


MainWindow::~MainWindow()
{

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
				std::transform(extStr.begin(), extStr.end(), extStr.begin(), std::tolower);
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
				std::transform(extStr.begin(), extStr.end(), extStr.begin(), std::tolower);
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
		auto name = result.splitRef(".");
		ui->shaderFileNameComboBox->addItem(name[0].toString());
		ui->shaderFileNameComboBox->setCurrentIndex(ui->shaderFileNameComboBox->findText(name[0].toString()));
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

}

void MainWindow::stopDSCP4()
{

}