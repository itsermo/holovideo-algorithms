#include "QDSCP4Settings.h"

QDSCP4Settings::QDSCP4Settings() : QDSCP4Settings(0, nullptr)
{


}

QDSCP4Settings::QDSCP4Settings(int argc, const char **argv) :
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

	int x = 0;
}

//void QDSCP4Settings::setValue(int value)
//{
//
//}

//void QDSCP4Settings::valueChanged(int newValue)
//{
//
//}