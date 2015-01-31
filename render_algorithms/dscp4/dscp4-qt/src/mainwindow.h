#ifndef mainwindow_h
#define mainwindow_h

#ifdef DSCP4_HAVE_LOG4CXX
#include <log4cxx/logger.h>
#include <log4cxx/patternlayout.h>
#include <log4cxx/writerappender.h>
#include <log4cxx/basicconfigurator.h>
#else
#define LOG4CXX_TRACE(logger, expression)    
#define LOG4CXX_DEBUG(logger, expression)    
#define LOG4CXX_INFO(logger, expression)   
#define LOG4CXX_WARN(logger, expression)    
#define LOG4CXX_ERROR(logger, expression)    
#define LOG4CXX_FATAL(logger, expression) 
#endif

#include <QMainWindow>
#include <QScopedPointer>
#include "QDSCP4Settings.h"
#include <assimp/Importer.hpp>

namespace Ui
{
    class MainWindow;
}

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
	MainWindow(QWidget *parent = 0);
	MainWindow(QDSCP4Settings* settings, QWidget *parent = 0);
    virtual ~MainWindow();

public slots:
	void populateModelFiles();
	void populateKernelFiles();
	void populateShaderFiles();

	void browseAndSetInputModelFile();
	void browseAndSetOpenCLKernelFile();
	void browseAndSetInstallPath();
	void browseAndSetBinPath();
	void browseAndSetModelsPath();
	void browseAndSetLibPath();
	void browseAndSetShadersPath();
	void browseAndSetKernelsPath();
	void browseAndSetShaderFileName();

	void startDSCP4();
	void stopDSCP4();

private:

	QString browseDir();
	QString browseFile(const char * title, QString currentDir, const char * filter);

	QDSCP4Settings * settings_;
    QScopedPointer<Ui::MainWindow> ui;

	Assimp::Importer assetImporter_;
	
	dscp4_context_t algorithmContext_;

	log4cxx::LoggerPtr logger_;

};

#endif
