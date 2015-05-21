#ifndef mainwindow_h
#define mainwindow_h

#include <thread>
#include <condition_variable>
#include <mutex>
#include <atomic>

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
#include <qprocess.h>
#include <qgraphicsscene.h>

#include <assimp/Importer.hpp>      
#include <assimp/scene.h>           
#include <assimp/postprocess.h> 

namespace Ui
{
    class MainWindow;
}

class QDSCP4Settings;

class MainWindow : public QMainWindow {
    Q_OBJECT

public:
	MainWindow(QWidget *parent = 0);
	MainWindow(int argc, const char ** argv, QWidget *parent = 0);
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
	void startX11();
	void stopX11();
	void startX11Vnc();
	void stopX11Vnc();

	void startNVIDIASettings();

	void enableUnchangeableUI();
	void disableUnchangeableUI();
	void enableControlsUI();
	void disableControlsUI();

	void dumpFramebufferToPNG();
	void forceRedraw();

	void translateModelX(int x);
	void translateModelY(int y);
	void translateModelZ(int z);

	void rotateModelX(int x);
	void rotateModelY(int y);
	void rotateModelZ(int z);

	void setPlaneZOffset(int zOffset);

	void setSpinOn(bool spinOn);
	void setRenderPreview();
	void pushNewRenderPreviewFrame(frame_data_t & frameData);
	static void dscp4RenderEvent(callback_type_t evt, void * parent, void * userData);

	void logX11();
	void logNVIDIASettings();
	void logX11Vnc();

	void clearLog();
	void handleLogMessage(QString logMessage);

signals:
	void dscp4IsRunningChanged(bool isRunning);
	void dscoUsRunningChangedRev(bool isRunning);

	void x11IsRunningChanged(bool isRunning);
	void nvidiaSettingsIsRunningChanged(bool isRunning);
	void x11vncIsRunningChanged(bool isRunning);

private:

	QString browseDir();
	QString browseFile(const char * title, QString currentDir, const char * filter);

	QDSCP4Settings * settings_;
    QScopedPointer<Ui::MainWindow> ui;

	dscp4_context_t algorithmContext_;

	log4cxx::LoggerPtr logger_;

	// Turn on all controls when DSCP4 is running
	QList<QWidget*> dscp4Controls_;

	// Turn off all UI elements below while DSCP4 is running
	QList<QWidget*> unchangeables_;

	Assimp::Importer assetImporter_;
	const aiScene* objectScene_;

	QProcess * x11Process_;
	QProcess * nvidiaSettingsProcess_;
	QProcess * x11vncProcess_;

	QGraphicsScene *renderPreviewScene_;
	QPixmap renderPreviewImage_;

	frame_data_t frameData_;

	QTimer *renderPreviewTimer_;
	std::condition_variable haveNewFrameCV_;
	std::atomic<bool> haveNewFrame_;
	std::mutex renderPreviewDataMutex_;
};

#endif
