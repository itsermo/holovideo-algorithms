#ifndef mainwindow_h
#define mainwindow_h

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

};

#endif
