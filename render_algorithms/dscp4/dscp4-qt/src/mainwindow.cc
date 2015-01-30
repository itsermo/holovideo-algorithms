#include "mainwindow.h"
#include "ui_mainwindow.h"

MainWindow::MainWindow(QWidget *parent)
: MainWindow(nullptr, parent)
{

}

MainWindow::MainWindow(QDSCP4Settings* settings, QWidget *parent) : QMainWindow(parent), ui(new Ui::MainWindow), settings_(settings)
{
	ui->setupUi(this);

	QObject::connect(ui->inputFileComboBox, SIGNAL(currentTextChanged(QString)), settings_, SLOT(setObjectFileName(QString)));

	QObject::connect(settings_, SIGNAL(objectFileNameChanged(QString)), ui->inputFileComboBox, SLOT(setCurrentText(QString)));

	QObject::connect(settings_, SIGNAL(generateNormalsChanged(QString)), ui->generateNormalsComboBox, SLOT(setCurrentText(QString)));
	QObject::connect(settings_, SIGNAL(triangulateMeshChanged(bool)), ui->triangulateMeshCheckBox, SLOT(setChecked(bool)));
	QObject::connect(settings_, SIGNAL(installPathChanged(QString)), ui->installPathLineEdit, SLOT(setText(QString)));
	QObject::connect(settings_, SIGNAL(binPathChanged(QString)), ui->binPathLineEdit, SLOT(setText(QString)));
	QObject::connect(settings_, SIGNAL(libPathChanged(QString)), ui->libPathLineEdit, SLOT(setText(QString)));
	QObject::connect(settings_, SIGNAL(modelsPathChanged(QString)), ui->modelsPathLineEdit, SLOT(setText(QString)));
	QObject::connect(settings_, SIGNAL(shadersPathChanged(QString)), ui->shadersPathLineEdit, SLOT(setText(QString)));
	QObject::connect(settings_, SIGNAL(kernelsPathChanged(QString)), ui->kernelsPathLineEdit, SLOT(setText(QString)));



	settings->populateSettings();
}


MainWindow::~MainWindow()
{

}
