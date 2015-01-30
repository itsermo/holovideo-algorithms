#ifndef mainwindow_h
#define mainwindow_h

#include <QMainWindow>
#include <QScopedPointer>
#include "QDSCP4Settings.h"

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

private:
	QDSCP4Settings * settings_;
    QScopedPointer<Ui::MainWindow> ui;
};

#endif
