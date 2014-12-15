#ifndef REMOTEQT_H
#define REMOTEQT_H

#include <QtGui/QWidget>
#include "ui_RemoteQT.h"
#include <QMainWindow>
#include "JSharedMemory.h"
#include "JDisplayState.h"
//#include "JPCLSharedmemAdaptor.h"
//#include "JVRPNClient.h"

class RemoteQT : public QMainWindow
{
    Q_OBJECT

public:
    RemoteQT(QWidget *parent = 0);
    ~RemoteQT();

public slots:
	void powerOnClicked();
	void powerOffClicked();

	void spinToggleChanged(int);

	/*
    void echoChanged(int);
    void validatorChanged(int);
    void alignmentChanged(int);
    void inputMaskChanged(int);
    void accessChanged(int);
    */
    void sliderTxChanged(int);
    void sliderTyChanged(int);
    void sliderTzChanged(int);

    void sliderRxChanged(int);
    void sliderRyChanged(int);
    void sliderRzChanged(int);

    void sliderGainChanged(int);

    void sliderScaleChanged(double);

    void startxClicked();
    void nvidiaSettingsClicked();

    void kinectClicked();

    void startClicked();
    void killClicked();

    void runPointsClicked();

    void flatCheck1Clicked(bool);
    void flatCheck2Clicked(bool);

    void sliderFlatDepth1Changed(int);
    void sliderFlatDepth2Changed(int);

    void spinStep();
    void statusUpdate();

    void debugSwitchToggled(bool);
    void shaderModeChanged(int);
    void runWafelClicked();

    void debugVariableChanged(int);
    void debugVarScaleChanged(double);
    void debugAdjustChanged(int);

    void updateDebugVariableValue();

    void loadCloud();
    void loadSequence();
    void endSequence();
    void expUpdate();

    void trackerUpdate();

    //saving state
    void saveCurrentState();
    void resetDefaults();
	void loadSavedState();
private:
    void sliderChanged(int val, int index);
    void init();

    void keyPressEvent( QKeyEvent *k );

    Ui::RemoteQTClass ui;
    QLineEdit *echoLineEdit;
    QLineEdit *validatorLineEdit;
    QLineEdit *alignmentLineEdit;
    QLineEdit *inputMaskLineEdit;
    QLineEdit *accessLineEdit;
    JSharedMemory *sharedmem;
    JSharedMemory *sharedstatus;
    JDisplayState statecopy;
    JDisplayStatus statuscopy;
    QTimer *spinTimer;
    QTimer *statusTimer;
    QTimer *expTimer;
    QTimer *trackerTimer;
    //JPCLSharedmemAdaptor *importer;
    int lastRenderer;

    bool lastButton;

    //JVRPNClient *vrpn;

};

#endif // REMOTEQT_H
