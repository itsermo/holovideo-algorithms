#include <QtGui>
#include <QWidget>
#include "RemoteQT.h"
#include "JSharedMemory.h"
#include "JHolovideoDisplay.h"
#include <stdlib.h>

#include <QSettings>

#define GUI_SETTINGS_FILENAME "RemoteQTSettings.ini"

void RemoteQT::saveCurrentState() {
	//Currently this doesn't work. Maybe by QApplication's session manager?

    QSettings settings( GUI_SETTINGS_FILENAME, QSettings::IniFormat );
	settings.beginGroup( "GuiState" );
		int version = 0;
		settings.setValue( "Saved", saveState(version) );
	settings.endGroup();
}

void RemoteQT::loadSavedState() {
	//Currently this doesn't work. Maybe by QApplication's session manager?

	QSettings settings( GUI_SETTINGS_FILENAME, QSettings::IniFormat );
	settings.beginGroup( "GuiState" );
	int version = 0;
	bool ok = restoreState( settings.value( "Saved", QByteArray() ).toByteArray(), version );
	settings.endGroup();
	printf("attempt to restore settings result was %s\n", ok?"successful":"unsuccessful");
}

void RemoteQT::init() {
	ui.setupUi(this);


	//set up shared memory & state

	//this is state we control. Don't bother synchronizing. we own it.
	if(sharedmem) delete sharedmem;
	sharedmem = new JSharedMemory(sizeof(JDisplayState), ALL_STATE_KEY);

	statecopy.filename1[0] = 0;
	statecopy.filename2[0] = 0;
	statecopy.xpos = 0;
	statecopy.ypos = 0;
	statecopy.zpos = 0;
	statecopy.xrot = 0;
	statecopy.yrot = 0;
	statecopy.zrot = 0;
	statecopy.gain = 1.0;
	statecopy.rendermode1 = 0;
	statecopy.rendermode2 = 0;
	statecopy.flatdepth1 = 1;
	statecopy.flatdepth2 = 1;
	statecopy.viewmask = -1;
	statecopy.shaderMode = 0;
	statecopy.scale = 1.0;



        for(int i=0;i<JSHAREDSTATE_DEBUG_VARS;i++)
        {
        //    statecopy.debug[i] = 0; //leave these alone so app can init on launch?
        }

        ui.VarSelectSpinbox->setMaximum(JSHAREDSTATE_DEBUG_VARS-1);

	sharedmem->write(&statecopy);

	//this is state from individual processes
	if(sharedstatus) delete sharedstatus;
	sharedstatus = new JSharedMemory(sizeof(JDisplayStatus), ALL_STATUS_KEY);

	//this is point clouds we upload
	//importer = new JPCLSharedmemAdaptor();
	lastRenderer = -1;
}


//Currently this doesn't work. Maybe by QApplication's session manager?
void RemoteQT::resetDefaults(){
	//set up canned UI
	//qDeleteAll(findChildren<QWidget*>());
/*
	QList<QWidget *> widgets = findChildren<QWidget *>();
	foreach(QWidget * widget, widgets)
	{
		if(widget->isWidgetType())
			delete widget;
	}
*/

	//QWidget* mview = centralWidget();
	if ( this->layout() != NULL ) {
		QLayoutItem* item;
			while ( ( item = this->layout()->takeAt( 0 ) ) != NULL ){
				delete item->widget();
				delete item;
			}
		delete this->layout();
	}
	init();

}

RemoteQT::RemoteQT(QWidget *parent)
    : QMainWindow(parent)
	,sharedmem(NULL)
	,sharedstatus(NULL)
{
	//resetDefaults();
	init();


/*	if(vrpn) {
		delete vrpn;
	}
	vrpn = new JVRPNClient();
	vrpn->connectToZspace("obmgzspace");*/


    spinTimer = new QTimer(this);
    connect(spinTimer, SIGNAL(timeout()), this, SLOT(spinStep()));


    //status of cloud and processes updated in GUI
    statusTimer = new QTimer(this);
    connect(statusTimer, SIGNAL(timeout()), this, SLOT(statusUpdate()));
    int statusUpdateMs=300;
    statusTimer->start(statusUpdateMs);


    //new head/stylus tracking values fetched from network
    trackerTimer = new QTimer(this);
    connect(trackerTimer, SIGNAL(timeout()), this, SLOT(trackerUpdate()));
    int trackerUpdateMs=5;
    trackerTimer->start(trackerUpdateMs);


    //experiment and display sequencer updated
    expTimer = new QTimer(this);
    connect(expTimer, SIGNAL(timeout()), this, SLOT(expUpdate()));
    int expUpdateMs=10; //about twice per frame
    expTimer->start(expUpdateMs);
}

void RemoteQT::powerOnClicked()
{
	system("./holovideo-enable.sh");
}

void RemoteQT::powerOffClicked()
{
	system("./holovideo-disable.sh");
}

void RemoteQT::spinToggleChanged(int value)
{

	int timerintervalms = 30;
	if (value)
		spinTimer->start(timerintervalms);
	else
		spinTimer->stop();
}

void RemoteQT::spinStep()
{
        int speed = 1;
	QSlider* s = qFindChild<QSlider*>(this, "rotSliderY");
        int newval = s->value() + speed;
        if (newval > s->maximum())
        {
            newval = s->minimum();
        }
        s->setValue(newval);

}

void RemoteQT::statusUpdate()
{
	QLineEdit* e0 = qFindChild<QLineEdit*>(this, "StatusDisplay0");
	QLineEdit* e1 = qFindChild<QLineEdit*>(this, "StatusDisplay1");
	QLineEdit* e2 = qFindChild<QLineEdit*>(this, "StatusDisplay2");

	QTextBrowser* sc = qFindChild<QTextBrowser*>(this, "StatusDisplayCloud");

	QSpinBox* spin = qFindChild<QSpinBox*>(this, "spinBoxSkipCloudPoints");

	if(sharedstatus->getDataCopyIfUnlocked(&statuscopy))
	{
		if (e0) e0->setText(statuscopy.statusMessage[0]);
		if (e1) e1->setText(statuscopy.statusMessage[1]);
		if (e2) e2->setText(statuscopy.statusMessage[2]);
	}

//	if(importer && sc && spin) {
//		sc->setHtml(QString::fromStdString(importer->printCloudPreviewString(8,spin->value())));
//	}
//
//
//	if(importer && ui.LineEditSequenceMessage) {
//		//ui.LineEditSequenceMessage->setText(QString::fromStdString(importer->getSequenceStatusString()));
//	}


}
void RemoteQT::debugSwitchToggled(bool st)
{
	QObject* sender = QObject::sender();
        int boxnum = 1;
	QCheckBox *box;
     if (box = qobject_cast<QCheckBox *>(sender))
     {
    	 boxnum = box->text().toInt();
     }
     if (st)
     {	//disable bit
    	 statecopy.viewmask &= ~(1<<(boxnum-1));
     }
     else
     {
    	 statecopy.viewmask |= 1<<(boxnum-1);
     }

 	sharedmem->write(&statecopy);
}

void RemoteQT::sliderTxChanged(int value)
{
	sliderChanged(value,0);
}
void RemoteQT::sliderTyChanged(int value)
{
	sliderChanged(value,1);
}
void RemoteQT::sliderTzChanged(int value)
{
	sliderChanged(value,2);
}
void RemoteQT::sliderRxChanged(int value)
{
	sliderChanged(value,3);
}
void RemoteQT::sliderRyChanged(int value)
{
	sliderChanged(value,4);
}
void RemoteQT::sliderRzChanged(int value)
{
	sliderChanged(value,5);
}

void RemoteQT::sliderGainChanged(int value)
{
	sliderChanged(value,6);
}

void RemoteQT::sliderScaleChanged(double value)
{
	statecopy.scale = value;
	sharedmem->write(&statecopy);
}

void RemoteQT::sliderChanged(int value, int index)
{
	float angle = value/100.0*360.0;
	float pos = 5.0 - 10.0*value/100.0;

	switch(index)
	{
	case 0:
		statecopy.xpos = pos;
		break;
	case 1:
		statecopy.ypos = pos;
		break;
	case 2:
		statecopy.zpos = pos;
		break;
	case 3:
		statecopy.xrot = angle;
		break;
	case 4:
		statecopy.yrot = angle;
		break;
	case 5:
		statecopy.zrot = angle;
		break;
	case 6:
		statecopy.gain = value/5.0;
		break;
	}
	sharedmem->write(&statecopy);
}

void RemoteQT::flatCheck1Clicked(bool b)
{
	statecopy.rendermode1 = b?1:0;
	sharedmem->write(&statecopy);

}
void RemoteQT::flatCheck2Clicked(bool b)
{
	statecopy.rendermode2 = b?1:0;
	sharedmem->write(&statecopy);
}

void RemoteQT::sliderFlatDepth1Changed(int i)
{
	statecopy.flatdepth1 = i/1.0;
	sharedmem->write(&statecopy);
}
void RemoteQT::sliderFlatDepth2Changed(int i)
{
	statecopy.flatdepth2 = i/1.0;
	sharedmem->write(&statecopy);

}

void RemoteQT::startxClicked()
{
	system("X &");
	system("./nvidia-framelock-enable.sh &");
}

void RemoteQT::nvidiaSettingsClicked()
{
	system("nvidia-settings -c :0.0 &");
}

void RemoteQT::kinectClicked() {
	system("killall KinectServer;sleep 2;xterm -e KinectServer &");
}

void RemoteQT::startClicked()
{
	system("killall pointCloudCudaHolo");
	system("killall holodepth");

	QString cmd = tr("killall ripgen-fbo; ./ripgen-fbo-start.sh ") + ui.filenameComboBox->currentText() + tr("&");
	char c[512];
	strcpy(c,cmd.toAscii().data());
	system(c);
}

void RemoteQT::killClicked()
{
	lastRenderer = 0;
	system("killall ripgen-fbo &");
	system("killall holodepth &");
	system("killall pointCloudCudaHolo &");

}

void RemoteQT::shaderModeChanged(int mode)
{
	statecopy.shaderMode = mode;
	sharedmem->write(&statecopy);
}

void RemoteQT::runWafelClicked(void)
{
	lastRenderer = 1;
	system("killall ripgen-fbo");
	system("killall holodepth");
	system("killall pointCloudCudaHolo");
	system("./holodepth-start.sh");
}

void RemoteQT::runPointsClicked(void)
{
	lastRenderer = 2;
	system("killall ripgen-fbo");
	system("killall holodepth");
	system("killall pointCloudCudaHolo");
	system("~/Dropbox/Holovideo/scripts/cudaHolo3");
}

RemoteQT::~RemoteQT()
{
	delete sharedmem;
	printf("exiting RemoteQT\n");

}

void RemoteQT::debugVariableChanged(int i)
{

    float newval = statecopy.debug[i];

    QDoubleSpinBox* spinscale = qFindChild<QDoubleSpinBox*>(this, "VarScaleSpinbox");
    QSlider* slide = qFindChild<QSlider*>(this, "VarAdjustSlider");

    double newscale = fabs(newval);
    int newslider = newval>0?100:-100;
    if(newscale == 0)
    {
        newscale = 1;
        newslider=0;
    }

    //fprintf(stderr,"setting new scale to %g\n", newscale);
    spinscale->setValue(newscale);
    slide->setValue(newslider);


    updateDebugVariableValue();
}

void RemoteQT::debugVarScaleChanged(double d)
{

    updateDebugVariableValue();
}

void RemoteQT::debugAdjustChanged(int i)
{
    updateDebugVariableValue();
}

void RemoteQT::updateDebugVariableValue()
{
    int var = 0;
    float scaleval = 1.0;
    float sliderval = 0;

    QSpinBox* spinvar = qFindChild<QSpinBox*>(this, "VarSelectSpinbox");
    QDoubleSpinBox* spinscale = qFindChild<QDoubleSpinBox*>(this, "VarScaleSpinbox");
    QSlider* slide = qFindChild<QSlider*>(this, "VarAdjustSlider");

    QLineEdit* valuebox = qFindChild<QLineEdit*>(this,"VarValueEdit");

    var = spinvar->value();
    scaleval = spinscale->value();
    sliderval = slide->value()/100.0;

    float newval = scaleval*float(sliderval);

    valuebox->setText(QString::number(newval,'g'));
    //fprintf(stderr,"setting text to reflect new value: %f\n", newval);
    statecopy.debug[var] = newval;
    sharedmem->write(&statecopy);
}

void RemoteQT::loadCloud()
{

	//importer->loadCloudFromFile("/home/holo/testscene.pcd");
//	float scale = ui.doubleSpinBoxCloudLoadScale->value();
//	float clipsize = 1e99;
//	if(ui.checkBoxClipCircle->isChecked()) {
//		clipsize = 0.0225;
//	}
//	importer->loadCloudFromFile(ui.comboBoxCloudFilename->currentText().toStdString(), scale, clipsize, -1);
//	importer->setPointingMode(false);
//	importer->sliceSceneToShmem();
}


void RemoteQT::loadSequence()
{
//	//get subject ID, run number
//	std::string  logname;
//	logname.append(ui.comboBoxSequenceFilename->currentText().toStdString());
//	logname.append("_MK2");
//	//sprintf(logname,"%s.log", ui.comboBoxSequenceFilename->currentText().toStdString());
//	//importer->setupExperiment(ui.comboBoxSequenceFilename->currentText().toStdString(), logname, ui.spinBoxSubjectId->value(),ui.spinBoxRunNumber->value() );
//	if(lastRenderer > 0) importer->setupTweaksForRenderer(lastRenderer);
}

void RemoteQT::endSequence()
{
	//importer->endExperiment();
}

void RemoteQT::expUpdate() {

//	if(importer) {
//		//importer->updateExperiment(0); //get experiment to update state (including preloading/display of models.
////		if(vrpn) {
////			//importer->updateStylus(vrpn->stylusx, vrpn->stylusy, vrpn->stylusz);
////		}
//	}
	//in special debug mode, continuously re-project scene
//	if(statecopy.shaderMode > 3) {
//		importer->sliceSceneToShmem(statecopy.zpos/10.0);
//	}
}

void RemoteQT::trackerUpdate() {
//	if(vrpn) {
//		vrpn->update();
//		if(vrpn->button0 != lastButton) {
//			if(vrpn->button0) {
//				printf("got stylus click. Sending as 8 key\n");fflush(stdout);
//				if(importer) {
//					//importer->updateExperiment('8');
//				}
//			}
//			lastButton = vrpn->button0;
//		}
//	}
}

void RemoteQT::keyPressEvent( QKeyEvent *k )
{

	switch (k->key()) {
	case Qt::Key_8:
		printf("got 8 key\n");fflush(stdout);
		//send this to sequencer
		//importer->updateExperiment('8');
		break;
	case Qt::Key_2:
		printf("got 2 key\n");fflush(stdout);
		//importer->updateExperiment('2');
		//send this to sequencer
		break;
	case Qt::Key_R:
//		if(vrpn) {
//			delete vrpn;
//		}
//		vrpn = new JVRPNClient();
//		vrpn->connectToZspace("obmgzspace");
		break;
	}
}

