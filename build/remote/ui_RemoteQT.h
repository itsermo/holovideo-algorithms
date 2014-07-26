/********************************************************************************
** Form generated from reading UI file 'RemoteQT.ui'
**
** Created by: Qt User Interface Compiler version 4.8.6
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_REMOTEQT_H
#define UI_REMOTEQT_H

#include <QtCore/QVariant>
#include <QtGui/QAction>
#include <QtGui/QApplication>
#include <QtGui/QButtonGroup>
#include <QtGui/QCheckBox>
#include <QtGui/QComboBox>
#include <QtGui/QDoubleSpinBox>
#include <QtGui/QFrame>
#include <QtGui/QGridLayout>
#include <QtGui/QGroupBox>
#include <QtGui/QHBoxLayout>
#include <QtGui/QHeaderView>
#include <QtGui/QLabel>
#include <QtGui/QLineEdit>
#include <QtGui/QPushButton>
#include <QtGui/QSlider>
#include <QtGui/QSpinBox>
#include <QtGui/QTabWidget>
#include <QtGui/QTextBrowser>
#include <QtGui/QWidget>

QT_BEGIN_NAMESPACE

class Ui_RemoteQTClass
{
public:
    QGroupBox *groupBox_2;
    QPushButton *pushButton_2;
    QWidget *horizontalLayoutWidget_3;
    QHBoxLayout *horizontalLayout_3;
    QLabel *label_12;
    QPushButton *pushButton;
    QPushButton *pushButton_5;
    QPushButton *pushButton_6;
    QWidget *horizontalLayoutWidget_4;
    QHBoxLayout *horizontalLayout_4;
    QLabel *label_13;
    QComboBox *filenameComboBox;
    QTabWidget *tabWidget;
    QWidget *tab;
    QGroupBox *ModelPositionGroup;
    QWidget *gridLayoutWidget;
    QGridLayout *gridLayout;
    QSlider *posSliderY;
    QSlider *posSliderZ;
    QSlider *rotSliderZ;
    QLabel *label;
    QSlider *posSliderX;
    QSlider *rotSliderX;
    QSlider *rotSliderY;
    QLabel *label_2;
    QCheckBox *spinCheckbox;
    QGroupBox *groupBox;
    QSlider *horizontalSlider;
    QGroupBox *groupBox_3;
    QWidget *gridLayoutWidget_2;
    QGridLayout *gridLayout_2;
    QCheckBox *flatCheck1;
    QCheckBox *flatCheck2;
    QSlider *horizontalSlider_3;
    QLabel *label_3;
    QLabel *label_4;
    QSlider *horizontalSlider_2;
    QGroupBox *groupBox_9;
    QDoubleSpinBox *doubleSpinBoxScale;
    QWidget *tab_2;
    QGroupBox *groupBox_4;
    QGroupBox *groupBox_5;
    QWidget *gridLayoutWidget_4;
    QGridLayout *gridLayout_4;
    QCheckBox *checkBox_01;
    QCheckBox *checkBox_02;
    QCheckBox *checkBox_03;
    QCheckBox *checkBox_04;
    QCheckBox *checkBox_05;
    QCheckBox *checkBox_06;
    QCheckBox *checkBox_07;
    QCheckBox *checkBox_08;
    QCheckBox *checkBox_09;
    QCheckBox *checkBox_10;
    QCheckBox *checkBox_11;
    QCheckBox *checkBox_12;
    QCheckBox *checkBox_13;
    QCheckBox *checkBox_14;
    QCheckBox *checkBox_15;
    QCheckBox *checkBox_16;
    QSpinBox *spinBox;
    QLabel *label_5;
    QSpinBox *VarSelectSpinbox;
    QSlider *VarAdjustSlider;
    QLineEdit *VarValueEdit;
    QDoubleSpinBox *VarScaleSpinbox;
    QFrame *line_3;
    QLabel *label_6;
    QLabel *label_7;
    QLabel *label_8;
    QLabel *label_9;
    QWidget *tab_6;
    QPushButton *pushButton_loadPcd;
    QGroupBox *groupBox_8;
    QTextBrowser *StatusDisplayCloud;
    QSpinBox *spinBoxSkipCloudPoints;
    QLabel *label_10;
    QComboBox *comboBoxCloudFilename;
    QDoubleSpinBox *doubleSpinBoxCloudLoadScale;
    QLabel *label_16;
    QLabel *label_17;
    QCheckBox *checkBoxClipCircle;
    QWidget *tab_7;
    QComboBox *comboBoxSequenceFilename;
    QPushButton *pushButton_runSequence;
    QSpinBox *spinBoxSubjectId;
    QLabel *label_11;
    QLabel *label_14;
    QLabel *label_15;
    QSpinBox *spinBoxRunNumber;
    QLineEdit *LineEditSequenceMessage;
    QPushButton *pushButton_endSequence;
    QGroupBox *groupBox_7;
    QWidget *horizontalLayoutWidget;
    QHBoxLayout *horizontalLayout;
    QPushButton *pushButton_saveState;
    QPushButton *pushButton_loadState;
    QPushButton *pushButton_resetState;
    QGroupBox *groupBox_10;
    QPushButton *pushButton_3;
    QPushButton *pushButton_4;
    QGroupBox *powerGroupBox;
    QPushButton *PowerOnButton;
    QPushButton *PowerOffButton;
    QTabWidget *tabWidget_2;
    QWidget *tab_3;
    QLineEdit *StatusDisplay0;
    QWidget *tab_4;
    QLineEdit *StatusDisplay1;
    QWidget *tab_5;
    QLineEdit *StatusDisplay2;

    void setupUi(QWidget *RemoteQTClass)
    {
        if (RemoteQTClass->objectName().isEmpty())
            RemoteQTClass->setObjectName(QString::fromUtf8("RemoteQTClass"));
        RemoteQTClass->resize(506, 729);
        groupBox_2 = new QGroupBox(RemoteQTClass);
        groupBox_2->setObjectName(QString::fromUtf8("groupBox_2"));
        groupBox_2->setGeometry(QRect(10, 80, 471, 101));
        pushButton_2 = new QPushButton(groupBox_2);
        pushButton_2->setObjectName(QString::fromUtf8("pushButton_2"));
        pushButton_2->setEnabled(true);
        pushButton_2->setGeometry(QRect(380, 47, 81, 30));
        horizontalLayoutWidget_3 = new QWidget(groupBox_2);
        horizontalLayoutWidget_3->setObjectName(QString::fromUtf8("horizontalLayoutWidget_3"));
        horizontalLayoutWidget_3->setGeometry(QRect(20, 20, 326, 41));
        horizontalLayout_3 = new QHBoxLayout(horizontalLayoutWidget_3);
        horizontalLayout_3->setSpacing(6);
        horizontalLayout_3->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_3->setObjectName(QString::fromUtf8("horizontalLayout_3"));
        horizontalLayout_3->setContentsMargins(0, 0, 0, 0);
        label_12 = new QLabel(horizontalLayoutWidget_3);
        label_12->setObjectName(QString::fromUtf8("label_12"));

        horizontalLayout_3->addWidget(label_12);

        pushButton = new QPushButton(horizontalLayoutWidget_3);
        pushButton->setObjectName(QString::fromUtf8("pushButton"));

        horizontalLayout_3->addWidget(pushButton);

        pushButton_5 = new QPushButton(horizontalLayoutWidget_3);
        pushButton_5->setObjectName(QString::fromUtf8("pushButton_5"));

        horizontalLayout_3->addWidget(pushButton_5);

        pushButton_6 = new QPushButton(horizontalLayoutWidget_3);
        pushButton_6->setObjectName(QString::fromUtf8("pushButton_6"));

        horizontalLayout_3->addWidget(pushButton_6);

        horizontalLayoutWidget_4 = new QWidget(groupBox_2);
        horizontalLayoutWidget_4->setObjectName(QString::fromUtf8("horizontalLayoutWidget_4"));
        horizontalLayoutWidget_4->setGeometry(QRect(20, 70, 351, 29));
        horizontalLayout_4 = new QHBoxLayout(horizontalLayoutWidget_4);
        horizontalLayout_4->setSpacing(6);
        horizontalLayout_4->setContentsMargins(11, 11, 11, 11);
        horizontalLayout_4->setObjectName(QString::fromUtf8("horizontalLayout_4"));
        horizontalLayout_4->setContentsMargins(0, 0, 0, 0);
        label_13 = new QLabel(horizontalLayoutWidget_4);
        label_13->setObjectName(QString::fromUtf8("label_13"));

        horizontalLayout_4->addWidget(label_13);

        filenameComboBox = new QComboBox(horizontalLayoutWidget_4);
        filenameComboBox->setObjectName(QString::fromUtf8("filenameComboBox"));
        filenameComboBox->setMinimumSize(QSize(122, 0));
        filenameComboBox->setEditable(true);

        horizontalLayout_4->addWidget(filenameComboBox);

        tabWidget = new QTabWidget(RemoteQTClass);
        tabWidget->setObjectName(QString::fromUtf8("tabWidget"));
        tabWidget->setGeometry(QRect(10, 190, 471, 411));
        tab = new QWidget();
        tab->setObjectName(QString::fromUtf8("tab"));
        ModelPositionGroup = new QGroupBox(tab);
        ModelPositionGroup->setObjectName(QString::fromUtf8("ModelPositionGroup"));
        ModelPositionGroup->setGeometry(QRect(10, 10, 451, 141));
        gridLayoutWidget = new QWidget(ModelPositionGroup);
        gridLayoutWidget->setObjectName(QString::fromUtf8("gridLayoutWidget"));
        gridLayoutWidget->setGeometry(QRect(10, 20, 371, 116));
        gridLayout = new QGridLayout(gridLayoutWidget);
        gridLayout->setSpacing(6);
        gridLayout->setContentsMargins(11, 11, 11, 11);
        gridLayout->setObjectName(QString::fromUtf8("gridLayout"));
        gridLayout->setContentsMargins(0, 0, 0, 0);
        posSliderY = new QSlider(gridLayoutWidget);
        posSliderY->setObjectName(QString::fromUtf8("posSliderY"));
        posSliderY->setMinimumSize(QSize(0, 22));
        posSliderY->setValue(50);
        posSliderY->setOrientation(Qt::Horizontal);

        gridLayout->addWidget(posSliderY, 4, 0, 1, 1);

        posSliderZ = new QSlider(gridLayoutWidget);
        posSliderZ->setObjectName(QString::fromUtf8("posSliderZ"));
        posSliderZ->setMinimumSize(QSize(0, 22));
        posSliderZ->setValue(50);
        posSliderZ->setOrientation(Qt::Horizontal);

        gridLayout->addWidget(posSliderZ, 5, 0, 1, 1);

        rotSliderZ = new QSlider(gridLayoutWidget);
        rotSliderZ->setObjectName(QString::fromUtf8("rotSliderZ"));
        rotSliderZ->setMinimumSize(QSize(0, 22));
        rotSliderZ->setOrientation(Qt::Horizontal);

        gridLayout->addWidget(rotSliderZ, 5, 1, 1, 1);

        label = new QLabel(gridLayoutWidget);
        label->setObjectName(QString::fromUtf8("label"));

        gridLayout->addWidget(label, 0, 0, 1, 1);

        posSliderX = new QSlider(gridLayoutWidget);
        posSliderX->setObjectName(QString::fromUtf8("posSliderX"));
        posSliderX->setMinimumSize(QSize(0, 22));
        posSliderX->setValue(50);
        posSliderX->setOrientation(Qt::Horizontal);

        gridLayout->addWidget(posSliderX, 2, 0, 1, 1);

        rotSliderX = new QSlider(gridLayoutWidget);
        rotSliderX->setObjectName(QString::fromUtf8("rotSliderX"));
        rotSliderX->setMinimumSize(QSize(0, 22));
        rotSliderX->setOrientation(Qt::Horizontal);

        gridLayout->addWidget(rotSliderX, 2, 1, 1, 1);

        rotSliderY = new QSlider(gridLayoutWidget);
        rotSliderY->setObjectName(QString::fromUtf8("rotSliderY"));
        rotSliderY->setMinimumSize(QSize(0, 22));
        rotSliderY->setOrientation(Qt::Horizontal);

        gridLayout->addWidget(rotSliderY, 4, 1, 1, 1);

        label_2 = new QLabel(gridLayoutWidget);
        label_2->setObjectName(QString::fromUtf8("label_2"));

        gridLayout->addWidget(label_2, 0, 1, 1, 1);

        spinCheckbox = new QCheckBox(ModelPositionGroup);
        spinCheckbox->setObjectName(QString::fromUtf8("spinCheckbox"));
        spinCheckbox->setGeometry(QRect(390, 90, 181, 22));
        groupBox = new QGroupBox(tab);
        groupBox->setObjectName(QString::fromUtf8("groupBox"));
        groupBox->setGeometry(QRect(10, 150, 221, 50));
        horizontalSlider = new QSlider(groupBox);
        horizontalSlider->setObjectName(QString::fromUtf8("horizontalSlider"));
        horizontalSlider->setGeometry(QRect(10, 20, 191, 31));
        horizontalSlider->setValue(20);
        horizontalSlider->setOrientation(Qt::Horizontal);
        groupBox_3 = new QGroupBox(tab);
        groupBox_3->setObjectName(QString::fromUtf8("groupBox_3"));
        groupBox_3->setGeometry(QRect(10, 210, 451, 111));
        gridLayoutWidget_2 = new QWidget(groupBox_3);
        gridLayoutWidget_2->setObjectName(QString::fromUtf8("gridLayoutWidget_2"));
        gridLayoutWidget_2->setGeometry(QRect(20, 20, 361, 91));
        gridLayout_2 = new QGridLayout(gridLayoutWidget_2);
        gridLayout_2->setSpacing(6);
        gridLayout_2->setContentsMargins(11, 11, 11, 11);
        gridLayout_2->setObjectName(QString::fromUtf8("gridLayout_2"));
        gridLayout_2->setContentsMargins(0, 0, 0, 0);
        flatCheck1 = new QCheckBox(gridLayoutWidget_2);
        flatCheck1->setObjectName(QString::fromUtf8("flatCheck1"));

        gridLayout_2->addWidget(flatCheck1, 1, 0, 1, 1);

        flatCheck2 = new QCheckBox(gridLayoutWidget_2);
        flatCheck2->setObjectName(QString::fromUtf8("flatCheck2"));

        gridLayout_2->addWidget(flatCheck2, 2, 0, 1, 1);

        horizontalSlider_3 = new QSlider(gridLayoutWidget_2);
        horizontalSlider_3->setObjectName(QString::fromUtf8("horizontalSlider_3"));
        horizontalSlider_3->setMinimum(-99);
        horizontalSlider_3->setValue(1);
        horizontalSlider_3->setOrientation(Qt::Horizontal);

        gridLayout_2->addWidget(horizontalSlider_3, 2, 1, 1, 1);

        label_3 = new QLabel(gridLayoutWidget_2);
        label_3->setObjectName(QString::fromUtf8("label_3"));

        gridLayout_2->addWidget(label_3, 0, 0, 1, 1);

        label_4 = new QLabel(gridLayoutWidget_2);
        label_4->setObjectName(QString::fromUtf8("label_4"));

        gridLayout_2->addWidget(label_4, 0, 1, 1, 1);

        horizontalSlider_2 = new QSlider(gridLayoutWidget_2);
        horizontalSlider_2->setObjectName(QString::fromUtf8("horizontalSlider_2"));
        horizontalSlider_2->setMinimum(-99);
        horizontalSlider_2->setValue(1);
        horizontalSlider_2->setOrientation(Qt::Horizontal);

        gridLayout_2->addWidget(horizontalSlider_2, 1, 1, 1, 1);

        groupBox_9 = new QGroupBox(tab);
        groupBox_9->setObjectName(QString::fromUtf8("groupBox_9"));
        groupBox_9->setGeometry(QRect(240, 150, 221, 50));
        doubleSpinBoxScale = new QDoubleSpinBox(groupBox_9);
        doubleSpinBoxScale->setObjectName(QString::fromUtf8("doubleSpinBoxScale"));
        doubleSpinBoxScale->setGeometry(QRect(10, 20, 81, 25));
        doubleSpinBoxScale->setDecimals(4);
        doubleSpinBoxScale->setMinimum(0);
        doubleSpinBoxScale->setMaximum(1000);
        doubleSpinBoxScale->setSingleStep(0.1);
        doubleSpinBoxScale->setValue(1);
        tabWidget->addTab(tab, QString());
        tab_2 = new QWidget();
        tab_2->setObjectName(QString::fromUtf8("tab_2"));
        groupBox_4 = new QGroupBox(tab_2);
        groupBox_4->setObjectName(QString::fromUtf8("groupBox_4"));
        groupBox_4->setGeometry(QRect(0, 10, 431, 231));
        groupBox_4->setFlat(false);
        groupBox_4->setCheckable(false);
        groupBox_4->setChecked(false);
        groupBox_5 = new QGroupBox(groupBox_4);
        groupBox_5->setObjectName(QString::fromUtf8("groupBox_5"));
        groupBox_5->setGeometry(QRect(20, 20, 401, 80));
        gridLayoutWidget_4 = new QWidget(groupBox_5);
        gridLayoutWidget_4->setObjectName(QString::fromUtf8("gridLayoutWidget_4"));
        gridLayoutWidget_4->setGeometry(QRect(10, 20, 412, 60));
        gridLayout_4 = new QGridLayout(gridLayoutWidget_4);
        gridLayout_4->setSpacing(6);
        gridLayout_4->setContentsMargins(11, 11, 11, 11);
        gridLayout_4->setObjectName(QString::fromUtf8("gridLayout_4"));
        gridLayout_4->setContentsMargins(0, 0, 0, 0);
        checkBox_01 = new QCheckBox(gridLayoutWidget_4);
        checkBox_01->setObjectName(QString::fromUtf8("checkBox_01"));

        gridLayout_4->addWidget(checkBox_01, 0, 0, 1, 1);

        checkBox_02 = new QCheckBox(gridLayoutWidget_4);
        checkBox_02->setObjectName(QString::fromUtf8("checkBox_02"));

        gridLayout_4->addWidget(checkBox_02, 0, 1, 1, 1);

        checkBox_03 = new QCheckBox(gridLayoutWidget_4);
        checkBox_03->setObjectName(QString::fromUtf8("checkBox_03"));

        gridLayout_4->addWidget(checkBox_03, 0, 2, 1, 1);

        checkBox_04 = new QCheckBox(gridLayoutWidget_4);
        checkBox_04->setObjectName(QString::fromUtf8("checkBox_04"));

        gridLayout_4->addWidget(checkBox_04, 0, 3, 1, 1);

        checkBox_05 = new QCheckBox(gridLayoutWidget_4);
        checkBox_05->setObjectName(QString::fromUtf8("checkBox_05"));

        gridLayout_4->addWidget(checkBox_05, 0, 4, 1, 1);

        checkBox_06 = new QCheckBox(gridLayoutWidget_4);
        checkBox_06->setObjectName(QString::fromUtf8("checkBox_06"));

        gridLayout_4->addWidget(checkBox_06, 0, 5, 1, 1);

        checkBox_07 = new QCheckBox(gridLayoutWidget_4);
        checkBox_07->setObjectName(QString::fromUtf8("checkBox_07"));

        gridLayout_4->addWidget(checkBox_07, 0, 6, 1, 1);

        checkBox_08 = new QCheckBox(gridLayoutWidget_4);
        checkBox_08->setObjectName(QString::fromUtf8("checkBox_08"));

        gridLayout_4->addWidget(checkBox_08, 0, 7, 1, 1);

        checkBox_09 = new QCheckBox(gridLayoutWidget_4);
        checkBox_09->setObjectName(QString::fromUtf8("checkBox_09"));

        gridLayout_4->addWidget(checkBox_09, 1, 0, 1, 1);

        checkBox_10 = new QCheckBox(gridLayoutWidget_4);
        checkBox_10->setObjectName(QString::fromUtf8("checkBox_10"));

        gridLayout_4->addWidget(checkBox_10, 1, 1, 1, 1);

        checkBox_11 = new QCheckBox(gridLayoutWidget_4);
        checkBox_11->setObjectName(QString::fromUtf8("checkBox_11"));

        gridLayout_4->addWidget(checkBox_11, 1, 2, 1, 1);

        checkBox_12 = new QCheckBox(gridLayoutWidget_4);
        checkBox_12->setObjectName(QString::fromUtf8("checkBox_12"));

        gridLayout_4->addWidget(checkBox_12, 1, 3, 1, 1);

        checkBox_13 = new QCheckBox(gridLayoutWidget_4);
        checkBox_13->setObjectName(QString::fromUtf8("checkBox_13"));

        gridLayout_4->addWidget(checkBox_13, 1, 4, 1, 1);

        checkBox_14 = new QCheckBox(gridLayoutWidget_4);
        checkBox_14->setObjectName(QString::fromUtf8("checkBox_14"));

        gridLayout_4->addWidget(checkBox_14, 1, 5, 1, 1);

        checkBox_15 = new QCheckBox(gridLayoutWidget_4);
        checkBox_15->setObjectName(QString::fromUtf8("checkBox_15"));

        gridLayout_4->addWidget(checkBox_15, 1, 6, 1, 1);

        checkBox_16 = new QCheckBox(gridLayoutWidget_4);
        checkBox_16->setObjectName(QString::fromUtf8("checkBox_16"));

        gridLayout_4->addWidget(checkBox_16, 1, 7, 1, 1);

        spinBox = new QSpinBox(groupBox_4);
        spinBox->setObjectName(QString::fromUtf8("spinBox"));
        spinBox->setGeometry(QRect(30, 130, 55, 27));
        spinBox->setMinimum(-99);
        label_5 = new QLabel(groupBox_4);
        label_5->setObjectName(QString::fromUtf8("label_5"));
        label_5->setGeometry(QRect(20, 110, 101, 17));
        VarSelectSpinbox = new QSpinBox(groupBox_4);
        VarSelectSpinbox->setObjectName(QString::fromUtf8("VarSelectSpinbox"));
        VarSelectSpinbox->setGeometry(QRect(30, 190, 57, 25));
        VarAdjustSlider = new QSlider(groupBox_4);
        VarAdjustSlider->setObjectName(QString::fromUtf8("VarAdjustSlider"));
        VarAdjustSlider->setGeometry(QRect(180, 190, 160, 22));
        VarAdjustSlider->setMinimum(-100);
        VarAdjustSlider->setMaximum(100);
        VarAdjustSlider->setOrientation(Qt::Horizontal);
        VarAdjustSlider->setTickPosition(QSlider::NoTicks);
        VarAdjustSlider->setTickInterval(1);
        VarValueEdit = new QLineEdit(groupBox_4);
        VarValueEdit->setObjectName(QString::fromUtf8("VarValueEdit"));
        VarValueEdit->setEnabled(false);
        VarValueEdit->setGeometry(QRect(360, 190, 61, 22));
        VarScaleSpinbox = new QDoubleSpinBox(groupBox_4);
        VarScaleSpinbox->setObjectName(QString::fromUtf8("VarScaleSpinbox"));
        VarScaleSpinbox->setGeometry(QRect(110, 190, 62, 25));
        VarScaleSpinbox->setMinimum(-10000);
        VarScaleSpinbox->setMaximum(10000);
        VarScaleSpinbox->setValue(1);
        line_3 = new QFrame(groupBox_4);
        line_3->setObjectName(QString::fromUtf8("line_3"));
        line_3->setGeometry(QRect(30, 160, 381, 16));
        line_3->setFrameShape(QFrame::HLine);
        line_3->setFrameShadow(QFrame::Sunken);
        label_6 = new QLabel(groupBox_4);
        label_6->setObjectName(QString::fromUtf8("label_6"));
        label_6->setGeometry(QRect(20, 170, 62, 16));
        label_7 = new QLabel(groupBox_4);
        label_7->setObjectName(QString::fromUtf8("label_7"));
        label_7->setGeometry(QRect(100, 170, 62, 16));
        label_8 = new QLabel(groupBox_4);
        label_8->setObjectName(QString::fromUtf8("label_8"));
        label_8->setGeometry(QRect(360, 170, 62, 16));
        label_9 = new QLabel(groupBox_4);
        label_9->setObjectName(QString::fromUtf8("label_9"));
        label_9->setGeometry(QRect(230, 170, 62, 16));
        label_9->setAlignment(Qt::AlignCenter);
        tabWidget->addTab(tab_2, QString());
        tab_6 = new QWidget();
        tab_6->setObjectName(QString::fromUtf8("tab_6"));
        pushButton_loadPcd = new QPushButton(tab_6);
        pushButton_loadPcd->setObjectName(QString::fromUtf8("pushButton_loadPcd"));
        pushButton_loadPcd->setGeometry(QRect(370, 290, 92, 27));
        groupBox_8 = new QGroupBox(tab_6);
        groupBox_8->setObjectName(QString::fromUtf8("groupBox_8"));
        groupBox_8->setGeometry(QRect(10, 0, 431, 271));
        StatusDisplayCloud = new QTextBrowser(groupBox_8);
        StatusDisplayCloud->setObjectName(QString::fromUtf8("StatusDisplayCloud"));
        StatusDisplayCloud->setGeometry(QRect(10, 30, 321, 231));
        QFont font;
        font.setPointSize(10);
        StatusDisplayCloud->setFont(font);
        StatusDisplayCloud->setAcceptDrops(false);
        StatusDisplayCloud->setFrameShadow(QFrame::Sunken);
        StatusDisplayCloud->setLineWidth(1);
        StatusDisplayCloud->setVerticalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        StatusDisplayCloud->setHorizontalScrollBarPolicy(Qt::ScrollBarAlwaysOff);
        spinBoxSkipCloudPoints = new QSpinBox(groupBox_8);
        spinBoxSkipCloudPoints->setObjectName(QString::fromUtf8("spinBoxSkipCloudPoints"));
        spinBoxSkipCloudPoints->setGeometry(QRect(336, 60, 91, 25));
        spinBoxSkipCloudPoints->setMaximum(9999999);
        label_10 = new QLabel(groupBox_8);
        label_10->setObjectName(QString::fromUtf8("label_10"));
        label_10->setGeometry(QRect(340, 40, 81, 16));
        QFont font1;
        font1.setPointSize(9);
        label_10->setFont(font1);
        comboBoxCloudFilename = new QComboBox(tab_6);
        comboBoxCloudFilename->setObjectName(QString::fromUtf8("comboBoxCloudFilename"));
        comboBoxCloudFilename->setGeometry(QRect(10, 290, 261, 26));
        comboBoxCloudFilename->setEditable(true);
        comboBoxCloudFilename->setFrame(true);
        doubleSpinBoxCloudLoadScale = new QDoubleSpinBox(tab_6);
        doubleSpinBoxCloudLoadScale->setObjectName(QString::fromUtf8("doubleSpinBoxCloudLoadScale"));
        doubleSpinBoxCloudLoadScale->setGeometry(QRect(290, 290, 62, 25));
        doubleSpinBoxCloudLoadScale->setDecimals(3);
        doubleSpinBoxCloudLoadScale->setMinimum(-1e+08);
        doubleSpinBoxCloudLoadScale->setMaximum(1e+08);
        doubleSpinBoxCloudLoadScale->setSingleStep(0.01);
        doubleSpinBoxCloudLoadScale->setValue(1);
        label_16 = new QLabel(tab_6);
        label_16->setObjectName(QString::fromUtf8("label_16"));
        label_16->setGeometry(QRect(20, 276, 62, 16));
        label_16->setFont(font);
        label_17 = new QLabel(tab_6);
        label_17->setObjectName(QString::fromUtf8("label_17"));
        label_17->setGeometry(QRect(280, 275, 91, 16));
        label_17->setFont(font);
        checkBoxClipCircle = new QCheckBox(tab_6);
        checkBoxClipCircle->setObjectName(QString::fromUtf8("checkBoxClipCircle"));
        checkBoxClipCircle->setGeometry(QRect(280, 330, 111, 20));
        tabWidget->addTab(tab_6, QString());
        tab_7 = new QWidget();
        tab_7->setObjectName(QString::fromUtf8("tab_7"));
        comboBoxSequenceFilename = new QComboBox(tab_7);
        comboBoxSequenceFilename->setObjectName(QString::fromUtf8("comboBoxSequenceFilename"));
        comboBoxSequenceFilename->setGeometry(QRect(30, 230, 261, 26));
        comboBoxSequenceFilename->setEditable(true);
        comboBoxSequenceFilename->setInsertPolicy(QComboBox::InsertAtTop);
        comboBoxSequenceFilename->setFrame(true);
        pushButton_runSequence = new QPushButton(tab_7);
        pushButton_runSequence->setObjectName(QString::fromUtf8("pushButton_runSequence"));
        pushButton_runSequence->setGeometry(QRect(330, 260, 114, 32));
        spinBoxSubjectId = new QSpinBox(tab_7);
        spinBoxSubjectId->setObjectName(QString::fromUtf8("spinBoxSubjectId"));
        spinBoxSubjectId->setGeometry(QRect(30, 290, 57, 25));
        label_11 = new QLabel(tab_7);
        label_11->setObjectName(QString::fromUtf8("label_11"));
        label_11->setGeometry(QRect(20, 275, 71, 10));
        label_14 = new QLabel(tab_7);
        label_14->setObjectName(QString::fromUtf8("label_14"));
        label_14->setGeometry(QRect(20, 210, 81, 16));
        label_15 = new QLabel(tab_7);
        label_15->setObjectName(QString::fromUtf8("label_15"));
        label_15->setGeometry(QRect(140, 273, 91, 16));
        spinBoxRunNumber = new QSpinBox(tab_7);
        spinBoxRunNumber->setObjectName(QString::fromUtf8("spinBoxRunNumber"));
        spinBoxRunNumber->setGeometry(QRect(140, 290, 57, 25));
        LineEditSequenceMessage = new QLineEdit(tab_7);
        LineEditSequenceMessage->setObjectName(QString::fromUtf8("LineEditSequenceMessage"));
        LineEditSequenceMessage->setGeometry(QRect(20, 340, 431, 22));
        QFont font2;
        font2.setPointSize(8);
        LineEditSequenceMessage->setFont(font2);
        LineEditSequenceMessage->setAcceptDrops(true);
        LineEditSequenceMessage->setFrame(true);
        LineEditSequenceMessage->setReadOnly(true);
        pushButton_endSequence = new QPushButton(tab_7);
        pushButton_endSequence->setObjectName(QString::fromUtf8("pushButton_endSequence"));
        pushButton_endSequence->setGeometry(QRect(330, 300, 114, 32));
        tabWidget->addTab(tab_7, QString());
        groupBox_7 = new QGroupBox(RemoteQTClass);
        groupBox_7->setObjectName(QString::fromUtf8("groupBox_7"));
        groupBox_7->setEnabled(false);
        groupBox_7->setGeometry(QRect(10, 720, 471, 80));
        horizontalLayoutWidget = new QWidget(groupBox_7);
        horizontalLayoutWidget->setObjectName(QString::fromUtf8("horizontalLayoutWidget"));
        horizontalLayoutWidget->setGeometry(QRect(30, 20, 351, 33));
        horizontalLayout = new QHBoxLayout(horizontalLayoutWidget);
        horizontalLayout->setSpacing(6);
        horizontalLayout->setContentsMargins(11, 11, 11, 11);
        horizontalLayout->setObjectName(QString::fromUtf8("horizontalLayout"));
        horizontalLayout->setContentsMargins(0, 0, 0, 0);
        pushButton_saveState = new QPushButton(horizontalLayoutWidget);
        pushButton_saveState->setObjectName(QString::fromUtf8("pushButton_saveState"));

        horizontalLayout->addWidget(pushButton_saveState);

        pushButton_loadState = new QPushButton(horizontalLayoutWidget);
        pushButton_loadState->setObjectName(QString::fromUtf8("pushButton_loadState"));

        horizontalLayout->addWidget(pushButton_loadState);

        pushButton_resetState = new QPushButton(horizontalLayoutWidget);
        pushButton_resetState->setObjectName(QString::fromUtf8("pushButton_resetState"));

        horizontalLayout->addWidget(pushButton_resetState);

        groupBox_10 = new QGroupBox(RemoteQTClass);
        groupBox_10->setObjectName(QString::fromUtf8("groupBox_10"));
        groupBox_10->setGeometry(QRect(210, 0, 270, 81));
        groupBox_10->setFlat(false);
        groupBox_10->setCheckable(false);
        pushButton_3 = new QPushButton(groupBox_10);
        pushButton_3->setObjectName(QString::fromUtf8("pushButton_3"));
        pushButton_3->setGeometry(QRect(50, 24, 161, 20));
        pushButton_4 = new QPushButton(groupBox_10);
        pushButton_4->setObjectName(QString::fromUtf8("pushButton_4"));
        pushButton_4->setEnabled(true);
        pushButton_4->setGeometry(QRect(50, 50, 161, 20));
        pushButton_4->setCheckable(false);
        pushButton_4->setFlat(false);
        powerGroupBox = new QGroupBox(RemoteQTClass);
        powerGroupBox->setObjectName(QString::fromUtf8("powerGroupBox"));
        powerGroupBox->setGeometry(QRect(10, 0, 187, 81));
        PowerOnButton = new QPushButton(powerGroupBox);
        PowerOnButton->setObjectName(QString::fromUtf8("PowerOnButton"));
        PowerOnButton->setGeometry(QRect(30, 25, 131, 20));
        PowerOffButton = new QPushButton(powerGroupBox);
        PowerOffButton->setObjectName(QString::fromUtf8("PowerOffButton"));
        PowerOffButton->setGeometry(QRect(30, 50, 131, 20));
        tabWidget_2 = new QTabWidget(RemoteQTClass);
        tabWidget_2->setObjectName(QString::fromUtf8("tabWidget_2"));
        tabWidget_2->setEnabled(true);
        tabWidget_2->setGeometry(QRect(10, 630, 471, 81));
        tab_3 = new QWidget();
        tab_3->setObjectName(QString::fromUtf8("tab_3"));
        StatusDisplay0 = new QLineEdit(tab_3);
        StatusDisplay0->setObjectName(QString::fromUtf8("StatusDisplay0"));
        StatusDisplay0->setEnabled(false);
        StatusDisplay0->setGeometry(QRect(20, 10, 401, 27));
        tabWidget_2->addTab(tab_3, QString());
        tab_4 = new QWidget();
        tab_4->setObjectName(QString::fromUtf8("tab_4"));
        StatusDisplay1 = new QLineEdit(tab_4);
        StatusDisplay1->setObjectName(QString::fromUtf8("StatusDisplay1"));
        StatusDisplay1->setEnabled(false);
        StatusDisplay1->setGeometry(QRect(20, 10, 401, 27));
        tabWidget_2->addTab(tab_4, QString());
        tab_5 = new QWidget();
        tab_5->setObjectName(QString::fromUtf8("tab_5"));
        StatusDisplay2 = new QLineEdit(tab_5);
        StatusDisplay2->setObjectName(QString::fromUtf8("StatusDisplay2"));
        StatusDisplay2->setEnabled(false);
        StatusDisplay2->setGeometry(QRect(20, 10, 401, 27));
        tabWidget_2->addTab(tab_5, QString());
        QWidget::setTabOrder(PowerOnButton, PowerOffButton);
        QWidget::setTabOrder(PowerOffButton, pushButton_3);
        QWidget::setTabOrder(pushButton_3, pushButton_4);
        QWidget::setTabOrder(pushButton_4, pushButton);
        QWidget::setTabOrder(pushButton, pushButton_5);
        QWidget::setTabOrder(pushButton_5, pushButton_6);
        QWidget::setTabOrder(pushButton_6, filenameComboBox);
        QWidget::setTabOrder(filenameComboBox, pushButton_2);
        QWidget::setTabOrder(pushButton_2, tabWidget);
        QWidget::setTabOrder(tabWidget, posSliderX);
        QWidget::setTabOrder(posSliderX, posSliderY);
        QWidget::setTabOrder(posSliderY, posSliderZ);
        QWidget::setTabOrder(posSliderZ, rotSliderX);
        QWidget::setTabOrder(rotSliderX, rotSliderY);
        QWidget::setTabOrder(rotSliderY, rotSliderZ);
        QWidget::setTabOrder(rotSliderZ, spinCheckbox);
        QWidget::setTabOrder(spinCheckbox, horizontalSlider);
        QWidget::setTabOrder(horizontalSlider, doubleSpinBoxScale);
        QWidget::setTabOrder(doubleSpinBoxScale, flatCheck1);
        QWidget::setTabOrder(flatCheck1, flatCheck2);
        QWidget::setTabOrder(flatCheck2, horizontalSlider_2);
        QWidget::setTabOrder(horizontalSlider_2, horizontalSlider_3);
        QWidget::setTabOrder(horizontalSlider_3, checkBox_01);
        QWidget::setTabOrder(checkBox_01, checkBox_02);
        QWidget::setTabOrder(checkBox_02, checkBox_03);
        QWidget::setTabOrder(checkBox_03, checkBox_04);
        QWidget::setTabOrder(checkBox_04, checkBox_05);
        QWidget::setTabOrder(checkBox_05, checkBox_06);
        QWidget::setTabOrder(checkBox_06, checkBox_07);
        QWidget::setTabOrder(checkBox_07, checkBox_08);
        QWidget::setTabOrder(checkBox_08, checkBox_09);
        QWidget::setTabOrder(checkBox_09, checkBox_10);
        QWidget::setTabOrder(checkBox_10, checkBox_11);
        QWidget::setTabOrder(checkBox_11, checkBox_12);
        QWidget::setTabOrder(checkBox_12, checkBox_13);
        QWidget::setTabOrder(checkBox_13, checkBox_14);
        QWidget::setTabOrder(checkBox_14, checkBox_15);
        QWidget::setTabOrder(checkBox_15, checkBox_16);
        QWidget::setTabOrder(checkBox_16, spinBox);
        QWidget::setTabOrder(spinBox, VarSelectSpinbox);
        QWidget::setTabOrder(VarSelectSpinbox, VarScaleSpinbox);
        QWidget::setTabOrder(VarScaleSpinbox, VarAdjustSlider);
        QWidget::setTabOrder(VarAdjustSlider, VarValueEdit);
        QWidget::setTabOrder(VarValueEdit, StatusDisplayCloud);
        QWidget::setTabOrder(StatusDisplayCloud, spinBoxSkipCloudPoints);
        QWidget::setTabOrder(spinBoxSkipCloudPoints, comboBoxCloudFilename);
        QWidget::setTabOrder(comboBoxCloudFilename, pushButton_loadPcd);
        QWidget::setTabOrder(pushButton_loadPcd, tabWidget_2);
        QWidget::setTabOrder(tabWidget_2, StatusDisplay0);
        QWidget::setTabOrder(StatusDisplay0, StatusDisplay1);
        QWidget::setTabOrder(StatusDisplay1, StatusDisplay2);
        QWidget::setTabOrder(StatusDisplay2, pushButton_saveState);
        QWidget::setTabOrder(pushButton_saveState, pushButton_loadState);
        QWidget::setTabOrder(pushButton_loadState, pushButton_resetState);

        retranslateUi(RemoteQTClass);
        QObject::connect(posSliderX, SIGNAL(valueChanged(int)), RemoteQTClass, SLOT(sliderTxChanged(int)));
        QObject::connect(posSliderY, SIGNAL(valueChanged(int)), RemoteQTClass, SLOT(sliderTyChanged(int)));
        QObject::connect(posSliderZ, SIGNAL(valueChanged(int)), RemoteQTClass, SLOT(sliderTzChanged(int)));
        QObject::connect(rotSliderX, SIGNAL(valueChanged(int)), RemoteQTClass, SLOT(sliderRxChanged(int)));
        QObject::connect(rotSliderY, SIGNAL(valueChanged(int)), RemoteQTClass, SLOT(sliderRyChanged(int)));
        QObject::connect(rotSliderZ, SIGNAL(valueChanged(int)), RemoteQTClass, SLOT(sliderRzChanged(int)));
        QObject::connect(horizontalSlider, SIGNAL(valueChanged(int)), RemoteQTClass, SLOT(sliderGainChanged(int)));
        QObject::connect(pushButton, SIGNAL(clicked()), RemoteQTClass, SLOT(startClicked()));
        QObject::connect(pushButton_2, SIGNAL(clicked()), RemoteQTClass, SLOT(killClicked()));
        QObject::connect(flatCheck1, SIGNAL(toggled(bool)), RemoteQTClass, SLOT(flatCheck1Clicked(bool)));
        QObject::connect(flatCheck2, SIGNAL(toggled(bool)), RemoteQTClass, SLOT(flatCheck2Clicked(bool)));
        QObject::connect(horizontalSlider_2, SIGNAL(valueChanged(int)), RemoteQTClass, SLOT(sliderFlatDepth1Changed(int)));
        QObject::connect(horizontalSlider_3, SIGNAL(valueChanged(int)), RemoteQTClass, SLOT(sliderFlatDepth2Changed(int)));
        QObject::connect(spinCheckbox, SIGNAL(stateChanged(int)), RemoteQTClass, SLOT(spinToggleChanged(int)));
        QObject::connect(checkBox_01, SIGNAL(toggled(bool)), RemoteQTClass, SLOT(debugSwitchToggled(bool)));
        QObject::connect(checkBox_02, SIGNAL(toggled(bool)), RemoteQTClass, SLOT(debugSwitchToggled(bool)));
        QObject::connect(checkBox_03, SIGNAL(toggled(bool)), RemoteQTClass, SLOT(debugSwitchToggled(bool)));
        QObject::connect(checkBox_04, SIGNAL(toggled(bool)), RemoteQTClass, SLOT(debugSwitchToggled(bool)));
        QObject::connect(checkBox_05, SIGNAL(toggled(bool)), RemoteQTClass, SLOT(debugSwitchToggled(bool)));
        QObject::connect(checkBox_06, SIGNAL(toggled(bool)), RemoteQTClass, SLOT(debugSwitchToggled(bool)));
        QObject::connect(checkBox_07, SIGNAL(toggled(bool)), RemoteQTClass, SLOT(debugSwitchToggled(bool)));
        QObject::connect(checkBox_08, SIGNAL(toggled(bool)), RemoteQTClass, SLOT(debugSwitchToggled(bool)));
        QObject::connect(checkBox_09, SIGNAL(toggled(bool)), RemoteQTClass, SLOT(debugSwitchToggled(bool)));
        QObject::connect(checkBox_10, SIGNAL(toggled(bool)), RemoteQTClass, SLOT(debugSwitchToggled(bool)));
        QObject::connect(checkBox_11, SIGNAL(toggled(bool)), RemoteQTClass, SLOT(debugSwitchToggled(bool)));
        QObject::connect(checkBox_12, SIGNAL(toggled(bool)), RemoteQTClass, SLOT(debugSwitchToggled(bool)));
        QObject::connect(checkBox_13, SIGNAL(toggled(bool)), RemoteQTClass, SLOT(debugSwitchToggled(bool)));
        QObject::connect(checkBox_14, SIGNAL(toggled(bool)), RemoteQTClass, SLOT(debugSwitchToggled(bool)));
        QObject::connect(checkBox_15, SIGNAL(toggled(bool)), RemoteQTClass, SLOT(debugSwitchToggled(bool)));
        QObject::connect(checkBox_16, SIGNAL(toggled(bool)), RemoteQTClass, SLOT(debugSwitchToggled(bool)));
        QObject::connect(pushButton_5, SIGNAL(clicked()), RemoteQTClass, SLOT(runWafelClicked()));
        QObject::connect(spinBox, SIGNAL(valueChanged(int)), RemoteQTClass, SLOT(shaderModeChanged(int)));
        QObject::connect(VarSelectSpinbox, SIGNAL(valueChanged(int)), RemoteQTClass, SLOT(debugVariableChanged(int)));
        QObject::connect(VarScaleSpinbox, SIGNAL(valueChanged(double)), RemoteQTClass, SLOT(debugVarScaleChanged(double)));
        QObject::connect(VarAdjustSlider, SIGNAL(valueChanged(int)), RemoteQTClass, SLOT(debugAdjustChanged(int)));
        QObject::connect(pushButton_saveState, SIGNAL(clicked()), RemoteQTClass, SLOT(saveCurrentState()));
        QObject::connect(pushButton_loadState, SIGNAL(clicked()), RemoteQTClass, SLOT(loadSavedState()));
        QObject::connect(pushButton_resetState, SIGNAL(clicked()), RemoteQTClass, SLOT(resetDefaults()));
        QObject::connect(pushButton_loadPcd, SIGNAL(clicked()), RemoteQTClass, SLOT(loadCloud()));
        QObject::connect(doubleSpinBoxScale, SIGNAL(valueChanged(double)), RemoteQTClass, SLOT(sliderScaleChanged(double)));
        QObject::connect(pushButton_3, SIGNAL(clicked()), RemoteQTClass, SLOT(startxClicked()));
        QObject::connect(pushButton_4, SIGNAL(clicked()), RemoteQTClass, SLOT(kinectClicked()));
        QObject::connect(pushButton_6, SIGNAL(clicked()), RemoteQTClass, SLOT(runPointsClicked()));
        QObject::connect(PowerOffButton, SIGNAL(clicked()), RemoteQTClass, SLOT(powerOffClicked()));
        QObject::connect(PowerOnButton, SIGNAL(clicked()), RemoteQTClass, SLOT(powerOnClicked()));
        QObject::connect(pushButton_runSequence, SIGNAL(clicked()), RemoteQTClass, SLOT(loadSequence()));
        QObject::connect(pushButton_endSequence, SIGNAL(clicked()), RemoteQTClass, SLOT(endSequence()));

        filenameComboBox->setCurrentIndex(0);
        tabWidget->setCurrentIndex(0);
        tabWidget_2->setCurrentIndex(2);


        QMetaObject::connectSlotsByName(RemoteQTClass);
    } // setupUi

    void retranslateUi(QWidget *RemoteQTClass)
    {
        RemoteQTClass->setWindowTitle(QApplication::translate("RemoteQTClass", "Holovideo Remote", 0, QApplication::UnicodeUTF8));
        groupBox_2->setTitle(QApplication::translate("RemoteQTClass", "Renderer", 0, QApplication::UnicodeUTF8));
        pushButton_2->setText(QApplication::translate("RemoteQTClass", "Stop all", 0, QApplication::UnicodeUTF8));
        label_12->setText(QApplication::translate("RemoteQTClass", "Launch:", 0, QApplication::UnicodeUTF8));
        pushButton->setText(QApplication::translate("RemoteQTClass", "RIP", 0, QApplication::UnicodeUTF8));
        pushButton_5->setText(QApplication::translate("RemoteQTClass", "Wafel", 0, QApplication::UnicodeUTF8));
        pushButton_6->setText(QApplication::translate("RemoteQTClass", "Points", 0, QApplication::UnicodeUTF8));
        label_13->setText(QApplication::translate("RemoteQTClass", "RIP model:", 0, QApplication::UnicodeUTF8));
        filenameComboBox->clear();
        filenameComboBox->insertItems(0, QStringList()
         << QApplication::translate("RemoteQTClass", "/models/cube-lincolnface.xml", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("RemoteQTClass", "/models/face.xml", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("RemoteQTClass", "/models/ermal.xml", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("RemoteQTClass", "/models/palmnest.xml", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("RemoteQTClass", "/models/mushroom.xml", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("RemoteQTClass", "/models/deathstar.xml", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("RemoteQTClass", "/models/cube-leaf.xml", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("RemoteQTClass", "/models/ribs.xml", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("RemoteQTClass", "/models/ventricles.xml", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("RemoteQTClass", "/models/brain.xml", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("RemoteQTClass", "/models/vessels.xml", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("RemoteQTClass", "/models/skrump.xml", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("RemoteQTClass", "/models/skrumpclose.xml", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("RemoteQTClass", "-i /models/prerendered/presence1VidFrames/frame", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("RemoteQTClass", "-i /models/prerendered/presence2VidFrames/frame", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("RemoteQTClass", "-i /models/prerendered/presence3VidFrames/frame", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("RemoteQTClass", "-i /models/prerendered/presence4VidFrames/frame", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("RemoteQTClass", "-i /models/prerendered/presence5VidFrames/frame", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("RemoteQTClass", "-i /models/prerendered/presence6VidFrames/frame", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("RemoteQTClass", "-i /models/prerendered/Numbered/frame", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("RemoteQTClass", "/models/letters/SLOAN/N.xml /models/letters/SLOAN/O.xml", 0, QApplication::UnicodeUTF8)
        );
        ModelPositionGroup->setTitle(QApplication::translate("RemoteQTClass", "Model Position", 0, QApplication::UnicodeUTF8));
        label->setText(QApplication::translate("RemoteQTClass", "Translation", 0, QApplication::UnicodeUTF8));
        label_2->setText(QApplication::translate("RemoteQTClass", "Orientation", 0, QApplication::UnicodeUTF8));
        spinCheckbox->setText(QApplication::translate("RemoteQTClass", "Spin", 0, QApplication::UnicodeUTF8));
        groupBox->setTitle(QApplication::translate("RemoteQTClass", "Gain", 0, QApplication::UnicodeUTF8));
        groupBox_3->setTitle(QApplication::translate("RemoteQTClass", "Flat Rendering", 0, QApplication::UnicodeUTF8));
        flatCheck1->setText(QApplication::translate("RemoteQTClass", "1", 0, QApplication::UnicodeUTF8));
        flatCheck2->setText(QApplication::translate("RemoteQTClass", "2", 0, QApplication::UnicodeUTF8));
        label_3->setText(QApplication::translate("RemoteQTClass", "Flat", 0, QApplication::UnicodeUTF8));
        label_4->setText(QApplication::translate("RemoteQTClass", "Depth", 0, QApplication::UnicodeUTF8));
        groupBox_9->setTitle(QApplication::translate("RemoteQTClass", "Scale", 0, QApplication::UnicodeUTF8));
        tabWidget->setTabText(tabWidget->indexOf(tab), QApplication::translate("RemoteQTClass", "Scene Adjust", 0, QApplication::UnicodeUTF8));
        groupBox_4->setTitle(QApplication::translate("RemoteQTClass", "Debugging", 0, QApplication::UnicodeUTF8));
        groupBox_5->setTitle(QApplication::translate("RemoteQTClass", "Disable view:", 0, QApplication::UnicodeUTF8));
        checkBox_01->setText(QApplication::translate("RemoteQTClass", "1", 0, QApplication::UnicodeUTF8));
        checkBox_02->setText(QApplication::translate("RemoteQTClass", "2", 0, QApplication::UnicodeUTF8));
        checkBox_03->setText(QApplication::translate("RemoteQTClass", "3", 0, QApplication::UnicodeUTF8));
        checkBox_04->setText(QApplication::translate("RemoteQTClass", "4", 0, QApplication::UnicodeUTF8));
        checkBox_05->setText(QApplication::translate("RemoteQTClass", "5", 0, QApplication::UnicodeUTF8));
        checkBox_06->setText(QApplication::translate("RemoteQTClass", "6", 0, QApplication::UnicodeUTF8));
        checkBox_07->setText(QApplication::translate("RemoteQTClass", "7", 0, QApplication::UnicodeUTF8));
        checkBox_08->setText(QApplication::translate("RemoteQTClass", "8", 0, QApplication::UnicodeUTF8));
        checkBox_09->setText(QApplication::translate("RemoteQTClass", "9", 0, QApplication::UnicodeUTF8));
        checkBox_10->setText(QApplication::translate("RemoteQTClass", "10", 0, QApplication::UnicodeUTF8));
        checkBox_11->setText(QApplication::translate("RemoteQTClass", "11", 0, QApplication::UnicodeUTF8));
        checkBox_12->setText(QApplication::translate("RemoteQTClass", "12", 0, QApplication::UnicodeUTF8));
        checkBox_13->setText(QApplication::translate("RemoteQTClass", "13", 0, QApplication::UnicodeUTF8));
        checkBox_14->setText(QApplication::translate("RemoteQTClass", "14", 0, QApplication::UnicodeUTF8));
        checkBox_15->setText(QApplication::translate("RemoteQTClass", "15", 0, QApplication::UnicodeUTF8));
        checkBox_16->setText(QApplication::translate("RemoteQTClass", "16", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_TOOLTIP
        label_5->setToolTip(QApplication::translate("RemoteQTClass", "<html><head/><body><p>For Wafel renderer, try -10 to see view-render output in framebuffer (no hologram) </p></body></html>", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_TOOLTIP
        label_5->setText(QApplication::translate("RemoteQTClass", "Shader Mode", 0, QApplication::UnicodeUTF8));
        label_6->setText(QApplication::translate("RemoteQTClass", "Variable", 0, QApplication::UnicodeUTF8));
        label_7->setText(QApplication::translate("RemoteQTClass", "Scale", 0, QApplication::UnicodeUTF8));
        label_8->setText(QApplication::translate("RemoteQTClass", "Value", 0, QApplication::UnicodeUTF8));
        label_9->setText(QApplication::translate("RemoteQTClass", "Adjust", 0, QApplication::UnicodeUTF8));
        tabWidget->setTabText(tabWidget->indexOf(tab_2), QApplication::translate("RemoteQTClass", "Debugging", 0, QApplication::UnicodeUTF8));
        pushButton_loadPcd->setText(QApplication::translate("RemoteQTClass", "Load Static", 0, QApplication::UnicodeUTF8));
        groupBox_8->setTitle(QApplication::translate("RemoteQTClass", "Cloud Inspector", 0, QApplication::UnicodeUTF8));
        label_10->setText(QApplication::translate("RemoteQTClass", "Inspector skip", 0, QApplication::UnicodeUTF8));
        comboBoxCloudFilename->clear();
        comboBoxCloudFilename->insertItems(0, QStringList()
         << QApplication::translate("RemoteQTClass", "/clouds/apr5_13/hinge80_0.pcd", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("RemoteQTClass", "/clouds/angletest.pcd", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("RemoteQTClass", "/clouds/barTest.pcd", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("RemoteQTClass", "/clouds/triangle.pcd", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("RemoteQTClass", "/clouds/vertplane.pcd", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("RemoteQTClass", "/clouds/testscene.pcd", 0, QApplication::UnicodeUTF8)
        );
        label_16->setText(QApplication::translate("RemoteQTClass", "Filename", 0, QApplication::UnicodeUTF8));
        label_17->setText(QApplication::translate("RemoteQTClass", "Scale On Load", 0, QApplication::UnicodeUTF8));
        checkBoxClipCircle->setText(QApplication::translate("RemoteQTClass", "Clip to Circle", 0, QApplication::UnicodeUTF8));
        tabWidget->setTabText(tabWidget->indexOf(tab_6), QApplication::translate("RemoteQTClass", "Pointclouds", 0, QApplication::UnicodeUTF8));
        comboBoxSequenceFilename->clear();
        comboBoxSequenceFilename->insertItems(0, QStringList()
         << QApplication::translate("RemoteQTClass", "/clouds/sep25_13_a", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("RemoteQTClass", "/clouds/sep19_13_a", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("RemoteQTClass", "/clouds/sep17_13_a", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("RemoteQTClass", "/clouds/aug19_13_b", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("RemoteQTClass", "/clouds/aug16_13_a", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("RemoteQTClass", "/clouds/may21_13_c", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("RemoteQTClass", "/clouds/jul20_13_a", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("RemoteQTClass", "/clouds/jul20_13_b", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("RemoteQTClass", "/clouds/jul20_13_c", 0, QApplication::UnicodeUTF8)
         << QApplication::translate("RemoteQTClass", "/clouds/jul20_13_d", 0, QApplication::UnicodeUTF8)
         << QString()
        );
        pushButton_runSequence->setText(QApplication::translate("RemoteQTClass", "Run", 0, QApplication::UnicodeUTF8));
        label_11->setText(QApplication::translate("RemoteQTClass", "Subject Id:", 0, QApplication::UnicodeUTF8));
        label_14->setText(QApplication::translate("RemoteQTClass", "Experiment:", 0, QApplication::UnicodeUTF8));
        label_15->setText(QApplication::translate("RemoteQTClass", "Run number:", 0, QApplication::UnicodeUTF8));
        pushButton_endSequence->setText(QApplication::translate("RemoteQTClass", "End", 0, QApplication::UnicodeUTF8));
        tabWidget->setTabText(tabWidget->indexOf(tab_7), QApplication::translate("RemoteQTClass", "Sequence", 0, QApplication::UnicodeUTF8));
        groupBox_7->setTitle(QApplication::translate("RemoteQTClass", "Settings", 0, QApplication::UnicodeUTF8));
        pushButton_saveState->setText(QApplication::translate("RemoteQTClass", "Save", 0, QApplication::UnicodeUTF8));
        pushButton_loadState->setText(QApplication::translate("RemoteQTClass", "Load", 0, QApplication::UnicodeUTF8));
        pushButton_resetState->setText(QApplication::translate("RemoteQTClass", "Reset", 0, QApplication::UnicodeUTF8));
        groupBox_10->setTitle(QApplication::translate("RemoteQTClass", "Setup", 0, QApplication::UnicodeUTF8));
        pushButton_3->setText(QApplication::translate("RemoteQTClass", "Start Xserver", 0, QApplication::UnicodeUTF8));
        pushButton_4->setText(QApplication::translate("RemoteQTClass", "Start Kinect Server", 0, QApplication::UnicodeUTF8));
#ifndef QT_NO_TOOLTIP
        powerGroupBox->setToolTip(QApplication::translate("RemoteQTClass", "Via IP Power Switch \n"
"(Can use manual switch instead)", 0, QApplication::UnicodeUTF8));
#endif // QT_NO_TOOLTIP
        powerGroupBox->setTitle(QApplication::translate("RemoteQTClass", "Mark II Laser + Electronics", 0, QApplication::UnicodeUTF8));
        PowerOnButton->setText(QApplication::translate("RemoteQTClass", "On", 0, QApplication::UnicodeUTF8));
        PowerOffButton->setText(QApplication::translate("RemoteQTClass", "Off", 0, QApplication::UnicodeUTF8));
        tabWidget_2->setTabText(tabWidget_2->indexOf(tab_3), QApplication::translate("RemoteQTClass", "Display :0.0", 0, QApplication::UnicodeUTF8));
        tabWidget_2->setTabText(tabWidget_2->indexOf(tab_4), QApplication::translate("RemoteQTClass", "Display :0.1", 0, QApplication::UnicodeUTF8));
        tabWidget_2->setTabText(tabWidget_2->indexOf(tab_5), QApplication::translate("RemoteQTClass", "Display :0.2", 0, QApplication::UnicodeUTF8));
    } // retranslateUi

};

namespace Ui {
    class RemoteQTClass: public Ui_RemoteQTClass {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_REMOTEQT_H
