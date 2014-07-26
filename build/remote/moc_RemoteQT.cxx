/****************************************************************************
** Meta object code from reading C++ file 'RemoteQT.h'
**
** Created by: The Qt Meta Object Compiler version 63 (Qt 4.8.6)
**
** WARNING! All changes made in this file will be lost!
*****************************************************************************/

#include "../../remote/RemoteQT.h"
#if !defined(Q_MOC_OUTPUT_REVISION)
#error "The header file 'RemoteQT.h' doesn't include <QObject>."
#elif Q_MOC_OUTPUT_REVISION != 63
#error "This file was generated using the moc from 4.8.6. It"
#error "cannot be used with the include files from this version of Qt."
#error "(The moc has changed too much.)"
#endif

QT_BEGIN_MOC_NAMESPACE
static const uint qt_meta_data_RemoteQT[] = {

 // content:
       6,       // revision
       0,       // classname
       0,    0, // classinfo
      38,   14, // methods
       0,    0, // properties
       0,    0, // enums/sets
       0,    0, // constructors
       0,       // flags
       0,       // signalCount

 // slots: signature, parameters, type, tag, flags
      10,    9,    9,    9, 0x0a,
      27,    9,    9,    9, 0x0a,
      45,    9,    9,    9, 0x0a,
      68,    9,    9,    9, 0x0a,
      89,    9,    9,    9, 0x0a,
     110,    9,    9,    9, 0x0a,
     131,    9,    9,    9, 0x0a,
     152,    9,    9,    9, 0x0a,
     173,    9,    9,    9, 0x0a,
     194,    9,    9,    9, 0x0a,
     217,    9,    9,    9, 0x0a,
     244,    9,    9,    9, 0x0a,
     260,    9,    9,    9, 0x0a,
     284,    9,    9,    9, 0x0a,
     300,    9,    9,    9, 0x0a,
     315,    9,    9,    9, 0x0a,
     329,    9,    9,    9, 0x0a,
     348,    9,    9,    9, 0x0a,
     372,    9,    9,    9, 0x0a,
     396,    9,    9,    9, 0x0a,
     425,    9,    9,    9, 0x0a,
     454,    9,    9,    9, 0x0a,
     465,    9,    9,    9, 0x0a,
     480,    9,    9,    9, 0x0a,
     505,    9,    9,    9, 0x0a,
     528,    9,    9,    9, 0x0a,
     546,    9,    9,    9, 0x0a,
     572,    9,    9,    9, 0x0a,
     601,    9,    9,    9, 0x0a,
     625,    9,    9,    9, 0x0a,
     652,    9,    9,    9, 0x0a,
     664,    9,    9,    9, 0x0a,
     679,    9,    9,    9, 0x0a,
     693,    9,    9,    9, 0x0a,
     705,    9,    9,    9, 0x0a,
     721,    9,    9,    9, 0x0a,
     740,    9,    9,    9, 0x0a,
     756,    9,    9,    9, 0x0a,

       0        // eod
};

static const char qt_meta_stringdata_RemoteQT[] = {
    "RemoteQT\0\0powerOnClicked()\0powerOffClicked()\0"
    "spinToggleChanged(int)\0sliderTxChanged(int)\0"
    "sliderTyChanged(int)\0sliderTzChanged(int)\0"
    "sliderRxChanged(int)\0sliderRyChanged(int)\0"
    "sliderRzChanged(int)\0sliderGainChanged(int)\0"
    "sliderScaleChanged(double)\0startxClicked()\0"
    "nvidiaSettingsClicked()\0kinectClicked()\0"
    "startClicked()\0killClicked()\0"
    "runPointsClicked()\0flatCheck1Clicked(bool)\0"
    "flatCheck2Clicked(bool)\0"
    "sliderFlatDepth1Changed(int)\0"
    "sliderFlatDepth2Changed(int)\0spinStep()\0"
    "statusUpdate()\0debugSwitchToggled(bool)\0"
    "shaderModeChanged(int)\0runWafelClicked()\0"
    "debugVariableChanged(int)\0"
    "debugVarScaleChanged(double)\0"
    "debugAdjustChanged(int)\0"
    "updateDebugVariableValue()\0loadCloud()\0"
    "loadSequence()\0endSequence()\0expUpdate()\0"
    "trackerUpdate()\0saveCurrentState()\0"
    "resetDefaults()\0loadSavedState()\0"
};

void RemoteQT::qt_static_metacall(QObject *_o, QMetaObject::Call _c, int _id, void **_a)
{
    if (_c == QMetaObject::InvokeMetaMethod) {
        Q_ASSERT(staticMetaObject.cast(_o));
        RemoteQT *_t = static_cast<RemoteQT *>(_o);
        switch (_id) {
        case 0: _t->powerOnClicked(); break;
        case 1: _t->powerOffClicked(); break;
        case 2: _t->spinToggleChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 3: _t->sliderTxChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 4: _t->sliderTyChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 5: _t->sliderTzChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 6: _t->sliderRxChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 7: _t->sliderRyChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 8: _t->sliderRzChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 9: _t->sliderGainChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 10: _t->sliderScaleChanged((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 11: _t->startxClicked(); break;
        case 12: _t->nvidiaSettingsClicked(); break;
        case 13: _t->kinectClicked(); break;
        case 14: _t->startClicked(); break;
        case 15: _t->killClicked(); break;
        case 16: _t->runPointsClicked(); break;
        case 17: _t->flatCheck1Clicked((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 18: _t->flatCheck2Clicked((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 19: _t->sliderFlatDepth1Changed((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 20: _t->sliderFlatDepth2Changed((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 21: _t->spinStep(); break;
        case 22: _t->statusUpdate(); break;
        case 23: _t->debugSwitchToggled((*reinterpret_cast< bool(*)>(_a[1]))); break;
        case 24: _t->shaderModeChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 25: _t->runWafelClicked(); break;
        case 26: _t->debugVariableChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 27: _t->debugVarScaleChanged((*reinterpret_cast< double(*)>(_a[1]))); break;
        case 28: _t->debugAdjustChanged((*reinterpret_cast< int(*)>(_a[1]))); break;
        case 29: _t->updateDebugVariableValue(); break;
        case 30: _t->loadCloud(); break;
        case 31: _t->loadSequence(); break;
        case 32: _t->endSequence(); break;
        case 33: _t->expUpdate(); break;
        case 34: _t->trackerUpdate(); break;
        case 35: _t->saveCurrentState(); break;
        case 36: _t->resetDefaults(); break;
        case 37: _t->loadSavedState(); break;
        default: ;
        }
    }
}

const QMetaObjectExtraData RemoteQT::staticMetaObjectExtraData = {
    0,  qt_static_metacall 
};

const QMetaObject RemoteQT::staticMetaObject = {
    { &QMainWindow::staticMetaObject, qt_meta_stringdata_RemoteQT,
      qt_meta_data_RemoteQT, &staticMetaObjectExtraData }
};

#ifdef Q_NO_DATA_RELOCATION
const QMetaObject &RemoteQT::getStaticMetaObject() { return staticMetaObject; }
#endif //Q_NO_DATA_RELOCATION

const QMetaObject *RemoteQT::metaObject() const
{
    return QObject::d_ptr->metaObject ? QObject::d_ptr->metaObject : &staticMetaObject;
}

void *RemoteQT::qt_metacast(const char *_clname)
{
    if (!_clname) return 0;
    if (!strcmp(_clname, qt_meta_stringdata_RemoteQT))
        return static_cast<void*>(const_cast< RemoteQT*>(this));
    return QMainWindow::qt_metacast(_clname);
}

int RemoteQT::qt_metacall(QMetaObject::Call _c, int _id, void **_a)
{
    _id = QMainWindow::qt_metacall(_c, _id, _a);
    if (_id < 0)
        return _id;
    if (_c == QMetaObject::InvokeMetaMethod) {
        if (_id < 38)
            qt_static_metacall(this, _c, _id, _a);
        _id -= 38;
    }
    return _id;
}
QT_END_MOC_NAMESPACE
