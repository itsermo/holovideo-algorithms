#include "RemoteQT.h"

#include <QtGui>
#include <QApplication>

int main(int argc, char *argv[])
{
    QApplication a(argc, argv);
    RemoteQT w;
    w.show();
    return a.exec();
}
