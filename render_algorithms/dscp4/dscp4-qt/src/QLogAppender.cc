#include "QLogAppender.h"
#include <stdio.h>

#include <log4cxx/helpers/transcoder.h>
#include <iostream>

using namespace log4cxx;
using namespace log4cxx::helpers;

// Register this class with log4cxx
IMPLEMENT_LOG4CXX_OBJECT(QLogAppender)

QLogAppender::QLogAppender()  {}

QLogAppender::QLogAppender(ObjectPtrT<log4cxx::Layout> const& layoutPtr)
{
	this->setLayout(layoutPtr);
}


QLogAppender::~QLogAppender() {}

void QLogAppender::append(const spi::LoggingEventPtr& event, Pool& p)
{
	log4cxx::LogString fMsg;

	this->layout->format(fMsg, event, p);

	LOG4CXX_DECODE_CHAR(fMsgStr, fMsg);

#ifdef WIN32
	emit gotNewLogMessage(QString::fromStdWString(fMsgStr.c_str()).remove("\r").remove("\n"));
	std::wcout << fMsgStr;
#else
	emit gotNewLogMessage(QString::fromStdString(fMsgStr.c_str()).remove("\r").remove("\n"));
	std::cout << fMsgStr;
#endif
}

void QLogAppender::close()
{
	if (this->closed)
	{
		return;
	}

	this->closed = true;
}
