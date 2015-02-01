#ifndef Q_LOG_APPENDER_H
#define Q_LOG_APPENDER_H

#include <qobject.h>
#include <log4cxx/appenderskeleton.h>
#include <log4cxx/spi/loggingevent.h>
#include <log4cxx/layout.h>
#include <string>

namespace log4cxx
{
	class QLogAppender : public QObject, public log4cxx::AppenderSkeleton
	{
		Q_OBJECT

	public:

		DECLARE_LOG4CXX_OBJECT(QLogAppender)
		BEGIN_LOG4CXX_CAST_MAP()
			LOG4CXX_CAST_ENTRY(QLogAppender)
			LOG4CXX_CAST_ENTRY_CHAIN(AppenderSkeleton)
		END_LOG4CXX_CAST_MAP()

		QLogAppender();
		QLogAppender(const LayoutPtr& layoutPtr);
		~QLogAppender();

		//This method is called by the AppenderSkeleton#doAppend method
		void append(const spi::LoggingEventPtr& event, log4cxx::helpers::Pool& p);

		void close();

		bool isClosed() const { return closed; }

		bool requiresLayout() const { return true; }

	signals:
		void gotNewLogMessage(QString message);

	};
}
#endif