#include <dscp4.h>

#ifdef DSCP4_HAVE_LOG4CXX
#include <log4cxx/logger.h>
#include <log4cxx/patternlayout.h>
#include <log4cxx/consoleappender.h>
#include <log4cxx/basicconfigurator.h>
#else
#define LOG4CXX_TRACE(logger, expression)    
#define LOG4CXX_DEBUG(logger, expression)    
#define LOG4CXX_INFO(logger, expression)   
#define LOG4CXX_WARN(logger, expression)    
#define LOG4CXX_ERROR(logger, expression)    
#define LOG4CXX_FATAL(logger, expression) 
#endif

#ifdef DSCP4_HAVE_LOG4CXX
log4cxx::LoggerPtr createLogger();
#endif

int main(int argc, const char* argv[])
{

#ifdef DSCP4_HAVE_LOG4CXX
	auto logger = createLogger();
#endif

	LOG4CXX_INFO(logger, "Starting DSCP4 test program...");

	CreateRenderer();

	return 0;
}

#ifdef DSCP4_HAVE_LOG4CXX
log4cxx::LoggerPtr createLogger()
{
	log4cxx::LoggerPtr logger(log4cxx::Logger::getLogger("edu.mit.media.obmg.holovideo.dscp4"));

#ifdef WIN32
	log4cxx::PatternLayoutPtr logLayoutPtr = new log4cxx::PatternLayout(L"%-5p %m%n");
#else
	log4cxx::PatternLayoutPtr logLayoutPtr = new log4cxx::PatternLayout("%-5p %m%n");
#endif

	log4cxx::ConsoleAppenderPtr logAppenderPtr = new log4cxx::ConsoleAppender(logLayoutPtr);
	log4cxx::BasicConfigurator::configure(logAppenderPtr);


	return logger;
}
#endif