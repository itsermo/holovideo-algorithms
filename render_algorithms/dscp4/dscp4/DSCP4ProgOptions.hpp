#pragma once

#include <iostream>
#include <string>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/property_tree/ptree.hpp>

#ifdef DSCP4_HAVE_LOG4CXX
#include <log4cxx/logger.h>
#endif

#define DSCP4_CONF_FILENAME "dscp4.conf"
#define DSCP4_PATH_PREFIX "dscp4"
#define DSCP4_DEFAULT_VERBOSITY 2
#define DSCP4_DEFAULT_OBJECT_FILENAME "bun_zipper.ply"

class DSCP4ProgramOptions
{
public:
	enum DSCP4_OPTIONS_TYPE {
		DSCP4_OPTIONS_TYPE_ALL,
		DSCP4_OPTIONS_TYPE_GENERAL,
		DSCP4_OPTIONS_TYPE_INPUT
	};

	DSCP4ProgramOptions();
	~DSCP4ProgramOptions();

	void parseCommandLine(int argc, const char* argv[]);

	void printOptions(DSCP4_OPTIONS_TYPE options);

	std::string getFileName() { return vm_["input-file"].as<std::string>(); }
	bool getWantHelp() { return vm_.count("help") == 0 ? false : true; }
	int getVerbosity() { return verbosity_; }

	boost::filesystem::path getInstallPath() { return pt_.get<std::string>("general.install_path"); }
	boost::filesystem::path getBinPath() { return pt_.get<std::string>("general.bin_path"); }
	boost::filesystem::path getLibPath() { return pt_.get<std::string>("general.lib_path"); }
	boost::filesystem::path getModelsPath() { return pt_.get<std::string>("general.models_path"); }
	boost::filesystem::path getShadersPath() { return pt_.get<std::string>("general.shaders_path"); }

private:
	boost::property_tree::ptree pt_;

	boost::program_options::variables_map vm_;

	boost::program_options::options_description allOptions_;
	boost::program_options::options_description generalOptions_;
	boost::program_options::options_description inputOptions_;

	int verbosity_;

#ifdef DSCP4_HAVE_LOG4CXX
	log4cxx::LoggerPtr logger_ = log4cxx::Logger::getLogger("edu.mit.media.obmg.holovideo.dscp4");
#endif

};