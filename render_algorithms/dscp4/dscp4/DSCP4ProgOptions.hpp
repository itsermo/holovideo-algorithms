#pragma once

#include <iostream>
#include <string>
#include <boost/program_options.hpp>
#include <boost/filesystem.hpp>
#include <boost/property_tree/ptree.hpp>

#ifdef DSCP4_HAVE_LOG4CXX
#include <log4cxx/logger.h>
#else
#define LOG4CXX_TRACE(logger, expression)    
#define LOG4CXX_DEBUG(logger, expression)    
#define LOG4CXX_INFO(logger, expression)   
#define LOG4CXX_WARN(logger, expression)    
#define LOG4CXX_ERROR(logger, expression)    
#define LOG4CXX_FATAL(logger, expression) 
#endif

#define DSCP4_CONF_FILENAME "dscp4.conf"
#define DSCP4_PATH_PREFIX "dscp4"
#define DSCP4_DEFAULT_VERBOSITY 2
#define DSCP4_INPUT_DEFAULT_OBJECT_FILENAME "bun_zipper_res4.ply"
#define DSCP4_INPUT_DEFAULT_GEN_NORMALS "flat"
#define DSCP4_INPUT_DEFAULT_TRIANGULATE_MESH true
#define DSCP4_RENDER_DEFAULT_MODE "holovideo"
#define DSCP4_RENDER_DEFAULT_AUTOSCALE true
#define DSCP4_RENDER_DEFAULT_SHADEMODEL "flat"

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

	void parseConfigFile();
	void parseCommandLine(int argc, const char* argv[]);

	void printOptions(DSCP4_OPTIONS_TYPE options);

	std::string getFileName() { return vm_["input-file"].as<std::string>(); }
	bool getWantHelp() { return vm_.count("help") == 0 ? false : true; }
	int getVerbosity() { return verbosity_; }
	std::string getGenerateNormals() { return vm_["generate-normals"].as<std::string>(); }
	bool getTriangulateMesh() { return vm_["triangulate-mesh"].as<bool>(); }
	bool getAutoscale() { return vm_["autoscale"].as<bool>(); }
	std::string getShadeModel() { return vm_["shade-model"].as<std::string>(); }
	std::string getRenderMode() { return vm_["render-mode"].as<std::string>(); }

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
	boost::program_options::options_description renderOptions_;

	int verbosity_;

#ifdef DSCP4_HAVE_LOG4CXX
	log4cxx::LoggerPtr logger_ = log4cxx::Logger::getLogger("edu.mit.media.obmg.holovideo.dscp4");
#endif

};