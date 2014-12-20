#pragma once

#include <iostream>
#include <string>
#include <boost/program_options.hpp>

#define DSCP4_DEFAULT_VERBOSITY 1
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
	int getVerbosity() { return vm_["verbosity"].as<int>(); }

private:

	boost::program_options::variables_map vm_;

	boost::program_options::options_description allOptions_;
	boost::program_options::options_description generalOptions_;
	boost::program_options::options_description inputOptions_;

};