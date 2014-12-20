#include "DSCP4ProgOptions.hpp"

DSCP4ProgramOptions::DSCP4ProgramOptions() :
generalOptions_("General options"),
inputOptions_("Input options")
{	
	generalOptions_.add_options()
		("verbosity,v", boost::program_options::value<int>()->default_value(DSCP4_DEFAULT_VERBOSITY), "level of detail for console output. valid values are [0-3] from least to most verbose")
		("help,h", "produce help message");

	inputOptions_.add_options()
		("input-file,i",
		boost::program_options::value<std::string>()->default_value(DSCP4_DEFAULT_OBJECT_FILENAME),
		"the input 3D object file path");

	allOptions_.add(generalOptions_).add(inputOptions_);
}

DSCP4ProgramOptions::~DSCP4ProgramOptions()
{

}

void DSCP4ProgramOptions::parseCommandLine(int argc, const char* argv[])
{
	boost::program_options::store(boost::program_options::parse_command_line(argc, argv, allOptions_), vm_);
}

void DSCP4ProgramOptions::printOptions(DSCP4_OPTIONS_TYPE options)
{

	switch (options)
	{
	case DSCP4ProgramOptions::DSCP4_OPTIONS_TYPE_ALL:
		std::cout << allOptions_;
		break;
	case DSCP4ProgramOptions::DSCP4_OPTIONS_TYPE_GENERAL:
		std::cout << generalOptions_;
		break;
	case DSCP4ProgramOptions::DSCP4_OPTIONS_TYPE_INPUT:
		std::cout << inputOptions_;
		break;
	default:
		break;
	}

}

