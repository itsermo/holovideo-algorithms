#include "DSCP4ProgOptions.hpp"
#include <boost/property_tree/ini_parser.hpp>


DSCP4ProgramOptions::DSCP4ProgramOptions() :
generalOptions_("General options"),
inputOptions_("Input options")
{	
	generalOptions_.add_options()
#ifdef DSCP4_HAVE_LOG4CXX
		("verbosity,v", boost::program_options::value<int>()->default_value(DSCP4_DEFAULT_VERBOSITY), "level of detail for console output. valid values are [0-3] from least to most verbose")("verbosity,v", boost::program_options::value<int>()->default_value(DSCP4_DEFAULT_VERBOSITY), "level of detail for console output. valid values are [0-3] from least to most verbose")
#endif
		("help,h", "produce help message");

	inputOptions_.add_options()
		("input-file,i",
		boost::program_options::value<std::string>()->default_value(DSCP4_INPUT_DEFAULT_OBJECT_FILENAME),
		"the input 3D object file path")
		("generate-normals,n",
		boost::program_options::value<std::string>()->default_value(DSCP4_INPUT_DEFAULT_GEN_NORMALS),
		"generate normals for 3D object file. valid values are \"off\" \"flat\" and \"smooth\"")
		("triangulate-mesh,t",
		boost::program_options::value<bool>()->default_value(DSCP4_INPUT_DEFAULT_TRIANGULATE_MESH),
		"triangulates the mesh (if it is made of quads or something else");

	renderOptions_.add_options()
		("autoscale,a",
		boost::program_options::value<bool>()->default_value(DSCP4_RENDER_DEFAULT_AUTOSCALE),
		"finds the 3d model bounding sphere, scales the model and places it at the origin")
		("shade-model,s",
		boost::program_options::value<std::string>()->default_value(DSCP4_RENDER_DEFAULT_SHADEMODEL),
		"the shading model. valid options are 'off' to turn off all lighting, 'flat' and 'smooth'")
		("render-mode,m",
		boost::program_options::value<std::string>()->default_value(DSCP4_RENDER_DEFAULT_MODE),
		"sets the render mode. valid options are 'viewing', 'stereogram', and 'holovideo'");

	allOptions_.add(generalOptions_).add(inputOptions_).add(renderOptions_);
}

DSCP4ProgramOptions::~DSCP4ProgramOptions()
{

}

void DSCP4ProgramOptions::parseConfigFile()
{
	//check for working path conf file first
	try
	{
		boost::property_tree::ini_parser::read_ini(DSCP4_CONF_FILENAME, pt_);
	}
	catch (std::exception)
	{
		boost::filesystem::path homePath;

#ifdef WIN32
		homePath /= getenv("HOMEDRIVE");
		homePath /= getenv("HOMEPATH");
#else
		homePath /= getenv("HOME");
#endif

		homePath /= std::string(".").append(DSCP4_PATH_PREFIX);
		homePath /= DSCP4_CONF_FILENAME;

		LOG4CXX_DEBUG(logger_, "Could not find '" << DSCP4_CONF_FILENAME << "' in current working dir '" << boost::filesystem::current_path().string() << "'")
		LOG4CXX_DEBUG(logger_, "Continuing search for conf file at '" << homePath.string() << "'...")
		try
		{
			boost::property_tree::ini_parser::read_ini(homePath.string(), pt_);
		}
		catch (std::exception)
		{
			boost::filesystem::path globalPath;
#ifdef WIN32
			globalPath /= getenv("PROGRAMDATA");
#else
			globalPath = "/etc";
#endif
			globalPath /= DSCP4_PATH_PREFIX;
			globalPath /= DSCP4_CONF_FILENAME;

			LOG4CXX_DEBUG(logger_, "Could not find conf file at '" << homePath.string() << "'")
			LOG4CXX_DEBUG(logger_, "Continuing search for conf file at '" << globalPath.string() << "'...")
			
			boost::property_tree::ini_parser::read_ini(globalPath.string(), pt_);
		}

	}

#ifdef DSCP4_HAVE_LOG4CXX
	try
	{
		verbosity_ = pt_.get<int>("general.verbosity");
	}
	catch (std::exception)
	{
		verbosity_ = DSCP4_DEFAULT_VERBOSITY;
	}
#endif

}

void DSCP4ProgramOptions::parseCommandLine(int argc, const char* argv[])
{
	boost::program_options::store(boost::program_options::parse_command_line(argc, argv, allOptions_), vm_);

#ifdef DSCP4_HAVE_LOG4CXX
	if (vm_.count("verbosity"))
		verbosity_ = vm_["verbosity"].as<int>();
#endif

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

