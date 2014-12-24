
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

#include "DSCP4ProgOptions.hpp"

#include <dscp4.h>

#include <assimp/Importer.hpp>      
#include <assimp/scene.h>           
#include <assimp/postprocess.h> 

#include <boost/filesystem.hpp>
#include <thread>

#ifdef DSCP4_HAVE_LOG4CXX
log4cxx::LoggerPtr createLogger();
#endif

int main(int argc, const char* argv[])
{
#ifdef DSCP4_HAVE_LOG4CXX
	auto logger = createLogger();
	logger->setLevel(log4cxx::Level::getError());
#endif

	int logLevel;
	boost::filesystem::path objectFilePath;
	Assimp::Importer objectFileImporter;
	const aiScene* objectScene;

	DSCP4ProgramOptions options;

	try {
		options.parseCommandLine(argc, argv);
	}
	catch (std::exception)
	{
		std::cout << "Uknown arguments detected" << std::endl;
		options.printOptions(DSCP4ProgramOptions::DSCP4_OPTIONS_TYPE_ALL);
		return -1;
	}

	if (options.getWantHelp())
	{
		std::cout << "Help requested. Here are all the options" << std::endl;
		options.printOptions(DSCP4ProgramOptions::DSCP4_OPTIONS_TYPE_ALL);
		return -1;
	}

	logLevel = options.getVerbosity();

	switch (logLevel)
	{
	case 0:
		logger->setLevel(log4cxx::Level::getError());
		break;
	case 1:
		logger->setLevel(log4cxx::Level::getInfo());
		break;
	case 2:
		logger->setLevel(log4cxx::Level::getDebug());
		break;
	case 3:
		logger->setLevel(log4cxx::Level::getAll());
		break;
	default:
		std::cout << "Invalid verbosity level" << std::endl;
		options.printOptions(DSCP4ProgramOptions::DSCP4_OPTIONS_TYPE_GENERAL);
		return -1;
		break;
	}

	objectFilePath = boost::filesystem::path(options.getFileName());

	if (!boost::filesystem::exists(objectFilePath))
	{
		std::cout << "Invalid 3D object input file path (file not found)" << std::endl;
		options.printOptions(DSCP4ProgramOptions::DSCP4_OPTIONS_TYPE_INPUT);
		return -1;
	}

	LOG4CXX_INFO(logger, "Starting DSCP4 test program...");

	LOG4CXX_INFO(logger, "Loading 3D object file \'" << objectFilePath.filename().string() << "\'...");
	objectScene = objectFileImporter.ReadFile(objectFilePath.string(), 0);


	if (!objectScene->HasMeshes())
	{
		LOG4CXX_FATAL(logger, "3D object file does not appear to have any meshes");
		return -1;
	}

	LOG4CXX_INFO(logger, "Starting DSCP4 lib");
	if (!dscp4_InitRenderer())
	{
		LOG4CXX_FATAL(logger, "Could not initialize DSCP4 lib");
		return -1;
	}

	for (unsigned int m = 0; m < objectScene->mNumMeshes; m++)
	{
		// if it has faces, treat as mesh, otherwise as point cloud
		if (objectScene->mMeshes[m]->HasFaces())
		{
			std::string meshID;
			meshID += std::string("Mesh ") += std::to_string(m);
			LOG4CXX_INFO(logger, "Found " << meshID << " from 3D object file '" << objectFilePath.string() << "'");
			LOG4CXX_INFO(logger, meshID << " has " << objectScene->mMeshes[m]->mNumVertices << " vertices");
			if (objectScene->mMeshes[m]->HasNormals())
				LOG4CXX_INFO(logger, meshID <<  " has normals")
			else
			LOG4CXX_WARN(logger, meshID << " does not have normals, lighting effects might be fucked up");

			LOG4CXX_INFO(logger, meshID << " has " << objectScene->mMeshes[m]->mNumFaces << " faces")

			if (objectScene->mMeshes[m]->HasVertexColors(0))
			{
				LOG4CXX_INFO(logger, meshID << " has vertex colors");
				dscp4_AddMesh(meshID.c_str(), objectScene->mMeshes[m]->mNumVertices, (float*)objectScene->mMeshes[m]->mVertices, (float*)objectScene->mMeshes[m]->mNormals, (float*)objectScene->mMeshes[m]->mColors[0]);
			}
			else
			{
				LOG4CXX_WARN(logger, meshID << " does not have vertex colors--it may look dull");
				dscp4_AddMesh(meshID.c_str(), objectScene->mMeshes[m]->mNumVertices, (float*)objectScene->mMeshes[m]->mVertices, (float*)objectScene->mMeshes[m]->mNormals);
			}
				
		}
		else
			LOG4CXX_DEBUG(logger, "Found mesh " << m << " with no faces.  Treating vertecies as point cloud");
	}

	for (unsigned int m = 0; m < objectScene->mNumMaterials; m++)
	{
		aiString key;
		for (unsigned int p = 0; p < objectScene->mMaterials[m]->mNumProperties; p++)
		{
			key = objectScene->mMaterials[m]->mProperties[p]->mKey;
		}

		aiColor3D color;
		objectScene->mMaterials[m]->Get(AI_MATKEY_COLOR_DIFFUSE, color);

		// if it has faces, treat as mesh, otherwise as point cloud
		if (true)
		{
			//LOG4CXX_DEBUG(logger, "Found mesh " << m << " with " << objectScene->mMeshes[m]->mNumFaces << " faces from 3D object file...");
			//AddMesh((float*)objectScene->mMeshes[m]->mVertices, objectScene->mMeshes[m]->mNumVertices);
		}
		else
			LOG4CXX_DEBUG(logger, "Found Mesh " << m << " with no faces.  Treating vertecies as point cloud");
	}


	for (size_t i = 0; i < 5; i++)
	{
		std::this_thread::sleep_for(std::chrono::milliseconds(1000));
	}

	dscp4_RemoveMesh("Mesh 0");

	for (size_t i = 0; i < 5; i++)
	{
		std::this_thread::sleep_for(std::chrono::milliseconds(1000));
	}

	dscp4_DeinitRenderer();

	LOG4CXX_INFO(logger, "DSCP4 test program successfully exited");

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