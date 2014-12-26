
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

#ifdef UNIX
#include <stdio.h>
#include <termios.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/time.h>
#else
#include <conio.h>
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

#ifdef UNIX
int _kbhit(void);
#endif

int main(int argc, const char* argv[])
{
#ifdef DSCP4_HAVE_LOG4CXX
	auto logger = createLogger();
	logger->setLevel(log4cxx::Level::getError());
	int logLevel;
#endif

	int key = 0;
	bool shouldRun = true;
	dscp4_context_t renderContext = nullptr;
	boost::filesystem::path objectFilePath;
	Assimp::Importer objectFileImporter;
	const aiScene* objectScene = nullptr;
	bool triangulateMesh = false;
	std::string generateNormals;
	unsigned int aiFlags = 0;
	bool autoScaleEnabled = false;
	render_mode_t renderMode;

	float translateX = 0, translateY = 0, translateZ = 0;
	float scaleX = 0, scaleY = 0, scaleZ = 0;
	float rotateAngleX = 0, rotateAngleY = 0;

	DSCP4ProgramOptions options;

	try
	{
		options.parseConfigFile();
	}
	catch (std::exception)
	{
		LOG4CXX_WARN(logger, "Could not find dscp4.conf file. Using default option values, but your milage will vary")
	}

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

#ifdef DSCP4_HAVE_LOG4CXX
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
#endif

	objectFilePath = boost::filesystem::path(options.getFileName());

	if (!boost::filesystem::exists(objectFilePath))
	{
		try {
			objectFilePath = options.getModelsPath() / objectFilePath;
		}
		catch (std::exception)
		{
			LOG4CXX_ERROR(logger, "Couldn't find model path from '" << DSCP4_CONF_FILENAME << "' file")
		}

		if (!boost::filesystem::exists(objectFilePath))
		{
			std::cout << "Invalid 3D object input file path (file not found)" << std::endl;
			options.printOptions(DSCP4ProgramOptions::DSCP4_OPTIONS_TYPE_INPUT);
			return -1;
		}
	}

	generateNormals = options.getGenerateNormals();
	if (generateNormals != "off")
	{
		if (generateNormals == "flat")
		{
			aiFlags |= aiProcess_GenNormals;
			LOG4CXX_DEBUG(logger, "Set flag for asset importer to generate flat normals (if don't exist)")
		}
		else if (generateNormals == "smooth")
		{
			aiFlags |= aiProcess_GenSmoothNormals;
			LOG4CXX_DEBUG(logger, "Set flag for asset importer to generate smooth normals (if don't exist)")
		}
		else
		{
			std::cout << "Invalid generate normals option: '" << generateNormals << "'. Valid choices are 'off', 'flat' or 'smooth'" << std::endl;
			options.printOptions(DSCP4ProgramOptions::DSCP4_OPTIONS_TYPE_INPUT);
			return -1;
		}
		
	}

	triangulateMesh = options.getTriangulateMesh();
	if (triangulateMesh)
	{
		aiFlags |= aiProcess_Triangulate;
		LOG4CXX_DEBUG(logger, "Set flag for asset importer to triangulate 3d object file mesh")
	}

	LOG4CXX_INFO(logger, "Starting DSCP4 test program...")

	LOG4CXX_INFO(logger, "Loading 3D object file \'" << objectFilePath.filename().string() << "\'...")
	objectScene = objectFileImporter.ReadFile(objectFilePath.string(), aiFlags);

	if (!objectScene->HasMeshes())
	{
		LOG4CXX_FATAL(logger, "3D object file does not appear to have any meshes")
		return -1;
	}

	LOG4CXX_INFO(logger, "Starting DSCP4 lib")
	renderContext = dscp4_CreateContext();

	std::string renderModeString = options.getRenderMode();
	renderMode = renderModeString == "viewing" ? DSCP4_RENDER_MODE_MODEL_VIEWING : renderModeString == "stereogram" ? DSCP4_RENDER_MODE_STEREOGRAM_VIEWING : DSCP4_RENDER_MODE_HOLOVIDEO_FRINGE;
	dscp4_SetRenderMode(renderContext, renderMode);

	switch (renderMode)
	{
	case DSCP4_RENDER_MODE_MODEL_VIEWING:
		LOG4CXX_INFO(logger, "Rendering mode set to model viewing mode (for testing renderer on a normal display)")
			break;
	case DSCP4_RENDER_MODE_STEREOGRAM_VIEWING:
		LOG4CXX_INFO(logger, "Rendering mode set to panoramagram viewing mode (for testing renderer on a normal display)")
			break;
	case DSCP4_RENDER_MODE_HOLOVIDEO_FRINGE:
		LOG4CXX_INFO(logger, "Rendering mode set to holovideo fringe computation (for holovideo output)")
	default:
		break;
	}

	if (!dscp4_InitRenderer(renderContext))
	{
		LOG4CXX_FATAL(logger, "Could not initialize DSCP4 lib")
		return -1;
	}

	std::string shadeModelString = options.getShadeModel();
	if (shadeModelString == "off")
	{
		dscp4_SetShadeModel(renderContext, DSCP4_SHADE_MODEL_OFF);
		LOG4CXX_DEBUG(logger, "Turned off all lighting effects for renderer")
	}
	else if (shadeModelString == "flat")
	{
		dscp4_SetShadeModel(renderContext, DSCP4_SHADE_MODEL_FLAT);
		LOG4CXX_DEBUG(logger, "Set renderer shading model to 'flat'")
	}
	else if (shadeModelString == "smooth")
	{
		dscp4_SetShadeModel(renderContext, DSCP4_SHADE_MODEL_SMOOTH);
		LOG4CXX_DEBUG(logger, "Set renderer shading model to 'smooth'")
	}

	if ((shadeModelString == "flat" && generateNormals == "smooth") ||
		(shadeModelString == "smooth" && generateNormals == "flat"))
	{
		LOG4CXX_WARN(logger, "Your normal generation mode and shading model are mis-matching.  Your model will probably look like crap")
	}

	autoScaleEnabled = options.getAutoscale();
	dscp4_SetAutoScaleEnabled(renderContext, autoScaleEnabled);
	if (autoScaleEnabled)
	{
		LOG4CXX_DEBUG(logger, "Renderer model autoscaling and centering enabled")
	}
	else
	{
		LOG4CXX_DEBUG(logger, "Renderer model autoscaling and centering disabled")
	}

	for (unsigned int m = 0; m < objectScene->mNumMeshes; m++)
	{
		// if it has faces, treat as mesh, otherwise as point cloud
		if (objectScene->mMeshes[m]->HasFaces())
		{
			std::string meshID;
			meshID += std::string("Mesh ") += std::to_string(m);
			LOG4CXX_INFO(logger, "Found " << meshID << " from 3D object file '" << objectFilePath.string() << "'")
			LOG4CXX_INFO(logger, meshID << " has " << objectScene->mMeshes[m]->mNumVertices << " vertices")
			if (objectScene->mMeshes[m]->HasNormals())
			{
				LOG4CXX_INFO(logger, meshID << " has normals")
			}
			else
			{
				LOG4CXX_WARN(logger, meshID << " does not have normals, lighting effects will look fucked up")
			}

			LOG4CXX_INFO(logger, meshID << " has " << objectScene->mMeshes[m]->mNumFaces << " faces")

			if (objectScene->mMeshes[m]->HasVertexColors(0))
			{
				LOG4CXX_INFO(logger, meshID << " has vertex colors")
				dscp4_AddMesh(renderContext, meshID.c_str(), objectScene->mMeshes[m]->mNumVertices, (float*)objectScene->mMeshes[m]->mVertices, (float*)objectScene->mMeshes[m]->mNormals, (float*)objectScene->mMeshes[m]->mColors[0]);
			}
			else
			{
				LOG4CXX_WARN(logger, meshID << " does not have vertex colors--it may look dull")
				dscp4_AddMesh(renderContext, meshID.c_str(), objectScene->mMeshes[m]->mNumVertices, (float*)objectScene->mMeshes[m]->mVertices, (float*)objectScene->mMeshes[m]->mNormals);
			}
				
		}
		else
		{
			LOG4CXX_DEBUG(logger, "Found mesh " << m << " with no faces.  Treating vertecies as point cloud")
		}
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
		{
			LOG4CXX_DEBUG(logger, "Found Mesh " << m << " with no faces.  Treating vertecies as point cloud")
		}
	}


	//for (size_t i = 0; i < 5; i++)
	while (shouldRun)
	{
		if (_kbhit())
		{
			key = _getch();

			if (key == 224) { // if the first value is esc
				switch (_getch()) { // the real value
				case 72:
					dscp4_RotateObject(renderContext, "Mesh 0", ++rotateAngleX, -1.0f, 0.0f, 0.0f);
					break;
				case 80:
					dscp4_RotateObject(renderContext, "Mesh 0", --rotateAngleX, -1.0f, 0.0f, 0.0f);
					// code for arrow down
					break;
				case 75:
					// code for arrow right
					dscp4_RotateObject(renderContext, "Mesh 0", ++rotateAngleY, 0.0f, -1.0f, 0.0f);
					break;
				case 77:
					// code for arrow left
					dscp4_RotateObject(renderContext, "Mesh 0", --rotateAngleY, 0.0f, -1.0f, 0.0f);
					break;
				}
			}
			else
			{
				switch (key)
				{
				case 'w':
					dscp4_TranslateObject(renderContext, "Mesh 0", translateX*0.01f, ++translateY*0.01f, translateZ*0.01f);
					break;
				case 's':
					dscp4_TranslateObject(renderContext, "Mesh 0", translateX*0.01f, --translateY*0.01f, translateZ*0.01f);
					break;
				case 'd':
					dscp4_TranslateObject(renderContext, "Mesh 0", ++translateX*0.01f, translateY*0.01f, translateZ*0.01f);
					break;
				case 'a':
					dscp4_TranslateObject(renderContext, "Mesh 0", --translateX*0.01f, translateY*0.01f, translateZ*0.01f);
					break;
				case '=':
					dscp4_ScaleObject(renderContext, "Mesh 0", ++scaleX*0.1f, ++scaleY*0.1f, ++scaleZ*0.1f);
					break;
				case '-':
					dscp4_ScaleObject(renderContext, "Mesh 0", --scaleX*0.1f, --scaleY*0.1f, --scaleZ*0.1f);
					break;
				case 'q':
					LOG4CXX_INFO(logger, "Quit key detected, quitting...")
						shouldRun = false;
					break;
				default:
					break;
				}
			}
		}
		else
			std::this_thread::sleep_for(std::chrono::milliseconds(10));
	}



	dscp4_RemoveMesh(renderContext, "Mesh 0");

	dscp4_DeinitRenderer(renderContext);

	dscp4_DestroyContext(&renderContext);

	LOG4CXX_INFO(logger, "DSCP4 test program successfully exited")

	return 0;
}

#ifdef UNIX
int _kbhit(void)
{
	struct timeval tv;
	fd_set rdfs;

	tv.tv_sec = 0;
	tv.tv_usec = 0;

	FD_ZERO(&rdfs);
	FD_SET(STDIN_FILENO, &rdfs);

	select(STDIN_FILENO + 1, &rdfs, NULL, NULL, &tv);
	return FD_ISSET(STDIN_FILENO, &rdfs);
}
#endif

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