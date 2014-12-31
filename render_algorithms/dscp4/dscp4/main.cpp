
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

#ifndef WIN32
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

#ifndef WIN32
char _getch(void);
int _kbhit(void);
#endif

int main(int argc, const char* argv[])
{
#ifdef DSCP4_HAVE_LOG4CXX
	auto logger = createLogger();
	logger->setLevel(log4cxx::Level::getError());
	int logLevel = DSCP4_DEFAULT_VERBOSITY;
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

	render_options_t renderOptions = { 0 };
	algorithm_options_t algorithmOptions = { 0 };
	display_options_t displayOptions = { 0 };
	std::string displayName;
	std::string shadersPath;
	std::string kernelsPath;
	std::string shaderFileNamePrefix;

	float translateX = 0, translateY = 0, translateZ = 0;
	float scaleX = 0, scaleY = 0, scaleZ = 0;
	float rotateAngleX = 0, rotateAngleY = 0;

	DSCP4ProgramOptions options;

	try
	{
		options.parseConfigFile();
	}
	catch (std::exception&)
	{
		LOG4CXX_FATAL(logger, "Could not parse dscp4.conf file. This is copied to /etc/dscp4 folder in Linux, or %PROGRAMDATA%\\\\dscp4 folder in Windows during the install process. Make sure you've built and installed dscp4 properly, and your account has permission to access the file. Exiting...")
		return -1;
	}

	try {
		options.parseCommandLine(argc, argv);
	}
	catch (std::exception&)
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
	try
	{
		logLevel = options.getVerbosity();
	}
	catch (std::exception& e)
	{
		LOG4CXX_ERROR(logger, "Could not parse verbosity setting from command line or conf file: " << e.what())
	}

	switch (logLevel)
	{
	case 0:
		logger->setLevel(log4cxx::Level::getFatal());
		break;
	case 1:
		logger->setLevel(log4cxx::Level::getWarn());
		break;
	case 2:
		logger->setLevel(log4cxx::Level::getError());
		break;
	case 3:
		logger->setLevel(log4cxx::Level::getInfo());
		break;
	case 4:
		logger->setLevel(log4cxx::Level::getDebug());
		break;
	case 5:
		logger->setLevel(log4cxx::Level::getTrace());
		break;
	case 6:
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
		catch (std::exception&)
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


	try{
		renderOptions.render_mode = options.getRenderMode() == "viewing" ? DSCP4_RENDER_MODE_MODEL_VIEWING : options.getRenderMode() == "stereogram" ? DSCP4_RENDER_MODE_STEREOGRAM_VIEWING : DSCP4_RENDER_MODE_HOLOVIDEO_FRINGE;
		renderOptions.shader_model = options.getShadeModel() == "off" ? DSCP4_SHADER_MODEL_OFF : options.getShadeModel() == "flat" ? DSCP4_SHADER_MODEL_FLAT : DSCP4_SHADER_MODEL_SMOOTH;
		renderOptions.auto_scale_enabled = options.getAutoscale();

		shadersPath = options.getShadersPath().string();
		kernelsPath = options.getKernelsPath().string();
		shaderFileNamePrefix = options.getShaderFileName();

		renderOptions.shaders_path = shadersPath.c_str();
		renderOptions.kernels_path = kernelsPath.c_str();
		renderOptions.shader_filename_prefix = shaderFileNamePrefix.c_str();
		renderOptions.light_pos_x = options.getLightPosX();
		renderOptions.light_pos_y = options.getLightPosY();
		renderOptions.light_pos_z = options.getLightPosZ();
	}
	catch (std::exception& e)
	{
		LOG4CXX_FATAL(logger, "Could not parse render options from command line or conf file: " << e.what())
		return -1;
	}

	LOG4CXX_DEBUG(logger, "Render options parsed")

	switch (renderOptions.render_mode)
	{
	case DSCP4_RENDER_MODE_MODEL_VIEWING:
		LOG4CXX_INFO(logger, "Rendering mode set to model viewing mode (for testing renderer on a normal display)")
			break;
	case DSCP4_RENDER_MODE_STEREOGRAM_VIEWING:
		LOG4CXX_INFO(logger, "Rendering mode set to panoramagram viewing mode (for testing renderer on a normal display)")
			break;
	case DSCP4_RENDER_MODE_HOLOVIDEO_FRINGE:
		LOG4CXX_INFO(logger, "Rendering mode set to holovideo fringe computation (for holovideo output)")
			break;
	default:
		break;
	}

	if (renderOptions.shader_model == DSCP4_SHADER_MODEL_OFF)
	{
		LOG4CXX_INFO(logger, "Turned off all lighting effects for renderer")
	}
	else if (renderOptions.shader_model == DSCP4_SHADER_MODEL_FLAT)
	{
		LOG4CXX_INFO(logger, "Set renderer shading model to 'flat'")
		if (generateNormals == "smooth")
		{
			LOG4CXX_WARN(logger, "Your render shading model is set to flat, but input options are to generate smooth normals. Your model will probably look ugly...")
		}
	}
	else if (renderOptions.shader_model == DSCP4_SHADER_MODEL_SMOOTH)
	{
		LOG4CXX_INFO(logger, "Set renderer shading model to 'smooth'")
		if (generateNormals == "flat")
		{
			LOG4CXX_WARN(logger, "Your render shading model is set to smooth, but input options are to generate flat normals. Your model will probably look ugly...")
		}
	}

	if (renderOptions.auto_scale_enabled)
	{
		LOG4CXX_INFO(logger, "Renderer model autoscaling and centering enabled")
	}
	else
	{
		LOG4CXX_INFO(logger, "Renderer model autoscaling and centering disabled")
	}

	try {
		algorithmOptions.num_views_x = options.getNumViewsX();
		algorithmOptions.num_views_y = options.getNumViewsY();
		algorithmOptions.num_wafels_per_scanline = options.getNumWafelsPerScanline();
		algorithmOptions.num_scanlines = options.getNumScanlines();
		algorithmOptions.fov_x = options.getFovX();
		algorithmOptions.fov_y = options.getFovY();
	}
	catch (std::exception& e)
	{
		LOG4CXX_FATAL(logger, "Could not parse algorithm options from conf file or command line: " << e.what())
		return -1;
	}

	LOG4CXX_INFO(logger, "Algorithm options parsed")
	LOG4CXX_INFO(logger, "Number of stereogram views in X: " << algorithmOptions.num_views_x)
	LOG4CXX_INFO(logger, "Number of stereogram views in Y: " << algorithmOptions.num_views_y)
	LOG4CXX_INFO(logger, "Number of wafels per scanline: " << algorithmOptions.num_wafels_per_scanline)
	LOG4CXX_INFO(logger, "Number of scanlines: " << algorithmOptions.num_scanlines)

	try{
		displayName = options.getDisplayName();
		displayOptions.name = displayName.c_str();
		displayOptions.num_heads = options.getNumHeads();
		displayOptions.head_res_x = options.getHeadResX();
		displayOptions.head_res_y = options.getHeadResY();
	}
	catch (std::exception& e)
	{
		LOG4CXX_FATAL(logger, "Could not parse display options from conf file or command line: " << e.what())
		return -1;
	}

	LOG4CXX_DEBUG(logger, "Display options parsed")
	LOG4CXX_INFO(logger, "Display name: " << displayOptions.name)
	LOG4CXX_INFO(logger, "Number of heads (physical ports): " << displayOptions.num_heads)
	LOG4CXX_INFO(logger, "Head horizontal resolution (in pixels): " << displayOptions.head_res_x)
	LOG4CXX_INFO(logger, "Head vertical resolution (in pixels): " << displayOptions.head_res_y)

	LOG4CXX_DEBUG(logger, "Starting DSCP4 lib")
	renderContext = dscp4_CreateContext(renderOptions, algorithmOptions, displayOptions, logLevel);

	if (!dscp4_InitRenderer(renderContext))
	{
		LOG4CXX_FATAL(logger, "Could not initialize DSCP4 lib")
		return -1;
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
				case 'z':
					dscp4_TranslateObject(renderContext, "Mesh 0", translateX*0.01f, translateY*0.01f, ++translateZ*0.01f);
					break;
				case 'x':
					dscp4_TranslateObject(renderContext, "Mesh 0", translateX*0.01f, translateY*0.01f, --translateZ*0.01f);
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

#ifndef WIN32

char _getch(){
	/*#include <unistd.h>   //_getch*/
	/*#include <termios.h>  //_getch*/
	char buf=0;
	struct termios old={0};
	fflush(stdout);
	if(tcgetattr(0, &old)<0)
		perror("tcsetattr()");
	old.c_lflag&=~ICANON;
	old.c_lflag&=~ECHO;
	old.c_cc[VMIN]=1;
	old.c_cc[VTIME]=0;
	if(tcsetattr(0, TCSANOW, &old)<0)
		perror("tcsetattr ICANON");
	if (read(0, &buf, 1)<0)
		perror("read()");
	old.c_lflag |= ICANON;
	old.c_lflag |= ECHO;
	if (tcsetattr(0, TCSADRAIN, &old)<0)
		perror("tcsetattr ~ICANON");
	//printf("%c\n", buf);
	return buf;
}

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
