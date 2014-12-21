//A simple class for handling GLSL shader compilation
//Auhtor: Movania Muhammad Mobeen
#pragma once

#ifdef DSCP4_HAVE_LOG4CXX
#include <log4cxx/logger.h>
#endif

#include <GL/glew.h>
#include <map>
#include <string>

namespace dscp4
{
	class GLSLShader
	{
	public:
		GLSLShader(void);
		~GLSLShader(void);
		void LoadFromString(GLenum whichShader, const std::string& source);
		void LoadFromFile(GLenum whichShader, const std::string& filename);
		void CreateAndLinkProgram();
		void Use();
		void UnUse();
		void AddAttribute(const std::string& attribute);
		void AddUniform(const std::string& uniform);
		GLuint GetProgram() const;
		//An indexer that returns the location of the attribute/uniform
		GLuint operator[](const std::string& attribute);
		GLuint operator()(const std::string& uniform);
		//Program deletion
		void DeleteProgram() { glDeleteProgram(program_); program_ = -1; }
	private:
		enum ShaderType { VERTEX_SHADER, FRAGMENT_SHADER, GEOMETRY_SHADER };
		GLuint	program_;
		int totalShaders_;
		GLuint shaders_[3];//0->vertexshader, 1->fragmentshader, 2->geometryshader
		std::map<std::string, GLuint> attributeList_;
		std::map<std::string, GLuint> uniformLocationList_;

#ifdef DSCP4_HAVE_LOG4CXX
		log4cxx::LoggerPtr logger_ = log4cxx::Logger::getLogger("edu.mit.media.obmg.dscp4.lib.render");
#endif

	};
}
