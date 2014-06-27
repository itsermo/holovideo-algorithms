#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include "setupglew.h"
#include <GL/gl.h>
#include <GL/glu.h>
#include <tiffio.h>
#include <string.h>


#include "orientation.h"
#include "light.h"
#include "texture.h"
#include "texturecoord.h"
#include "material.h"
#include "drawable.h"
#include "triangles.h"
#include "primitives.h"
#include "object.h"
#include "model.h"
#include "render.h"
#include "holoren.h"
#include "parser.h"
#include "utils.h"

#ifdef OLDLINUX

#include <util/PlatformUtils.hpp>
#include <parsers/DOMParser.hpp>
#include <dom/DOM_Node.hpp>
#include <dom/DOM_NamedNodeMap.hpp>

#else

#include <xercesc/util/PlatformUtils.hpp>
#include <xercesc/dom/deprecated/DOMParser.hpp>
#include <xercesc/dom/deprecated/DOM_Document.hpp>
#include <xercesc/dom/deprecated/DOM_Node.hpp>
#include <xercesc/dom/deprecated/DOM_NamedNodeMap.hpp>
using namespace xercesc;

#endif

#define C_ROOT 0
#define C_HOLOREN 1
#define C_RENDER 2
#define C_LIGHTING 3
#define C_MATERIALS 4
#define C_MODEL 5
#define C_LIGHT 6
#define C_MATERIAL 7
#define C_ORIENTATION 8
#define C_OBJECTS 9
#define C_OBJECT 10
#define C_TEXTURES 11
#define C_TEXTURE 12
#define C_MODELS 13
#define C_TEXTURES_VAL 14
#define C_TEXTURE_COORD 15
#define C_TEXTURE_COORDS 16

int parseHoloRen(DOM_Node root, void *_data);
int parseRender(DOM_Node root, void *_data);
int parseLighting(DOM_Node root, void *_data);
int parseMaterials(DOM_Node root, void *_data);
int parseModel(DOM_Node root, void *_data);
int parseLight(DOM_Node root, void *_data);
int parseMaterial(DOM_Node root, void *_data);
int parseOrientation(DOM_Node root, void *_data);
int parseObjects(DOM_Node root, void *_data);
int parseObject(DOM_Node root, void *_data);
int parseTexturesVal(DOM_Node root, void *_data);
int parseTextureCoord(DOM_Node root, void *_data);
int parseTextureCoords(DOM_Node root, void *_data);

int parseNode(DOM_Node root, int context, void *data);
void loopChildren(DOM_Node root, int context, void *data);

float DOMStringToF(DOMString s);
int DOMStringToD(DOMString s);
int DOMStringToBool(DOMString s);
	
void loopChildren(DOM_Node root, int context, void *data)
{
	DOM_Node child = root.getFirstChild();
	while(child != 0)
	{
		parseNode(child, context, data);
		child = child.getNextSibling();
	}
}

float DOMStringToF(DOMString s)
{
	char *t = s.transcode();
	float v = (float) atof(t);
	//delete [] t;
	return v;
}

int DOMStringToD(DOMString s)
{
	char *t = s.transcode();
	int v = atoi(t);
	//delete [] t;
	return v;
}

int DOMStringToBool(DOMString s)
{
	char *t = s.transcode();
	int v = (!strcmp(t, "true")) || (!strcmp(t, "TRUE")) || (!strcmp(t, "1"));
	//delete [] t;
	return v;
}

int parseHoloRen(DOM_Node root, void *_data)
{
	holoConf *data = (holoConf *) _data;
	char *nodeName = root.getNodeName().transcode();

	if(!strcmp(nodeName, "render"))
	{
		if(data->ren) delete data->ren;
		data->ren = new renderConf;

		DOM_NamedNodeMap attributes = root.getAttributes();
		int attrCount = attributes.getLength();
		for(int i = 0; i < attrCount; i++)
		{
			DOM_Node attr = attributes.item(i);

			char *attrName = attr.getNodeName().transcode();

			if(!strcmp(attrName, "double_camera")) data->ren->doubleCamera = DOMStringToBool(attr.getNodeValue());
			else if(!strcmp(attrName, "recenter")) data->ren->recenter = DOMStringToBool(attr.getNodeValue());
			
			//delete [] attrName;
		}
		
		loopChildren(root, C_RENDER, (void *) data->ren);
	}
	else if(!strcmp(nodeName, "lighting"))
	{
		if(data->lighting) delete data->lighting;
		data->lighting = new lightingConf;

		DOM_NamedNodeMap attributes = root.getAttributes();
		int attrCount = attributes.getLength();
		for(int i = 0; i < attrCount; i++)
		{
			DOM_Node attr = attributes.item(i);


			char *attrName = attr.getNodeName().transcode();

			if(!strcmp(attrName, "nlights")) data->lighting->setNLights(DOMStringToD(attr.getNodeValue()));
			
			//delete [] attrName;
		}
		
		loopChildren(root, C_LIGHTING, (void *) data->lighting);
	}
	else if(!strcmp(nodeName, "textures"))
	{
		if(data->textures) delete data->textures;
		data->textures = new texturesConf;

		DOM_NamedNodeMap attributes = root.getAttributes();
		int attrCount = attributes.getLength();
		for(int i = 0; i < attrCount; i++)
		{
			DOM_Node attr = attributes.item(i);

			char *attrName = attr.getNodeName().transcode();

			if(!strcmp(attrName, "ntextures")) data->textures->setNTextures(DOMStringToD(attr.getNodeValue()));
			else if(!strcmp(attrName, "path"))
			{
				char *t = attr.getNodeValue().transcode();
				strcpy(data->texPath, t);
				//delete [] t;
			}
			
			//delete [] attrName;
		}
		
		loopChildren(root, C_TEXTURES, (void *) data->textures);
	}
	else if(!strcmp(nodeName, "materials"))
	{
		if(data->materials) delete data->materials;
		data->materials = new materialsConf;

		DOM_NamedNodeMap attributes = root.getAttributes();
		int attrCount = attributes.getLength();
		for(int i = 0; i < attrCount; i++)
		{
			DOM_Node attr = attributes.item(i);

			char *attrName = attr.getNodeName().transcode();

			if(!strcmp(attrName, "nmaterials")) data->materials->setNMaterials(DOMStringToD(attr.getNodeValue()));
			
			//delete [] attrName;
		}
		
		loopChildren(root, C_MATERIALS, (void *) data->materials);
	}
	else if(!strcmp(nodeName, "models"))
	{
		if(data->models) delete data->models;
		data->models = new modelsConf;

		DOM_NamedNodeMap attributes = root.getAttributes();
		int attrCount = attributes.getLength();
		for(int i = 0; i < attrCount; i++)
		{
			DOM_Node attr = attributes.item(i);

			char *attrName = attr.getNodeName().transcode();

			if(!strcmp(attrName, "nmodels")) data->models->setNModels(DOMStringToD(attr.getNodeValue()));
			
			//delete [] attrName;
		}
		
		loopChildren(root, C_MODELS, (void *) data->models);
	}

	//delete [] nodeName;

	return 0;
}

int parseLight(DOM_Node root, void *_data)
{
	lightConf *data = (lightConf *) _data;
	char *nodeName = root.getNodeName().transcode();

	if(!strcmp(nodeName, "position"))
	{
		DOM_NamedNodeMap attributes = root.getAttributes();
		int attrCount = attributes.getLength();
		for(int i = 0; i < attrCount; i++)
		{
			DOM_Node attr = attributes.item(i);

			char *attrName = attr.getNodeName().transcode();

			if(!strcmp(attrName, "x")) data->position[0] = DOMStringToF(attr.getNodeValue());
			else if(!strcmp(attrName, "y")) data->position[1] = DOMStringToF(attr.getNodeValue());
			else if(!strcmp(attrName, "z")) data->position[2] = DOMStringToF(attr.getNodeValue());
			else if(!strcmp(attrName, "w")) data->position[3] = DOMStringToF(attr.getNodeValue());

			//delete [] attrName;
		}
	}
	else if(!strcmp(nodeName, "direction"))
	{
		DOM_NamedNodeMap attributes = root.getAttributes();
		int attrCount = attributes.getLength();
		for(int i = 0; i < attrCount; i++)
		{
			DOM_Node attr = attributes.item(i);

			char *attrName = attr.getNodeName().transcode();

			if(!strcmp(attrName, "x")) data->direction[0] = DOMStringToF(attr.getNodeValue());
			else if(!strcmp(attrName, "y")) data->direction[1] = DOMStringToF(attr.getNodeValue());
			else if(!strcmp(attrName, "z")) data->direction[2] = DOMStringToF(attr.getNodeValue());
			else if(!strcmp(attrName, "w")) data->direction[3] = DOMStringToF(attr.getNodeValue());

			//delete [] attrName;
		}
	}
	else if(!strcmp(nodeName, "ambient"))
	{
		DOM_NamedNodeMap attributes = root.getAttributes();
		int attrCount = attributes.getLength();
		for(int i = 0; i < attrCount; i++)
		{
			DOM_Node attr = attributes.item(i);

			char *attrName = attr.getNodeName().transcode();

			if(!strcmp(attrName, "r")) data->ambient[0] = DOMStringToF(attr.getNodeValue());
			else if(!strcmp(attrName, "g")) data->ambient[1] = DOMStringToF(attr.getNodeValue());
			else if(!strcmp(attrName, "b")) data->ambient[2] = DOMStringToF(attr.getNodeValue());
			else if(!strcmp(attrName, "a")) data->ambient[3] = DOMStringToF(attr.getNodeValue());

			//delete [] attrName;
		}
	}
	else if(!strcmp(nodeName, "diffuse"))
	{
		DOM_NamedNodeMap attributes = root.getAttributes();
		int attrCount = attributes.getLength();
		for(int i = 0; i < attrCount; i++)
		{
			DOM_Node attr = attributes.item(i);

			char *attrName = attr.getNodeName().transcode();

			if(!strcmp(attrName, "r")) data->diffuse[0] = DOMStringToF(attr.getNodeValue());
			else if(!strcmp(attrName, "g")) data->diffuse[1] = DOMStringToF(attr.getNodeValue());
			else if(!strcmp(attrName, "b")) data->diffuse[2] = DOMStringToF(attr.getNodeValue());
			else if(!strcmp(attrName, "a")) data->diffuse[3] = DOMStringToF(attr.getNodeValue());

			//delete [] attrName;
		}
	}
	else if(!strcmp(nodeName, "specular"))
	{
		DOM_NamedNodeMap attributes = root.getAttributes();
		int attrCount = attributes.getLength();
		for(int i = 0; i < attrCount; i++)
		{
			DOM_Node attr = attributes.item(i);

			char *attrName = attr.getNodeName().transcode();

			if(!strcmp(attrName, "r")) data->specular[0] = DOMStringToF(attr.getNodeValue());
			else if(!strcmp(attrName, "g")) data->specular[1] = DOMStringToF(attr.getNodeValue());
			else if(!strcmp(attrName, "b")) data->specular[2] = DOMStringToF(attr.getNodeValue());
			else if(!strcmp(attrName, "a")) data->specular[3] = DOMStringToF(attr.getNodeValue());

			//delete [] attrName;
		}
	}
	else if(!strcmp(nodeName, "spotexponent"))
	{
		DOM_NamedNodeMap attributes = root.getAttributes();
		int attrCount = attributes.getLength();
		for(int i = 0; i < attrCount; i++)
		{
			DOM_Node attr = attributes.item(i);

			char *attrName = attr.getNodeName().transcode();

			if(!strcmp(attrName, "val")) data->spotExponent = DOMStringToF(attr.getNodeValue());

			//delete [] attrName;
		}
	}
	else if(!strcmp(nodeName, "spotcuttoff"))
	{
		DOM_NamedNodeMap attributes = root.getAttributes();
		int attrCount = attributes.getLength();
		for(int i = 0; i < attrCount; i++)
		{
			DOM_Node attr = attributes.item(i);

			char *attrName = attr.getNodeName().transcode();

			if(!strcmp(attrName, "val")) data->spotCutoff = DOMStringToF(attr.getNodeValue());

			//delete [] attrName;
		}
	}
	else if(!strcmp(nodeName, "constantattenuation"))
	{
		DOM_NamedNodeMap attributes = root.getAttributes();
		int attrCount = attributes.getLength();
		for(int i = 0; i < attrCount; i++)
		{
			DOM_Node attr = attributes.item(i);

			char *attrName = attr.getNodeName().transcode();

			if(!strcmp(attrName, "val")) data->constantAttenuation = DOMStringToF(attr.getNodeValue());

			//delete [] attrName;
		}
	}
	else if(!strcmp(nodeName, "linearattenuation"))
	{
		DOM_NamedNodeMap attributes = root.getAttributes();
		int attrCount = attributes.getLength();
		for(int i = 0; i < attrCount; i++)
		{
			DOM_Node attr = attributes.item(i);

			char *attrName = attr.getNodeName().transcode();

			if(!strcmp(attrName, "val")) data->linearAttenuation = DOMStringToF(attr.getNodeValue());

			//delete [] attrName;
		}
	}
	else if(!strcmp(nodeName, "quadraticattenuation"))
	{
		DOM_NamedNodeMap attributes = root.getAttributes();
		int attrCount = attributes.getLength();
		for(int i = 0; i < attrCount; i++)
		{
			DOM_Node attr = attributes.item(i);

			char *attrName = attr.getNodeName().transcode();

			if(!strcmp(attrName, "val")) data->quadraticAttenuation = DOMStringToF(attr.getNodeValue());

			//delete [] attrName;
		}
	}

	//delete [] nodeName;
	
	return 0;
}

int parseMaterials(DOM_Node root, void *_data)
{
	materialsConf *data = (materialsConf*) _data;
	char *nodeName = root.getNodeName().transcode();

	if(!strcmp(nodeName, "material"))
	{
		DOM_NamedNodeMap attributes = root.getAttributes();
		int materialN = 0;
		
		int attrCount = attributes.getLength();
		for(int i = 0; i < attrCount; i++)
		{
			DOM_Node attr = attributes.item(i);

			char *attrName = attr.getNodeName().transcode();

			if(!strcmp(attrName, "num")) materialN = DOMStringToD(attr.getNodeValue());

			//delete [] attrName;
		}
		
		if(materialN < data->nMaterials)
		{
			if(data->materials[materialN])
			{
				delete data->materials[materialN];
				data->materials[materialN] = new materialConf();
			}
			loopChildren(root, C_MATERIAL, data->materials[materialN]);
		}
	}
	
	//delete [] nodeName;
	

	return 0;
}

int parseTexture(DOM_Node root, void *_data)
{
	textureConf *data = (textureConf *) _data;
	char *nodeName = root.getNodeName().transcode();
	
	if(!strcmp(nodeName, "data"))
	{
		DOM_NamedNodeMap attributes = root.getAttributes();
		
		int attrCount = attributes.getLength();
		for(int i = 0; i < attrCount; i++)
		{
			DOM_Node attr = attributes.item(i);

			char *attrName = attr.getNodeName().transcode();

			if(!strcmp(attrName, "file"))
			{
				char *t = attr.getNodeValue().transcode();


				char actualpath[PATH_MAX +1];
				strcpy(data->fName, t);
				//delete [] t;
			}
			else if(!strcmp(attrName, "mode"))
			{
				char *t = attr.getNodeValue().transcode();
				
				if(!strcmp(t, "normal")) data->mode = TEX_NORMAL;
				else if(!strcmp(t, "cubemap")) data->mode = TEX_CUBEMAP;

				//delete [] t;
			}
			else if(!strcmp(attrName, "front"))
			{
				char *t = attr.getNodeValue().transcode();
				strcpy(data->cmFName[TEX_FRONT], t);
				//delete [] t;
			}
			else if(!strcmp(attrName, "back"))
			{
				char *t = attr.getNodeValue().transcode();
				strcpy(data->cmFName[TEX_BACK], t);
				//delete [] t;
			}
			else if(!strcmp(attrName, "left"))
			{
				char *t = attr.getNodeValue().transcode();
				strcpy(data->cmFName[TEX_LEFT], t);
				//delete [] t;
			}
			else if(!strcmp(attrName, "right"))
			{
				char *t = attr.getNodeValue().transcode();
				strcpy(data->cmFName[TEX_RIGHT], t);
				//delete [] t;
			}
			else if(!strcmp(attrName, "top"))
			{
				char *t = attr.getNodeValue().transcode();
				strcpy(data->cmFName[TEX_TOP], t);
				//delete [] t;
			}
			else if(!strcmp(attrName, "bottom"))
			{
				char *t = attr.getNodeValue().transcode();
				strcpy(data->cmFName[TEX_BOTTOM], t);
				//delete [] t;
			}

			//delete [] attrName;
		}
	}
	
	//delete [] nodeName;
	
	return 0;
}

int parseTextures(DOM_Node root, void *_data)
{
	texturesConf *data = (texturesConf *) _data;
	char *nodeName = root.getNodeName().transcode();
	
	if(!strcmp(nodeName, "texture"))
	{
		DOM_NamedNodeMap attributes = root.getAttributes();
		int textureN = 0;
		
		int attrCount = attributes.getLength();
		for(int i = 0; i < attrCount; i++)
		{
			DOM_Node attr = attributes.item(i);

			char *attrName = attr.getNodeName().transcode();

			if(!strcmp(attrName, "num")) textureN = DOMStringToD(attr.getNodeValue());

			//delete [] attrName;
		}
		
		if(textureN < data->nTextures)
		{
			if(data->textures[textureN])
			{
				delete data->textures[textureN];
				data->textures[textureN] = new textureConf();
			}
			loopChildren(root, C_TEXTURE, data->textures[textureN]);
		}
	}
	
	//delete [] nodeName;
	
	return 0;
}

int parseLighting(DOM_Node root, void *_data)
{
	lightingConf *data = (lightingConf *) _data;
	char *nodeName = root.getNodeName().transcode();

	if(!strcmp(nodeName, "globalambient"))
	{
		DOM_NamedNodeMap attributes = root.getAttributes();
		int attrCount = attributes.getLength();
		for(int i = 0; i < attrCount; i++)
		{
			DOM_Node attr = attributes.item(i);

			char *attrName = attr.getNodeName().transcode();

			if(!strcmp(attrName, "r")) data->globalAmbient[0] = DOMStringToF(attr.getNodeValue());
			else if(!strcmp(attrName, "g")) data->globalAmbient[1] = DOMStringToF(attr.getNodeValue());
			else if(!strcmp(attrName, "b")) data->globalAmbient[2] = DOMStringToF(attr.getNodeValue());
			else if(!strcmp(attrName, "a")) data->globalAmbient[3] = DOMStringToF(attr.getNodeValue());

			//delete [] attrName;
		}

	}
	else if(!strcmp(nodeName, "light"))
	{
		DOM_NamedNodeMap attributes = root.getAttributes();
		int lightN = 0;
		
		int attrCount = attributes.getLength();
		for(int i = 0; i < attrCount; i++)
		{
			DOM_Node attr = attributes.item(i);

			char *attrName = attr.getNodeName().transcode();

			if(!strcmp(attrName, "num")) lightN = DOMStringToD(attr.getNodeValue());

			//delete [] attrName;
		}
		
		if(lightN < data->nLights)
		{
			if(data->lights[lightN])
			{
				delete data->lights[lightN];
				data->lights[lightN] = new lightConf();
			}
			loopChildren(root, C_LIGHT, data->lights[lightN]);
		}
	}

	//delete [] nodeName;
	
	return 0;
}

int parseTexturesVal(DOM_Node root, void *_data)
{
	texturesValConf *data = (texturesValConf *) _data;
	char *nodeName = root.getNodeName().transcode();
	int num = 0;
	int tex = 0;

	if(!strcmp(nodeName, "texture"))
	{		
		DOM_NamedNodeMap attributes = root.getAttributes();
		int attrCount = attributes.getLength();
		for(int i = 0; i < attrCount; i++)
		{
			DOM_Node attr = attributes.item(i);

			char *attrName = attr.getNodeName().transcode();

			if(!strcmp(attrName, "num")) num = DOMStringToD(attr.getNodeValue());
			else if(!strcmp(attrName, "tex")) tex = DOMStringToD(attr.getNodeValue());

			//delete [] attrName;
		}
		
		data->texturesIndex[num] = tex;
	}
	
	//delete [] nodeName;
	
	return 0;
}

int parseMaterial(DOM_Node root, void *_data)
{
	materialConf *data = (materialConf *) _data;
	char *nodeName = root.getNodeName().transcode();

	if(!strcmp(nodeName, "ambient"))
	{
		DOM_NamedNodeMap attributes = root.getAttributes();
		int attrCount = attributes.getLength();
		for(int i = 0; i < attrCount; i++)
		{
			DOM_Node attr = attributes.item(i);

			char *attrName = attr.getNodeName().transcode();

			if(!strcmp(attrName, "r")) data->ambient[0] = DOMStringToF(attr.getNodeValue());
			else if(!strcmp(attrName, "g")) data->ambient[1] = DOMStringToF(attr.getNodeValue());
			else if(!strcmp(attrName, "b")) data->ambient[2] = DOMStringToF(attr.getNodeValue());
			else if(!strcmp(attrName, "a")) data->ambient[3] = DOMStringToF(attr.getNodeValue());

			//delete [] attrName;
		}
	}
	else if(!strcmp(nodeName, "diffuse"))
	{
		DOM_NamedNodeMap attributes = root.getAttributes();
		int attrCount = attributes.getLength();
		for(int i = 0; i < attrCount; i++)
		{
			DOM_Node attr = attributes.item(i);

			char *attrName = attr.getNodeName().transcode();

			if(!strcmp(attrName, "r")) data->diffuse[0] = DOMStringToF(attr.getNodeValue());
			else if(!strcmp(attrName, "g")) data->diffuse[1] = DOMStringToF(attr.getNodeValue());
			else if(!strcmp(attrName, "b")) data->diffuse[2] = DOMStringToF(attr.getNodeValue());
			else if(!strcmp(attrName, "a")) data->diffuse[3] = DOMStringToF(attr.getNodeValue());

			//delete [] attrName;
		}
	}
	else if(!strcmp(nodeName, "specular"))
	{
		DOM_NamedNodeMap attributes = root.getAttributes();
		int attrCount = attributes.getLength();
		for(int i = 0; i < attrCount; i++)
		{
			DOM_Node attr = attributes.item(i);

			char *attrName = attr.getNodeName().transcode();

			if(!strcmp(attrName, "r")) data->specular[0] = DOMStringToF(attr.getNodeValue());
			else if(!strcmp(attrName, "g")) data->specular[1] = DOMStringToF(attr.getNodeValue());
			else if(!strcmp(attrName, "b")) data->specular[2] = DOMStringToF(attr.getNodeValue());
			else if(!strcmp(attrName, "a")) data->specular[3] = DOMStringToF(attr.getNodeValue());

			//delete [] attrName;
		}
	}
	else if(!strcmp(nodeName, "shininess"))
	{
		DOM_NamedNodeMap attributes = root.getAttributes();
		int attrCount = attributes.getLength();
		for(int i = 0; i < attrCount; i++)
		{
			DOM_Node attr = attributes.item(i);

			char *attrName = attr.getNodeName().transcode();

			if(!strcmp(attrName, "val")) data->shininess = DOMStringToF(attr.getNodeValue());

			//delete [] attrName;
		}
	}
	else if(!strcmp(nodeName, "emission"))
	{
		DOM_NamedNodeMap attributes = root.getAttributes();
		int attrCount = attributes.getLength();
		for(int i = 0; i < attrCount; i++)
		{
			DOM_Node attr = attributes.item(i);

			char *attrName = attr.getNodeName().transcode();

			if(!strcmp(attrName, "r")) data->emission[0] = DOMStringToF(attr.getNodeValue());
			else if(!strcmp(attrName, "g")) data->emission[1] = DOMStringToF(attr.getNodeValue());
			else if(!strcmp(attrName, "b")) data->emission[2] = DOMStringToF(attr.getNodeValue());
			else if(!strcmp(attrName, "a")) data->emission[3] = DOMStringToF(attr.getNodeValue());

			//delete [] attrName;
		}
	}
	else if(!strcmp(nodeName, "textures"))	
	{
		if(data->textures) delete data->textures;
		data->textures = new texturesValConf;

		DOM_NamedNodeMap attributes = root.getAttributes();
		int attrCount = attributes.getLength();
		for(int i = 0; i < attrCount; i++)
		{
			DOM_Node attr = attributes.item(i);

			char *attrName = attr.getNodeName().transcode();

			if(!strcmp(attrName, "ntextures")) data->textures->setNTextures(DOMStringToD(attr.getNodeValue()));
			
			//delete [] attrName;
		}

		loopChildren(root, C_TEXTURES_VAL, (void *) data->textures);
	}
	else if(!strcmp(nodeName, "transparency"))
	{
		DOM_NamedNodeMap attributes = root.getAttributes();
		int attrCount = attributes.getLength();
		for(int i = 0; i < attrCount; i++)
		{
			DOM_Node attr = attributes.item(i);

			char *attrName = attr.getNodeName().transcode();

			if(!strcmp(attrName, "val")) data->transparency = DOMStringToBool(attr.getNodeValue());
			
			//delete [] attrName;
		}
	}

	//delete [] nodeName;
	
	return 0;
}

int parseOrientation(DOM_Node root, void *_data)
{
	orientation *data = (orientation *) _data;
	char *nodeName = root.getNodeName().transcode();

	if(!strcmp(nodeName, "magnification"))
	{
		DOM_NamedNodeMap attributes = root.getAttributes();
		int attrCount = attributes.getLength();
		for(int i = 0; i < attrCount; i++)
		{
			DOM_Node attr = attributes.item(i);

			char *attrName = attr.getNodeName().transcode();

			if(!strcmp(attrName, "val")) data->magnification = DOMStringToF(attr.getNodeValue());

			//delete [] attrName;
		}
	}
	else if(!strcmp(nodeName, "scale"))
	{
		DOM_NamedNodeMap attributes = root.getAttributes();
		int attrCount = attributes.getLength();
		for(int i = 0; i < attrCount; i++)
		{
			DOM_Node attr = attributes.item(i);

			char *attrName = attr.getNodeName().transcode();

			if(!strcmp(attrName, "x")) data->scale[0] = DOMStringToF(attr.getNodeValue());
			else if(!strcmp(attrName, "y")) data->scale[1] = DOMStringToF(attr.getNodeValue());
			else if(!strcmp(attrName, "z")) data->scale[2] = DOMStringToF(attr.getNodeValue());

			//delete [] attrName;
		}
	}
	else if(!strcmp(nodeName, "rotate"))
	{
		DOM_NamedNodeMap attributes = root.getAttributes();
		int attrCount = attributes.getLength();
		for(int i = 0; i < attrCount; i++)
		{
			DOM_Node attr = attributes.item(i);

			char *attrName = attr.getNodeName().transcode();

			if(!strcmp(attrName, "x")) data->rotate[0] = DOMStringToF(attr.getNodeValue());
			else if(!strcmp(attrName, "y")) data->rotate[1] = DOMStringToF(attr.getNodeValue());
			else if(!strcmp(attrName, "z")) data->rotate[2] = DOMStringToF(attr.getNodeValue());

			//delete [] attrName;
		}
	}
	else if(!strcmp(nodeName, "translate"))
	{
		DOM_NamedNodeMap attributes = root.getAttributes();
		int attrCount = attributes.getLength();
		for(int i = 0; i < attrCount; i++)
		{
			DOM_Node attr = attributes.item(i);

			char *attrName = attr.getNodeName().transcode();

			if(!strcmp(attrName, "x")) data->translate[0] = DOMStringToF(attr.getNodeValue());
			else if(!strcmp(attrName, "y")) data->translate[1] = DOMStringToF(attr.getNodeValue());
			else if(!strcmp(attrName, "z")) data->translate[2] = DOMStringToF(attr.getNodeValue());

			//delete [] attrName;
		}
	}

	//delete [] nodeName;
	
	return 0;
}

int parseObjects(DOM_Node root, void *_data)
{
	objectsConf *data = (objectsConf *) _data;
	char *nodeName = root.getNodeName().transcode();

	if(!strcmp(nodeName, "object"))
	{
		DOM_NamedNodeMap attributes = root.getAttributes();
		int objectN = 0;
		
		int attrCount = attributes.getLength();
		for(int i = 0; i < attrCount; i++)
		{
			DOM_Node attr = attributes.item(i);

			char *attrName = attr.getNodeName().transcode();

			if(!strcmp(attrName, "num")) objectN = DOMStringToD(attr.getNodeValue());

			//delete [] attrName;
		}
		
		if(objectN < data->nObjects)
		{
			if(data->objects[objectN])
			{
				delete data->objects[objectN];
				data->objects[objectN] = new objectConf();
			}
			loopChildren(root, C_OBJECT, data->objects[objectN]);
		}
	}
	
	//delete [] nodeName;

	return 0;
}

int parseTextureCoords(DOM_Node root, void *_data)
{
	textureCoordsConf *data = (textureCoordsConf *) _data;
	char *nodeName = root.getNodeName().transcode();

	if(!strcmp(nodeName, "texcoord"))
	{
		DOM_NamedNodeMap attributes = root.getAttributes();
		int texN = 0;
		
		int attrCount = attributes.getLength();
		for(int i = 0; i < attrCount; i++)
		{
			DOM_Node attr = attributes.item(i);

			char *attrName = attr.getNodeName().transcode();

			if(!strcmp(attrName, "num")) texN = DOMStringToD(attr.getNodeValue());

			//delete [] attrName;
		}
		
		if(texN < data->nTextureCoords)
		{
			if(data->texCoords[texN])
			{
				delete data->texCoords[texN];
				data->texCoords[texN] = new textureCoordConf();
			}
			loopChildren(root, C_TEXTURE_COORD, data->texCoords[texN]);
		}
	}
	
	//delete [] nodeName;

	return 0;
}

int parseTextureCoord(DOM_Node root, void *_data)
{
	textureCoordConf *data = (textureCoordConf *) _data;
	char *nodeName = root.getNodeName().transcode();
	
	if(!strcmp(nodeName, "xtile"))
	{
		DOM_NamedNodeMap attributes = root.getAttributes();
		
		int attrCount = attributes.getLength();
		for(int i = 0; i < attrCount; i++)
		{
			DOM_Node attr = attributes.item(i);

			char *attrName = attr.getNodeName().transcode();

			if(!strcmp(attrName, "val"))
			{
				char *t = attr.getNodeValue().transcode();

				if(!strcmp(t, "repeat")) data->xTile = GL_REPEAT;
				else if(!strcmp(t, "stretch")) data->xTile = GL_CLAMP;
				
				//delete [] t;
			}

			//delete [] attrName;
		}
	}
	else if(!strcmp(nodeName, "ytile"))
	{
		DOM_NamedNodeMap attributes = root.getAttributes();
		
		int attrCount = attributes.getLength();
		for(int i = 0; i < attrCount; i++)
		{
			DOM_Node attr = attributes.item(i);

			char *attrName = attr.getNodeName().transcode();

			if(!strcmp(attrName, "val"))
			{
				char *t = attr.getNodeValue().transcode();

				if(!strcmp(t, "repeat")) data->yTile = GL_REPEAT;
				else if(!strcmp(t, "stretch")) data->yTile = GL_CLAMP;
				
				//delete [] t;
			}

			//delete [] attrName;
		}
	}
	else if(!strcmp(nodeName, "space"))
	{
		DOM_NamedNodeMap attributes = root.getAttributes();
		
		int attrCount = attributes.getLength();
		for(int i = 0; i < attrCount; i++)
		{
			DOM_Node attr = attributes.item(i);

			char *attrName = attr.getNodeName().transcode();

			if(!strcmp(attrName, "val"))
			{
				char *t = attr.getNodeValue().transcode();

				if(!strcmp(t, "object")) data->space = GL_OBJECT_LINEAR;
				else if(!strcmp(t, "eye")) data->space = GL_EYE_LINEAR;
				
				//delete [] t;
			}

			//delete [] attrName;
		}
	}
	else if(!strcmp(nodeName, "plane"))
	{
		DOM_NamedNodeMap attributes = root.getAttributes();
		int attrCount = attributes.getLength();
		for(int i = 0; i < attrCount; i++)
		{
			DOM_Node attr = attributes.item(i);

			char *attrName = attr.getNodeName().transcode();

			if(!strcmp(attrName, "s_x")) data->texS[0] = DOMStringToF(attr.getNodeValue());
			else if(!strcmp(attrName, "s_y")) data->texS[1] = DOMStringToF(attr.getNodeValue());
			else if(!strcmp(attrName, "s_z")) data->texS[2] = DOMStringToF(attr.getNodeValue());
			else if(!strcmp(attrName, "s_w")) data->texS[2] = DOMStringToF(attr.getNodeValue());
			else if(!strcmp(attrName, "t_x")) data->texT[0] = DOMStringToF(attr.getNodeValue());
			else if(!strcmp(attrName, "t_y")) data->texT[1] = DOMStringToF(attr.getNodeValue());
			else if(!strcmp(attrName, "t_z")) data->texT[2] = DOMStringToF(attr.getNodeValue());
			else if(!strcmp(attrName, "t_w")) data->texT[2] = DOMStringToF(attr.getNodeValue());
			
			//delete [] attrName;
		}
	}
	else if(!strcmp(nodeName, "combination"))
	{
		DOM_NamedNodeMap attributes = root.getAttributes();
		
		int attrCount = attributes.getLength();
		for(int i = 0; i < attrCount; i++)
		{
			DOM_Node attr = attributes.item(i);

			char *attrName = attr.getNodeName().transcode();

			if(!strcmp(attrName, "val"))
			{
				char *t = attr.getNodeValue().transcode();

				if(!strcmp(t, "decal")) data->combination = GL_DECAL;
				else if(!strcmp(t, "replace")) data->combination = GL_REPLACE;
				else if(!strcmp(t, "modulate")) data->combination = GL_MODULATE;
				else if(!strcmp(t, "blend")) data->combination = GL_BLEND;
				
				//delete [] t;
			}

			//delete [] attrName;
		}
	}
	else if(!strcmp(nodeName, "type"))
	{
		DOM_NamedNodeMap attributes = root.getAttributes();
		
		int attrCount = attributes.getLength();
		for(int i = 0; i < attrCount; i++)
		{
			DOM_Node attr = attributes.item(i);

			char *attrName = attr.getNodeName().transcode();
			
			if(!strcmp(attrName, "mode"))
			{
				char *t = attr.getNodeValue().transcode();

				if(!strcmp(t, "file")) data->mode = TEX_COORD_FILE;
				else if(!strcmp(t, "gen")) data->mode = TEX_COORD_GENERATE;
				else if(!strcmp(t, "cubemapgen")) data->mode = TEX_COORD_CUBEMAP_GENERATE;
				else if(!strcmp(t, "cylindrical")) data->mode = TEX_COORD_CYLIN;
				else if(!strcmp(t, "embedded")) data->mode = TEX_COORD_EMBED;
								
				//delete [] t;
			}
			else if(!strcmp(attrName, "file"))
			{
				char *t = attr.getNodeValue().transcode();
				strcpy(data->fName, t);
				//delete [] t;
			}
			else if(!strcmp(attrName, "format"))
			{
				char *t = attr.getNodeValue().transcode();

				if(!strcmp(t, "tcd")) data->format = TEX_COORD_TCD;
				
				//delete [] t;
			}
			else if(!strcmp(attrName, "angle0"))
			{
				data->ang0 = DOMStringToF(attr.getNodeValue());
			}
			else if(!strcmp(attrName, "nwraps"))
			{
				data->nWraps = DOMStringToF(attr.getNodeValue());
			}
			else if(!strcmp(attrName, "ybottom"))
			{
				data->yBottom = DOMStringToF(attr.getNodeValue());
			}
			else if(!strcmp(attrName, "ytop"))
			{
				data->yTop = DOMStringToF(attr.getNodeValue());
			}			

			//delete [] attrName;
		}
	}
	else if(!strcmp(nodeName, "orientation"))
	{
		if(data->coordOrient)
		{
			delete data->coordOrient;
			data->coordOrient = new orientation();
		}
		loopChildren(root, C_ORIENTATION, data->coordOrient);
	}
	
	//delete [] nodeName;
	
	return 0;
}

int parseObject(DOM_Node root, void *_data)
{
	objectConf *data = (objectConf *) _data;
	char *nodeName = root.getNodeName().transcode();
	
	if(!strcmp(nodeName, "data"))
	{
		DOM_NamedNodeMap attributes = root.getAttributes();
		
		int attrCount = attributes.getLength();
		for(int i = 0; i < attrCount; i++)
		{
			DOM_Node attr = attributes.item(i);

			char *attrName = attr.getNodeName().transcode();

			if(!strcmp(attrName, "file"))
			{
				char *t = attr.getNodeValue().transcode();
				strcpy(data->fName, t);
				//delete [] t;
			}
			else if(!strcmp(attrName, "format"))
			{
				char *t = attr.getNodeValue().transcode();
			
				if(!strcmp(t, "alias")) data->drawType = DRAWABLE_ALIAS;
				else if(!strcmp(t, "raw")) data->drawType = DRAWABLE_RAW;
				else if(!strcmp(t, "square")) data->drawType = DRAWABLE_SQUARE;
				else if(!strcmp(t, "triangle")) data->drawType = DRAWABLE_TRIANGLE;
				else if(!strcmp(t, "cube")) data->drawType = DRAWABLE_CUBE;
				else if(!strcmp(t, "sphere")) data->drawType = DRAWABLE_SPHERE;
				
				//delete [] t;
			}
			else if(!strcmp(attrName, "texcoords"))
			{
				data->useTexCoords = DOMStringToBool(attr.getNodeValue());
			}
				

			//delete [] attrName;
		}
	}
	else if(!strcmp(nodeName, "texcoords"))
	{
		if(data->texCoords) delete data->texCoords;
		data->texCoords = new textureCoordsConf;

		DOM_NamedNodeMap attributes = root.getAttributes();
		int attrCount = attributes.getLength();
		for(int i = 0; i < attrCount; i++)
		{
			DOM_Node attr = attributes.item(i);

			char *attrName = attr.getNodeName().transcode();

			if(!strcmp(attrName, "ntexcoords")) data->texCoords->setNTextureCoords(DOMStringToD(attr.getNodeValue()));
			
			//delete [] attrName;
		}

		loopChildren(root, C_TEXTURE_COORDS, (void *) data->texCoords);
	}
	else if(!strcmp(nodeName, "material"))
	{
		DOM_NamedNodeMap attributes = root.getAttributes();
		int attrCount = attributes.getLength();
		for(int i = 0; i < attrCount; i++)
		{
			DOM_Node attr = attributes.item(i);

			char *attrName = attr.getNodeName().transcode();

			if(!strcmp(attrName, "num")) data->material = DOMStringToD(attr.getNodeValue());

			//delete [] attrName;
		}
	}
	else if(!strcmp(nodeName, "orientation"))
	{
		if(data->orient)
		{
			delete data->orient;
			data->orient = new orientation();
		}
		loopChildren(root, C_ORIENTATION, data->orient);
	}
	
	//delete [] nodeName;
	
	return 0;
}

int parseModels(DOM_Node root, void *_data)
{
	modelsConf *data = (modelsConf *) _data;
	char *nodeName = root.getNodeName().transcode();

	if(!strcmp(nodeName, "model"))
	{
		DOM_NamedNodeMap attributes = root.getAttributes();
		int modelN = 0;

		
		int attrCount = attributes.getLength();
		for(int i = 0; i < attrCount; i++)
		{
			DOM_Node attr = attributes.item(i);

			char *attrName = attr.getNodeName().transcode();

			if(!strcmp(attrName, "num")) modelN = DOMStringToD(attr.getNodeValue());

			//delete [] attrName;
		}
		
		if(modelN < data->nModels)
		{
			if(data->models[modelN])
			{
				delete data->models[modelN];
				data->models[modelN] = new modelConf();
			}
			loopChildren(root, C_MODEL, data->models[modelN]);
		}
	}
		
	//delete [] nodeName;
	
	return 0;
}

int parseModel(DOM_Node root, void *_data)
{
	modelConf *data = (modelConf *) _data;
	char *nodeName = root.getNodeName().transcode();
	
	if(!strcmp(nodeName, "objects"))
	{
		if(data->objects) delete data->objects;
		data->objects = new objectsConf;

		DOM_NamedNodeMap attributes = root.getAttributes();
		int attrCount = attributes.getLength();
		for(int i = 0; i < attrCount; i++)
		{
			DOM_Node attr = attributes.item(i);

			char *attrName = attr.getNodeName().transcode();

			if(!strcmp(attrName, "nobjects")) data->objects->setNObjects(DOMStringToD(attr.getNodeValue()));
			else if(!strcmp(attrName, "path"))
			{
				char *t = attr.getNodeValue().transcode();
				strcpy(data->path, t);
				//delete [] t;
			}
			
			//delete [] attrName;
		}

		loopChildren(root, C_OBJECTS, (void *) data->objects);
	}
	else if(!strcmp(nodeName, "orientation"))
	{
		if(data->orient)
		{
			delete data->orient;
			data->orient = new orientation();
		}

		loopChildren(root, C_ORIENTATION, data->orient);
	}
	
	//delete [] nodeName;

	return 0;
}

int parseRender(DOM_Node root, void *_data)
{
	renderConf *data = (renderConf *) _data;
	char *nodeName = root.getNodeName().transcode();
	
	if(!strcmp(nodeName, "screen_resolution"))
	{
		DOM_NamedNodeMap attributes = root.getAttributes();
		int attrCount = attributes.getLength();
		for(int i = 0; i < attrCount; i++)
		{
			DOM_Node attr = attributes.item(i);

			char *attrName = attr.getNodeName().transcode();

			if(!strcmp(attrName, "width")) data->screenX = DOMStringToD(attr.getNodeValue());
			else if(!strcmp(attrName, "height")) data->screenY = DOMStringToD(attr.getNodeValue());

			//delete [] attrName;
		}
	}
	else if(!strcmp(nodeName, "hologram_size"))
	{
		DOM_NamedNodeMap attributes = root.getAttributes();
		int attrCount = attributes.getLength();
		for(int i = 0; i < attrCount; i++)
		{
			DOM_Node attr = attributes.item(i);

			char *attrName = attr.getNodeName().transcode();

			if(!strcmp(attrName, "width")) data->holoX = DOMStringToF(attr.getNodeValue());
			else if(!strcmp(attrName, "height")) data->holoY = DOMStringToF(attr.getNodeValue());

			//delete [] attrName;
		}
	}
	else if(!strcmp(nodeName, "camera_track_size"))
	{
		DOM_NamedNodeMap attributes = root.getAttributes();
		int attrCount = attributes.getLength();
		for(int i = 0; i < attrCount; i++)
		{
			DOM_Node attr = attributes.item(i);

			char *attrName = attr.getNodeName().transcode();

			if(!strcmp(attrName, "width")) data->cameraPlaneX = DOMStringToF(attr.getNodeValue());

			//delete [] attrName;
		}
	}
	else if(!strcmp(nodeName, "nviews"))
	{
		DOM_NamedNodeMap attributes = root.getAttributes();
		int attrCount = attributes.getLength();
		for(int i = 0; i < attrCount; i++)
		{
			DOM_Node attr = attributes.item(i);

			char *attrName = attr.getNodeName().transcode();

			if(!strcmp(attrName, "horiz")) data->viewsX = DOMStringToD(attr.getNodeValue());

			//delete [] attrName;
		}
	}
	else if(!strcmp(nodeName, "eyez"))
	{
		DOM_NamedNodeMap attributes = root.getAttributes();
		int attrCount = attributes.getLength();
		for(int i = 0; i < attrCount; i++)
		{
			DOM_Node attr = attributes.item(i);

			char *attrName = attr.getNodeName().transcode();

			if(!strcmp(attrName, "z")) data->eyeZ = DOMStringToF(attr.getNodeValue());

			//delete [] attrName;
		}
	}
	else if(!strcmp(nodeName, "far_clip"))
	{
		DOM_NamedNodeMap attributes = root.getAttributes();

		int attrCount = attributes.getLength();
		for(int i = 0; i < attrCount; i++)
		{
			DOM_Node attr = attributes.item(i);

			char *attrName = attr.getNodeName().transcode();

			if(!strcmp(attrName, "z")) data->farClip = DOMStringToF(attr.getNodeValue());

			//delete [] attrName;
		}
	}
	else if(!strcmp(nodeName, "near_clip_scale"))
	{
		DOM_NamedNodeMap attributes = root.getAttributes();

		int attrCount = attributes.getLength();

		for(int i = 0; i < attrCount; i++)
		{
			DOM_Node attr = attributes.item(i);

			char *attrName = attr.getNodeName().transcode();

			if(!strcmp(attrName, "zinverse")) data->nearClip = DOMStringToF(attr.getNodeValue());

			//delete [] attrName;
		}
	}
	
	//delete [] nodeName;
	
	return 0;
}

int parseNode(DOM_Node root, int context, void *data)
{
	switch(root.getNodeType())
	{
	case DOM_Node::TEXT_NODE:
		break;
	case DOM_Node::DOCUMENT_NODE:
		loopChildren(root, C_ROOT, data);

		break;
	case DOM_Node::ELEMENT_NODE:
		switch(context)
		{
			case C_ROOT:
				loopChildren(root, C_HOLOREN, data);
				break;
			case C_HOLOREN:
				parseHoloRen(root, data);
				break;
			case C_RENDER:
				parseRender(root, data);
				break;
			case C_LIGHTING:
				parseLighting(root, data);
				break;
			case C_TEXTURES:
				parseTextures(root, data);
				break;
			case C_TEXTURE:
				parseTexture(root, data);
			case C_MATERIALS:
				parseMaterials(root, data);
				break;
			case C_MODELS:
				parseModels(root, data);
				break;
			case C_MODEL:
				parseModel(root, data);
				break;
			case C_LIGHT:
				parseLight(root, data);
				break;
			case C_MATERIAL:
				parseMaterial(root, data);
				break;
			case C_ORIENTATION:
				parseOrientation(root, data);
				break;
			case C_OBJECTS:
				parseObjects(root, data);
				break;
			case C_OBJECT:
				parseObject(root, data);
				break;
			case C_TEXTURES_VAL:
				parseTexturesVal(root, data);
				break;
			case C_TEXTURE_COORD:
				parseTextureCoord(root, data);
				break;
			case C_TEXTURE_COORDS:
				parseTextureCoords(root, data);
				break;
		}
	}

	return 0;
}

void holoConf::_parseConfigFile(char *fName)
{
	XMLPlatformUtils::Initialize();

	DOMParser parser;
	parser.setDoValidation(false);

	bool errorsOccured = false;
	parser.parse(fName);
	
	if(!errorsOccured)
	{
		DOM_Node doc = parser.getDocument();
		parseNode(doc, 0, (void *) this);
	}
	else return;
	
	printf("configuration file parsed\n");
}
