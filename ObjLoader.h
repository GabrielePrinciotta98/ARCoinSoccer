#pragma once
#include <GL/glew.h>
#include <string>

class ObjModel
{
public:
	ObjModel(std::string obj) : objFile(obj), textureFile("") { }
	ObjModel(std::string obj, std::string texture) : objFile(obj), textureFile(texture) { }
	~ObjModel() { }

	void init();
	void draw();

protected:
	std::string objFile, textureFile;
	GLuint callList, texture;
};