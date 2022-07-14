#pragma once
#include <GL/glew.h>
#include <string>

class Texture
{
public:
	Texture(std::string file) : file(file), wrappingRepeat(false) { }
	Texture(std::string file, bool repeats) : file(file), wrappingRepeat(repeats) { }

	void init();
	void bind();
	void drawQuad(float x, float y, float height);
	void drawQuad(float x, float y, float width, float height);
	bool valid();
	unsigned int getWidth() { return width; }
	unsigned int getHeight() { return height; }

protected:
	GLuint texture = 0;
	std::string file;
	unsigned int width = 1, height = 1;
	bool initialized = false;
	bool wrappingRepeat;
};

class ObjModel
{
public:
	ObjModel(std::string obj) : objFile(obj), texture("") { }
	ObjModel(std::string obj, std::string texture) : objFile(obj), texture(texture) { }
	ObjModel(std::string obj, std::initializer_list<cv::Vec4f> colors) : objFile(obj), texture(""), colors(colors) { }
	~ObjModel() { }

	void init();
	void draw();

protected:
	std::string objFile;
	GLuint callList;
	Texture texture;
	bool initialized = false;
	std::vector<cv::Vec4f> colors{};
};