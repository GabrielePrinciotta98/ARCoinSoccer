#pragma once
#include <string>
#include <vector>
#include <opencv/cv.hpp>

#include "ObjLoader.h"

struct Particle
{
	float emitTime;
	float lifeTime;

	cv::Vec3f startPos, velocity;
	float startSize, deltaSize;
	float startRot, rotVelocity;
	cv::Vec4f startColor, deltaColor;
};

class ParticleSystem
{
public:
	ParticleSystem(std::string tex) : texture(tex) { }
	~ParticleSystem() { }

	void init(float time);
	void moveToPos(cv::Vec3f p, float time);
	void draw(float time);
	void emit(cv::Vec3f pos, cv::Vec3f pos2, int count, float deltaTime);

	bool emitting = true;
	float emitPerSecond = 5, emitPerDistance = 0;
	cv::Vec3f pos = cv::Vec3f(0, 0, 0);
	float minEmitRad = 0, maxEmitRad = 0;
	cv::Vec3f emitSize = cv::Vec3f(0, 0, 0);
	cv::Vec3f minVel = cv::Vec3f(0, 0, 0);
	cv::Vec3f maxVel = cv::Vec3f(0, 0, 0);
	cv::Vec3f gravity = cv::Vec3f(0, 0, 0);
	float minRadVel = 0, maxRadVel = 0.1F;
	float minStartSize = 0.1F, maxStartSize = 0.1F;
	float minEndSize = 0.1F, maxEndSize = 0.1F;
	float minStartRot = 0, maxStartRot = 0;
	float minRotVelocity = 0, maxRotVelocity = 0;
	float minLifeTime = 1, maxLifeTime = 1;
	float fadeTime = 0.F;
	cv::Vec4f startColor1 = cv::Vec4f(1, 1, 1, 1), startColor2 = cv::Vec4f(1, 1, 1, 1);
	cv::Vec4f endColor1 = cv::Vec4f(1, 1, 1, 1), endColor2 = cv::Vec4f(1, 1, 1, 1);

protected:
	Texture texture;
	float lastUpdateTime = 0;
	float lastDist = 0;
	std::vector<Particle> particles;
};