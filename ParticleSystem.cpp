#include "ParticleSystem.h"
#include <iostream>
#include <ctime>

// Some really basic particle effects

void ParticleSystem::init(float time)
{
	std::srand(std::time(nullptr));

	texture.init();
	lastUpdateTime = time;
}

void ParticleSystem::moveToPos(cv::Vec3f p, float time)
{
	if (emitting)
	{
		float dx = p[0] - pos[0];
		float dy = p[1] - pos[1];
		float dz = p[2] - pos[2];

		float dist = sqrt(dx * dx + dy * dy + dz * dz) + lastDist;

		int emitCount = (int)(emitPerDistance * dist) - (int)(emitPerDistance * lastDist);
		emit(pos, p, emitCount, time - lastUpdateTime);

		lastDist = dist;
	}

	pos = p;
}

void ParticleSystem::draw(float time)
{
	particles.erase(std::remove_if(particles.begin(),particles.end(),[time](const Particle &p) -> bool { return p.emitTime + p.lifeTime < time; }), particles.end());

	if (emitting)
	{
		int emitCount = (int)(emitPerSecond * time) - (int)(emitPerSecond * lastUpdateTime);
		emit(pos, pos, emitCount, time - lastUpdateTime);
	}

	if (particles.size() > 0)
	{
		float ratio = texture.getWidth() / (float)texture.getHeight();

		glEnable(GL_TEXTURE_2D);
		glDepthMask(false);
		texture.bind();
		glBegin(GL_QUADS);
		for (const Particle& p : particles)
		{
			float life = time - p.emitTime;
			if (life < 0 || life > p.lifeTime)
				std::cout << "Life out of bounds";

			float t = life / p.lifeTime;
			cv::Vec3f pos = p.startPos + p.velocity * life + gravity * life * life * 0.5;
			float rot = p.startRot + p.rotVelocity * life;
			float size = p.startSize + p.deltaSize * t;
			float a = life < fadeTime ? life / fadeTime : life > p.lifeTime - fadeTime ? (p.lifeTime - life) / fadeTime : 1;
			cv::Vec4f color = p.startColor + p.deltaColor * t;
			glColor4f(color[0], color[1], color[2], color[3] * a);

			float c = cos(rot) * size * ratio;
			float s = sin(rot) * size;
			glTexCoord2f(1, 0);
			glVertex3f(pos[0] + c, pos[1] + s, pos[2]);
			glTexCoord2f(0, 0);
			glVertex3f(pos[0] - s, pos[1] + c, pos[2]);
			glTexCoord2f(0, 1);
			glVertex3f(pos[0] - c, pos[1] - s, pos[2]);
			glTexCoord2f(1, 1);
			glVertex3f(pos[0] + s, pos[1] - c, pos[2]);
		}
		glEnd();
		glDisable(GL_TEXTURE_2D);
		glDepthMask(true);
	}

	lastUpdateTime = time;
}

float randFloat(float min, float max)
{
	return min + std::rand() / (float)RAND_MAX * (max - min);
}

#ifndef M_PI
#define M_PI 3.1415926535897932384626433832795
#endif

void ParticleSystem::emit(cv::Vec3f pos1, cv::Vec3f pos2, int count, float deltaTime)
{
	for (int i = 0; i < count; ++i)
	{
		Particle p;
		float t = randFloat(0, 1);
		float ang = randFloat(0, M_PI * 2);
		float rad = randFloat(minEmitRad, maxEmitRad);
		float radVel = randFloat(minRadVel, maxRadVel);
		p.emitTime = lastUpdateTime + deltaTime * t;
		p.lifeTime = randFloat(minLifeTime, maxLifeTime);
		p.startPos = pos1 + (pos2 - pos1) * t + cv::Vec3f(randFloat(-emitSize[0], emitSize[0]) + cos(ang) * rad, randFloat(-emitSize[1], emitSize[1]) + sin(ang) * rad, randFloat(-emitSize[2], emitSize[2]));
		p.velocity = cv::Vec3f(randFloat(minVel[0], maxVel[0]) + cos(ang) * radVel, randFloat(minVel[1], maxVel[1]) + sin(ang) * radVel, randFloat(minVel[2], maxVel[2]));
		p.startSize = randFloat(minStartSize, maxStartSize);
		p.deltaSize = randFloat(minEndSize, maxEndSize) - p.startSize;
		p.startRot = (randFloat(minStartRot, maxStartRot) + 45) / 180 * M_PI;
		p.rotVelocity = randFloat(minRotVelocity, maxRotVelocity) / 180 * M_PI;
		p.startColor = startColor1 + (startColor2 - startColor1) * randFloat(0, 1);
		p.deltaColor = endColor1 + (endColor2 - endColor1) * randFloat(0, 1) - p.startColor;

		particles.push_back(p);
	}
}
