#pragma once
#include <opencv\cv.hpp>

struct Coin
{
	size_t id;
	cv::Vec2f pos2D;
	cv::Vec3f pos3D;
	cv::Vec3f vel;
	int framesTracked;
};

class CoinTracker {

	const std::string kWinName = "Coins";

public:
	CoinTracker(double coinSize) : coinSize(coinSize) { init();  }
	~CoinTracker() { cleanup(); }

	void findCoins(cv::Mat& img, std::vector<Coin>& lastCoins, std::vector<Coin>& coins);

	double coinSize;

protected:
	size_t idCounter;
	void init();
	void cleanup();
};