#pragma once
#include <opencv/cv.hpp>

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
	CoinTracker(double coinSize, double houghParam1, double houghParam2) : coinSize(coinSize), houghParam1(houghParam1), houghParam2(houghParam2) { init();  }
	~CoinTracker() { cleanup(); }

	void findCoins(cv::Mat& img, std::vector<Coin>& lastCoins, std::vector<Coin>& coins);

	double coinSize;
	double houghParam1, houghParam2;

protected:
	size_t idCounter;
	void init();
	void cleanup();
};
