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
	CoinTracker(double coinSize, double houghParam1, double houghParam2) : coinSize(coinSize), coinSize2(coinSize * 1.5), coinDist(coinSize * 2), houghParam1(houghParam1), houghParam2(houghParam2) { }
	CoinTracker(double coinSize, double coinSize2, double coinDist, double houghParam1, double houghParam2) : coinSize(coinSize), coinSize2(coinSize2), coinDist(coinDist), houghParam1(houghParam1), houghParam2(houghParam2) { }

	~CoinTracker() { }

	void findCoins(cv::Mat& img, std::vector<Coin>& lastCoins, std::vector<Coin>& coins);
	void init();
	void cleanup();

	double coinSize, coinSize2, coinDist;
	double houghParam1, houghParam2;

protected:
	size_t idCounter = 0;
	int slider1 = 0, slider2 = 0, slider3 = 0, slider4 = 0, slider5 = 0, slider6 = 0, slider7 = 0;

};
