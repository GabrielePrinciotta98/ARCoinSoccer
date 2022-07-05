#pragma once
#include <opencv/cv.hpp>

class CoinTracker {

	const std::string kWinName = "Coins";

public:
	CoinTracker(double coinSize) : coinSize(coinSize) { init();  }
	~CoinTracker() { cleanup(); }

	void findCoins(cv::Mat& img, std::vector<cv::Vec3f>& circles);

protected:
	double coinSize;
	void init();
	void cleanup();
};
