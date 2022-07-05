#include <opencv\cv.hpp>
#include "CoinTracker.h"

using namespace cv;
using namespace std;

void CoinTracker::findCoins(cv::Mat& img, vector<Vec3f>& circles)
{
	circles.clear();

	cv::Mat grayScale;
	cv::cvtColor(img, grayScale, CV_BGR2GRAY);
	//GaussianBlur(grayScale, grayScale, Size(9, 9), 2, 2);
	//imshow(windowNormalGray, grayScale);

	HoughCircles(grayScale, circles, HOUGH_GRADIENT, 2, coinSize * 2, 100.0, 30.0, coinSize, coinSize * 1.5);

	for (size_t i = 0; i < circles.size(); i++)
	{
		Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
		int radius = cvRound(circles[i][2]);

		// draw the circle center
		circle(img, center, 3, Scalar(0, 255, 0), -1, 8, 0);
		// draw the circle outline
		circle(img, center, radius, Scalar(0, 0, 255), 3, 8, 0);
	}

	std::string nCircles = to_string(circles.size());
	putText(img, nCircles, Point(100, 100), FONT_HERSHEY_SIMPLEX, 4, (0, 0, 0), 2);

	imshow(kWinName, img);
};

void CoinTracker::init()
{
	namedWindow(kWinName, CV_WINDOW_NORMAL);
};

void CoinTracker::cleanup()
{
	destroyWindow(kWinName);
};

