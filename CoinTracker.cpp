#include <opencv/cv.hpp>
#include <iostream>
#include "CoinTracker.h"

using namespace cv;
using namespace std;

#define OPENCV_WINDOWS

void CoinTracker::findCoins(cv::Mat& img, vector<Coin>& lastCoins, vector<Coin>& coins)
{
	cv::Mat grayScale;
	cv::cvtColor(img, grayScale, CV_BGR2GRAY);
	//GaussianBlur(grayScale, grayScale, Size(9, 9), 2, 2);
	//imshow(windowNormalGray, grayScale);

	vector<Vec3f> circles;

	HoughCircles(grayScale, circles, HOUGH_GRADIENT, 2, coinDist, houghParam1, houghParam2, coinSize, coinSize2);

	coins.clear();
	for (size_t i = 0; i < lastCoins.size(); i++)
	{
		const Coin& lastCoin = lastCoins[i];
		const Vec2f &circle2 = lastCoin.pos2D;

		Coin c;
		size_t closest = -1;
		float closestDist = -1;
		for (size_t j = 0; j < circles.size(); j++)
		{
			const Vec3f& circle = circles[j];

			float dx = circle[0] - circle2[0];
			float dy = circle[1] - circle2[1];
			float sqDist = dx * dx + dy * dy;
			if ((closest == -1 || sqDist < closestDist) && sqDist < coinSize * coinSize * 6 * 6)
			{
				closest = j;
				closestDist = sqDist;
				c.vel = Vec3f(dx, dy, sqDist);
			}
		}
		// Assume that closest is the same coin as lastCoin
		if (closest != -1)
		{
			const Vec3f circle = circles[closest];

			circles.erase(circles.begin() + closest);
			if (c.vel[2] < 5 * 5)
				c.pos2D = Vec2f(circle[0] - c.vel[0] * 0.9, circle[1] - c.vel[1] * 0.9);
			else
				c.pos2D = Vec2f(circle[0], circle[1]);
			c.id = lastCoin.id;
			c.framesTracked = lastCoin.framesTracked + 1;
			coins.push_back(c);

			Point start(cvRound(circle[0]), cvRound(circle[1]));
			Point end(cvRound(circle2[0]), cvRound(circle2[1]));

			line(img, start, end, Scalar(255, 0, 0), 3);

			int radius = cvRound(circle[2]);
			cv::circle(img, end, radius, Scalar(0, 255, 0), 2, 8, 0);
		}
		// Keep coins alive for one frame after we lost them
		/*else if (lastCoin.framesTracked > 0)
		{
			c.pos2D = Vec2f(lastCoin.pos2D[0], lastCoin.pos2D[1]);
			c.vel = Vec3f(0, 0, 0);
			c.framesTracked = -(lastCoin.framesTracked + 1);
			coins.push_back(c);

			Point start(cvRound(c.pos2D[0]), cvRound(c.pos2D[1]));
			int radius = cvRound(circles.size() > 0 ? circles[0][2] : coinSize);
			cv::circle(img, start, radius, Scalar(255, 0, 255), 2, 8, 0);
		}*/
		else
		{
			Point center(cvRound(lastCoin.pos2D[0]), cvRound(lastCoin.pos2D[1]));
			int radius = cvRound(circles.size() > 0 ? circles[0][2] : coinSize);
			cv::circle(img, center, radius, Scalar(0, 0, 255), 2, 8, 0);
			//std::cout << "Lost tracking after " << lastCoin.framesTracked << " frames: " << lastCoin.pos2D[0] << " " << lastCoin.pos2D[1] << "\n";
		}
	}
	for (size_t j = 0; j < circles.size(); j++)
	{
		const Vec3f& circle = circles[j];

		//if (coins.size() < 3)
		{
			Coin c;
			c.id = idCounter++;
			c.vel = Vec3f(0, 0, 0);
			c.pos2D = Vec2f(circle[0], circle[1]);
			c.framesTracked = 0;
			coins.push_back(c);


			Point center(cvRound(circle[0]), cvRound(circle[1]));
			int radius = cvRound(circle[2]);
			cv::circle(img, center, radius, Scalar(0, 255, 255), 2, 8, 0);
		}
		/*
		else
		{
			Point center(cvRound(circle[0]), cvRound(circle[1]));
			int radius = cvRound(circle[2]);
			cv::circle(img, center, radius, Scalar(255, 0, 255), 2, 8, 0);
		}
		*/
	}

	std::sort(coins.begin(), coins.end(), [](const Coin& l, const Coin& r) -> bool { return abs(l.framesTracked) > abs(r.framesTracked); });

	std::string nCircles = to_string(circles.size());
	putText(img, nCircles, Point(100, 100), FONT_HERSHEY_SIMPLEX, 3, (0, 0, 0), 2);

#ifdef OPENCV_WINDOWS
	imshow(kWinName, img);
#endif
};

void handler(int pos, void* slider_value) {
	*((double*)slider_value) = max(1, pos);
}

void CoinTracker::init()
{
#ifdef OPENCV_WINDOWS
	idCounter = 0;
	namedWindow(kWinName, CV_WINDOW_NORMAL);

	int max = 100;
	slider1 = coinSize;
	slider2 = coinSize2;
	slider3 = coinDist;
	slider4 = houghParam1;
	slider5 = houghParam2;

	cv::createTrackbar("Coin size", kWinName, &slider1, max, handler, &coinSize);
	cv::createTrackbar("Max Coin size", kWinName, &slider2, max, handler, &coinSize2);
	cv::createTrackbar("Coin dist", kWinName, &slider3, max, handler, &coinDist);
	cv::createTrackbar("Hough param 1", kWinName, &slider4, max, handler, &houghParam1);
	cv::createTrackbar("Hough param 2", kWinName, &slider5, max, handler, &houghParam2);
#endif
};

void CoinTracker::cleanup()
{
#ifdef OPENCV_WINDOWS
	destroyWindow(kWinName);
#endif
};

