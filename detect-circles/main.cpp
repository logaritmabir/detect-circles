#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include "opencv2/imgcodecs.hpp"

#include <vector>
#include <numeric>
#include <algorithm>
#include <thread>
#include <iostream>

using namespace std;
using namespace cv;

int find_nearest_circle(vector<double>& distances);

int main(void) {

	VideoCapture capture;
	capture.open(0);

	if (!capture.isOpened()) {
		cerr << "Camera has not been opened" << endl;
		return EXIT_FAILURE;
	}

	Mat input = imread("C:\\Users\\steam\\OneDrive\\Masaüstü\\ball-images\\circles.png", IMREAD_COLOR);
	Point origin(0, 0);

	while (1) {
		Mat frame;
		capture.read(frame);
		input = frame;

		GaussianBlur(input, input, Size(3, 3), 2, 2, BORDER_REFLECT_101);/*clear noise*/

		float gamma = 0.8f;
		Mat lookUpTable(1, 256, CV_8U);
		uchar* p = lookUpTable.ptr();
		for (int i = 0; i < 256; ++i)
			p[i] = saturate_cast<uchar>(pow(i / 255.0, gamma) * 255.0);
		LUT(input, lookUpTable, input);

		vector<Mat> channels;
		split(input, channels);

		for (int i = 0; i < channels.size(); i++) {
			equalizeHist(channels[i], channels[i]);
		}
		merge(channels, input);

		Mat hsv_input, hsv_input_gray;
		cvtColor(input, hsv_input, COLOR_BGR2HSV); /*conver input to hsv*/
		cvtColor(hsv_input, hsv_input_gray, COLOR_BGR2GRAY);/*gray hsv for HoughCircle()*/

		/*Masking for red*/
		Mat mask1, mask2;
		inRange(hsv_input, Scalar(0, 70, 50), Scalar(10, 255, 255), mask1);/*red channel masking*/
		inRange(hsv_input, Scalar(170, 70, 50), Scalar(180, 255, 255), mask2);
		Mat mask = (mask1 | mask2);/*combining masks*/

		Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(9, 9)); /*morphological enhancement*/
		morphologyEx(mask, mask, MORPH_OPEN, kernel);
		morphologyEx(mask, mask, MORPH_CLOSE, kernel);

		Mat hsv_input_gray_masked = hsv_input_gray & mask; /*masking*/

		Mat mask3d, hsv_input_masked;
		cvtColor(mask, mask3d, COLOR_GRAY2BGR);
		hsv_input_masked = mask3d & hsv_input; /*masking colorful image masking*/

		vector<Vec3f> circles; /*element of circle_centers -> (x,y,radius) */

		int pieces = 5;
		int radius = pieces * 2;
		int min_dist = input.cols / pieces;
		int min_radius = input.cols / radius;
		int max_radius = input.cols / (radius - 2);

		HoughCircles(hsv_input_gray_masked, circles, HOUGH_GRADIENT, 1.5, min_dist, 50, 40, min_radius, max_radius);

		Mat roi = Mat::zeros(input.size(), CV_8UC1);
		vector<double> distances;
		for (int i = 0; i < circles.size(); i++) {
			Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
			int radius = cvRound(circles[i][2]);
			circle(input, center, 3, Scalar(0, 255, 0), -1, 8, 0);/*center of circle*/
			circle(input, center, radius, Scalar(255, 0, 0), 3, 8, 0);/*outside of circle*/
			circle(roi, center, radius, Scalar(255, 255, 255), -1, 8, 0);/*masking the circle*/
			double dist = sqrt(pow(origin.x - center.x, 2) + pow(origin.y - center.y, 2));
			distances.push_back(dist);
		}

		Mat roi3d;
		cvtColor(roi, roi3d, COLOR_GRAY2BGR);
		Mat pixels = hsv_input_masked & roi3d; /*görüntünün dairesel kısmının renkli hsv olarak alınması*/

		vector<int>index_list;
		for (int i = 0; i < distances.size(); i++) {
			int nearest = find_nearest_circle(distances);
			index_list.push_back(nearest);
		}

		for (int i = 0; i < index_list.size(); i++) {
			Point center = Point(cvRound(circles[index_list.at(i)][0]), cvRound(circles[index_list.at(i)][1]));
			putText(input, to_string(i), center, FONT_HERSHEY_PLAIN, 1.5, Scalar(255, 200, 100), 2);
		}

		Mat bla;
		cvtColor(pixels, bla, COLOR_HSV2BGR);
		imshow("bla", bla);
		imshow("roi", roi);
		imshow("pixels", pixels);
		imshow("input", input);
		imshow("Masked HSV Input", hsv_input_masked);
		waitKey(1);

		distances.clear();
		index_list.clear();
	}
}
int find_nearest_circle(vector<double>& distances) {
	int min = 0;
	for (int i = 0; i < distances.size(); i++) {
		if (distances.at(i) < distances.at(min))
			min = i;
	}
	distances.at(min) = INT_MAX; /*hmm*/
	return min;
}
//int find_farthest_circle(vector<double>& distances) {
//	int max_index = 0;
//	for (int i = 0; i < distances.size(); i++) {
//		if (distances.at(i) > distances.at(max_index))
//			max_index = i;
//	}
//	distances.at(max_index) = -1;
//	return max_index;
//}