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
		Mat pixels = hsv_input_masked & roi3d;

		vector<float> hues, saturations, values;
		for (int i = 0; i < pixels.rows; i++) {
			for (int j = 0; j < pixels.cols; j++) {
				if (mask.at<uchar>(i, j) != 0) {
					Vec3b pixel = pixels.at<Vec3b>(i, j);
					float hue = pixel[0];
					float sat = pixel[1];
					float val = pixel[2];

					hues.push_back(hue);
					saturations.push_back(sat);
					values.push_back(val);
				}
			}
		}

		float total_hues = accumulate(hues.begin(), hues.end(), 0);
		float total_saturations = accumulate(saturations.begin(), saturations.end(), 0);
		float total_values = accumulate(values.begin(), values.end(), 0);

		float average_hue = total_hues / hues.size();
		float average_saturation = total_saturations / saturations.size();
		float average_value = total_values / values.size();

		float min_hue;
		float min_sat;
		float min_val;

		float max_hue;
		float max_sat;
		float max_val;

		if (!(hues.empty() || saturations.empty() || values.empty())) {
			vector<float>::iterator it_min_hue = min_element(hues.begin(), hues.end());
			vector<float>::iterator it_min_sat = min_element(saturations.begin(), saturations.end());
			vector<float>::iterator it_min_val = min_element(values.begin(), values.end());

			vector<float>::iterator it_max_hue = max_element(hues.begin(), hues.end());
			vector<float>::iterator it_max_sat = max_element(saturations.begin(), saturations.end());
			vector<float>::iterator it_max_val = max_element(values.begin(), values.end());

			min_hue = *it_min_hue;
			min_sat = *it_min_sat;
			min_val = *it_min_val;

			max_hue = *it_max_hue;
			max_sat = *it_max_sat;
			max_val = *it_max_val;
		}
		else {
			min_hue = 0;
			min_sat = 0;
			min_val = 0;

			max_hue = 0;
			max_sat = 0;
			max_val = 0;
		}

		vector<Mat> channels;
		split(pixels, channels);

		equalizeHist(channels[2], channels[2]); /*Histogram equalization on Value channel*/

		Mat equalized_hsv_input_masked;
		merge(channels, equalized_hsv_input_masked);

		vector<float> equalized_hues;
		vector<float> equalized_saturations;
		vector<float> equalized_values;
		for (int i = 0; i < pixels.rows; i++) {
			for (int j = 0; j < pixels.cols; j++) {
				if (roi.at<uchar>(i, j) != 0) {
					Vec3b pixel = pixels.at<Vec3b>(i, j);
					float hue = pixel[0];
					float sat = pixel[1];
					float val = pixel[2];

					equalized_hues.push_back(hue);
					equalized_saturations.push_back(sat);
					equalized_values.push_back(val);
				}
			}
		}

		float equalized_total_hues = accumulate(equalized_hues.begin(), equalized_hues.end(), 0);
		float equalized_total_saturations = accumulate(equalized_saturations.begin(), equalized_saturations.end(), 0);
		float equalized_total_values = accumulate(equalized_values.begin(), equalized_values.end(), 0);

		float equalized_average_hue = equalized_total_hues / equalized_hues.size();
		float equalized_average_saturation = equalized_total_saturations / equalized_saturations.size();
		float equalized_average_value = equalized_total_values / equalized_values.size();

		float equalized_min_hue;
		float equalized_min_sat;
		float equalized_min_val;

		float equalized_max_hue;
		float equalized_max_sat;
		float equalized_max_val;

		if (!(equalized_hues.empty() || equalized_saturations.empty() || equalized_values.empty())) {
			vector<float>::iterator it_min_hue = min_element(equalized_hues.begin(), equalized_hues.end());
			vector<float>::iterator it_min_sat = min_element(equalized_saturations.begin(), equalized_saturations.end());
			vector<float>::iterator it_min_val = min_element(equalized_values.begin(), equalized_values.end());

			vector<float>::iterator it_max_hue = max_element(equalized_hues.begin(), equalized_hues.end());
			vector<float>::iterator it_max_sat = max_element(equalized_saturations.begin(), equalized_saturations.end());
			vector<float>::iterator it_max_val = max_element(equalized_values.begin(), equalized_values.end());

			equalized_min_hue = *it_min_hue;
			equalized_min_sat = *it_min_sat;
			equalized_min_val = *it_min_val;

			equalized_max_hue = *it_max_hue;
			equalized_max_sat = *it_max_sat;
			equalized_max_val = *it_max_val;
		}
		else {
			equalized_min_hue = 0;
			equalized_min_sat = 0;
			equalized_min_val = 0;

			equalized_max_hue = 0;
			equalized_max_sat = 0;
			equalized_max_val = 0;
		}

		cout << "Average Hue : " << average_hue << " Average Saturation : " << average_saturation << " Average Value : " << average_value << endl;
		cout << "Equalized Average Hue : " << equalized_average_hue << " Equalized Average Saturation : " << equalized_average_saturation << " Equalized Average Value : " << equalized_average_value << endl;
		cout << "Hue Range in Input : " << min_hue << "-" << max_hue << " Saturation Range in Input : " << min_sat << "-" << max_sat << " Value Range in Input :" << min_val << "-" << max_val << endl;
		cout << "Hue Range in Histogram Equalized HSV : " << equalized_min_hue << "-" << equalized_max_hue << " Saturation Range in Histogram Equalized HSV : " << equalized_min_sat << "-" << equalized_max_sat << " Value Histogram Equalized HSV in Input :" << equalized_min_val << "-" << equalized_max_val << endl;
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
		cvtColor(equalized_hsv_input_masked, bla, COLOR_HSV2BGR);
		imshow("bla", bla);
		imshow("roi", roi);
		imshow("pixels", pixels);
		imshow("trackHough", input);
		imshow("Masked HSV Input", hsv_input_masked);
		//imshow("equalized_hsv_input_masked", equalized_hsv_input_masked);
		waitKey(1);

		distances.clear();
		index_list.clear();
		hues.clear();
		saturations.clear();
		values.clear();
		equalized_hues.clear();
		equalized_saturations.clear();
		equalized_values.clear();
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