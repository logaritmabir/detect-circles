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

		GaussianBlur(input, input, Size(3, 3), 2, 2, BORDER_REFLECT_101); /*gürültü giderme*/

		Mat hsv_input, hsv_input_gray;
		cvtColor(input, hsv_input, COLOR_BGR2HSV);
		cvtColor(hsv_input, hsv_input_gray, COLOR_BGR2GRAY);

		Mat mask1, mask2;

		//inRange(hsv_input, Scalar(0, 70, 50), Scalar(10, 255, 255), mask1);
		//inRange(hsv_input, Scalar(170, 70, 50), Scalar(180, 255, 255), mask2);
		//Mat mask = (mask1 | mask2);
		inRange(hsv_input, Scalar(90, 50, 50), Scalar(130, 255, 255), mask1); /*mavi maske aralığı*/
		Mat mask = (mask1);

		Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(9, 9));  /*maskelenen bölgedeki gürültülerin kaldırılması*/
		morphologyEx(mask, mask, MORPH_OPEN, kernel);
		morphologyEx(mask, mask, MORPH_CLOSE, kernel);

		Mat hsv_input_gray_masked = hsv_input_gray & mask; /*hough dönüşümü için gri hsv girdiyi kırp*/

		Mat mask3d, hsv_input_masked;
		cvtColor(mask, mask3d, COLOR_GRAY2BGR);
		hsv_input_masked = mask3d & hsv_input; 

		vector<Vec3f> circles; /*daire merkezleri*/

		int pieces = 5; /*enine ya da boyuna olacak şekilde kamera kadrajına kaç tane daire sığabilir, bu parametre tespit edilecek dairelerin yarıçapını belirleyecek*/
		int radius = pieces * 2;
		int min_dist = input.cols / pieces;
		int min_radius = input.cols / radius;
		int max_radius = input.cols / (radius - 2);

		HoughCircles(hsv_input_gray_masked, circles, HOUGH_GRADIENT, 1.5, min_dist, 50, 40, min_radius, max_radius); /*dairelerin tespiti*/

		Mat roi = Mat::zeros(input.size(), CV_8UC1);
		vector<double> distances;
		for (int i = 0; i < circles.size(); i++) { /*dairelerin merkez noktaya uzaklığının hesaplanması ve uzaklıklarına göre indislenmesi*/
			Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
			int radius = cvRound(circles[i][2]);
			circle(input, center, 3, Scalar(0, 255, 0), -1, 8, 0);
			circle(input, center, radius, Scalar(255, 0, 0), 3, 8, 0);
			circle(roi, center, radius, Scalar(255, 255, 255), -1, 8, 0); /*dairesel bölgenin kesilmesi*/
			double dist = sqrt(pow(origin.x - center.x, 2) + pow(origin.y - center.y, 2)); /*merkez noktaya uzaklığın hesaplanması*/
			distances.push_back(dist);
		}

		Mat roi3d;
		cvtColor(roi, roi3d, COLOR_GRAY2BGR);
		Mat pixels = hsv_input_masked & roi3d; 
		Mat roi_masked = roi3d & hsv_input_masked; /*dairesel cismin iç bölge piksellerinin alınması*/

		vector<float> hues, saturations, values;
		for (int i = 0; i < pixels.rows; i++) { 
			for (int j = 0; j < pixels.cols; j++) {
				if (roi_masked.at<Vec3b>(i, j) != Vec3b(0,0,0)) {
					Vec3b pixel = pixels.at<Vec3b>(i, j);
					float val = pixel[2];/*histogram eşitliği uygulanacak kanal values*/
					values.push_back(val);
				}
			}
		}
		float total_values = accumulate(values.begin(), values.end(), 0);
		float average_value = total_values / values.size();


		float min_val;
		float max_val;

		if (!(values.empty())) {
			vector<float>::iterator it_min_val = min_element(values.begin(), values.end());
			vector<float>::iterator it_max_val = max_element(values.begin(), values.end());

			min_val = *it_min_val;
			max_val = *it_max_val;
		}
		else {
			min_val = 0;
			max_val = 0;
		}

		vector<Mat> channels;
		split(hsv_input, channels);
		equalizeHist(channels[2], channels[2]); /*histogram eşitleme*/

		Mat equalized_hsv_input;
		merge(channels, equalized_hsv_input);

		vector<float> equalized_hues;
		vector<float> equalized_saturations;
		vector<float> equalized_values;
		for (int i = 0; i < equalized_hsv_input.rows; i++) { /*eşitleme sonrası yeni değerlerin okunması*/
			for (int j = 0; j < equalized_hsv_input.cols; j++) {
				if (roi_masked.at<Vec3b>(i, j) != Vec3b(0, 0, 0)) {
					Vec3b pixel = equalized_hsv_input.at<Vec3b>(i, j);
					float val = pixel[2];
					equalized_values.push_back(val);
				}
			}
		}

		float equalized_total_values = accumulate(equalized_values.begin(), equalized_values.end(), 0);
		float equalized_average_value = equalized_total_values / equalized_values.size();

		float equalized_min_val;
		float equalized_max_val;

		if (!(equalized_values.empty())) {
			vector<float>::iterator it_min_val = min_element(equalized_values.begin(), equalized_values.end());
			vector<float>::iterator it_max_val = max_element(equalized_values.begin(), equalized_values.end());

			equalized_min_val = *it_min_val;
			equalized_max_val = *it_max_val;
		}
		else {
			equalized_min_val = 0;
			equalized_max_val = 0;
		}

		cout << "Average Value : " << average_value << endl;
		cout << "Equalized Average Value : " << equalized_average_value << endl;
		cout << "Value Range in Input :" << min_val << "-" << max_val << endl;
		cout << "Value Range in Histogram Equalized HSV :" << equalized_min_val << "-" << equalized_max_val << endl;
		vector<int>index_list; /*merkez noktaya olan uzaklıklar için indis listesi*/
		for (int i = 0; i < distances.size(); i++) {
			int nearest = find_nearest_circle(distances);
			index_list.push_back(nearest);
		}

		for (int i = 0; i < index_list.size(); i++) { /*dairelerin indislenmesi*/
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
	distances.at(min) = INT_MAX;
	return min;
}