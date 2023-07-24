#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <vector>
#include <algorithm>

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
	Mat frame;

	Mat input = imread("C:\\Users\\steam\\OneDrive\\Masaüstü\\ball-images\\circles.png", IMREAD_COLOR);
	Mat hsv_input, input_gray, hsv_input_gray;
	Mat edge;
	Mat mask1, mask2, input_masked, hsv_input_gray_masked;
	Point origin(0, 0);

	vector<double> distances;
	vector<int>index_list;

	while (1) {
		capture.read(frame);
		input = frame;

		distances.clear();
		index_list.clear();

		GaussianBlur(input, input, Size(3, 3), 2, 2, BORDER_REFLECT_101);

		cvtColor(input, hsv_input, COLOR_BGR2HSV);
		cvtColor(input, input_gray, COLOR_BGR2GRAY);
		cvtColor(hsv_input, hsv_input_gray, COLOR_BGR2GRAY);

		/*Masking for red*/
		inRange(hsv_input, Scalar(0, 70, 50), Scalar(10, 255, 255), mask1);
		inRange(hsv_input, Scalar(170, 70, 50), Scalar(180, 255, 255), mask2);
		input_masked = (mask1 | mask2);

		Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(9, 9));
		morphologyEx(input_masked, input_masked, MORPH_OPEN, kernel);
		morphologyEx(input_masked, input_masked, MORPH_CLOSE, kernel);

		hsv_input_gray_masked = hsv_input_gray & input_masked;

		//Canny(input_gray, edge, 50, 150, 3);

		vector<Vec3f> circles; /*element of circle_centers -> (x,y,radius) */

		int pieces = 5;
		int radius = pieces * 2;
		int min_dist = input_gray.cols / pieces;
		int min_radius = input_gray.cols / radius;
		int max_radius = input_gray.cols / (radius - 2);

		HoughCircles(hsv_input_gray_masked, circles, HOUGH_GRADIENT,1.5, min_dist,50,40, min_radius, max_radius);

		for (int i = 0; i < circles.size(); i++) {
			Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
			int radius = cvRound(circles[i][2]);
			circle(input, center, 3, Scalar(0, 255, 0), -1, 8, 0);
			circle(input, center, radius, Scalar(255, 0, 0), 3, 8, 0); 
			double dist = sqrt(pow(origin.x - center.x, 2) + pow(origin.y - center.y, 2));
			distances.push_back(dist);
		}

		for (int i = 0; i < distances.size(); i++) {
			int nearest = find_nearest_circle(distances);
			index_list.push_back(nearest);
		}

		for (int i = 0; i < index_list.size(); i++) {
			Point center = Point(cvRound(circles[index_list.at(i)][0]), cvRound(circles[index_list.at(i)][1]));
			putText(input, to_string(i), center, FONT_HERSHEY_PLAIN, 1.5, Scalar(255, 200, 100), 2);
		}
		//imshow("trackEdge", edge);
		imshow("trackHough", input);
		imshow("Input HSV Masked Red Channel", input_masked);
		waitKey(1);
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