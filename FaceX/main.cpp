#include <iostream>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include "face_x.h"

using namespace std;

int main()
{
	const string kModelFileName = "model.xml.gz";
	const string kAlt2 = "haarcascade_frontalface_alt2.xml";
	const string kTestImage = "test.jpg";

	FaceX face_x;
	if (!face_x.OpenModel(kModelFileName))
	{
		cout << "Cannot open model file \"" << kModelFileName << "\"!" << endl;
		return -1;
	}

	cv::Mat image = cv::imread(kTestImage);
	cv::Mat gray_image;
	cv::cvtColor(image, gray_image, CV_BGR2GRAY);
	cv::CascadeClassifier cc(kAlt2);
	cv::equalizeHist(gray_image, gray_image);
	vector<cv::Rect> faces;
	cc.detectMultiScale(gray_image, faces);

	for (cv::Rect face : faces)
	{
		cv::rectangle(image, face, cv::Scalar(0, 0, 255), 2);
		vector<cv::Point2d> landmarks = face_x.Alignment(gray_image, face);
		for (cv::Point2d landmark : landmarks)
		{
			cv::rectangle(image, cv::Rect(landmark.x, landmark.y, 2, 2),
				cv::Scalar(0, 255, 0), 2);
		}
	}
	cv::imshow("Alignment result", image);
	cv::waitKey();
}