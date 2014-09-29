#include <iostream>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/objdetect/objdetect.hpp>

#include "face_x.h"

using namespace std;

const string kModelFileName = "model.xml.gz";
const string kAlt2 = "haarcascade_frontalface_alt2.xml";
const string kTestImage = "test.jpg";

void AlignImage(const FaceX & face_x)
{
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
			cv::circle(image, landmark, 1, cv::Scalar(0, 255, 0), 2);
		}
	}
	cv::imshow("Alignment result", image);
	cv::waitKey();
}

void Tracking(const FaceX & face_x)
{
	cout << "Press \"r\" to re-initialize the face location." << endl;
	cv::Mat frame;
	cv::Mat img;
	cv::VideoCapture vc(0);
	vc >> frame;
	cv::CascadeClassifier cc(kAlt2);
	cv::vector<cv::Point2d> landmarks(face_x.landmarks_count());

	for (;;)
	{
		vc >> frame;
		cv::cvtColor(frame, img, cv::COLOR_BGR2GRAY);
		cv::equalizeHist(img, img);
		cv::imshow("test", img);

		cv::vector<cv::Point2d> original_landmarks = landmarks;
		landmarks = face_x.Alignment(img, landmarks);

		for (int i = 0; i < landmarks.size(); ++i)
		{
			landmarks[i].x = (landmarks[i].x + original_landmarks[i].x) / 2;
			landmarks[i].y = (landmarks[i].y + original_landmarks[i].y) / 2;
		}

		for (cv::Point2d p : landmarks)
		{
			cv::circle(frame, p, 1, cv::Scalar(0, 255, 0), 2);
		}

		cv::imshow("\"r\" to re-initialize, \"q\" to exit", frame);
		int key = cv::waitKey(10);
		if (key == 'q')
			break;
		else if (key == 'r')
		{
			vector<cv::Rect> faces;
			cc.detectMultiScale(img, faces);
			if (!faces.empty())
			{
				landmarks = face_x.Alignment(img, faces[0]);
			}
		}
	}
}

int main()
{
	FaceX face_x;
	if (!face_x.OpenModel(kModelFileName))
	{
		cout << "Cannot open model file \"" << kModelFileName << "\"!" << endl;
		return -1;
	}

	cout << "Choice: " << endl;
	cout << "1. Align " << kTestImage << " in the current working directory." << endl;
	cout << "2. Align video from web camera." << endl;
	cout << "Please select one [1/2]: ";
	int choice;
	cin >> choice;
	switch (choice)
	{
	case 1:
		AlignImage(face_x);
		break;
	case 2:
		Tracking(face_x);
		break;
	}
}