/*
The MIT License(MIT)

Copyright(c) 2015 Yang Cao

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files(the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and / or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions :

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
*/

#include <iostream>
#include <string>

#include <opencv2/opencv.hpp>

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
	if(cc.empty())
	{
		cout << "Cannot open model file " << kAlt2 << " for OpenCV face detector!" << endl;
		return;
	}
	vector<cv::Rect> faces;
	double start_time = cv::getTickCount();
	cc.detectMultiScale(gray_image, faces);
	cout << "Detection time: " << (cv::getTickCount() - start_time) / cv::getTickFrequency()
		<< "s" << endl;

	for (cv::Rect face : faces)
	{
		cv::rectangle(image, face, cv::Scalar(0, 0, 255), 2);
		start_time = cv::getTickCount();
		vector<cv::Point2d> landmarks = face_x.Alignment(gray_image, face);
		cout << "Alignment time: " 
			<< (cv::getTickCount() - start_time) / cv::getTickFrequency()
			<< "s" << endl;
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
	vector<cv::Point2d> landmarks(face_x.landmarks_count());

	for (;;)
	{
		vc >> frame;
		cv::cvtColor(frame, img, cv::COLOR_BGR2GRAY);
		cv::imshow("Gray image", img);

		vector<cv::Point2d> original_landmarks = landmarks;
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
	try
	{
		FaceX face_x(kModelFileName);

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
	catch (const runtime_error& e)
	{
		cerr << e.what() << endl;
	}
}
