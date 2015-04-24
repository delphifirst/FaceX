/*
FaceX-Train is a tool to train model file for FaceX, which is an open
source face alignment library.

Copyright(C) 2015  Yang Cao

This program is free software : you can redistribute it and / or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef FACE_X_UTILS_TRAIN_H_
#define FACE_X_UTILS_TRAIN_H_

#include <vector>
#include <string>
#include <utility>

#include <opencv2/opencv.hpp>

struct TrainingParameters
{
	/* General parameters */
	std::string training_data_root = "";
	int landmark_count = -1;
	int left_eye_index = -1;
	int right_eye_index = -1;
	std::string output_model_pathname = "";

	/* Model parameters */
	int T = -1;
	int K = -1;
	int P = -1;
	double Kappa = -1;
	int F = -1;
	int Beta = -1;
	int TestInitShapeCount = -1;
	int ArgumentDataFactor = -1;
	int Base = -1;
	int Q = -1;
};

struct DataPoint
{
	cv::Mat image;
	cv::Rect face_rect;
	std::vector<cv::Point2d> landmarks;
	std::vector<cv::Point2d> init_shape;
};

struct Transform
{
	cv::Matx22d scale_rotation;
	cv::Matx21d translation;

	void Apply(std::vector<cv::Point2d> *x, bool need_translation = true);
};

template<typename T>
inline T Sqr(T a)
{
	return a * a;
}

Transform Procrustes(const std::vector<cv::Point2d> &x,
	const std::vector<cv::Point2d> &y);

std::vector<cv::Point2d> MeanShape(std::vector<std::vector<cv::Point2d>> shapes, 
	const TrainingParameters &tp);

std::vector<cv::Point2d> ShapeDifference(const std::vector<cv::Point2d> &s1,
	const std::vector<cv::Point2d> &s2);

std::vector<cv::Point2d> ShapeAdjustment(const std::vector<cv::Point2d> &shape,
	const std::vector<cv::Point2d> &offset);

double Covariance(double *x, double * y, const int size);

std::vector<cv::Point2d> MapShape(cv::Rect original_face_rect,
	const std::vector<cv::Point2d> original_landmarks,
	cv::Rect new_face_rect);

std::vector<std::pair<int, double>> OMP(cv::Mat x, cv::Mat base, int coeff_count);

std::string TrimStr(const std::string &s, const std::string &space = "\t ");

#endif