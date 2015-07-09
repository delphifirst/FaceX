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

#ifndef FACE_X_FERN_TRAIN_H_
#define FACE_X_FERN_TRAIN_H_

#include<vector>
#include<utility>

#include<opencv2/opencv.hpp>

#include "utils_train.h"

struct FernTrain
{
	FernTrain(const TrainingParameters &tp);
	void Regress(std::vector<std::vector<cv::Point2d>> *targets, 
		cv::Mat pixels_val, cv::Mat pixels_cov);
	std::vector<cv::Point2d> Apply(cv::Mat features)const;
	void ApplyMini(cv::Mat features, std::vector<double> &coeffs)const;

	void write(cv::FileStorage &fs)const;

	std::vector<double> thresholds;
	std::vector<std::pair<int, int>> features_index;
	std::vector<std::vector<cv::Point2d>> outputs;
	std::vector<std::vector<std::pair<int, double>>> outputs_mini;
private:
	const TrainingParameters &training_parameters;
};

void write(cv::FileStorage& fs, const std::string&, const FernTrain& f);

#endif