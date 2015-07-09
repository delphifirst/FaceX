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

#ifndef FACE_X_REGRESSOR_TRAIN_H_
#define FACE_X_REGRESSOR_TRAIN_H_

#include<vector>
#include<utility>
#include<string>

#include<opencv2/opencv.hpp>

#include "utils_train.h"
#include "fern_train.h"

class RegressorTrain
{
public:
	RegressorTrain(const TrainingParameters &tp);
	void Regress(const std::vector<cv::Point2d> &mean_shape, 
		std::vector<std::vector<cv::Point2d>> *targets,
		const std::vector<DataPoint> & training_data);
	std::vector<cv::Point2d> Apply(const std::vector<cv::Point2d> &mean_shape, 
		const DataPoint &data) const;

	void write(cv::FileStorage &fs)const;

private:
	std::vector<std::pair<int, cv::Point2d>> pixels_;
	std::vector<FernTrain> ferns_;
	cv::Mat base_;
	const TrainingParameters &training_parameters_;

	void CompressFerns();
};

void write(cv::FileStorage& fs, const std::string&, const RegressorTrain& r);

#endif