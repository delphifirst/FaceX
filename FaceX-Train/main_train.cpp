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

#include <iostream>
#include <fstream>
#include <string>
#include <set>
#include <map>
#include <algorithm>
#include <numeric>
#include <stdexcept>

#include <opencv2/opencv.hpp>

#include "regressor_train.h"
#include "utils_train.h"

using namespace std;

TrainingParameters ReadParameters(const string &filename)
{
	ifstream fin(filename);
	TrainingParameters result;
	if (fin)
	{
		map<string, string> items;
		string line;
		int line_no = 0;
		while (getline(fin, line))
		{
			++line_no;
			line = TrimStr(line);
			if (line.empty() || line[0] == '#')
				continue;

			int colon_pos = line.find(':');
			if (colon_pos == string::npos)
			{
				throw runtime_error("Illegal line " + to_string(line_no) +
					" in config file " + filename);
			}
			
			items[TrimStr(line.substr(0, colon_pos))] = TrimStr(
				line.substr(colon_pos + 1));
		}

		result.training_data_root = items.at("training_data_root");
		result.landmark_count = stoi(items.at("landmark_count"));
		if (result.landmark_count <= 0)
			throw invalid_argument("landmark_count must be positive.");
		result.left_eye_index = stoi(items.at("left_eye_index"));
		if (result.left_eye_index < 0 || result.left_eye_index >= result.landmark_count)
			throw out_of_range("left_eye_index not in range.");
		result.right_eye_index = stoi(items.at("right_eye_index"));
		if (result.right_eye_index < 0 || result.right_eye_index >= result.landmark_count)
			throw out_of_range("right_eye_index not in range.");
		result.output_model_pathname = items.at("output_model_pathname");
		result.T = stoi(items.at("T"));
		if (result.T <= 0)
			throw invalid_argument("T must be positive.");
		result.K = stoi(items.at("K"));
		if (result.K <= 0)
			throw invalid_argument("K must be positive.");
		result.P = stoi(items.at("P"));
		if (result.P <= 0)
			throw invalid_argument("P must be positive.");
		result.Kappa = stod(items.at("Kappa"));
		if (result.Kappa < 0.01 || result.Kappa > 1)
			throw out_of_range("Kappa must be in [0.01, 1].");
		result.F = stoi(items.at("F"));
		if (result.F <= 0)
			throw invalid_argument("F must be positive.");
		result.Beta = stoi(items.at("Beta"));
		if (result.Beta <= 0)
			throw invalid_argument("Beta must be positive.");
		result.TestInitShapeCount = stoi(items.at("TestInitShapeCount"));
		if (result.TestInitShapeCount <= 0)
			throw invalid_argument("TestInitShapeCount must be positive.");
		result.ArgumentDataFactor = stoi(items.at("ArgumentDataFactor"));
		if (result.ArgumentDataFactor <= 0)
			throw invalid_argument("ArgumentDataFactor must be positive.");
		result.Base = stoi(items.at("Base"));
		if (result.Base <= 0)
			throw invalid_argument("Base must be positive.");
		result.Q = stoi(items.at("Q"));
		if (result.Q <= 0)
			throw invalid_argument("Q must be positive.");
	}
	else
		throw runtime_error("Cannot open config file: " + filename);

	return result;
}

vector<DataPoint> GetTrainingData(const TrainingParameters &tp)
{
	const string label_pathname = tp.training_data_root + "/labels.txt";
	ifstream fin(label_pathname);
	if (!fin)
		throw runtime_error("Cannot open label file " + label_pathname + " (Pay attention to path separator!)");

	vector<DataPoint> result;
	string current_image_pathname;
	int count = 0;
	while (fin >> current_image_pathname)
	{
		DataPoint current_data_point;
		current_data_point.image = cv::imread(tp.training_data_root + "/" +
			current_image_pathname, CV_LOAD_IMAGE_GRAYSCALE);
		if (current_data_point.image.data == nullptr)
			throw runtime_error("Cannot open image file " + current_image_pathname + " (Pay attention to path separator!)");
		int left, right, top, bottom;
		fin >> left >> right >> top >> bottom;
		current_data_point.face_rect =
			cv::Rect(left, top, right - left + 1, bottom - top + 1);

		for (int i = 0; i < tp.landmark_count; ++i)
		{
			cv::Point2d p;
			fin >> p.x >> p.y;
			current_data_point.landmarks.push_back(p);
		}
		result.push_back(current_data_point);
		++count;
	}
	return result;
}

vector<vector<cv::Point2d>> CreateTestInitShapes(
	const vector<DataPoint> &training_data, const TrainingParameters &tp)
{
	if (tp.TestInitShapeCount > training_data.size())
	{
		throw invalid_argument("TestInitShapeCount is larger than training image count"
			", which is not allowed.");
	}
	const int kLandmarksSize = training_data[0].landmarks.size();
	cv::Mat all_landmarks(training_data.size(), kLandmarksSize * 2, CV_32FC1);
	for (int i = 0; i < training_data.size(); ++i)
	{
		vector<cv::Point2d> landmarks = MapShape(training_data[i].face_rect,
			training_data[i].landmarks, cv::Rect(0, 0, 1, 1));
		for (int j = 0; j < kLandmarksSize; ++j)
		{
			all_landmarks.at<float>(i, j * 2) = static_cast<float>(landmarks[j].x);
			all_landmarks.at<float>(i, j * 2 + 1) = static_cast<float>(landmarks[j].y);
		}
	}
	cv::Mat labels, centers;
	cv::kmeans(all_landmarks, tp.TestInitShapeCount, labels, 
		cv::TermCriteria(cv::TermCriteria::COUNT, 50, 0), 
		10, cv::KMEANS_RANDOM_CENTERS | cv::KMEANS_PP_CENTERS, centers);

	vector<vector<cv::Point2d>> result;
	for (int i = 0; i < tp.TestInitShapeCount; ++i)
	{
		vector<cv::Point2d> landmarks;
		for (int j = 0; j < kLandmarksSize; ++j)
		{
			landmarks.push_back(cv::Point2d(
				centers.at<float>(i, j * 2), centers.at<float>(i, j * 2 + 1)));
		}
		result.push_back(landmarks);
	}
	return result;
}

vector<DataPoint> ArgumentData(const vector<DataPoint> &training_data, int factor)
{
	if (training_data.size() < 2 * factor)
	{
		throw invalid_argument("You should provide training data with at least "
			"2*ArgumentDataFactor images.");
	}
	vector<DataPoint> result(training_data.size() * factor);
	for (int i = 0; i < training_data.size(); ++i)
	{
		set<int> shape_indices;
		while (shape_indices.size() < factor)
		{
			int rand_index = cv::theRNG().uniform(0, training_data.size());
			if (rand_index != i)
				shape_indices.insert(rand_index);
		}

		auto it = shape_indices.cbegin();
		for (int j = i * factor; j < (i + 1) * factor; ++j, ++it)
		{
			result[j] = training_data[i];
			result[j].init_shape = MapShape(training_data[*it].face_rect, 
				training_data[*it].landmarks, result[j].face_rect);
		}
	}
	return result;
}

vector<vector<cv::Point2d>> ComputeNormalizedTargets(
	const vector<cv::Point2d> mean_shape, const vector<DataPoint> &data)
{
	vector<vector<cv::Point2d>> result;

	for (const DataPoint& dp : data)
	{
		vector<cv::Point2d> error = ShapeDifference(dp.landmarks, dp.init_shape);
		Transform t = Procrustes(mean_shape, dp.init_shape);
		t.Apply(&error, false);
		result.push_back(error);
	}

	return result;
}


void TrainModel(const vector<DataPoint> &training_data, const TrainingParameters &tp)
{
	cout << "Training data count: " << training_data.size() << endl;

	vector<vector<cv::Point2d>> shapes;
	for (const DataPoint &dp : training_data)
		shapes.push_back(dp.landmarks);

	vector<cv::Point2d> mean_shape = MeanShape(shapes, tp);

	vector<vector<cv::Point2d>> test_init_shapes = 
		CreateTestInitShapes(training_data, tp);

	vector<DataPoint> argumented_training_data = 
		ArgumentData(training_data, tp.ArgumentDataFactor);

	vector<RegressorTrain> stage_regressors(tp.T, RegressorTrain(tp));
	for (int i = 0; i < tp.T; ++i)
	{
		long long s = cv::getTickCount();

		vector<vector<cv::Point2d>> normalized_targets = 
			ComputeNormalizedTargets(mean_shape, argumented_training_data);
		stage_regressors[i].Regress(mean_shape, &normalized_targets, 
			argumented_training_data);
		for (DataPoint &dp : argumented_training_data)
		{
			vector<cv::Point2d> offset = 
				stage_regressors[i].Apply(mean_shape, dp);
			Transform t = Procrustes(dp.init_shape, mean_shape);
			t.Apply(&offset, false);
			dp.init_shape = ShapeAdjustment(dp.init_shape, offset);
		}

		cout << "(^_^) Finish training " << i + 1 << " regressor. Using " 
			<< (cv::getTickCount() - s) / cv::getTickFrequency() 
			<< "s. " << tp.T << " in total." << endl;
	}

	cv::FileStorage model_file;
	model_file.open(tp.output_model_pathname, cv::FileStorage::WRITE);
	model_file << "mean_shape" << mean_shape;
	model_file << "test_init_shapes" << "[";
	for (auto it = test_init_shapes.begin(); it != test_init_shapes.end(); ++it)
	{
		model_file << *it;
	}
	model_file << "]";
	model_file << "stage_regressors" << "[";
	for (auto it = stage_regressors.begin(); it != stage_regressors.end(); ++it)
		model_file << *it;
	model_file << "]";
	model_file.release();
}

int main(int argc, char *argv[])
{
	if (argc != 2)
	{
		cout << "Usage: FaceX-Train config.txt" << endl;
		return 0;
	}

	try
	{
		TrainingParameters tp = ReadParameters(argv[1]);

		cout << "Training begin." << endl;
		vector<DataPoint> training_data = GetTrainingData(tp);
		TrainModel(training_data, tp);
	}
	catch (const exception &e)
	{
		cout << e.what() << endl;
		return -1;
	}
}
