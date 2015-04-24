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

#include "face_x.h"

#include <algorithm>
#include <stdexcept>

#include "utils.h"

using namespace std;

FaceX::FaceX(const string & filename)
{
	cv::FileStorage model_file;
	model_file.open(filename, cv::FileStorage::READ);
	if (!model_file.isOpened())
		throw runtime_error("Cannot open model file \"" + filename + "\".");

	model_file["mean_shape"] >> mean_shape_;
	cv::FileNode fn = model_file["test_init_shapes"];
	for (auto it = fn.begin(); it != fn.end(); ++it)
	{
		vector<cv::Point2d> shape;
		*it >> shape;
		test_init_shapes_.push_back(shape);
	}
	fn = model_file["stage_regressors"];
	for (auto it = fn.begin(); it != fn.end(); ++it)
	{
		Regressor r;
		*it >> r;
		stage_regressors_.push_back(r);
	}
}

vector<cv::Point2d> FaceX::Alignment(cv::Mat image, cv::Rect face_rect) const
{
	vector<vector<double>> all_results(test_init_shapes_[0].size() * 2);
	for (int i = 0; i < test_init_shapes_.size(); ++i)
	{
		vector<cv::Point2d> init_shape = MapShape(cv::Rect(0, 0, 1, 1),
			test_init_shapes_[i], face_rect);
		for (int j = 0; j < stage_regressors_.size(); ++j)
		{
			Transform t = Procrustes(init_shape, mean_shape_);
			vector<cv::Point2d> offset =
				stage_regressors_[j].Apply(t, image, init_shape);
			t.Apply(&offset, false);
			init_shape = ShapeAdjustment(init_shape, offset);
		}

		for (int i = 0; i < init_shape.size(); ++i)
		{
			all_results[i * 2].push_back(init_shape[i].x);
			all_results[i * 2 + 1].push_back(init_shape[i].y);
		}
	}

	vector<cv::Point2d> result(test_init_shapes_[0].size());
	for (int i = 0; i < result.size(); ++i)
	{
		nth_element(all_results[i * 2].begin(),
			all_results[i * 2].begin() + test_init_shapes_.size() / 2,
			all_results[i * 2].end());
		result[i].x = all_results[i * 2][test_init_shapes_.size() / 2];
		nth_element(all_results[i * 2 + 1].begin(),
			all_results[i * 2 + 1].begin() + test_init_shapes_.size() / 2,
			all_results[i * 2 + 1].end());
		result[i].y = all_results[i * 2 + 1][test_init_shapes_.size() / 2];
	}
	return result;
}

vector<cv::Point2d> FaceX::Alignment(cv::Mat image,
	vector<cv::Point2d> initial_landmarks) const
{
	vector<vector<double>> all_results(test_init_shapes_[0].size() * 2);
	for (int i = 0; i < test_init_shapes_.size(); ++i)
	{
		Transform t = Procrustes(initial_landmarks, test_init_shapes_[i]);
		vector<cv::Point2d> init_shape = test_init_shapes_[i];
		t.Apply(&init_shape);
		for (int j = 0; j < stage_regressors_.size(); ++j)
		{
			Transform t = Procrustes(init_shape, mean_shape_);
			vector<cv::Point2d> offset =
				stage_regressors_[j].Apply(t, image, init_shape);
			t.Apply(&offset, false);
			init_shape = ShapeAdjustment(init_shape, offset);
		}

		for (int i = 0; i < init_shape.size(); ++i)
		{
			all_results[i * 2].push_back(init_shape[i].x);
			all_results[i * 2 + 1].push_back(init_shape[i].y);
		}
	}

	vector<cv::Point2d> result(test_init_shapes_[0].size());
	for (int i = 0; i < result.size(); ++i)
	{
		nth_element(all_results[i * 2].begin(),
			all_results[i * 2].begin() + test_init_shapes_.size() / 2,
			all_results[i * 2].end());
		result[i].x = all_results[i * 2][test_init_shapes_.size() / 2];
		nth_element(all_results[i * 2 + 1].begin(),
			all_results[i * 2 + 1].begin() + test_init_shapes_.size() / 2,
			all_results[i * 2 + 1].end());
		result[i].y = all_results[i * 2 + 1][test_init_shapes_.size() / 2];
	}
	return result;
}