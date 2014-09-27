#include "face_x.h"

#include <algorithm>

#include "utils.h"

using namespace std;

bool FaceX::OpenModel(const string & filename)
{
	cv::FileStorage model_file;
	model_file.open(filename, cv::FileStorage::READ);
	if (!model_file.isOpened())
		return false;

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

	is_loaded_ = true;
	return true;
}

vector<cv::Point2d> FaceX::Alignment(cv::Mat image, cv::Rect face_rect)
{
	CV_Assert(is_loaded_);

	vector<vector<double>> all_results(test_init_shapes_[0].size() * 2);
	for (int i = 0; i < test_init_shapes_.size(); ++i)
	{
		vector<cv::Point2d> init_shape = MapShape(cv::Rect(0, 0, 1, 1),
			test_init_shapes_[i], face_rect);
		for (int j = 0; j < stage_regressors_.size(); ++j)
		{
			vector<cv::Point2d> offset =
				stage_regressors_[j].Apply(mean_shape_, image, init_shape);
			Transform t = Procrustes(init_shape, mean_shape_);
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