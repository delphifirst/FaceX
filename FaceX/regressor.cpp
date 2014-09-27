#include "regressor.h"

#include <utility>
#include <iostream>
#include <algorithm>

#include "utils.h"


using namespace std;


vector<cv::Point2d> Regressor::Apply(const vector<cv::Point2d> &mean_shape, 
	cv::Mat image, const std::vector<cv::Point2d> &init_shape) const
{
	cv::Mat pixels_val(1, pixels.size(), CV_64FC1);
	Transform t = Procrustes(init_shape, mean_shape);
	vector<cv::Point2d> offsets(pixels.size());
	for (int j = 0; j < pixels.size(); ++j)
		offsets[j] = pixels[j].second;
	t.Apply(&offsets, false);

	double *p = pixels_val.ptr<double>(0);
	for (int j = 0; j < pixels.size(); ++j)
	{
		cv::Point pixel_pos = init_shape[pixels[j].first] + offsets[j];
		if (pixel_pos.inside(cv::Rect(0, 0, image.cols, image.rows)))
			p[j] = image.at<uchar>(pixel_pos);
		else
			p[j] = 0;
	}

	vector<double> coeffs(base.cols);
	for (int i = 0; i < ferns.size(); ++i)
		ferns[i].ApplyMini(pixels_val, coeffs);

	cv::Mat result_mat = cv::Mat::zeros(mean_shape.size() * 2, 1, CV_64FC1);
	for (int i = 0; i < base.cols; ++i)
		result_mat += coeffs[i] * base.col(i);
	vector<cv::Point2d> result(mean_shape.size());
	for (int i = 0; i < result.size(); ++i)
	{
		result[i].x = result_mat.at<double>(i * 2);
		result[i].y = result_mat.at<double>(i * 2 + 1);
	}
	return result;
}

void Regressor::read(const cv::FileNode &fn)
{
	pixels.clear();
	ferns.clear();
	cv::FileNode pixels_node = fn["pixels"];
	for (auto it = pixels_node.begin(); it != pixels_node.end(); ++it)
	{
		pair<int, cv::Point2d> pixel;
		(*it)["first"] >> pixel.first;
		(*it)["second"] >> pixel.second;
		pixels.push_back(pixel);
	}
	cv::FileNode ferns_node = fn["ferns"];
	for (auto it = ferns_node.begin(); it != ferns_node.end(); ++it)
	{
		Fern f;
		*it >> f;
		ferns.push_back(f);
	}
	fn["base"] >> base;
}

void read(const cv::FileNode& node, Regressor& r, const Regressor& default_value)
{
	if (node.empty())
	{
		r = default_value;
		cout << "One default Regressor. Model file is corrupt!" << endl;
	}
	else
		r.read(node);
}