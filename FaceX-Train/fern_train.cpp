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

#include "fern_train.h"

#include<iostream>
#include<cstdlib>
#include<memory>
#include<algorithm>

using namespace std;

FernTrain::FernTrain(const TrainingParameters &tp) : training_parameters(tp)
{
}

void FernTrain::Regress(vector<vector<cv::Point2d>> *targets, 
	cv::Mat pixels_val, cv::Mat pixels_cov)
{
	cv::Mat Y(targets->size(), (*targets)[0].size() * 2, CV_64FC1);
	for (int i = 0; i < Y.rows; ++i)
	{
		for (int j = 0; j < Y.cols; j += 2)
		{
			Y.at<double>(i, j) = (*targets)[i][j / 2].x;
			Y.at<double>(i, j + 1) = (*targets)[i][j / 2].y;
		}
	}
	features_index.assign(training_parameters.F, pair<int, int>());
	thresholds.assign(training_parameters.F, 0);

	for (int i = 0; i < training_parameters.F; ++i)
	{
		cv::Mat projection(Y.cols, 1, CV_64FC1);
		cv::theRNG().fill(projection, cv::RNG::NORMAL, cv::Scalar(0), cv::Scalar(1));
		unique_ptr<double[]> Y_proj_data(new double[Y.rows + 3]);
		cv::Mat Y_proj(Y.rows, 1, CV_64FC1, cv::alignPtr(Y_proj_data.get(), 32));
		static_cast<cv::Mat>(Y * projection).copyTo(Y_proj);
		double Y_proj_cov = Covariance(Y_proj.ptr<double>(0), 
			Y_proj.ptr<double>(0), Y_proj.total());	// Use AVXCovariance if you want.
		vector<double> Y_pixels_cov(pixels_val.rows);
		for (int j = 0; j < pixels_val.rows; ++j)
		{
			Y_pixels_cov[j] = Covariance(Y_proj.ptr<double>(0), 
				pixels_val.ptr<double>(j), Y_proj.total());	// Use AVXCovariance if you want.
		}
		double max_corr = -1;
		for (int j = 0; j < pixels_val.rows; ++j)
		{
			for (int k = 0; k < pixels_val.rows; ++k)
			{
				double corr = (Y_pixels_cov[j] - Y_pixels_cov[k]) / sqrt(
					Y_proj_cov * (pixels_cov.at<double>(j, j) 
					+ pixels_cov.at<double>(k, k) 
					- 2 * pixels_cov.at<double>(j, k)));
				if (corr > max_corr)
				{
					max_corr = corr;
					features_index[i].first = j;
					features_index[i].second = k;
				}
			}
		}

		double threshold_max = -1000000;
		double threshold_min = 1000000;
		for (int j = 0; j < pixels_val.cols; ++j)
		{
			double val = pixels_val.at<double>(features_index[i].first, j)
				- pixels_val.at<double>(features_index[i].second, j);
			threshold_max = max(threshold_max, val);
			threshold_min = min(threshold_min, val);
		}
		thresholds[i] = (threshold_max + threshold_min) / 2 
			+ cv::theRNG().uniform(-(threshold_max - threshold_min) * 0.1, 
			(threshold_max - threshold_min) * 0.1);
	}

	int outputs_count = 1 << training_parameters.F;
	outputs.assign(outputs_count, vector<cv::Point2d>((*targets)[0].size()));
	vector<int> each_output_count(outputs_count);

	for (int i = 0; i < targets->size(); ++i)
	{
		int mask = 0;
		for (int j = 0; j < training_parameters.F; ++j)
		{
			double p1 = pixels_val.at<double>(features_index[j].first, i);
			double p2 = pixels_val.at<double>(features_index[j].second, i);
			mask |= (p1 - p2 > thresholds[j]) << j;
		}
		outputs[mask] = ShapeAdjustment(outputs[mask], (*targets)[i]);
		++each_output_count[mask];
	}

	for (int i = 0; i < outputs_count; ++i)
	{
		for (cv::Point2d &p : outputs[i])
			p *= 1.0 / (each_output_count[i] + training_parameters.Beta);
	}
}

vector<cv::Point2d> FernTrain::Apply(cv::Mat features)const
{
	int outputs_index = 0;
	for (int i = 0; i < training_parameters.F; ++i)
	{
		pair<int, int> feature = features_index[i];
		double p1 = features.at<double>(feature.first);
		double p2 = features.at<double>(feature.second);
		outputs_index |= (p1 - p2 > thresholds[i]) << i;
	}
	return outputs[outputs_index];
}

void FernTrain::ApplyMini(cv::Mat features, std::vector<double> &coeffs)const
{
	int outputs_index = 0;
	for (int i = 0; i < training_parameters.F; ++i)
	{
		pair<int, int> feature = features_index[i];
		double p1 = features.at<double>(feature.first);
		double p2 = features.at<double>(feature.second);
		outputs_index |= (p1 - p2 > thresholds[i]) << i;
	}

	const vector<pair<int, double>> &output = outputs_mini[outputs_index];
	for (int i = 0; i < training_parameters.Q; ++i)
		coeffs[output[i].first] += output[i].second;
}



void FernTrain::write(cv::FileStorage &fs)const
{
	fs << "{";
	fs << "thresholds" << thresholds;
	fs << "features_index";
	fs << "[";
	for (auto it = features_index.begin(); it != features_index.end(); ++it)
		fs << "{" << "first" << it->first << "second" << it->second << "}";
	fs << "]";
	fs << "outputs_mini";
	fs << "[";
	for (const auto &output: outputs_mini)
	{
		fs << "[";
		for (int i = 0; i < training_parameters.Q; ++i)
		{
			fs << "{" << "index" << output[i].first <<
				"coeff" << output[i].second << "}";
		}
		fs << "]";
	}
	fs << "]";
	fs << "}";
}

void write(cv::FileStorage& fs, const string&, const FernTrain &f)
{
	f.write(fs);
}