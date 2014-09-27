#include "fern.h"

#include<iostream>
#include<cstdlib>
#include<memory>
#include<algorithm>

#include "utils.h"

using namespace std;

void Fern::ApplyMini(cv::Mat features, std::vector<double> &coeffs)const
{
	int outputs_index = 0;
	for (int i = 0; i < features_index.size(); ++i)
	{
		pair<int, int> feature = features_index[i];
		double p1 = features.at<double>(feature.first);
		double p2 = features.at<double>(feature.second);
		outputs_index |= (p1 - p2 > thresholds[i]) << i;
	}

	const vector<pair<int, double>> &output = outputs_mini[outputs_index];
	for (int i = 0; i < output.size(); ++i)
		coeffs[output[i].first] += output[i].second;
}

void Fern::read(const cv::FileNode &fn)
{
	thresholds.clear();
	features_index.clear();
	outputs_mini.clear();
	fn["thresholds"] >> thresholds;
	cv::FileNode features_index_node = fn["features_index"];
	for (auto it = features_index_node.begin(); it != features_index_node.end(); ++it)
	{
		pair<int, int> feature_index;
		(*it)["first"] >> feature_index.first;
		(*it)["second"] >> feature_index.second;
		features_index.push_back(feature_index);
	}
	cv::FileNode outputs_mini_node = fn["outputs_mini"];
	for (auto it = outputs_mini_node.begin(); it != outputs_mini_node.end(); ++it)
	{
		vector<std::pair<int, double>> output;
		cv::FileNode output_node = *it;
		for (auto it2 = output_node.begin(); it2 != output_node.end(); ++it2)
			output.push_back(make_pair((*it2)["index"], (*it2)["coeff"]));
		outputs_mini.push_back(output);
	}
}

void read(const cv::FileNode& node, Fern &f, const Fern& default_value)
{
	if (node.empty())
	{
		f = default_value;
		cout << "! One default Regressor." << endl;
	}
	else
		f.read(node);
}