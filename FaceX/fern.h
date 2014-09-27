#ifndef FACE_X_FERN_H_
#define FACE_X_FERN_H_

#include<vector>
#include<utility>

#include<opencv2/core/core.hpp>

struct Fern
{
	void ApplyMini(cv::Mat features, std::vector<double> &coeffs)const;

	void read(const cv::FileNode &fn);

	std::vector<double> thresholds;
	std::vector<std::pair<int, int>> features_index;
	std::vector<std::vector<cv::Point2d>> outputs;
	std::vector<std::vector<std::pair<int, double>>> outputs_mini;
};

void read(const cv::FileNode& node, Fern& f,
	const Fern& default_value = Fern());

#endif