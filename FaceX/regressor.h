#ifndef FACE_X_REGRESSOR_H_
#define FACE_X_REGRESSOR_H_

#include<vector>
#include<utility>
#include<string>

#include<opencv2/core/core.hpp>

#include "fern.h"

class Regressor
{
public:
	std::vector<cv::Point2d> Apply(const std::vector<cv::Point2d> &mean_shape, 
		cv::Mat image, const std::vector<cv::Point2d> &init_shape) const;

	void read(const cv::FileNode &fn);

private:

	std::vector<std::pair<int, cv::Point2d>> pixels;
	std::vector<Fern> ferns;
	cv::Mat base;
};

void read(const cv::FileNode& node, Regressor& r, 
	const Regressor& default_value = Regressor());

#endif