#ifndef FACE_X_UTILS_H_
#define FACE_X_UTILS_H_

#include<vector>

#include<opencv2/core/core.hpp>

struct Transform
{
	cv::Matx22d scale_rotation;
	cv::Matx21d translation;

	void Apply(std::vector<cv::Point2d> *x, bool need_translation = true);
};

template<typename T>
inline T Sqr(T a)
{
	return a * a;
}

Transform Procrustes(const std::vector<cv::Point2d> &x,
	const std::vector<cv::Point2d> &y);

std::vector<cv::Point2d> ShapeAdjustment(const std::vector<cv::Point2d> &shape,
	const std::vector<cv::Point2d> &offset);

std::vector<cv::Point2d> MapShape(cv::Rect original_face_rect,
	const std::vector<cv::Point2d> original_landmarks,
	cv::Rect new_face_rect);

#endif