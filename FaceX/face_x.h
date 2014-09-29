#ifndef FACE_X_H_
#define FACE_X_H_

#include <string>
#include <vector>

#include <opencv2/core/core.hpp>

#include "regressor.h"

class FaceX
{
public:
	// Load model from file.
	//
	// filename: The file name of the model file.
	//
	// Return true if the model is successfully loaded, false otherwise.
	bool OpenModel(const std::string &filename);

	// Do face alignment.
	//
	// image: The image which contains face. Must be 8 bits gray image.
	// face_rect: Where the face locates.
	//
	// Return the landmarks. The number and positions of landmarks depends on
	// the model.
	std::vector<cv::Point2d> Alignment(cv::Mat image, cv::Rect face_rect) const;

	// Do face alignment incrementally. Useful for videos.
	//
	// image: The image which contains face. Must be 8 bits gray image.
	// initial_landmarks: Initial guess of where each landmark is.
	//
	// Return the landmarks. The number and positions of landmarks depends on
	// the model.
	std::vector<cv::Point2d> Alignment(cv::Mat image,
		std::vector<cv::Point2d> initial_landmarks) const;

	// Return how many landmarks the model provides for a face.
	int landmarks_count() const
	{
		return mean_shape_.size();
	}

private:
	std::vector<cv::Point2d> mean_shape_;
	std::vector<std::vector<cv::Point2d>> test_init_shapes_;
	std::vector<Regressor> stage_regressors_;
	bool is_loaded_ = false;
};

#endif