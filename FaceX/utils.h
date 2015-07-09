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

#ifndef FACE_X_UTILS_H_
#define FACE_X_UTILS_H_

#include<vector>

#include<opencv2/opencv.hpp>

struct Transform
{
	cv::Matx22d scale_rotation;
	cv::Matx21d translation;

	void Apply(std::vector<cv::Point2d> *x, bool need_translation = true) const;
};

template<typename T>
inline T Sqr(T a)
{
	return a * a;
}

// Find the transform from y to x
Transform Procrustes(const std::vector<cv::Point2d> &x,
	const std::vector<cv::Point2d> &y);

std::vector<cv::Point2d> ShapeAdjustment(const std::vector<cv::Point2d> &shape,
	const std::vector<cv::Point2d> &offset);

std::vector<cv::Point2d> MapShape(cv::Rect original_face_rect,
	const std::vector<cv::Point2d> original_landmarks,
	cv::Rect new_face_rect);

#endif