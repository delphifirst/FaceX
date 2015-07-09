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

#ifndef FACE_X_REGRESSOR_H_
#define FACE_X_REGRESSOR_H_

#include<vector>
#include<utility>
#include<string>

#include<opencv2/opencv.hpp>

#include "fern.h"
#include "utils.h"

class Regressor
{
public:
	std::vector<cv::Point2d> Apply(const Transform &t, 
		cv::Mat image, const std::vector<cv::Point2d> &init_shape) const;

	void read(const cv::FileNode &fn);

private:

	std::vector<std::pair<int, cv::Point2d>> pixels_;
	std::vector<Fern> ferns_;
	cv::Mat base_;
};

void read(const cv::FileNode& node, Regressor& r, const Regressor&);

#endif