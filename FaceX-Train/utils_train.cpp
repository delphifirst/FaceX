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

#include "utils_train.h"

#include<cmath>
#include<iostream>
#include<cassert>

using namespace std;


void Transform::Apply(vector<cv::Point2d> *x, bool need_translation)
{
	for (cv::Point2d &p : *x)
	{
		cv::Matx21d v;
		v(0) = p.x;
		v(1) = p.y;
		v = scale_rotation * v;
		if (need_translation)
			v += translation;
		p.x = v(0);
		p.y = v(1);
	}
}

Transform Procrustes(const vector<cv::Point2d> &x, const vector<cv::Point2d> &y)
{
	assert(x.size() == y.size());
	int landmark_count = x.size();
	double X1 = 0, X2 = 0, Y1 = 0, Y2 = 0, Z = 0, W = landmark_count;
	double C1 = 0, C2 = 0;

	for (int i = 0; i < landmark_count; ++i)
	{
		X1 += x[i].x;
		X2 += y[i].x;
		Y1 += x[i].y;
		Y2 += y[i].y;
		Z += Sqr(y[i].x) + Sqr(y[i].y);
		C1 += x[i].x * y[i].x + x[i].y * y[i].y;
		C2 += x[i].y * y[i].x - x[i].x * y[i].y;
	}

	cv::Matx44d A(X2, -Y2, W, 0,
		Y2, X2, 0, W,
		Z, 0, X2, Y2,
		0, Z, -Y2, X2);
	cv::Matx41d b(X1, Y1, C1, C2);
	cv::Matx41d solution = A.inv() * b;

	Transform result;
	result.scale_rotation(0, 0) = solution(0);
	result.scale_rotation(0, 1) = -solution(1);
	result.scale_rotation(1, 0) = solution(1);
	result.scale_rotation(1, 1) = solution(0);
	result.translation(0) = solution(2);
	result.translation(1) = solution(3);
	return result;
}

static void Normalize(vector<cv::Point2d> * shape, const TrainingParameters &tp)
{

	cv::Point2d center;
	for (const cv::Point2d p : *shape)
		center += p;
	center *= 1.0 / shape->size();
	for (cv::Point2d &p : *shape)
		p -= center;

	cv::Point2d left_eye = (*shape).at(tp.left_eye_index);
	cv::Point2d right_eye = (*shape).at(tp.right_eye_index);
	double eyes_distance = cv::norm(left_eye - right_eye);
	double scale = 1 / eyes_distance;

	double theta = -atan((right_eye.y - left_eye.y) / (right_eye.x - left_eye.x));

	// Must do translation first, and then rotation.
	// Therefore, translation is done separately
	Transform t;
	t.scale_rotation(0, 0) = scale * cos(theta);
	t.scale_rotation(0, 1) = -scale * sin(theta);
	t.scale_rotation(1, 0) = scale * sin(theta);
	t.scale_rotation(1, 1) = scale * cos(theta);


	t.Apply(shape, false);
}

vector<cv::Point2d> MeanShape(vector<vector<cv::Point2d>> shapes, 
	const TrainingParameters &tp)
{
	const int kIterationCount = 10;
	vector<cv::Point2d> mean_shape = shapes[0];
	
	for (int i = 0; i < kIterationCount; ++i)
	{
		for (vector<cv::Point2d> & shape: shapes)
		{
			Transform t = Procrustes(mean_shape, shape);
			t.Apply(&shape);
		}
		
		for (cv::Point2d & p : mean_shape)
			p.x = p.y = 0;
		
		for (const vector<cv::Point2d> & shape : shapes)
			for (int j = 0; j < mean_shape.size(); ++j)
			{
				mean_shape[j].x += shape[j].x;
				mean_shape[j].y += shape[j].y;
			}

		for (cv::Point2d & p : mean_shape)
			p *= 1.0 / shapes.size();

		Normalize(&mean_shape, tp);
	}

	return mean_shape;
}

vector<cv::Point2d> ShapeDifference(const vector<cv::Point2d> &s1,
	const vector<cv::Point2d> &s2)
{
	assert(s1.size() == s2.size());
	vector<cv::Point2d> result(s1.size());
	for (int i = 0; i < s1.size(); ++i)
		result[i] = s1[i] - s2[i];
	return result;
}

vector<cv::Point2d> ShapeAdjustment(const vector<cv::Point2d> &shape,
	const vector<cv::Point2d> &offset)
{
	assert(shape.size() == offset.size());
	vector<cv::Point2d> result(shape.size());
	for (int i = 0; i < shape.size(); ++i)
		result[i] = shape[i] + offset[i];
	return result;
}

double Covariance(double *x, double * y, const int size)
{
	double a = 0, b = 0, c = 0;
	for (int i = 0; i < size; ++i)
	{
		a += x[i];
		b += y[i];
		c += x[i] * y[i];
	}

	return c / size - (a / size) * (b / size);
}

/*
The function below has the same functionality as the one above. It uses 
AVX2 instructions to speed up computation. Use it if your processor is 
newer than or equal to Intel Haswell, and your compiler support AVX2.
*/

/*
#include<intrin.h>

double Covariance(double *x, double * y, const int size)
{
	const int kBlockSize = 4;
	const int kBlocksCount = size / kBlockSize;

	double *p = x, *q = y;
	__m256d sum1 = _mm256_setzero_pd();
	__m256d sum2 = _mm256_setzero_pd();
	__m256d sum = _mm256_setzero_pd();
	__m256d load1, load2;
	for (int i = 0; i < kBlocksCount ; ++i, p += kBlockSize, q += kBlockSize)
	{
		load1 = _mm256_load_pd(p);
		load2 = _mm256_load_pd(q);
		sum1 = _mm256_add_pd(sum1, load1);
		sum2 = _mm256_add_pd(sum2, load2);
		sum = _mm256_fmadd_pd(load1, load2, sum);
	}
	p = reinterpret_cast<double*>(&sum1);
	double mean_x = (p[0] + p[1] + p[2] + p[3]) / size;

	q = reinterpret_cast<double*>(&sum2);
	double mean_y = (q[0] + q[1] + q[2] + q[3]) / size;

	p = reinterpret_cast<double*>(&sum);
	double result = (p[0] + p[1] + p[2] + p[3]) / size;
	return result - mean_x * mean_y;
}*/

vector<cv::Point2d> MapShape(cv::Rect original_face_rect,
	const vector<cv::Point2d> original_landmarks, cv::Rect new_face_rect)
{
	vector<cv::Point2d> result;
	for (const cv::Point2d &landmark: original_landmarks)
	{
		result.push_back(landmark);
		result.back() -= cv::Point2d(original_face_rect.x, original_face_rect.y);
		result.back().x *= 
			static_cast<double>(new_face_rect.width) / original_face_rect.width;
		result.back().y *= 
			static_cast<double>(new_face_rect.height) / original_face_rect.height;
		result.back() += cv::Point2d(new_face_rect.x, new_face_rect.y);
	}
	return result;
}

vector<pair<int, double>> OMP(cv::Mat x, cv::Mat base, int coeff_count)
{
	vector<pair<int, double>> result;
	cv::Mat residual = x.clone();
	for (int i = 0; i < coeff_count; ++i)
	{
		int max_index = 0;
		double max_value = 0;
		for (int j = 0; j < base.cols; ++j)
		{
			double current_value = abs(
				static_cast<cv::Mat>(residual.t() * base.col(j)).at<double>(0));
			if (current_value > max_value)
			{
				max_value = current_value;
				max_index = j;
			}
		}

		result.push_back(make_pair(max_index, 0));
		cv::Mat sparse_base(base.rows, result.size(), CV_64FC1);
		for (int j = 0; j < result.size(); ++j)
			base.col(result[j].first).copyTo(sparse_base.col(j));

		cv::Mat beta;
		cv::solve(sparse_base.t() * sparse_base, 
			sparse_base.t() * x, beta, cv::DECOMP_SVD);
		for (int j = 0; j < result.size(); ++j)
			result[j].second = beta.at<double>(j);
		residual -= sparse_base * beta;
	}

	return result;
}

string TrimStr(const string &s, const string &space)
{
	string result = s;
	result = result.erase(0, result.find_first_not_of(space));
	return result.erase(result.find_last_not_of(space) + 1);
}
