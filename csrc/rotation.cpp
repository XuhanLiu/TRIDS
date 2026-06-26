/*
 Copyright (c) 2024-2026 @  Shenzhen Bay Laboratory &
							Peking University &
							Changping Laboratory &
							XtalPi Technologies Co., Ltd

 This code is a part of TRIDS:
 The unified molecular docking framework integrated with deep learning-based site
	binding, sampling and scoring.

 TRIDS is open-source software for molecular docking with PyTorch-based DL models:
 (https://www.github.cn/xuhanliu/trids)

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

 http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.

 Author:		Dr. Xuhan Liu
 Email:			xuhanliu@qq.com
*/

#include "rotation.h"


namespace F = torch::nn::functional;


at::Tensor Rotation::normalize_angle(at::Tensor x) {
	x = torch::fmod(x, 2 * PI);
	x = torch::where(x >= PI, x - 2 * PI, x);
	x = torch::where(x < -PI, x + 2 * PI, x);
	return x;
}


at::Tensor Rotation::euler_to_quaternion(at::Tensor euler) {
	auto half = euler * 0.5;
	auto c = half.cos();
	auto s = half.sin();
	auto w = c[2] * c[1] * c[0] + s[2] * s[1] * s[0];
	auto x = c[2] * c[1] * s[0] - s[2] * s[1] * c[0];
	auto y = s[2] * c[1] * s[0] + c[2] * s[1] * c[0];
	auto z = s[2] * c[1] * c[0] - c[2] * s[1] * s[0];
	return torch::stack({ w, x, y, z });
}

at::Tensor Rotation::quaternion_to_euler(at::Tensor qt) {
	auto w = qt[0];
	auto x = qt[1];
	auto y = qt[2];
	auto z = qt[3];
	auto r_ = torch::atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y));
	auto p_ = torch::asin(2 * (w * y - z * x));
	auto y_ = torch::atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
	return torch::stack({ r_, p_, y_ });
}

at::Tensor Rotation::quaternion_to_matrix(at::Tensor qt) {
	auto w = qt[0];
	auto x = qt[1];
	auto y = qt[2];
	auto z = qt[3];
	auto w2 = w * w;
	auto x2 = x * x;
	auto y2 = y * y;
	auto z2 = z * z;
	auto xy = x * y;
	auto xz = x * z;
	auto yz = y * z;
	auto wx = w * x;
	auto wy = w * y;
	auto wz = w * z;

	auto R = torch::stack({
		w2 + x2 - y2 - z2, 2 * (xy - wz), 2 * (wy + xz),
		2 * (wz + xy), w2 - x2 + y2 - z2, 2 * (yz - wx),
		2 * (xz - wy), 2 * (wx + yz), w2 - x2 - y2 + z2 }).reshape({ 3, 3 });
		
	return R;
}

at::Tensor Rotation::rot_vec_to_matrix(at::Tensor axis, at::Tensor theta) {
	auto zero = torch::tensor(0, axis.options());
	auto eye = torch::eye(3, axis.options());
	auto cos = torch::cos(theta);
	auto sin = torch::sin(theta);
	auto x = axis[0];
	auto y = axis[1];
	auto z = axis[2];
	auto R_ = torch::stack({
		zero, 	-z,		y,
		z, 		zero, 	-x,
		-y, 	x, 		zero }).reshape({ 3, 3 });
	auto R = eye * cos + (1 - cos) * axis.unsqueeze(1).mm(axis.unsqueeze(0)) + R_ * sin;
	// std::cout << R << std::endl;
	return R;
}

at::Tensor Rotation::rot_vec_to_quaternion(at::Tensor axis, at::Tensor theta) {
	assert(axis.size(0) == 3);
	assert(axis.ndimension() == 1);
	assert(theta.numel() == 1);
	auto half = theta * 0.5;
	auto cos = torch::cos(half);
	auto sin = torch::sin(half);
	auto w = torch::tensor({ 1 }, axis.options());
	auto qt = torch::cat({ w * cos, axis * sin });
	return qt;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> Rotation::euler_to_matrix(at::Tensor euler) {
	auto zero = torch::tensor(0, euler.options());
	auto one = torch::tensor(1, euler.options());
	auto cos = euler.cos();
	auto sin = euler.sin();
	auto cos_r = cos[0];
	auto sin_r = sin[0];
	auto cos_p = cos[1];
	auto sin_p = sin[1];
	auto cos_y = cos[2];
	auto sin_y = sin[2];
	auto Rx = torch::stack({
		one,	zero,	zero,
		zero,	cos_r,	-sin_r,
		zero,	sin_r,	cos_r
						}).reshape({ 3, 3 });
	auto Ry = torch::stack({
		cos_p, 	zero,	sin_p,
		zero,	one,	zero,
		-sin_p,	zero,	cos_p
						}).reshape({ 3, 3 });
	auto Rz = torch::stack({
		cos_y,	-sin_y,	zero,
		sin_y,	cos_y,	zero,
		zero,	zero,	one
						}).reshape({ 3, 3 });
	return std::make_tuple(Rx, Ry, Rz);
}

// still has some problems for rotation
at::Tensor Rotation::matrix_from_euler(at::Tensor euler) {
	auto cos = euler.cos();
	auto sin = euler.sin();
	auto cos_r = cos[0];
	auto sin_r = sin[0];
	auto cos_p = cos[1];
	auto sin_p = sin[1];
	auto cos_y = cos[2];
	auto sin_y = sin[2];
	auto R = torch::stack({
		cos_y * cos_p, 	-cos_r * sin_y + cos_y * sin_p * sin_r,		cos_y * cos_r * sin_p + sin_y * sin_r,
		cos_p * sin_y, 	cos_y * cos_r + sin_y * sin_p * sin_r,		cos_r * sin_y * sin_p - cos_y * sin_r,
		-sin_p, 		cos_p * sin_r,								cos_p * cos_r,
						}).reshape({ 3,3 });

	return R;
}


at::Tensor Rotation::quaternion_increment(at::Tensor qt1, at::Tensor qt2) {
	auto qt_ = torch::stack({
		qt1[0] * qt2[0] - qt1[1] * qt2[1] - qt1[2] * qt2[2] - qt1[3] * qt2[3],
		qt1[0] * qt2[1] + qt1[1] * qt2[0] + qt1[2] * qt2[3] - qt1[3] * qt2[2],
		qt1[0] * qt2[2] - qt1[1] * qt2[3] + qt1[2] * qt2[0] + qt1[3] * qt2[1],
		qt1[0] * qt2[3] + qt1[1] * qt2[2] - qt1[2] * qt2[1] + qt1[3] * qt2[0] });
	auto qt = F::normalize(qt_, F::NormalizeFuncOptions().dim(0));
	return qt;
}

at::Tensor Rotation::matrix_from_rot_vec(at::Tensor axis, at::Tensor theta) {
	auto sin = torch::sin(theta);
	auto cos = torch::cos(theta);
	auto cos_1 = 1. - cos;
	auto x = axis[0];
	auto y = axis[1];
	auto z = axis[2];
	auto R = torch::stack({
		cos_1 * x * x + cos,			cos_1 * x * y - sin * z,		cos_1 * x * z + sin * y,
		cos_1 * x * y + sin * z,		cos_1 * y * y + cos,			cos_1 * y * z - sin * x,
		cos_1 * x * z - sin * y,		cos_1 * y * z + sin * x,		cos_1 * z * z + cos
						}).reshape({ 3, 3 });
	return R;
}

///////////////////////////////////////////////////////////////////////////////////////////

at::Tensor Rotation::batched_euler_to_quaternion(at::Tensor euler) {
	auto half = euler * 0.5;
	auto c = half.cos();
	auto s = half.sin();
	auto c0 = c.select(1, 0);
	auto c1 = c.select(1, 1);
	auto c2 = c.select(1, 2);
	auto s0 = s.select(1, 0);
	auto s1 = s.select(1, 1);
	auto s2 = s.select(1, 2);
	auto w = c2 * c1 * c0 + s2 * s1 * s0;
	auto x = c2 * c1 * s0 - s2 * s1 * c0;
	auto y = s2 * c1 * s0 + c2 * s1 * c0;
	auto z = s2 * c1 * c0 - c2 * s1 * s0;
	return torch::stack({ w, x, y, z }, 1);
}

at::Tensor Rotation::batched_quaternion_to_euler(at::Tensor qt) {
	auto w = qt.select(1, 0);
	auto x = qt.select(1, 1);
	auto y = qt.select(1, 2);
	auto z = qt.select(1, 3);
	auto r_ = torch::atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y));
	auto p_ = torch::asin(2 * (w * y - z * x));
	auto y_ = torch::atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
	return torch::stack({ r_, p_, y_ }, 1);
}

at::Tensor Rotation::batched_quaternion_to_matrix(at::Tensor qt) {
	auto w = qt.select(1, 0);
	auto x = qt.select(1, 1);
	auto y = qt.select(1, 2);
	auto z = qt.select(1, 3);
	auto w2 = w * w;
	auto x2 = x * x;
	auto y2 = y * y;
	auto z2 = z * z;
	auto xy = x * y;
	auto xz = x * z;
	auto yz = y * z;
	auto wx = w * x;
	auto wy = w * y;
	auto wz = w * z;
	auto R = torch::stack({
		w2 + x2 - y2 - z2, 2 * (xy - wz), 2 * (wy + xz),
		2 * (wz + xy), w2 - x2 + y2 - z2, 2 * (yz - wx),
		2 * (xz - wy), 2 * (wx + yz), w2 - x2 - y2 + z2 }, 1).reshape({ -1, 3, 3 });
	return R.reshape({ -1, 3, 3 });
}

at::Tensor Rotation::batched_rot_vec_to_matrix(at::Tensor axis, at::Tensor theta) {
	auto batch_size = axis.size(0);
	auto axis2 = axis.unsqueeze(-1).bmm(axis.unsqueeze(1));
	auto zero = torch::zeros(batch_size, axis.options());
	auto R_ = torch::stack({
		zero, 				-axis.select(1, 2), axis.select(1, 1),
		axis.select(1, 2), 	zero, 				-axis.select(1, 0),
		-axis.select(1, 1), axis.select(1, 0), 	zero }, 1
	).reshape({ -1, 3, 3 });
	// std::cout << R_ << std::endl;
	auto cos = torch::cos(theta).reshape({ -1, 1, 1 });
	auto sin = torch::sin(theta).reshape({ -1, 1, 1 });
	auto eye = torch::eye(3, axis.options()).unsqueeze(0).expand({ batch_size, 3, 3 });
	auto R = eye * cos + R_ * sin + (1 - cos) * axis2;
	return R;
}

at::Tensor Rotation::batched_rot_vec_to_quaternion(at::Tensor axis, at::Tensor theta) {
	auto batch_size = axis.size(0);
	assert(axis.size(1) == 3);
	assert(axis.ndimension() == 2);
	assert(theta.numel() == batch_size);

	auto half = theta.unsqueeze(1) * 0.5;
	auto cos = torch::cos(half);
	auto sin = torch::sin(half);
	auto qt = torch::cat({ cos, axis * sin }, 1);
	return qt;
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> Rotation::batched_euler_to_matrix(at::Tensor euler) {
	auto batch_size = euler.size(0);
	auto zero = torch::zeros(batch_size, euler.options());
	auto one = torch::ones(batch_size, euler.options());
	auto cos = euler.cos();
	auto sin = euler.sin();
	auto cos_r = cos.select(1, 0);
	auto sin_r = sin.select(1, 0);
	auto cos_p = cos.select(1, 1);
	auto sin_p = sin.select(1, 1);
	auto cos_y = cos.select(1, 2);
	auto sin_y = sin.select(1, 2);
	auto Rx = torch::stack({
		one,	zero,	zero,
		zero,	cos_r,	-sin_r,
		zero,	sin_r,	cos_r }, 1).reshape({ -1, 3, 3 });
	auto Ry = torch::stack({
		cos_p, 	zero,	sin_p,
		zero,		one,	zero,
		-sin_p,	zero,	cos_p }, 1).reshape({ -1, 3, 3 });
	auto Rz = torch::stack({
		cos_y,	-sin_y,	zero,
		sin_y,	cos_y,	zero,
		zero,	zero,	one }, 1).reshape({ -1, 3, 3 });
	return std::make_tuple(Rx, Ry, Rz);
}


at::Tensor Rotation::batched_matrix_from_euler(at::Tensor euler) {
	auto cos = euler.cos();
	auto sin = euler.sin();
	auto cos_r = cos.select(1, 0);
	auto sin_r = sin.select(1, 0);
	auto cos_p = cos.select(1, 1);
	auto sin_p = sin.select(1, 1);
	auto cos_y = cos.select(1, 2);
	auto sin_y = sin.select(1, 2);
	auto R = torch::stack({
		cos_y * cos_p, 	-cos_r * sin_y + cos_y * sin_p * sin_r,		cos_y * cos_r * sin_p + sin_y * sin_r,
		cos_p * sin_y, 	cos_y * cos_r + sin_y * sin_p * sin_r,		cos_r * sin_y * sin_p - cos_y * sin_r,
		-sin_p, 			cos_p * sin_r,								cos_p * cos_r }, 1
	).reshape({ -1, 3, 3 });
	return R;
}


at::Tensor Rotation::batched_quaternion_increment(at::Tensor qt1, at::Tensor qt2) {
	auto w1 = qt1.select(1, 0);
	auto x1 = qt1.select(1, 1);
	auto y1 = qt1.select(1, 2);
	auto z1 = qt1.select(1, 3);
	auto w2 = qt2.select(1, 0);
	auto x2 = qt2.select(1, 1);
	auto y2 = qt2.select(1, 2);
	auto z2 = qt2.select(1, 3);
	auto qt_ = torch::stack({
		w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
		w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
		w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
		w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2 }, 1);
	auto qt = F::normalize(qt_, F::NormalizeFuncOptions().dim(1));
	return qt;
}

at::Tensor Rotation::batched_matrix_from_rot_vec(at::Tensor axis, at::Tensor theta) {
	auto sin = torch::sin(theta);
	auto cos = torch::cos(theta);
	auto cos_1 = 1. - cos;
	auto x = axis.select(1, 0);
	auto y = axis.select(1, 1);
	auto z = axis.select(1, 2);
	auto x2 = x * x;
	auto y2 = y * y;
	auto z2 = z * z;
	auto xy = x * y;
	auto xz = x * z;
	auto yz = y * z;
	auto R = torch::stack({
		cos_1 * x2 + cos,			cos_1 * xy - sin * z,		cos_1 * xz + sin * y,
		cos_1 * xy + sin * z,		cos_1 * y2 + cos,			cos_1 * yz - sin * x,
		cos_1 * xz - sin * y,		cos_1 * yz + sin * x,		cos_1 * z2 + cos
						}, 1).reshape({ -1, 3, 3 });
	return R;
}