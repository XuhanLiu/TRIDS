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

#include <ctime> // for time (for seeding)
#include <random>
#include <assert.h>
#include "rand.h"
#include "constant.h"

#if defined(_WIN32)
#include <Windows.h>
#include <process.h>
#else
#include <unistd.h>
#endif

using namespace std;

float randf(float a, float b, RNG& rng) {
	assert(a < b);
	uniform_real_distribution<> distr(a, b);
	float tmp = distr(rng);
	assert(tmp >= a);
	assert(tmp <= b);
	return tmp;
}

float randn(float mean, float sigma, RNG& rng) {
	assert(sigma >= 0);
	normal_distribution<> distr(mean, sigma);
	return distr(rng);
}

int randi(int a, int b, RNG& rng) {
	assert(a <= b);
	uniform_int_distribution<> distr(a, b);
	int tmp = distr(rng);
	assert(tmp >= a);
	assert(tmp <= b);
	return tmp;
}

size_t randsz(size_t a, size_t b, RNG& rng) {
	assert(a <= b);
	assert(int(a) >= 0);
	assert(int(b) >= 0);
	int i = randi(int(a), int(b), rng);
	assert(i >= 0);
	assert(i >= int(a));
	assert(i <= int(b));
	return static_cast<size_t>(i);
}


at::Tensor rand_sphere(RNG& rng, torch::TensorOptions option) {
	while (true) { // on average, this will have to be run about twice
		float r1 = randf(-1, 1, rng);
		float r2 = randf(-1, 1, rng);
		float r3 = randf(-1, 1, rng);

		auto tmp = torch::tensor({ r1, r2, r3 }, option);
		if (tmp.norm().item<float>() < 1)
			return tmp;
	}
}

// expects corner1[i] < corner2[i]
at::Tensor rand_inbox(torch::Tensor& corner1, torch::Tensor& corner2, RNG& rng) {
	at::Tensor tmp = torch::empty({ 3 });
	for (auto i = 0; (i) < tmp.size(0); i++) {
		tmp[i] = randf(corner1[i].item<float>(), corner2[i].item<float>(), rng);
	}
	return tmp.to(corner1.options());
}


int my_pid() {
#if defined(_WIN32)
	return GetCurrentProcessId();
#else
	return getpid();
#endif
}

int auto_seed() { // make seed from PID and time
	return my_pid() * int(time(nullptr));
}