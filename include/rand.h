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
#pragma once

#include <random>
#include <torch/torch.h>


using namespace std;

typedef mt19937 RNG;

// expects a < b, returns rand in [a, b]
float randf(float a, float b, RNG& rng);

// expects sigma >= 0
float randn(float mean, float sigma, RNG& rng);

// expects a <= b, returns rand in [a, b]
int randi(int a, int b, RNG& rng);

// expects a <= b, returns rand in [a, b]
size_t randsz(size_t a, size_t b, RNG& rng);

// returns a random Crd inside the sphere centered at 0 with radius 1
at::Tensor rand_sphere(RNG& rng, torch::TensorOptions option = {});

// expects corner1[i] < corner2[i]
at::Tensor rand_inbox(const torch::Tensor& corner1, const torch::Tensor& corner2, RNG& rng);

// make seed from PID and time
int auto_seed();


int my_pid();
