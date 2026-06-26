#pragma once

#include <torch/torch.h>
#include <torch/cuda.h>
#include <c10/cuda/CUDAStream.h>
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

#include <memory>
#include <c10/cuda/CUDAGuard.h>
#include <openbabel/mol.h>
#include "conformer.h"
#include "result.h"

class Task {
	std::optional<at::cuda::CUDAStream> stream;

	std::unique_ptr<at::cuda::CUDAStreamGuard> guard;

public:
	Dof conf;

	Result result;

	Task() = default;

	Task(Dof conf);

	void init(torch::Device device);

	void synchronize();

};
