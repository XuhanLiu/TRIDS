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

#include <torch/torch.h>
#include <ATen/cuda/CUDAEvent.h>
#include <c10/cuda/CUDAStream.h>
#include "conformer.h"
#include "rand.h"
#include "rotation.h"


#define THREADS 256
#define BLOCKS(B) (B + THREADS - 1) / THREADS

using namespace torch::autograd;
namespace F = torch::nn::functional;

class Sampling : public Function<Sampling> {

public:
    static at::Tensor forward(AutogradContext* ctx, at::Tensor trans, at::Tensor rots, at::Tensor tors,
                              Conformer* conformer);

    static tensor_list backward(AutogradContext* ctx, tensor_list grad_outputs);
};
