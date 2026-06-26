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
#include "rand.h"
#include "constant.h"

namespace Rotation {
    at::Tensor normalize_angle(at::Tensor x);

    at::Tensor euler_to_quaternion(at::Tensor euler);

    at::Tensor quaternion_to_matrix(at::Tensor qt);

    at::Tensor rot_vec_to_matrix(at::Tensor axis, at::Tensor theta);

    at::Tensor matrix_from_rot_vec(at::Tensor axis, at::Tensor theta);

    at::Tensor rot_vec_to_quaternion(at::Tensor axis, at::Tensor theta);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> euler_to_matrix(at::Tensor euler);

    at::Tensor matrix_from_euler(at::Tensor euler);

    at::Tensor quaternion_to_euler(at::Tensor qt);

    at::Tensor quaternion_increment(at::Tensor qt1, at::Tensor qt2);

    ////////////////////////////////////////////////////////////////////////////

    at::Tensor batched_euler_to_quaternion(at::Tensor euler);

    at::Tensor batched_quaternion_to_matrix(at::Tensor qt);

    at::Tensor batched_rot_vec_to_matrix(at::Tensor axis, at::Tensor theta);

    at::Tensor batched_matrix_from_rot_vec(at::Tensor axis, at::Tensor theta);

    at::Tensor batched_rot_vec_to_quaternion(at::Tensor axis, at::Tensor theta);

    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor> batched_euler_to_matrix(at::Tensor euler);

    at::Tensor batched_matrix_from_euler(at::Tensor euler);

    at::Tensor batched_quaternion_to_euler(at::Tensor qt);

    at::Tensor batched_quaternion_increment(at::Tensor qt1, at::Tensor qt2);
}