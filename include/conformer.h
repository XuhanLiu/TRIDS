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
#include <openbabel/bond.h>
#include <openbabel/mol.h>
#include <openbabel/builder.h>
#include "rand.h"


class Dof {
public:
    // {x, y, z}, shape: { 3 }
    at::Tensor trans;
    // should not be quaternion number but euler angles: {r, p, y}
    at::Tensor rots;
    // {tor_1, tor_2, ..., tor_n}, shape: { n }
    at::Tensor tors;

    Dof() = default;

    void to(torch::TensorOptions options);

    void require_grad(bool flag);

    void detach();

    void clear();

    Dof clone();

    void copy(Dof origin);

    void init(at::Tensor box_min, at::Tensor box_max, unsigned num_tors);

    void mutate(at::Tensor box_min, at::Tensor box_max, float amplitude, RNG& rng);

    void mutate_trans(at::Tensor box_min, at::Tensor box_max, float amplitude, RNG& rng);

    void mutate_rots(float amplitude, RNG& rng);

    void mutate_tors(int which, RNG& rng);

};


class Dofs : public Dof {
public:
    // trans shape: {batch_size, 1, 3}
    // rots euler angles {r, p, y}, shape: {batch_size, 3}
    // rots shape: {batch_size, num_tors}
    unsigned batch_size = 0;

    Dofs() = default;

    void init(at::Tensor box_min, at::Tensor box_max, unsigned batch_size_, unsigned num_tors);

    void mutate(at::Tensor box_min, at::Tensor box_max, RNG& rng);

    Dofs clone();

};

// todo:: inherit from torch::CustomClassHolder, and then register to libtorch
class Conformer {
public:
    OpenBabel::OBMol* lig = nullptr;
    at::Tensor lig_pos;
    at::Tensor init_pos;
    at::Tensor pos;

    int pivot = -1;

    // num_tors * 2
    std::vector<int64_t> tor_bonds;

    std::vector<torch::Tensor> tor_masks;

    // float amplitude = 2;
    torch::TensorOptions options;

    Conformer() = default;

    Conformer(OpenBabel::OBMol* lig, int num_tasks, torch::TensorOptions options_);

    inline float gyration_radius();

    void get_torsions(OpenBabel::OBMol* mol, const std::optional<std::vector<int> >& norotate = std::nullopt);

    unsigned num_tors();

    void update_coord(Dof& conf, int i);

    void batched_auto_udpate_coord(Dofs& dofs);

    void batched_cuda_update_coord(Dofs& dofs);

    void to(torch::TensorOptions options_);
};