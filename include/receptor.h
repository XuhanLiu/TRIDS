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
#include <openbabel/mol.h>
#include "graph.h"
#include "grid.cuh"
#include "constant.h"

class Receptor {
public:
    at::TensorOptions options;
    Graph graph;
    // index of residues each atom belongs to
    at::Tensor idx;
    // Begin and end index of each residues in the atom list
    at::Tensor ptr;
    at::Tensor vdw;
    at::Tensor hydro;
    at::Tensor ha;
    at::Tensor hd;
    at::Tensor atomic;
    at::Tensor pos;
    at::Tensor box_max;
    at::Tensor box_min;
    std::unique_ptr<Grid> grid;
    int num_res = 0;
    int num_atoms = 0;

    Receptor() = default;

    ~Receptor() = default;

    Receptor(OpenBabel::OBMol* rec, at::Tensor pos, bool is_vina = false, float cutoff = TRIDS_POCKET_CUTOFF, at::TensorOptions options_ = {});

    void to(at::TensorOptions options_);
};