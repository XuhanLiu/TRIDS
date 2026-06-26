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
#include "scorer.h"
#include "constant.h"

class VinaScorer : public Scorer {

public:
    constexpr static float border_strength = 100.;
    at::Tensor inner_hydro;
    at::Tensor inner_hbond;
    at::Tensor inter_hydro;
    at::Tensor inter_hbond;
    at::Tensor inner_vdw;
    at::Tensor inter_vdw;

    at::Tensor lig_vdw;
    at::Tensor lig_hd;
    at::Tensor lig_ha;
    at::Tensor lig_hydro;
    at::Tensor lig_atom;

public:
    VinaScorer() = default;
    
    VinaScorer(torch::TensorOptions options_ = {}, float cutoff_ = 8, float beta_ = 0.838718)
        : Scorer(options_, cutoff_, beta_) {
    }

    virtual ~VinaScorer() = default;

    at::Tensor calc_energy(at::Tensor dist, at::Tensor hydro_mask, at::Tensor hbond_mask);

    virtual void set_ligand(OpenBabel::OBMol* ligand);

    virtual void make_grid();

    virtual void profiling();

    virtual at::Tensor scoring(at::Tensor lig_pos);

    virtual at::Tensor batched_auto_scoring(at::Tensor lig_pos);

    virtual at::Tensor batched_accurate_scoring(at::Tensor lig_pos);

    virtual at::Tensor batched_fast_scoring(at::Tensor lig_pos);

    virtual at::Tensor loss(at::Tensor lig_pos);

    virtual void to(torch::TensorOptions options);

};