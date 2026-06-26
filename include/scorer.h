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
#include <memory>
#include "graph.h"
#include "grid.cuh"
#include "constant.h"
#include "mdn.h"
#include "receptor.h"

class Scorer {
public:
    torch::TensorOptions options;
    at::Tensor inner_nbr;
    Receptor* rec;
    float cutoff;
    float cutoff2;
    float beta;
    int num_tors;

    Scorer(torch::TensorOptions options_ = {}, float cutoff_ = 8, float beta_ = 1)
        : cutoff(cutoff_), cutoff2(cutoff_ * cutoff_), options(options_), beta(beta_) {
    }

    Scorer() = default;
    ~Scorer() = default;

    virtual at::Tensor get_neighbors(OpenBabel::OBMol* lig);

    virtual at::Tensor scoring(at::Tensor lig_pos) = 0;

    virtual void set_ligand(OpenBabel::OBMol* ligand);

    virtual void set_pocket(Receptor& receptor);

    virtual void make_grid() = 0;

    virtual void profiling();

    virtual at::Tensor batched_auto_scoring(at::Tensor lig_pos) = 0;

    virtual at::Tensor batched_accurate_scoring(at::Tensor lig_pos) = 0;

    virtual at::Tensor batched_fast_scoring(at::Tensor lig_pos) = 0;

    virtual at::Tensor loss(at::Tensor lig_pos) = 0;

    virtual void to(torch::TensorOptions options) = 0;
};

class TriScorer : public Scorer {
    // num_atom, 4, grid, grid, grid;
    at::Tensor forcefield;
    Graph lig_graph;
    TriScore model = nullptr;
    
public:
    at::Tensor log_rho;
    at::Tensor mu;
    at::Tensor var;
    at::Tensor log_scale;
    at::Tensor data;
    
    at::Tensor lig_vdw;
    at::Tensor lig_ha;
    at::Tensor lig_hd;
    at::Tensor lig_hydro;
    

    TriScorer() = default;
    
    TriScorer(TriScore model_, torch::TensorOptions options_ = {}, 
              float cutoff_ = 10, float beta_ = 0.69314718 / 10): 
        Scorer(options_, cutoff_, beta_), model(model_) {
    }

    virtual void make_grid();

    virtual at::Tensor interpolate(at::Tensor lig_pos);

    virtual at::Tensor scoring(at::Tensor lig_pos);
    
    virtual void set_ligand(OpenBabel::OBMol* ligand);

    virtual void profiling();
    
    virtual at::Tensor batched_auto_scoring(at::Tensor lig_pos);

    virtual at::Tensor batched_accurate_scoring(at::Tensor lig_pos);

    virtual at::Tensor batched_fast_scoring(at::Tensor lig_pos);

    virtual at::Tensor loss(at::Tensor lig_pos);

    virtual void to(torch::TensorOptions options);
};