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
#include <torch/optim.h>
#include <torch/cuda.h>
#include "conformer.h"
#include "rand.h"
#include "result.h"
#include "scorer.h"
#include "rotation.h"
#include <cuda_runtime.h>


class Engine {
public:
    std::shared_ptr<Scorer> scorer;
    torch::TensorOptions options;
    int num_tasks;
    int max_depth;
    float topn;
    at::Tensor curr_score;

    Engine() = default;

    Engine(std::shared_ptr<Scorer> scorer_, int num_tasks_ = 512, 
           int max_depth_ = 32, int topn_ = 10, torch::TensorOptions options_ = {})
        : scorer(scorer_), num_tasks(num_tasks_), max_depth(max_depth_), topn(topn_), options(options_) {
        curr_score = torch::empty({ num_tasks, 2 }, options);
    }

    void refine_structures(Conformer& conformer, Dofs& dofs, torch::optim::Optimizer& optimizer);

    at::Tensor metropolis_accept(at::Tensor curr_score, at::Tensor prev_score,
                                 float beta, RNG& rng);

    void add_result_to_accepted_list(list<Result>& results, Result& result);

    at::Tensor scoring(at::Tensor lig_pos);

    std::tuple<torch::Tensor, torch::Tensor> docking(Conformer& conformer, RNG& rng);
};
