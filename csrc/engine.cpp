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

#include <functional>
#include <ATen/cuda/CUDAGraph.h>
#include <c10/cuda/CUDAStream.h>
#include <spdlog/spdlog.h>
#include "engine.h"
#include "constant.h"
#include "sampling.cuh"
#include "rand.h"


void Engine::refine_structures(Conformer& conformer, Dofs& dofs, torch::optim::Optimizer& optimizer) {
    dofs.require_grad(true);
    optimizer.zero_grad();
    // auto closure = [&]() -> at::Tensor {
    //     conformer.batched_cuda_update_coord(dofs);
    //     curr_score = scorer->batched_accurate_scoring(conformer.pos);
    //     // auto auto_score = scorer->batched_auto_scoring(coords);
    //     auto total_score = curr_score.select(1, 1).sum();
    //     total_score.backward();
    //     curr_score.detach_();
    //     return total_score;
    // };

    // This section is test code for comparsion auto and manual calculation of gradients 
    conformer.batched_cuda_update_coord(dofs);
    curr_score = scorer->batched_accurate_scoring(conformer.pos);
    curr_score.select(1, 1).sum().backward();
    optimizer.step();

    // Dofs dof1 = dofs.clone();
    // dof1.trans = dof1.trans.detach();
    // dof1.rots = dof1.rots.detach();
    // dof1.tors = dof1.tors.detach();
    // dof1.require_grad(true);
    // auto coord1 = conformer->batched_auto_udpate_coord(dof1);
    // // coord1 = coord1.detach();
    // // coord1.requires_grad_(true);
    // auto curr_score1 = scorer->batched_auto_scoring(coord1);
    // curr_score1.sum().backward();

    // auto grad_crd1 = coord1.grad();
    // auto grad_crds = coords.grad();
    // auto equal_crd = torch::abs(grad_crds - grad_crd1) > 0.01;
    // std::cout << torch::where(equal_crd) << std::endl;
    // std::cout << torch::allclose(grad_crds, grad_crd1, 0.1, 0.01) << std::endl;

    // auto grad_tran1 = dof1.trans.grad();
    // auto grad_trans = dofs.trans.grad();
    // auto equal_tran = torch::abs(grad_trans - grad_tran1) > 0.01;
    // std::cout << torch::where(equal_tran) << std::endl;
    // std::cout << torch::allclose(grad_trans, grad_tran1, 0.1, 0.01) << std::endl;

    // auto grad_rot1 = dof1.rots.grad();
    // auto grad_rots = dofs.rots.grad();
    // auto equal_rot = torch::abs(grad_rots - grad_rot1) > 0.01;
    // std::cout << torch::where(equal_rot) << std::endl;
    // std::cout << torch::allclose(grad_rots, grad_rot1, 0.1, 0.01) << std::endl;

    // auto grad_tor1 = dof1.tors.grad();
    // auto grad_tors = dofs.tors.grad();
    // auto equal_tors = torch::abs(grad_tors - grad_tor1) > 0.01;
    // std::cout << torch::where(equal_tors) << std::endl;
    // std::cout << torch::allclose(grad_tors, grad_tor1, 0.1, 0.01) << std::endl;

    // optimizer.step(closure);
    
    Rotation::normalize_angle(dofs.rots);
    Rotation::normalize_angle(dofs.tors);
    dofs.require_grad(false);
}

at::Tensor Engine::metropolis_accept(at::Tensor curr_score, at::Tensor prev_score,
                                      float beta, RNG& rng) {
    auto acceptance_probability = ((prev_score - curr_score) * beta).exp();
    auto mask_rejected = acceptance_probability < torch::rand({ num_tasks }, options);
    // std::cout << acceptance_probability << std::endl;
    // std::cout << mask_rejected << std::endl;
    return mask_rejected;
}

void Engine::add_result_to_accepted_list(list<Result>& results, Result& result) {
    auto score = result.getScore();
    for (auto it = results.begin(); it != results.end(); it++) {
        auto& result_i = (*it);
        auto score_i = result_i.getScore();
        if (score <= score_i) {
            results.insert(it, result);
            break;
        }
    }

    if (results.size() < topn) {
        results.push_back(result);
        return;
    }

    if (results.size() > topn) {
        results.pop_back();
    }
}

at::Tensor Engine::scoring(at::Tensor lig_pos) {
    return scorer->scoring(lig_pos) / (1 + scorer->num_tors * TRIDS_OMEGA);
}

std::tuple<torch::Tensor, torch::Tensor> Engine::docking(Conformer& conformer, RNG& rng) {
    clock_t start = std::clock();
    float duration;

    auto best_score = at::full_like(curr_score, FLT_MAX);
    auto best_coord = conformer.pos.detach().clone();

    Dofs dofs;
    dofs.init(scorer->rec->box_min, scorer->rec->box_max, num_tasks, conformer.num_tors());
    auto option = torch::optim::AdamOptions().lr(0.1);
    auto optimizer = torch::optim::Adam({ dofs.trans, dofs.rots, dofs.tors }, option);

    auto run_metropolis_loop = [&](const std::function<void()>& refine_step) {
        Dofs prev_dofs;
        prev_dofs.copy(dofs);
        auto prev_score = curr_score.clone();
        for (auto depth = 0; depth < max_depth; depth++) {
            dofs.mutate(scorer->rec->box_min, scorer->rec->box_max, rng);

            for (auto i = 0; i < 50; i++) {
                refine_step();
            }
            curr_score.detach_();
            conformer.pos.detach_();
            auto update_ids = (curr_score.select(1, 1) < best_score.select(1, 1)).nonzero().squeeze(1);
            if (update_ids.size(0) > 0) {
                best_score.index_put_({ update_ids }, curr_score.index_select(0, update_ids));
                best_coord.index_put_({ update_ids }, conformer.pos.index_select(0, update_ids));
            }

            auto mask_rejected = metropolis_accept(curr_score.select(1, 1), prev_score.select(1, 1), scorer->beta, rng);
            auto reject_ids = mask_rejected.nonzero().squeeze(1);

            if (reject_ids.size(0) > 0) {
                curr_score.index_put_({ reject_ids }, prev_score.index_select(0, reject_ids));
                dofs.trans.index_put_({ reject_ids }, prev_dofs.trans.index_select(0, reject_ids));
                dofs.rots.index_put_({ reject_ids }, prev_dofs.rots.index_select(0, reject_ids));
                dofs.tors.index_put_({ reject_ids }, prev_dofs.tors.index_select(0, reject_ids));
            }
            if (depth < max_depth - 1) {
                prev_dofs.copy(dofs);
                prev_score.copy_(curr_score);
            }
        }
        optimizer.state().clear();
        conformer.pos.detach_();
    };

    if (options.device().is_cuda()) {
        spdlog::info("CUDA Graph warming up begin ...");

        static at::cuda::CUDAStream capture_stream = at::cuda::getStreamFromPool();
        at::cuda::CUDAStream default_stream = at::cuda::getCurrentCUDAStream();

        default_stream.synchronize();
        at::cuda::setCurrentCUDAStream(capture_stream);

        at::cuda::CUDAGraph graph;
        refine_structures(conformer, dofs, optimizer);
        duration = static_cast<float>(std::clock() - start) / CLOCKS_PER_SEC;
        spdlog::info("CUDA Graph warming up end ({:.4f}s)", duration);

        optimizer.zero_grad();
        start = std::clock();
        spdlog::info("CUDA Graph Capturing begin ...");
        graph.capture_begin();
        refine_structures(conformer, dofs, optimizer);
        graph.capture_end();
        duration = static_cast<float>(std::clock() - start) / CLOCKS_PER_SEC;
        spdlog::info("CUDA Graph capturing end ({:.4f}s)", duration);

        start = std::clock();
        spdlog::info("Sampling with constructed CUDA Graph begin ...");
        run_metropolis_loop([&]() { graph.replay(); });
        graph.reset();
        duration = static_cast<float>(std::clock() - start) / CLOCKS_PER_SEC;
        spdlog::info("Sampling with constructed CUDA Graph end ({:.4f}s)", duration);

        capture_stream.synchronize();
        at::cuda::setCurrentCUDAStream(default_stream);
    } else {
        spdlog::info("CPU sampling begin ...");
        refine_structures(conformer, dofs, optimizer);
        start = std::clock();
        run_metropolis_loop([&]() { refine_structures(conformer, dofs, optimizer); });
        duration = static_cast<float>(std::clock() - start) / CLOCKS_PER_SEC;
        spdlog::info("CPU sampling end ({:.4f}s)", duration);
    }

    return { best_coord.detach(), best_score.detach() / (1 + scorer->num_tors * TRIDS_OMEGA) };
}
