
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
#include "sampling.cuh"

namespace F = torch::nn::functional;
using namespace torch::indexing;


at::Tensor Sampling::forward(AutogradContext* ctx, at::Tensor trans, at::Tensor rots, at::Tensor tors,
                             Conformer* conformer) {
    conformer->pos.copy_(conformer->init_pos);
    for (auto i = 0; i < conformer->tor_masks.size(); i++) {
        auto u = conformer->tor_bonds[i * 2];
        auto v = conformer->tor_bonds[i * 2 + 1];
        
        auto pos_v = conformer->pos.select(1, v);
        auto rot_vec = conformer->pos.select(1, u) - pos_v;
        rot_vec = F::normalize(rot_vec, F::NormalizeFuncOptions().dim(-1));

        // std::cout << "tor mat begin ..." << std::endl;
        // std::cout << tors << std::endl;
        auto tor = tors.select(1, i);
        // std::cout << rot_vec << std::endl;
        auto tor_mat = Rotation::batched_matrix_from_rot_vec(rot_vec, tor);
        // std::cout << "tor mat end ..." << std::endl;

        // std::cout << "rotated pos begin ..." << std::endl;
        auto rotated_pos = conformer->pos.index_select(1, conformer->tor_masks[i]);
        // std::cout << "rotated pos end ..." << std::endl;

        pos_v = pos_v.unsqueeze(1);
        auto updated_pos = (rotated_pos - pos_v).bmm(tor_mat.permute({ 0, 2, 1 })) + pos_v;
        conformer->pos.index_put_({ torch::indexing::Slice(), conformer->tor_masks[i] }, updated_pos);
    }

    auto rot_mat = Rotation::batched_matrix_from_euler(rots);
    conformer->pos.copy_(conformer->pos.bmm(rot_mat.permute({ 0, 2, 1 })) + trans);

    ctx->saved_data["pos"] = conformer->pos;
    ctx->saved_data["rots"] = rots;
    ctx->saved_data["tors"] = tors;
    ctx->saved_data["pivot"] = conformer->pivot;
    ctx->saved_data["tor_bonds"] = at::IntArrayRef(conformer->tor_bonds.data(), conformer->tor_bonds.size());
    ctx->saved_data["tor_mask"] = conformer->tor_masks;
    return conformer->pos;
}

tensor_list Sampling::backward(AutogradContext* ctx, tensor_list grad_outputs) {
    auto crd = ctx->saved_data["pos"].toTensor();
    auto rots = ctx->saved_data["rots"].toTensor();
    auto tors = ctx->saved_data["tors"].toTensor();
    auto pivot = ctx->saved_data["pivot"].toInt();
    auto tor_bonds = ctx->saved_data["tor_bonds"].toIntVector();
    auto tor_mask = ctx->saved_data["tor_mask"].toTensorVector();

    auto batch_size = crd.size(0);
    auto frc = grad_outputs[0];

    // gradients for the freedom of translations
    auto grad_trans = frc.sum(1, true);

    // gradients for the freedom of rotations
    crd = crd - crd.select(1, pivot).unsqueeze(1);
    auto grad_rots = torch::empty({ batch_size, 3 }, rots.options());

    auto cos = rots.cos();
    auto sin = rots.sin();
    auto cos_c = cos.select(1, 2);
    auto sin_c = sin.select(1, 2);

    auto cos_b = cos.select(1, 1);
    auto sin_b = sin.select(1, 1);

    auto zero = torch::zeros(batch_size, rots.options());
    auto one = torch::ones(batch_size, rots.options());
    auto axis_z = torch::stack({ zero, zero, one }).t().unsqueeze(1);
    auto axis_y = torch::stack({ -sin_c, cos_c, zero }).t().unsqueeze(1);
    auto axis_x = torch::stack({ cos_c * cos_b, sin_c * cos_b, -sin_b }).t().unsqueeze(1);

    auto cross = torch::linalg_cross(crd, frc);
    grad_rots.select(1, 0) = (axis_x * cross).sum({ 1, 2 });
    grad_rots.select(1, 1) = (axis_y * cross).sum({ 1, 2 });
    grad_rots.select(1, 2) = (axis_z * cross).sum({ 1, 2 });

    // gradients for the freedom of torsions
    auto grad_tors = torch::empty({ batch_size, (int)tor_mask.size() }, tors.options());
    for (auto i = 0u; i < tor_mask.size(); i++) {
        auto u = tor_bonds[i * 2];
        auto v = tor_bonds[i * 2 + 1];

        // convention : positive rotation if pointing inwards
        auto pos_v = crd.select(1, v);
        auto rot_vec = crd.select(1, u) - pos_v;
        rot_vec = F::normalize(rot_vec, F::NormalizeFuncOptions().dim(-1)).unsqueeze(1);

        auto rotated_crd = crd.index_select(1, tor_mask[i]) - pos_v.unsqueeze(1);
        auto rotated_frc = frc.index_select(1, tor_mask[i]);
        auto cross_ = torch::linalg_cross(rot_vec, rotated_crd);
        auto grad_tor = (rotated_frc * cross_).sum(-1).sum(1);
        grad_tors.select(1, i) = grad_tor;
    }
    return { grad_trans, grad_rots, grad_tors, torch::Tensor() };
}
