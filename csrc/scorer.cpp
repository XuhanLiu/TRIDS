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

#include <openbabel/elements.h>
#include <openbabel/atom.h>
#include <openbabel/obiter.h>
#include <spdlog/spdlog.h>
#include "scorer.h"
#include "feat.h"
#include "scoring.cuh"
#include "utility.h"

namespace F = torch::nn::functional;

void Scorer::set_pocket(Receptor& rec) {
    this->rec = &rec;
}

void Scorer::set_ligand(OpenBabel::OBMol* lig) {
    inner_nbr = get_neighbors(lig);
    num_tors = static_cast<int>(OpenBabel::num_tors(lig));
}

void Scorer::profiling() {
    // make_grid();
}

at::Tensor Scorer::get_neighbors(OpenBabel::OBMol* lig) {
    inner_nbr = torch::eye({ lig->NumAtoms() });
    FOR_BONDS_OF_MOL(bond, lig) {
        int i = bond->GetBeginAtom()->GetIndex();
        int j = bond->GetEndAtom()->GetIndex();
        inner_nbr[i][j] = 1;
        inner_nbr[j][i] = 1;
    }
    inner_nbr = ~inner_nbr.mm(inner_nbr).mm(inner_nbr).to(torch::kBool);
    return inner_nbr.to(options.device());
}

void TriScorer::set_ligand(OpenBabel::OBMol* lig) {
    auto start = clock();
    spdlog::info("Extract ligand features begin ...");
    Scorer::set_ligand(lig);
    lig_vdw = torch::empty({ lig->NumAtoms() });
    lig_hd = torch::empty({ lig->NumAtoms() }, torch::kBool);
    lig_ha = torch::empty({ lig->NumAtoms() }, torch::kBool);
    FOR_ATOMS_OF_MOL(atom, lig) {
        int id = atom->GetIndex();
        float vdw = OpenBabel::OBElements::GetVdwRad(atom->GetAtomicNum());
        lig_vdw[id] = vdw;

        bool is_hd = OpenBabel::IsHbondDonor(*atom);
        bool is_ha = atom->IsHbondAcceptor();
        lig_hd[id] = is_hd;
        lig_ha[id] = is_ha;
    }
    lig_vdw = lig_vdw.to(options);
    lig_hd = lig_hd.to(options.device());
    lig_ha = lig_ha.to(options.device());
    
    lig_graph = Feature::obmol_to_graph(lig);
    lig_graph.to(options);

    auto duration = static_cast<float>(clock() - start) / CLOCKS_PER_SEC;
    spdlog::info("Extract ligand features end ({:.4f}s)", duration);
}

void TriScorer::profiling() {
    // the following computation do not need gradient
    spdlog::info("Construct scoring profile begin ...");
    auto start = clock();
    torch::NoGradGuard guard;
    // lig_graph.save("../lig_graph.pt");
    // rec_graph.save("../rec_graph.pt");
    auto profile = model->single(lig_graph, rec->graph).detach().view({ 3, -1, rec->num_res, 10 });
    log_rho = profile[0].log_();
    mu = profile[1];
    var = (profile[2] * profile[2]).reciprocal_();
    log_scale = profile[2].log_();
    data = log_scale.neg().sub_(log_sqrt_pi).add_(log_rho);
    Scorer::profiling();
    auto duration = static_cast<float>(clock() - start) / CLOCKS_PER_SEC;
    spdlog::info("Construct scoring profile end ({:.4f}s)", duration);
}

void TriScorer::make_grid() {
    // int grid_dim = 100;
    // forcefield = torch::empty({ grid_dim, grid_dim, grid_dim, mu.size(0), 4 }, options);

    // auto grid = torch::linspace(-1, 1, grid_dim, options.device()).unsqueeze(1);
    // grid = ((box_max - box_min) * (grid + 1) / 2 + box_min).t();
    // // grid, grid, grid, 3 
    // auto mesh = torch::stack(torch::meshgrid({ grid[0], grid[1], grid[2] }, "xy"), -1).unsqueeze(-2);

    // int j = 1;
    // for (int i = 0; i < grid_dim / j; i++) {
    //    // i, grid, grid, num_rec_atom, 3
    //     auto vector = mesh.slice(0, i, i + j) - rec_pos.view({ 1, 1, 1, rec_pos.size(0), 3 });
    //     auto distances = vector.norm(2, -1, true);
    //     auto out = scatter_min(distances, res_idx, -2, std::nullopt, num_res);
    //     // i, grid, grid, num_res, 1
    //     distances = get<0>(out);
    //     // i, grid, grid, num_res, 1
    //     auto dist_idx = get<1>(out);
    //     // i, grid, grid, num_res, 3
    //     vector = vector.gather(-2, dist_idx.repeat({ 1, 1, 1, 1, 3 }));
    //     auto mask = (distances <= cutoff);

    //     distances = distances.index({ mask });
    //     vector = vector.index({ mask.expand_as(vector) }).view({ -1, 3 });
    //     // normalize the vector
    //     vector.div_(distances.unsqueeze(-1).clip_(FLT_EPSILON));

    //     mask = mask.squeeze(-1).nonzero().t();
    //     auto batch_mu = mu.index_select(1, mask[-1]);
    //     auto batch_var = var.index_select(1, mask[-1]);
    //     auto batch_data = data.index_select(1, mask[-1]);
    //     // 22, mask, 10
    //     auto prob = distances.unsqueeze(0).unsqueeze(-1).repeat({ mu.size(0), 1, mu.size(-1) });
    //     // prob = -(prob - batch_mu).pow(2) * (batch_var * 0.5) - batch_scale - log_sqrt_pi + batch_rho;
    //     prob.sub_(batch_mu).pow_(2).mul_(-batch_var.mul_(0.5)).add_(batch_data).exp_();

    //     // i, grid, grid, num_atom
    //     auto score = torch::sparse_coo_tensor(mask, prob.sum(-1).t(), options).to_dense().sum(3);
    //     forcefield.slice(0, i, i + j).select(-1, 0) = score;
    //     score.reset();

    //     prob.mul_(batch_mu - distances.view({ 1, -1, 1 })).mul_(batch_var);
    //     // 22, mask, 3
    //     auto grad = -prob.sum(-1, true) * vector;

    //     // i, grid, grid, num_atom, 3
    //     grad = torch::sparse_coo_tensor(mask, grad.permute({ 1, 0, 2 }), options).to_dense().sum(3);
    //     forcefield.slice(0, i, i + j).slice(-1, 1) = grad;
    // }
    // // num_atom, 4, grid, grid, grid
    // forcefield = forcefield.permute({ 3, 4, 0, 1, 2 });
}

at::Tensor TriScorer::interpolate(at::Tensor pos) {
    auto batch_size = pos.size(0);
    auto num_atom = pos.size(1);
    auto norm_pos = 2 * (pos - rec->box_min) / (rec->box_max - rec->box_min) - 1;
    // Shape: { num_atom, batch_size, 1, 1, 3 }
    norm_pos = norm_pos.permute({ 1, 0, 2 }).unsqueeze(2).unsqueeze(2);;
    // Shape: { num_atom, 4, batch_size, 1, 1 }
    auto output = F::grid_sample(forcefield, norm_pos, F::GridSampleFuncOptions().align_corners(true));
    // Shape: {batch_size, num_atom, 4 }
    return output.squeeze(-1).squeeze(-1).permute({ 2, 0, 1 });
}

at::Tensor TriScorer::scoring(at::Tensor pos) {
    auto distances = torch::cdist(pos, rec->pos);
    auto dists = torch::empty({ pos.size(0), rec->num_res }, options);
    auto index = rec->idx.view({ 1, -1 }).expand({ pos.size(0), -1 });
    distances = dists.scatter_reduce(-1, index, distances, "amin", false);

    distances = distances.unsqueeze(-1);
    // std::cout << data << std::endl;
    auto prob = distances.repeat({ 1, 1, mu.size(-1) });
    prob.sub_(mu).pow_(2).mul_(-var.mul_(0.5)).add_(data).exp_();
    // auto prob = -(distances` - mu).pow(2) * (var * 0.5) - log_scale - log_sqrt_pi + log_rho;
    // std::cout << prob << std::endl;
    prob = prob.sum(-1, true) * (distances <= TRIDS_MDN_CUTOFF);
    auto score = prob.sum();

    // penalty for atom clashing between ligand and recetpor
    auto mask = distances < TRIDS_CLASH_CUTOFF;
    auto inter_penalty = torch::log(distances / TRIDS_CLASH_CUTOFF).masked_fill(~mask, 0);
    inter_penalty = inter_penalty.sum();
    
    // penalty for atom clashing within the ligand.
    auto inner_dist = torch::cdist(pos, pos).clip(FLT_EPSILON);
    mask = (inner_dist < TRIDS_CLASH_CUTOFF) & inner_nbr;
    auto inner_penalty = torch::log(inner_dist / TRIDS_CLASH_CUTOFF).masked_fill(~mask, 0);
    inner_penalty = inner_penalty.sum() * 0.5f;

    auto total_score = score + inter_penalty + inner_penalty; 
    return -total_score;
}

at::Tensor TriScorer::batched_auto_scoring(at::Tensor pos) {
    auto batch_size = pos.size(0);
    auto num_atoms = pos.size(1);

    auto distances = at::cdist(pos, rec->pos);
    auto dists = at::full({ batch_size, num_atoms, rec->num_res }, FLT_MAX, options);
    auto index = rec->idx.view({ 1, 1, -1 }).expand({ batch_size, num_atoms, -1 });
    distances = dists.scatter_reduce(-1, index, distances, "amin", false).clip(FLT_EPSILON);
    
    // long range interaction only for sampling effectively
    auto dist = distances - TRIDS_CLASH_CUTOFF - GAUSS2_C;
	auto long_range = TRIDS_LONG_WEIGHT * at::exp(-GAUSS1_2 * dist * dist);
    long_range = long_range.sum({-2, -1});

    auto prob = (-(distances.unsqueeze(-1) - mu).pow(2) * (var * 0.5) + data).exp();
    auto score = prob.sum(-1) * (distances < TRIDS_MDN_CUTOFF);
    // std::cout << score.sum(-1) << std::endl;
    score = score.sum({ -2, -1 });
    
    // penalty for atom clashing between ligand and recetpor
    auto mask = distances < TRIDS_CLASH_CUTOFF;
    auto inter_penalty = at::log(distances / TRIDS_CLASH_CUTOFF).masked_fill(~mask, 0);
    inter_penalty = TRIDS_CLASH_WEIGHT * inter_penalty.sum({ -2, -1 });
    
    // penalty for atom clashing within the ligand.
    auto inner_dist = at::cdist(pos, pos).clip(FLT_EPSILON);
    mask = (inner_dist < TRIDS_CLASH_CUTOFF) & inner_nbr;
    auto inner_penalty = at::log(inner_dist / TRIDS_CLASH_CUTOFF).masked_fill(~mask, 0);
    inner_penalty = TRIDS_CLASH_WEIGHT * inner_penalty.sum({ -2, -1 }) * 0.5f;

    score += inter_penalty + inner_penalty;

    auto total_score = at::stack({score, score + long_range}, 1);
    return -total_score;
}

at::Tensor TriScorer::batched_accurate_scoring(at::Tensor lig_pos) {
    // return TriAccurateScoring::apply(lig_pos, this);
    return TriAccurateScoring::apply(lig_pos, this);
}

at::Tensor TriScorer::batched_fast_scoring(at::Tensor lig_pos) {
    return TriFastScoring::apply(lig_pos, this);
}

at::Tensor TriScorer::loss(at::Tensor lig_pos) {
    return batched_auto_scoring(lig_pos);
}

void TriScorer::to(torch::TensorOptions options) {
    log_rho = log_rho.to(options);
    mu = mu.to(options);
    var = var.to(options);
    log_scale = log_scale.to(options);
    forcefield = forcefield.to(options);
    rec->to(options);
    inner_nbr = inner_nbr.to(options);
    lig_vdw = lig_vdw.to(options);
    
    lig_hd = lig_hd.to(options);
    lig_ha = lig_ha.to(options);
}