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
#include <float.h>
#include <openbabel/elements.h>
#include <openbabel/atom.h>
#include <openbabel/obiter.h>
#include "vina_scorer.h"
#include "scoring.cuh"
#include "bucket.cuh"
#include "utility.h"
#include "grid.cuh"

void VinaScorer::set_ligand(OpenBabel::OBMol* lig) {
    Scorer::set_ligand(lig);
    lig_vdw = torch::empty({ lig->NumAtoms() });
    lig_hydro = torch::empty({ lig->NumAtoms() }, torch::kBool);
    lig_hd = torch::empty({ lig->NumAtoms() }, torch::kBool);
    lig_ha = torch::empty({ lig->NumAtoms() }, torch::kBool);
    lig_atom = torch::empty({ lig->NumAtoms() }, torch::kInt);
    FOR_ATOMS_OF_MOL(atom, lig) {
        int id = atom->GetIndex();
        float vdw = OpenBabel::OBElements::GetVdwRad(atom->GetAtomicNum());
        lig_vdw[id] = vdw;
        bool is_hd = OpenBabel::IsHbondDonor(*atom);
        bool is_ha = atom->IsHbondAcceptor();
        bool is_hydro = OpenBabel::IsHydrophobic(*atom);
        lig_hydro[id] = is_hydro;   
        lig_hd[id] = is_hd;
        lig_ha[id] = is_ha;
        lig_atom[id] = OpenBabel::atom2type(*atom, is_hydro, is_hd, is_ha);
    }

    lig_vdw = lig_vdw.to(options);
    lig_vdw = lig_vdw.to(options.device());
    lig_ha = lig_ha.to(options.device());
    lig_hd = lig_hd.to(options.device());
    lig_atom = lig_atom.to(options.device());
    lig_hydro = lig_hydro.to(options.device());
}

void VinaScorer::profiling() {
    Scorer::profiling();
    inter_vdw = rec->vdw.unsqueeze(0) + lig_vdw.unsqueeze(1);
    inter_hbond = (rec->ha.unsqueeze(0) & lig_hd.unsqueeze(1)) | (rec->hd.unsqueeze(0) & lig_ha.unsqueeze(1));
    inter_hydro = rec->hydro.unsqueeze(0) & lig_hydro.unsqueeze(1);
    inner_vdw = lig_vdw.unsqueeze(0) + lig_vdw.unsqueeze(1);
    inner_hydro = lig_hydro.unsqueeze(0) & lig_hydro.unsqueeze(1);
    inner_hbond = (lig_ha.unsqueeze(0) & lig_hd.unsqueeze(1)) | (lig_hd.unsqueeze(0) & lig_ha.unsqueeze(1));
}

void VinaScorer::make_grid() {
    Bucket bucket({ 400.f, 400.f, 400.f }, cutoff, 2.f);
    bucket.put_atom_into_bucket(rec->pos);
    rec->grid = std::make_unique<Grid>(100, cutoff, options);
    rec->grid->make_grid(rec->pos, rec->hd, rec->ha, rec->hydro, rec->vdw, rec->box_min, bucket);
}

at::Tensor VinaScorer::calc_energy(at::Tensor dist, at::Tensor hydro_mask, at::Tensor hbond_mask) {
    // GAUSS1
    auto total_score = GAUSS1 * torch::exp(-GAUSS1_2 * dist.pow(2));
    // GAUSS2
    auto dp = dist - GAUSS2_C;
    auto score = GAUSS2 * torch::exp(-GAUSS2_2 * dp.pow(2));
    total_score += score;

    // repulsion
    score = REPULSION * (dist < 0) * dist.pow(2);
    total_score += score;

    // hydrophobic
    score = HYDRO_UA * (dist <= HYDRO_A) + HYDRO_UB * (dist >= HYDRO_B)
        + (HYDRO_RANGE * (dist - HYDRO_A) + HYDRO_UA)
        * ((HYDRO_A < dist) & (dist < HYDRO_B));
    score *= HYDRO * hydro_mask;
    total_score += score;

    // H_BOND
    score = H_BOND_UA * (dist <= H_BOND_A) + H_BOND_UB * (dist >= H_BOND_B)
        + (H_BOND_RANGE * (dist - H_BOND_A) + H_BOND_UA)
        * ((H_BOND_A < dist) & (dist < H_BOND_B));
    score *= H_BOND * hbond_mask;
    total_score += score;
    return total_score;
}

at::Tensor VinaScorer::scoring(at::Tensor lig_pos) {
    auto dist0 = torch::cdist(lig_pos, rec->pos);
    auto dist = dist0 - inter_vdw;
    // std::cout << torch::nonzero(dist != dist) << std::endl;
    auto inter_score = calc_energy(dist, inter_hydro, inter_hbond) * (dist0 < cutoff);
    
    // dist0 = torch::cdist(lig_pos, lig_pos);
    // dist = dist0 - inner_vdw;
    // auto inner_score = calc_energy(dist, inner_hydro, inner_hbond) * inner_nbr * (dist0 < cutoff);
    // return inner_score.sum() + inter_score.sum();
    return inter_score.sum();
}

at::Tensor VinaScorer::batched_auto_scoring(at::Tensor lig_pos) {
    auto batch_size = lig_pos.size(0);
    auto num_atoms = lig_pos.size(1);
    auto dist0 = torch::cdist(lig_pos, rec->pos);
    auto dist = dist0 - inter_vdw.unsqueeze(0); 
    auto inter_score = calc_energy(dist, inter_hydro, inter_hbond) * (dist0 < cutoff);

    dist0 = torch::cdist(lig_pos, lig_pos);
    dist = dist0 - inner_vdw.unsqueeze(0);
    auto inner_score = calc_energy(dist, inner_hydro.unsqueeze(0), inner_hbond.unsqueeze(0)) * inner_nbr.unsqueeze(0) * (dist0 < cutoff);
    auto total_score = at::stack({inter_score.sum(-1), inner_score.sum(-1) * 0.5f + inter_score.sum(-1)}, 1);
    return total_score;
}

at::Tensor VinaScorer::batched_accurate_scoring(at::Tensor lig_pos) {
    return VinaAccurateScoring::apply(lig_pos, this);
}

at::Tensor VinaScorer::batched_fast_scoring(at::Tensor lig_pos) {
    return VinaFastScoring::apply(lig_pos, this);
}

at::Tensor VinaScorer::loss(at::Tensor lig_pos) {
    return batched_auto_scoring(lig_pos);
}

void VinaScorer::to(torch::TensorOptions options_) {
    options = options_;
    inner_nbr = inner_nbr.to(options.device());
    inner_hydro = inner_hydro.to(options.device());
    inner_hbond = inner_hbond.to(options.device());
    inter_hydro = inter_hydro.to(options.device());
    inter_hbond = inter_hbond.to(options.device());

    inner_vdw = inner_vdw.to(options);
    inter_vdw = inter_vdw.to(options);

    lig_vdw = lig_vdw.to(options);
    lig_hd = lig_hd.to(options.device());
    lig_ha = lig_ha.to(options.device());
    lig_hydro = lig_hydro.to(options.device());
    lig_atom = lig_atom.to(options.device());

    rec->to(options);
}