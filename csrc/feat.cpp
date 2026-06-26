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

#include "feat.h"
#include "utility.h"

using at::indexing::Slice;
using at::indexing::None;

inline at::Tensor GetCoordinate(OpenBabel::OBAtom* atom) {
    auto crd = at::tensor({ atom->x(), atom->y(), atom->z() });
    return crd;
}

template<typename T>
inline at::Tensor one_of_k_encoding_unk(T t, const vector<T>& allowable_set) {
    int size = allowable_set.size();
    at::Tensor out = at::zeros({ size });
    auto it = std::find(allowable_set.begin(), allowable_set.end(), t);
    if (it != allowable_set.end()) {
        out[it - allowable_set.begin()] = 1;
    } else {
        out[-1] = 1;
    }

    return out;
}

Graph Feature::obmol_to_graph(OpenBabel::OBMol* mol) {
    auto facade = OpenBabel::OBStereoFacade(mol);
    int size = SYMBOL.size() + DEGREE.size() + HYB.size() + NUM_H.size() + 3 + 3;
    auto atom_feats = torch::zeros({ mol->NumAtoms(), 42 });

    FOR_ATOMS_OF_MOL(atom, mol) {
        auto atom_id = atom->GetIndex();
        int idx1 = 0, idx2 = SYMBOL.size();
        auto symbol = one_of_k_encoding_unk(string(OpenBabel::OBElements::GetSymbol(atom->GetAtomicNum())), SYMBOL);
        atom_feats[atom_id].slice(0, idx1, idx2) = symbol;
        idx1 = idx2;
        idx2 += DEGREE.size();
        auto degree = one_of_k_encoding_unk(atom->GetExplicitDegree(), DEGREE);
        atom_feats[atom_id].slice(0, idx1, idx2) = degree;

        idx1 = idx2;
        idx2++;
        float charge = atom->GetFormalCharge();
        atom_feats[atom_id][idx1] = charge;

        idx1 = idx2;
        idx2++;
        auto spin = atom->GetSpinMultiplicity();
        atom_feats[atom_id][idx1] = static_cast<int>(spin);

        idx1 = idx2;
        idx2 += HYB.size();

        auto hyb = one_of_k_encoding_unk(atom->GetHyb(), HYB);
        atom_feats[atom_id].slice(0, idx1, idx2) = hyb;

        idx1 = idx2;
        idx2++;
        bool aromatic = atom->IsAromatic();
        atom_feats[atom_id][idx1] = aromatic;

        idx1 = idx2;
        idx2 += NUM_H.size();

        auto implicit_h = atom->GetImplicitHCount();
        auto num_h = one_of_k_encoding_unk(implicit_h, NUM_H);
        atom_feats[atom_id].slice(0, idx1, idx2) = num_h;

        auto stereo = facade.GetTetrahedralStereo(atom->GetIndex());
        if (stereo == nullptr) {
            atom_feats[atom_id][idx2 + 2] = 1;
        } else if (stereo->GetConfig().winding == OpenBabel::OBStereo::Clockwise) {
            atom_feats[atom_id][idx2] = 1;
        } else if (stereo->GetConfig().winding == OpenBabel::OBStereo::AntiClockwise) {
            atom_feats[atom_id][idx2 + 1] = 1;
        } else {
            atom_feats[atom_id][idx2 + 2] = 1;
        }
    };

    auto bond_feats = torch::zeros({ mol->NumBonds() * 2, 9 });
    auto edge_index = at::zeros({ 2, mol->NumBonds() * 2 }, at::kLong);
    FOR_BONDBFS_OF_MOL(bond, mol) {
        auto bond_id = bond->GetIdx();
        edge_index[0][bond_id * 2] = static_cast<int>(bond->GetBeginAtom()->GetIndex());
        edge_index[1][bond_id * 2] = static_cast<int>(bond->GetEndAtom()->GetIndex());
        edge_index[0][bond_id * 2 + 1] = static_cast<int>(bond->GetEndAtom()->GetIndex());
        edge_index[1][bond_id * 2 + 1] = static_cast<int>(bond->GetBeginAtom()->GetIndex());

        bond_feats[bond_id * 2][0] = bond->GetBondOrder() == 1;
        bond_feats[bond_id * 2][1] = bond->GetBondOrder() == 2;
        bond_feats[bond_id * 2][2] = bond->GetBondOrder() == 3;
        bond_feats[bond_id * 2][3] = bond->IsAromatic();
        bond_feats[bond_id * 2][4] = bond->IsInRing();

        auto stereo = facade.GetCisTransStereo(bond_id);
        if (stereo == nullptr) {
            bond_feats[bond_id * 2][5] = 1;
        } else if (stereo->GetConfig().shape == OpenBabel::OBStereo::Shape4) {
            bond_feats[bond_id * 2][6] = 1;
        } else if (stereo->GetConfig().shape == OpenBabel::OBStereo::ShapeZ) {
            bond_feats[bond_id * 2][7] = 1;
        } else if (stereo->GetConfig().shape == OpenBabel::OBStereo::ShapeU) {
            bond_feats[bond_id * 2][8] = 1;
        } else {
            bond_feats[bond_id * 2][5] = 1;
        }
        bond_feats[bond_id * 2 + 1] = bond_feats[bond_id * 2];
    }
    Graph graph(atom_feats, bond_feats, edge_index);
    return graph;
}

string Feature::obtain_resname(OpenBabel::OBResidue* res) {
    string resname{ res->GetName().substr(0, 2) };
    const vector<string> metals{ "CA", "FE", "CU" };

    if (std::count(metals.begin(), metals.end(), resname)) {
        return resname;
    } else {
        resname = res->GetName();
        resname.erase(0, resname.find_first_not_of(" "));
        resname.erase(resname.find_last_not_of(" ") + 1);

        if (std::count(METAL.begin(), METAL.end(), resname)) {
            return "M";
        } else {
            return resname;
        }
    }
}

at::Tensor Feature::obtain_self_dist(OpenBabel::OBResidue* res) {
    float dist_max = 0, dist_min = FLT_MAX;
    float dist_ca_o = 0, dist_o_n = 0, dist_n_c = 0;
    int num_atoms = res->GetNumAtoms();
    for (auto it1 = res->BeginAtoms(); it1 != res->EndAtoms(); it1++) {
        auto atom1 = *it1;
        for (auto it2 = it1 + 1; it2 != res->EndAtoms(); it2++) {
            auto atom2 = *it2;
            float dist = atom1->GetDistance(atom2);
            if (res->GetAtomID(atom1) == " CA " && res->GetAtomID(atom2) == " O  ") {
                dist_ca_o = dist;
            } else if (res->GetAtomID(atom1) == " N  " && res->GetAtomID(atom2) == " O  ") {
                dist_o_n = dist;
            } else if (res->GetAtomID(atom1) == " N  " && res->GetAtomID(atom2) == " C  ") {
                dist_n_c = dist;
            }
            if (dist > dist_max) {
                dist_max = dist;
            }
            if (dist < dist_min) {
                dist_min = dist;
            }
        }
    }
    if (res->GetNumAtoms() < 2) {
        dist_min = 0;
    }
    return at::tensor({ dist_max , dist_min, dist_ca_o, dist_o_n, dist_n_c }) * 0.1;
}


Feature::Coord Feature::coordinate(OpenBabel::OBResidue* res) {
    auto coords = at::empty({ res->GetNumAtoms(), 3 });
    at::Tensor ca_crd;
    auto mass = at::empty({ res->GetNumAtoms(), 1 });
    auto find_ca = false;
    for (auto iter = res->BeginAtoms(); iter != res->EndAtoms(); iter++) {
        auto atom = *iter;
        int atom_id = iter - res->BeginAtoms();
        auto coord = GetCoordinate(atom);
        mass[atom_id] = atom->GetExactMass();
        coords[atom_id] = coord;
        if (res->GetAtomID(atom) == " CA ") {
            ca_crd = GetCoordinate(atom);
            find_ca = true;
        }
    }
    if (!find_ca) {
        if (obtain_resname(res) == "M") {
            ca_crd = coords[0];
        } else {
            ca_crd = coords.mean(0);
        }
    }
    auto ce_crd = (coords * mass).sum(0) / mass.sum();
    Coord coord{ coords, ca_crd, ce_crd };
    return coord;
}

float Feature::check_connect(OpenBabel::OBResidue* res1, OpenBabel::OBResidue* res2) {
    int i = res1->GetIdx();
    int j = res2->GetIdx();
    if (std::abs(i - j) != 1) {
        return 0;
    } else {
        for (auto& bond : res1->GetBonds()) {
            auto atom1 = bond->GetBeginAtom();
            auto atom2 = bond->GetEndAtom();

            if (atom1 == nullptr || atom2 == nullptr) {
                continue;
            }
            if (atom1->GetResidue() == res2 || atom2->GetResidue() == res2) {
                return 1;
            }
        }
    }
    return 0;
}

inline OpenBabel::OBAtom* get_atom_by_name(OpenBabel::OBResidue* res, const string& name) {
    OpenBabel::OBAtom* atom{ nullptr };
    for (auto it = res->BeginAtoms(); it != res->EndAtoms(); it++) {
        auto resname = res->GetAtomID(*it);
        if (resname == name) {
            atom = *it;
            break;
        }
    }
    return atom;
}

Graph Feature::pdb_to_graph(OpenBabel::OBMol* mol, float cutoff) {
    std::vector<OpenBabel::OBResidue*> residues;
    residues.reserve(mol->NumResidues());
    for (auto iter = mol->BeginResidues(); iter != mol->EndResidues(); ++iter) {
        residues.push_back(*iter);
    }
    return pdb_to_graph(mol, residues, cutoff);
}

Graph Feature::pdb_to_graph(OpenBabel::OBMol* mol, const std::vector<OpenBabel::OBResidue*>& residues, float cutoff) {
    vector<int64_t> edge_list;
    vector<float> edge_attr;
    
    int num_residues = static_cast<int>(residues.size());
    auto node_feats = at::zeros({ num_residues, 41 });
    
    // ==================== Optimization: pre-compute all residue coordinates ====================
    // Avoid repeated coordinate() calls in nested loops
    std::vector<Coord> all_coords(num_residues);
    std::vector<OpenBabel::OBResidue*> all_residues(num_residues);
    auto all_ca_coords = at::empty({ num_residues, 3 });  // For batch distance calculation
    
    for (int res_idx = 0; res_idx < num_residues; res_idx++) {
        auto res = residues[res_idx];
        all_residues[res_idx] = res;
        all_coords[res_idx] = coordinate(res);
        all_ca_coords[res_idx] = all_coords[res_idx].ca_crd;
    }
    
    // ==================== Optimization: batch CA distance calculation for pre-filtering ====================
    // Use CA distance < (cutoff + max_residue_radius) to pre-filter, avoid unnecessary precise distance calculations
    // Typical residue radius is ~5-8 Å, use 15 Å as safe threshold
    constexpr float CA_FILTER_THRESHOLD = 25.0f;  // cutoff(10) + 2*max_residue_radius(~7.5)
    auto ca_dist_matrix = at::cdist(all_ca_coords, all_ca_coords);
    
    // ==================== First pass: compute node features ====================
    for (int curr_id = 0; curr_id < num_residues; curr_id++) {
        auto res = residues[curr_id];

        // Calculate psi, phi, chi1 and omega
        int idx1 = 0, idx2 = RES3.size();
        auto resname = obtain_resname(res);
        auto elem = one_of_k_encoding_unk(resname, RES3);
        node_feats[curr_id].slice(0, idx1, idx2) = elem;

        idx1 = idx2;
        idx2 += 5;
        if (std::count(RES3.begin(), RES3.end(), resname)) {
            auto dist = obtain_self_dist(res);
            node_feats[curr_id].slice(0, idx1, idx2) = dist;
        }

        auto n_curr = get_atom_by_name(res, " N  ");
        auto ca_curr = get_atom_by_name(res, " CA ");
        auto c_curr = get_atom_by_name(res, " C  ");

        auto prev_res = curr_id > 0 ? residues[curr_id - 1] : nullptr;
        auto next_res = curr_id + 1 < num_residues ? residues[curr_id + 1] : nullptr;

        // phi
        if (prev_res != nullptr && res->GetNum() - prev_res->GetNum() == 1) {
            auto c_prev = get_atom_by_name(prev_res, " C  ");
            if (c_prev != nullptr && n_curr != nullptr && ca_curr != nullptr && c_curr != nullptr) {
                float phi = mol->GetTorsion(c_prev, n_curr, ca_curr, c_curr) * 0.01;
                node_feats[curr_id][-4] = phi;
            }
        }

        // psi and omega
        if (next_res != nullptr && next_res->GetNum() - res->GetNum() == 1) {
            auto n_next = get_atom_by_name(next_res, " N  ");
            auto ca_next = get_atom_by_name(next_res, " CA ");
            if (c_curr != nullptr && ca_curr != nullptr && c_curr != nullptr && n_next != nullptr) {
                float psi = mol->GetTorsion(n_curr, ca_curr, c_curr, n_next) * 0.01;
                node_feats[curr_id][-3] = psi;
            }

            if (ca_curr != nullptr && c_curr != nullptr && n_next != nullptr && ca_next != nullptr) {
                float omega = mol->GetTorsion(ca_curr, c_curr, n_next, ca_next) * 0.01;
                node_feats[curr_id][-2] = omega;
            }
        }

        // chi
        auto cb_curr = get_atom_by_name(res, " CB ");
        auto s_curr = get_atom_by_name(res, " CG ");
        if (s_curr == nullptr) {
            s_curr = get_atom_by_name(res, " CG ");
        }
        if (s_curr == nullptr) {
            s_curr = get_atom_by_name(res, " CG1");
        }
        if (s_curr == nullptr) {
            s_curr = get_atom_by_name(res, " OG ");
        }
        if (s_curr == nullptr) {
            s_curr = get_atom_by_name(res, " OG1");
        }
        if (n_curr != nullptr && ca_curr != nullptr && cb_curr != nullptr && s_curr != nullptr) {
            float chi1 = mol->GetTorsion(n_curr, ca_curr, cb_curr, s_curr) * 0.01;
            node_feats[curr_id][-1] = chi1;
        }
    }
    
    // ==================== Second pass: compute edge features (with pre-filtering) ====================
    for (int i = 0; i < num_residues; i++) {
        auto res1 = all_residues[i];
        const auto& out1 = all_coords[i];
        auto crd1 = out1.coords;
        auto ca_crd1 = out1.ca_crd;
        auto center1 = out1.ce_crd;
        
        for (int j = i + 1; j < num_residues; j++) {
            // Optimization: pre-filter using CA distance, skip obviously distant residue pairs
            float ca_dist_val = ca_dist_matrix[i][j].item<float>();
            if (ca_dist_val > CA_FILTER_THRESHOLD) {
                continue;  // Skip, no need to calculate precise atomic distances
            }
            
            auto res2 = all_residues[j];
            const auto& out2 = all_coords[j];
            auto crd2 = out2.coords;
            auto ca_crd2 = out2.ca_crd;
            auto center2 = out2.ce_crd;
            
            // Only calculate precise atomic distances for residue pairs that pass pre-filtering
            auto dists = at::cdist(crd1, crd2);
            float min_dist_val = dists.min().item<float>();

            if (min_dist_val <= cutoff) {
                edge_list.push_back(i);
                edge_list.push_back(j);
                edge_list.push_back(j);
                edge_list.push_back(i);

                float ca_dist = ca_dist_val * 0.1;
                float ce_dist = at::norm(center1 - center2).item<float>() * 0.1;
                float min_dist = min_dist_val * 0.1;
                float max_dist = dists.max().item<float>() * 0.1;

                for (auto k = 0; k < 2; k++) {
                    edge_attr.push_back(check_connect(res1, res2));
                    edge_attr.push_back(ca_dist);
                    edge_attr.push_back(ce_dist);
                    edge_attr.push_back(min_dist);
                    edge_attr.push_back(max_dist);
                }
            }
        }
    }

    auto edge_feats = at::from_blob(edge_attr.data(), { int(edge_attr.size()) / 5, 5 }).clone();
    auto edge_index = at::from_blob(edge_list.data(), { int(edge_list.size()) / 2, 2 }, at::kLong).t().clone();
    Graph graph(node_feats, edge_feats, edge_index);
    return graph;
}