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

#include <vector>
#include <openbabel/residue.h>
#include <openbabel/elements.h>
#include <openbabel/atom.h>
#include <openbabel/obiter.h>
#include "receptor.h"
#include "feat.h"
#include "utility.h"


inline bool is_pocket_residue(OpenBabel::OBResidue* resi, const double* grid, int num_points, float cutoff2) {
	for (auto iter1 = resi->BeginAtoms(); iter1 != resi->EndAtoms(); iter1++) {
		auto* atom = *iter1;
		auto* crd = atom->GetCoordinate();
		for (int i = 0; i < num_points; i++) {
			const double dx = crd[0] - grid[3 * i];
			const double dy = crd[1] - grid[3 * i + 1];
			const double dz = crd[2] - grid[3 * i + 2];
			const double rij = dx * dx + dy * dy + dz * dz;
			// Prody uses "<" rather than "<=" at the cutoff boundary.
			if (rij < cutoff2) {
				return true;
			}
		}
	}
	return false;
}

Receptor::Receptor(OpenBabel::OBMol* rec, at::Tensor ref_pos, bool only_vina, float cutoff) {
	rec->AddHydrogens(true);
	if (ref_pos.dtype() != at::kDouble) {
		ref_pos = ref_pos.to(at::kDouble);
	}
	if (!ref_pos.is_contiguous()) {
		ref_pos = ref_pos.contiguous();
	}

	const double* grid = ref_pos.data_ptr<double>();
	const int num_points = static_cast<int>(ref_pos.size(0));
	const float cutoff2 = cutoff * cutoff;

	std::vector<OpenBabel::OBResidue*> pocket_residues;

	for (auto iter = rec->BeginResidues(); iter != rec->EndResidues(); ++iter) {
		auto* resi = *iter;
		if (is_pocket_residue(resi, grid, num_points, cutoff2)) {
			pocket_residues.push_back(resi);
			FOR_ATOMS_OF_RESIDUE(atom, *resi) {
				if (atom->GetAtomicNum() > 1) {
					++num_atoms;
				}
			}
		}
	}

	num_res = static_cast<int>(pocket_residues.size());
	if (num_atoms == 0 || num_res == 0) {
		return;
	}

	idx = at::empty({ num_atoms }, at::kLong);
	vdw = at::empty({ num_atoms }, at::kFloat);
	hydro = at::empty({ num_atoms }, at::kBool);
	ha = at::empty({ num_atoms }, at::kBool);
	hd = at::empty({ num_atoms }, at::kBool);
	atomic = at::empty({ num_atoms }, at::kInt);
	at::Tensor coords = at::empty({ num_atoms, 3 }, at::kFloat);

	int atom_i = 0;
	for (int res_i = 0; res_i < num_res; res_i++) {
		auto* res = pocket_residues[res_i];
		FOR_ATOMS_OF_RESIDUE(atom, *res) {
			if (atom->GetAtomicNum() <= 1) {
				continue;
			}
			idx[atom_i] = res_i;

            const bool is_hd = OpenBabel::IsHbondDonor(*atom);
			const bool is_ha = OpenBabel::IsHbondAcceptor(*atom);
			const bool is_hydro = OpenBabel::IsHydrophobic(*atom);
            
			hd[atom_i] = is_hd;
			ha[atom_i] = is_ha;
			hydro[atom_i] = is_hydro;

			auto* c = atom->GetCoordinate();
			coords[atom_i][0] = static_cast<float>(c[0]);
			coords[atom_i][1] = static_cast<float>(c[1]);
			coords[atom_i][2] = static_cast<float>(c[2]);
			if (only_vina) {
				auto atom_type = OpenBabel::atom2type(*atom, is_hydro, is_ha, is_hd);
				vdw[atom_i] = VINA_VDW.at(atom_type);
				atomic[atom_i] = atom_type;
			} else {
				vdw[atom_i] = OpenBabel::OBElements::GetVdwRad(atom->GetAtomicNum());
			}
			atom_i++;
		}
	}

	pos = coords.to(OPTIONS);
	vdw = vdw.to(OPTIONS);
	idx = idx.to(OPTIONS.device());
	hd = hd.to(OPTIONS.device());
	ha = ha.to(OPTIONS.device());
	hydro = hydro.to(OPTIONS.device());
	atomic = atomic.to(OPTIONS.device());
	ptr = torch::_convert_indices_from_coo_to_csr(idx.cpu(), num_res).to(at::kInt).to(OPTIONS.device());
	if (!only_vina) {
		rec->DeleteHydrogens();
		graph = Feature::pdb_to_graph(rec, pocket_residues);
		graph.to(OPTIONS);
	}
}

void Receptor::to(at::TensorOptions options) {
	graph.to(options);
	vdw = vdw.to(options);
	idx = idx.to(options.device());
	ptr = ptr.to(options.device());
	hd = hd.to(options.device());
	ha = ha.to(options.device());
	hydro = hydro.to(options.device());
	atomic = atomic.to(options.device());
	pos = pos.to(options);
	box_max = box_max.to(options);
	box_min = box_min.to(options);
}
