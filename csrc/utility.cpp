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

 #include <algorithm>
 #include <unordered_set>
 #include <vector>
 #include <openbabel/math/vector3.h>
 #include "utility.h"

static std::vector<int> component_without_bond(OpenBabel::OBMol* mol, int start, int u, int v) {
	const int n = static_cast<int>(mol->NumAtoms());
	std::vector<bool> visited(n, false);
	std::vector<int> component;
	std::vector<int> stack = { start };

	while (!stack.empty()) {
		const int node = stack.back();
		stack.pop_back();
		if (visited[node]) {
			continue;
		}
		visited[node] = true;
		component.push_back(node);
		OpenBabel::OBAtom* atom = mol->GetAtom(node + 1);
		FOR_NBORS_OF_ATOM(nbr, atom) {
			const int next = nbr->GetIndex();
			if (visited[next]) {
				continue;
			}
			if ((node == u && next == v) || (node == v && next == u)) {
				continue;
			}
			stack.push_back(next);
		}
	}
	return component;
}

static bool fragments_split_by_bond(OpenBabel::OBMol* mol, int src_id, int dst_id,
									std::vector<int>& comp_src, std::vector<int>& comp_dst) {
	comp_src = component_without_bond(mol, src_id, src_id, dst_id);
	comp_dst = component_without_bond(mol, dst_id, src_id, dst_id);
	return comp_src.size() + comp_dst.size() == mol->NumAtoms();
}

inline bool is_amide_bond(OpenBabel::OBBond& bond) {
	OpenBabel::OBAtom* a = bond.GetBeginAtom();
	OpenBabel::OBAtom* b = bond.GetEndAtom();
	OpenBabel::OBAtom* c = (a->GetAtomicNum() == 6 && b->GetAtomicNum() == 7) ? a
		: (a->GetAtomicNum() == 7 && b->GetAtomicNum() == 6) ? b : nullptr;
	if (c == nullptr || (c->GetExplicitDegree() != 3 && c->GetHvyDegree() != 3)) {
		return false;
	}
	FOR_NBORS_OF_ATOM(nbr, c) {
		if (nbr->GetAtomicNum() == 8 && nbr->GetHvyDegree() == 1) {
			return true;
		}
	}
	return false;
}

inline bool is_guanidinium_bond(OpenBabel::OBBond& bond) {
	for (OpenBabel::OBAtom* atom : {bond.GetBeginAtom(), bond.GetEndAtom()}) {
		if (atom->GetAtomicNum() != 6 || atom->GetHvyDegree() != 3) {
			continue;
		}
		bool all_n = true;
		int in_cycle = 0;
		FOR_NBORS_OF_ATOM(nbr, atom) {
			if (nbr->GetAtomicNum() > 1 && nbr->GetAtomicNum() != 7) {
				all_n = false;
				break;
			}
		}
		if (!all_n) {
			continue;
		}
		FOR_BONDS_OF_ATOM(bb, atom) {
			in_cycle += static_cast<int>(bb->IsInRing());
		}
		if (in_cycle < 2) {
			return true;
		}
	}
	return false;
}

inline bool tors_order_ok(OpenBabel::OBBond& bond) {
	const int order = bond.GetBondOrder();
	if (order == 1) {
		return true;
	}
	if (order != 2) {
		return false;
	}
	const int za = bond.GetBeginAtom()->GetAtomicNum();
	const int zb = bond.GetEndAtom()->GetAtomicNum();
	const double len = bond.GetLength();
	if (za == 6 && zb == 6) {
		return len >= 1.42;
	}
	return ((za == 6 && zb == 7) || (za == 7 && zb == 6)) && len >= 1.35;
}

inline bool is_leaf_atom(OpenBabel::OBAtom* atom) {
	return atom->GetExplicitDegree() == 1
		|| (atom->GetAtomicNum() == 6 && atom->GetHvyDegree() == 1);
}

inline bool only_carbon_or_h(OpenBabel::OBAtom* atom) {
	FOR_NBORS_OF_ATOM(nbr, atom) {
		if (nbr->GetAtomicNum() != 1 && nbr->GetAtomicNum() != 6) {
			return false;
		}
	}
	return true;
}

inline bool is_polyene_double(OpenBabel::OBBond& bond) {
	OpenBabel::OBAtom* a = bond.GetBeginAtom();
	OpenBabel::OBAtom* b = bond.GetEndAtom();
	if (a->GetAtomicNum() != 6 || b->GetAtomicNum() != 6) {
		return false;
	}
	if (!only_carbon_or_h(a) || !only_carbon_or_h(b)) {
		return false;
	}
	return bond.GetLength() < 1.38;
}

inline bool is_sulfonamide_bond(OpenBabel::OBBond& bond) {
	OpenBabel::OBAtom* a = bond.GetBeginAtom();
	OpenBabel::OBAtom* b = bond.GetEndAtom();
	OpenBabel::OBAtom* sulfur = (a->GetAtomicNum() == 16 && b->GetAtomicNum() == 7) ? a
		: (a->GetAtomicNum() == 7 && b->GetAtomicNum() == 16) ? b : nullptr;
	if (sulfur == nullptr) {
		return false;
	}
	int n_o = 0;
	FOR_NBORS_OF_ATOM(nbr, sulfur) {
		n_o += static_cast<int>(nbr->GetAtomicNum() == 8);
	}
	return n_o >= 3;
}


bool OpenBabel::is_rot_bond(OpenBabel::OBBond& bond) {
	if (bond.IsInRing() || !tors_order_ok(bond)) {
		return false;
	}

	if (is_leaf_atom(bond.GetBeginAtom()) || is_leaf_atom(bond.GetEndAtom())) {
		return false;
	}

	if (is_amide_bond(bond) || is_guanidinium_bond(bond)
		|| is_polyene_double(bond) || is_sulfonamide_bond(bond)) {
		return false;
	}

	return true;
}

int OpenBabel::find_pivot(OpenBabel::OBMol* mol) {
	const int n = static_cast<int>(mol->NumAtoms());
	unsigned shortest_max_subgraph = mol->NumAtoms();
	int best_pivot = 0;
	for (int removed = 0; removed < n; ++removed) {
		std::vector<bool> visited(n, false);
		visited[removed] = true;
		int max_frag = 0;
		OpenBabel::OBAtom* removed_atom = mol->GetAtom(removed + 1);
		FOR_NBORS_OF_ATOM(nbr, removed_atom) {
			const int neighbor = nbr->GetIndex();
			if (visited[neighbor]) {
				continue;
			}
			int count = 0;
			std::vector<int> stack = { neighbor };
			while (!stack.empty()) {
				const int node = stack.back();
				stack.pop_back();
				if (visited[node]) {
					continue;
				}
				visited[node] = true;
				count++;
				OpenBabel::OBAtom* atom = mol->GetAtom(node + 1);
				FOR_NBORS_OF_ATOM(next, atom) {
					if (!visited[next->GetIndex()]) {
						stack.push_back(next->GetIndex());
					}
				}
			}
			max_frag = std::max(max_frag, count);
		}
		if (max_frag < static_cast<int>(shortest_max_subgraph)) {
			shortest_max_subgraph = static_cast<unsigned>(max_frag);
			best_pivot = removed;
		}
	}
	return best_pivot;
}

std::vector<OpenBabel::RotatableBond> OpenBabel::find_rotatable_bonds(
		OpenBabel::OBMol* mol, int pivot, const std::vector<int>* norotate) {
	std::unordered_set<int> norot;
	if (norotate != nullptr) {
		norot = { norotate->begin(), norotate->end() };
	}

	std::vector<RotatableBond> bonds;
	FOR_BONDS_OF_MOL(bond, mol) {
		auto src = bond->GetBeginAtom();
		auto dst = bond->GetEndAtom();
		const int src_id = src->GetIndex();
		const int dst_id = dst->GetIndex();

		if (norot.count(src_id) && norot.count(dst_id)) {
			continue;
		}
		if (src->HasData("Fixed") && dst->HasData("Fixed")) {
			continue;
		}
		if (!is_rot_bond(*bond)) {
			continue;
		}

		if (bond->GetBeginAtom()->GetHvyDegree() == 1|| bond->GetEndAtom()->GetHvyDegree() == 1) {
			continue;
		}

		std::vector<int> comp_src;
		std::vector<int> comp_dst;
		if (!fragments_split_by_bond(mol, src_id, dst_id, comp_src, comp_dst)) {
			continue;
		}

		const bool pivot_in_src = std::find(comp_src.begin(), comp_src.end(), pivot) != comp_src.end();
		const std::vector<int>& rotating = pivot_in_src ? comp_dst : comp_src;

		RotatableBond entry;
		if (std::find(rotating.begin(), rotating.end(), dst_id) != rotating.end()) {
			entry.src = src_id;
			entry.dst = dst_id;
		} else {
			entry.src = dst_id;
			entry.dst = src_id;
		}
		entry.rotating = rotating;
		bonds.push_back(std::move(entry));
	}
	return bonds;
}

float OpenBabel::num_tors(OpenBabel::OBMol* mol) {
	float num_tors = 0;
	FOR_BONDS_OF_MOL(bond, mol) {
		if (!is_rot_bond(*bond)) {
			continue;
		} 
		const bool hrot = bond->GetBeginAtom()->GetHvyDegree() == 1
			|| bond->GetEndAtom()->GetHvyDegree() == 1;
		num_tors += hrot ? 0.5f : 1.f;
	}
	return num_tors;
}

int OpenBabel::NumOuterElec(int atomic_num) {
	if (1 <= atomic_num <= 2) {
		return atomic_num;
	} else if (3 <= atomic_num <= 10) {
		return atomic_num - 2;
	} else if (11 <= atomic_num <= 18) {
		return atomic_num - 10;
	} else if (19 <= atomic_num <= 30) {
		return atomic_num - 18;
	} else if (31 <= atomic_num <= 36) {
		return atomic_num - 26;
	} else if (37 <= atomic_num <= 48) {
		return atomic_num - 36;
	} else if (49 <= atomic_num <= 54) {
		return atomic_num - 46;
	} else if (55 <= atomic_num <= 70) {
		return atomic_num - 54;
	} else if (71 <= atomic_num <= 80) {
		return atomic_num - 68;
	} else if (81 <= atomic_num <= 86) {
		return atomic_num - 78;
	} else if (87 <= atomic_num <= 102) {
		return atomic_num - 86;
	} else if (103 <= atomic_num <= 112) {
		return atomic_num - 100;
	} else {
		return atomic_num - 110;
	}
}

bool OpenBabel::isAtomConjugCand(OBAtom* at) {
	// return false for neutral atoms where the current valence exceeds the
	// minimal valence for the atom. logic: if we're hypervalent we aren't
	// conjugated
	const auto vals = OBElements::GetMaxBonds(at->GetAtomicNum());
	if (!at->GetFormalCharge() && vals > 0 && at->GetTotalValence() > vals) {
		return false;
	}

	// the second check here is for Issue211, where the c-P bonds in
	// Pc1ccccc1 were being marked as conjugated.  This caused the P atom
	// itself to be SP2 hybridized.  This is wrong.  For now we'll do a quick
	// hack and forbid this check from adding conjugation to anything out of
	// the first row of the periodic table.  (Conjugation in aromatic rings
	// has already been attended to, so this is safe.)
	int outer = NumOuterElec(at->GetAtomicNum());
	auto res = ((at->GetAtomicNum() <= 10) || (outer != 5 && outer != 6) ||
				(outer == 6 && at->GetTotalDegree() < 2u)) && at->GetAtomicNum() != 0;
	return res;
}

std::set<unsigned> OpenBabel::markConjAtomBonds(OBMol* mol) {
	std::set<unsigned> conjugates;
	FOR_ATOMS_OF_MOL(it0, mol) {
		auto at0 = *it0;
		if (!isAtomConjugCand(&at0)) {
			continue;
		}

		// make sure that have either 2 or 3 substitutions on this atom
		int sbo = at0.GetTotalDegree();
		if (sbo != 2 && sbo != 3) {
			continue;
		}
		FOR_NBORS_OF_ATOM(it1, &at0) {
			auto at1 = *it1;
			auto bond1 = mol->GetBond(&at0, &at1);
			if (bond1->GetBondOrder() == 1 || !isAtomConjugCand(&at1)) {
				continue;
			}

			FOR_NBORS_OF_ATOM(it2, at1) {
				auto at2 = *it2;
				auto bond2 = mol->GetBond(&at1, &at2);
				if (bond1 == bond2) {
					continue;
				}
				sbo = at2.GetTotalDegree();
				if (sbo > 3) {
					continue;
				}
				if (isAtomConjugCand(&at2)) {
					conjugates.insert(bond1->GetIdx());
					conjugates.insert(bond2->GetIdx());
				}
			}
		}
	}
	return conjugates;
}

bool OpenBabel::HasConjugatedBond(OpenBabel::OBAtom* at, std::set<unsigned>& conjugate) {
	FOR_BONDS_OF_ATOM(b, at) {
		auto bond = *b;
		if (conjugate.count(bond.GetIdx())) {
			return true;
		}
	}
	return false;
}

int OpenBabel::NumBondsPlusLonePairs(OBAtom* at) {
	int deg = at->GetTotalDegree();

	if (at->GetAtomicNum() <= 1) {
		return deg;
	}
	int nouter = OpenBabel::NumOuterElec(at->GetAtomicNum());
	int totalValence = at->GetTotalValence();
	int chg = at->GetFormalCharge();

	int numFreeElectrons = nouter - (totalValence + chg);
	if (totalValence + nouter - chg < 8) {
		// we're below an octet, so we need to think
		// about radicals:
		unsigned spin = at->GetSpinMultiplicity();
		unsigned radical = (spin > 0) ? (spin - 1) : 0;
		int numRadicals = radical;
		int numLonePairs = (numFreeElectrons - numRadicals) / 2;
		return deg + numLonePairs + numRadicals;
	} else {
		int numLonePairs = numFreeElectrons / 2;
		return deg + numLonePairs;
	}
}

unsigned OpenBabel::Hybridization(OBAtom* atom, OBStereoFacade& facade, std::set<unsigned>& conjugate) {
	if (atom->GetAtomicNum() == 0) {
		return 0;
	} else if (facade.HasTetrahedralStereo(atom->GetId())) {
		// if the stereo spec matches the coordination number, this is easy
		if (atom->GetTotalDegree() == 4) {
			return 3;
		} else if (atom->GetTotalDegree() == 5) {
			return 4;
		} else if (atom->GetTotalDegree() == 6) {
			return 5;
		}
	} else if (facade.HasSquarePlanarStereo(atom->GetId())) {
		if (atom->GetTotalDegree() <= 4 && atom->GetTotalDegree() >= 2) {
			return 0; // Atom::SP2D
		}
	}
	// otherwise we have to do some work
	int norbs;
	// try to be smart for early elements, but for later
	// ones just use the degree
	// FIX: we should probably also be using the degree for metals
	if (atom->GetAtomicNum() < 89) {
		norbs = NumBondsPlusLonePairs(atom);
	} else {
		norbs = atom->GetTotalDegree();
	}
	switch (norbs) {
	case 0:
	case 1:
		return 0; //Atom::S;
	case 2:
		return 1; //Atom::SP;
	case 3:
		return 2; //Atom::SP2;
	case 4:
		// potentially SP3, but we'll set it down to SP2
		// if we have a conjugated bond (like the second O in O=CO)
		// we'll also avoid setting the hybridization down to
		// SP2 in the case of an atom with degree higher than 3
		// (e.g. things like CP1(C)=CC=CN=C1C, where the P
		//   has norbs = 4, and a conjugated bond, but clearly should
		//   not be SP2)
		// This is Issue276
		if (atom->GetTotalDegree() > 3 || !HasConjugatedBond(atom, conjugate)) {
			return 3;
		} else {
			return 2;
		}
		break;
	case 5:
		return 4;
		break;
	case 6:
		return 5;
		break;
	default:
		return 0;
	}
}

bool OpenBabel::IsHalogen(OBAtom& atom) {
	auto elem = atom.GetAtomicNum();
	return elem == 9 || elem == 17 || elem == 35 || elem == 53;
}

bool OpenBabel::IsHbondDonor(OBAtom& atom) {
	auto elem = atom.GetAtomicNum();
	if (elem == 7 || elem == 8) {
		FOR_NBORS_OF_ATOM(nbr, atom) {
			if (nbr->IsPolarHydrogen()) {
				return true;
			}
		}
		
		if (atom.GetImplicitHCount() > 0) {
			return true;
		} 
	}
	return false;
}

bool OpenBabel::IsHbondAcceptor(OBAtom& atom) {
	// AD4 / Vina XS: OA and NA. DSDP ignores SA/S/F.
	const int elem = atom.GetAtomicNum();
	if (elem != 7 && elem != 8) {
		return false;
	}
	if (atom.GetTotalDegree() >= 4) {
		return false;
	}
	if (elem == 8) {
		return true;
	}

	// NA: directional lone pair. Not Npl / Ntr / N3+ / azide terminal.
	if (atom.GetFormalCharge() > 0) {
		return false;
	}

	OpenBabel::OBAtom* nbrs[3];
	unsigned n_explicit = 0;
	unsigned n_oxygen = 0;
	bool has_heavy = false;
	bool all_c3 = true;
	bool bonded_to_n = false;
	FOR_NBORS_OF_ATOM(nbr, atom) {
		if (n_explicit < 3) {
			nbrs[n_explicit] = &*nbr;
		}
		++n_explicit;
		const int z = nbr->GetAtomicNum();
		if (z == 8) {
			++n_oxygen;
		} else if (z == 7) {
			bonded_to_n = true;
		}
		if (z == 1) {
			continue;
		}
		has_heavy = true;
		bool sp3_c = (z == 6 && !nbr->IsAromatic() && nbr->GetHyb() != 2);
		if (sp3_c) {
			FOR_NBORS_OF_ATOM(n2, *nbr) {
				if (n2->GetAtomicNum() == 8 && n2->GetHvyDegree() == 1) {
					sp3_c = false;
					break;
				}
			}
		}
		if (!sp3_c) {
			all_c3 = false;
		}
	}
	if (n_oxygen >= 2) {
		return false;  // nitro
	}
	if (atom.GetTotalDegree() == 1 && bonded_to_n) {
		return false;  // azide / diazo terminal; nitrile stays NA
	}
	if (has_heavy && all_c3 && (atom.GetTotalDegree() == 3 || n_explicit == 3)) {
		return false;  // aliphatic N3 / N3+
	}
	if (n_explicit == 3) {
		const OpenBabel::vector3 origin = atom.GetVector();
		const OpenBabel::vector3 v0 = nbrs[0]->GetVector() - origin;
		const OpenBabel::vector3 v1 = nbrs[1]->GetVector() - origin;
		const OpenBabel::vector3 v2 = nbrs[2]->GetVector() - origin;
		if (v0.length() >= 1e-6 && v1.length() >= 1e-6 && v2.length() >= 1e-6) {
			const double angle_sum = OpenBabel::vectorAngle(v0, v1) +
									 OpenBabel::vectorAngle(v0, v2) +
									 OpenBabel::vectorAngle(v1, v2);
			if (angle_sum >= 345.0) {
				return false;  // Npl: amide, aniline, pyrrole
			}
		}
	} else if (atom.GetTotalDegree() == 3 &&
			   (atom.GetHyb() == 2 || atom.IsAromatic())) {
		return false;  // implicit-H fallback when angles are unavailable
	}
	return true;
}

bool OpenBabel::IsHydrophobic(OBAtom& atom) {
	bool is_hydro = false;
	// if C atom is connected to any non-H or C atom, it is not hydrophobic
	if (IsHalogen(atom)) {
		is_hydro = true;
	} else if (atom.GetAtomicNum() == 6) {
		bool has_polar = false;
		FOR_NBORS_OF_ATOM(nbr, atom) {
			if (nbr->GetAtomicNum() != 6 && nbr->GetAtomicNum() != 1) {
				has_polar = true;
				break;
			}
		}
		is_hydro = !has_polar;
	}
	return is_hydro;
}

int OpenBabel::atom2type(OpenBabel::OBAtom& atom, bool is_hydro, bool is_ha, bool is_hd) {
	auto symbol = std::string(OpenBabel::OBElements::GetSymbol(atom.GetAtomicNum()));
	int atom_type = -1;
	if (symbol == "C") {
		atom_type = is_hydro ? 0 : 1;
	} else if (symbol == "N") {
		if (!(is_ha || is_hd)) {
			atom_type = 2;
		} else if ((!is_ha) && is_hd) {
			atom_type = 3;
		} else if (is_ha and (!is_hd)) {
			atom_type = 4;
		} else {
			atom_type = 5;
		}
	} else if (symbol == "O") {
		if (!(is_ha || is_hd)) {
			atom_type = 6;
		} else if ((!is_ha) && is_hd) {
			atom_type = 7;
		} else if (is_ha and (!is_hd)) {
			atom_type = 8;
		} else {
			atom_type = 9;
		}
	} else if (symbol == "S") {
		atom_type = 10;
	} else if (symbol == "P") {
		atom_type = 11;
	} else if (symbol == "F") {
		atom_type = 12;
	} else if (symbol == "Cl") {
		atom_type = 13;
	} else if (symbol == "Br") {
		atom_type = 14;
	} else if (symbol == "I") {
		atom_type = 15;
	} else if (symbol == "H") {
		atom_type = 17;
	} else {
		std::cout << "Warning: new atom type " << symbol << " found in Vina force field!" << std::endl;
		atom_type = 16;
	}
	return atom_type;
}