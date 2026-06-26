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

// RMSD-based deduplication of docked poses (ported from DSDP DSDP_SORT).
class Cluster {

public:
	int selected_numbers;
	at::Tensor selected_score;
	at::Tensor selected_coord;

	Cluster() = default;
	~Cluster() = default;

	// coords: {record_numbers, atom_numbers, 3}
	// scores: {record_numbers}, lower is better; sorted internally by ascending energy.
	// rmsd_cutoff: structures within this heavy-atom RMSD are duplicates (typically 2.f).
	// forward_comparing_numbers: compare each candidate with at most this many recent
	// 		selected poses (0 = compare all selected; DSDP default is often 20).
	// topn: max diverse poses to keep (e.g. 50).
	// Results are written to selected_crd {selected_numbers, atom_numbers, 3} and
	// selected_energy {selected_numbers}, on the same device/dtype as crd.
	void sort_structures(
		at::Tensor coords,
		at::Tensor scores,
		float rmsd_cutoff,
		int topn);
};
