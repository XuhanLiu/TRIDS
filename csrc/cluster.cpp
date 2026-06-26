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
#include <vector>
#include "cluster.h"

inline at::Tensor calc_rmsd(
	at::Tensor selected_crd,
	at::Tensor candidate_crd) {

	auto diff = selected_crd - candidate_crd;
	auto sq_dist = (diff * diff).sum();
	return sq_dist.div(selected_crd.size(0)).sqrt();
}

void Cluster::sort_structures(at::Tensor coords, at::Tensor scores, 
							  float rmsd_cutoff, int topn) {
	TORCH_CHECK(coords.dim() == 3 && coords.size(2) == 3, "coordinates must be {N, M, 3}");
	TORCH_CHECK(scores.dim() == 2 && scores.size(1) == 2, "scores must be {N, 2}");
	TORCH_CHECK(coords.size(0) == scores.size(0), "scores dimension mismatch coordinates");

	auto order = scores.select(1, 1).argsort();

	auto record_numbers = coords.size(0);
	auto atom_numbers = coords.size(1);
	selected_numbers = 0;

	coords = coords.cpu();
	scores = scores.cpu();

	selected_coord = at::empty({topn, atom_numbers, 3}, at::kFloat);
	selected_score = at::empty({ topn }, at::kFloat);

	for (auto frame_i = 0; frame_i < record_numbers; frame_i++) {
		int pose_id = order[frame_i].item<int>();
		auto candidate = coords[pose_id];
		bool has_similar = false;
		for (auto pose_i = 0; pose_i < selected_numbers; pose_i++) {
			auto rmsd = calc_rmsd(selected_coord[pose_i], candidate);
			if (rmsd.item<float>() < rmsd_cutoff) {
				has_similar = true;
			}
		}

		if (!has_similar) {
			selected_coord[selected_numbers] = candidate;
			selected_score[selected_numbers] = scores[pose_id][0];
			selected_numbers += 1;
			if (selected_numbers >= topn) {
				break;
			}
		}
	}
}
