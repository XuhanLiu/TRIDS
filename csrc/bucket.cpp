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

#include <tuple>
#include <iostream>
#include <fstream>
#include "bucket.cuh"

Bucket::Bucket(float3 box_len_, float cutoff_, float skin_) :
	box_len(box_len_), cutoff(cutoff_ + skin_), skin(skin_) {
	skin2 = skin_ * skin_;
	cutoff2 = cutoff * cutoff;

	bucket_dim.x = floorf(box_len.x / cutoff);
	bucket_dim.y = floorf(box_len.y / cutoff);
	bucket_dim.z = floorf(box_len.z / cutoff);

	num_buckets = bucket_dim.x * bucket_dim.y * bucket_dim.z;
	num_layers = bucket_dim.y * bucket_dim.z;

	bucket_len = { box_len.x / bucket_dim.x, box_len.y / bucket_dim.y , box_len.z / bucket_dim.z };
	bucket_len_inv = { 1.f / bucket_len.x, 1.f / bucket_len.y , 1.f / bucket_len.z };

	// clear();
}

void Bucket::put_atom_into_bucket(at::Tensor rec_pos) {
	auto int_opt = at::TensorOptions().dtype(at::kInt).device(rec_pos.device());
	neighbors = at::full({ num_buckets, MAX_NBR_ATOM }, -1, int_opt);
	neighbors.slice(1, 0, 1) = 0;
	auto num_atoms = rec_pos.size(0);
	auto bucket_len_ = at::tensor({ bucket_len.x, bucket_len.y, bucket_len.z }, rec_pos.options());
	auto bucket_dim_ = at::tensor({ bucket_dim.x, bucket_dim.y, bucket_dim.z }, rec_pos.options());
	offset = std::get<0>(rec_pos.min(0)) - skin;
	auto temp_idx = at::floor((rec_pos - offset) / bucket_len_);
	auto dim = at::tensor({ num_layers, bucket_dim.z, 1 }, temp_idx.options()).unsqueeze(1);
	auto grid_ids = temp_idx.mm(dim).squeeze(1);
	// std::cout << temp_idx << std::endl;
	// std::cout << std::get<0>(rec_pos.max(0)) - std::get<0>(rec_pos.min(0)) << std::endl;
	for (int i = 0; i < num_atoms; i++) {
		if ((temp_idx[i] < 0).any().item<bool>() || (temp_idx[i] >= bucket_dim_).any().item<bool>()) {
			printf("atom coordinate %f %f %f is out of range of neighbor box\n",
				   rec_pos[i][0].item<float>(), rec_pos[i][1].item<float>(), rec_pos[i][2].item<float>());
		}
		auto grid_idx = grid_ids[i].item<int>();
		neighbors[grid_idx][0] += 1;
		neighbors[grid_idx][neighbors[grid_idx][0]] = i;
		
		if (neighbors[grid_idx][0].item<int>() >= MAX_NBR_ATOM) {
			printf("nbr_bucket %d max\n", MAX_NBR_ATOM);
		}
	}
	// std::string out_path = "/home/liuxh/trids/tensor.log";
	// std::ofstream fout(out_path);
    // fout << neighbors << std::endl;
    // fout.close();
}

void Bucket::clear() {
	for (int i = 0; i < num_buckets; i++) {
		neighbors[i][0] = 1;
	}
}
