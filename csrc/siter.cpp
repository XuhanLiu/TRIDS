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

#include <openbabel/residue.h>
#include <openbabel/atom.h>
#include <openbabel/obiter.h>
#include <openbabel/elements.h>
#include <spdlog/spdlog.h>
#include "siter.h"
#include "constant.h"
#include "surface.cuh"


using namespace torch::indexing;

Siter::~Siter() {
	for (auto& pocket : pockets) {
		delete pocket;
	}
	pockets.clear();
}

std::tuple<at::Tensor, at::Tensor> Siter::get_box(at::Tensor points) {
	auto box_min = std::get<0>(points.min(0)) - 2;
	auto box_max = std::get<0>(points.max(0)) + 2;
	return {box_min.to(options), box_max.to(options) };
}

void Siter::init(OpenBabel::OBMol* rec, bool is_vina) {
	for (auto current = 0; current < num_clusters; current++) {
		auto ref_pos = this->points(current);
		auto receptor = this->extract(rec, ref_pos, is_vina);
		if (receptor->num_res < 3) {
			delete receptor;
			spdlog::warn("Extract binding pocket {} fail", current);
			continue;
		}
		std::tie(receptor->box_min, receptor->box_max) = this->get_box(ref_pos);
		pockets.push_back(receptor);
	}
}

RefSiter::~RefSiter() {
	for (auto& ref : refs) {
		delete ref;
	}
	refs.clear();
}

at::Tensor RefSiter::points(int current) {
	auto ref = refs[current];
	auto pos = torch::from_blob(ref->GetCoordinates(), { ref->NumAtoms(), 3 }, at::kDouble).clone();
	return pos;
}

Receptor* RefSiter::extract(OpenBabel::OBMol* rec, at::Tensor pos, bool is_vina) {
	if (pos.dtype() != at::kDouble) {
		pos = pos.to(at::kDouble);
	}
	if (!pos.is_contiguous()) {
		pos = pos.contiguous();
	}
	return new Receptor(rec, pos, is_vina, cutoff, options);
}

at::Tensor GridSiter::points(int current) {
	auto pos = cluster[current].to(at::kDouble);
	return pos;
}

Receptor* GridSiter::extract(OpenBabel::OBMol* rec, at::Tensor pos, bool is_vina) {
	if (pos.dtype() != at::kDouble) {
		pos = pos.to(at::kDouble);
	}
	if (!pos.is_contiguous()) {
		pos = pos.contiguous();
	}
	return new Receptor(rec, pos, is_vina, cutoff, options);
}

ModelSiter::ModelSiter(torch::jit::Module model_, OpenBabel::OBMol* rec, torch::TensorOptions options_, float cutoff_)
	: GridSiter(options_, cutoff_), model(model_) {
	
	std::vector<float> pos_list;
	std::vector<float> feat_list;
	int num_atoms = 0;
	feat_list.reserve(rec->NumAtoms());
	std::vector<int> atomicnum;
	atomicnum.reserve(rec->NumAtoms());
	FOR_RESIDUES_OF_MOL(resi, rec) {
        if (resi->GetName() == "HOH") {
            continue;
        }
		for (auto it1 = resi->BeginAtoms(); it1 != resi->EndAtoms(); it1++) {
			auto atom = *it1;
			pos_list.push_back(atom->GetX());
			pos_list.push_back(atom->GetY());
			pos_list.push_back(atom->GetZ());
			
			atomicnum.push_back(atom->GetAtomicNum());

			num_atoms++;
		}
	}

	auto pos = torch::from_blob(pos_list.data(), {num_atoms, 3}).clone();
	auto surface = Surface(pos, atomicnum.data());
	int i = 0;
	FOR_RESIDUES_OF_MOL(resi, rec) {
		auto res_name = resi->GetName();
		if (res_name == "HOH") {
            continue;
        }
		res_name.erase(0, res_name.find_first_not_of(" "));			
		res_name.erase(res_name.find_last_not_of(" ") + 1);

		// Todo: FOR_ATOMS_RESIDUE has a bug, the following GetAtomID method does not work
		for (auto it1 = resi->BeginAtoms(); it1 != resi->EndAtoms(); it1++) {
        	auto atom = *it1;
			auto atom_name = resi->GetAtomID(atom);
			atom_name.erase(0, atom_name.find_first_not_of(" "));
			atom_name.erase(atom_name.find_last_not_of(" ") + 1);

			auto index = res_name + "_" + atom_name;
			std::vector<float> feat;
			if (Feature::RESI_ATOM_FEAT.count(index)) {
				feat = Feature::RESI_ATOM_FEAT.at(index);
			} else {
				feat = std::vector<float>(17, 0);
			}

			// require to calculate SASA for each atom of the receptor
			float sasa = surface.atom_is_near_surface[i] * 1.0f;
			feat.push_back(sasa);

			feat_list.insert(feat_list.end(), feat.begin(), feat.end());
			i++;
		}
	}
	
	auto features = torch::from_blob(feat_list.data(), {i, 18}).clone();
	auto centroid = pos.mean(0);
	pos -= centroid;
	auto grid_feat = make_grid(pos, features).to(options);

	auto grid = model.forward({ grid_feat }).toTensor();

	// torch::save(features, "../feat.pt");
	// torch::save(pos, "../pos.pt");
	// torch::save(grid_feat, "../grid_feat.pt");
	// torch::save(grid, "../grid.pt");
	// std::cout << pos << std::endl;
	// std::cout << features << std::endl;
	grid_to_cluster(grid, centroid);
}

at::Tensor ModelSiter::make_grid(at::Tensor coords, at::Tensor features) {
	if (coords.ndimension() != 2 || coords.size(1) != 3) {
		throw std::runtime_error("coords must be an array of floats of shape (N, 3)");
	}
	auto n_atoms = coords.size(0);
	if (features.ndimension() != 2 || features.size(0) != n_atoms) {
		throw std::runtime_error("features must be an array of floats of shape (N, F)");
	}

	if (resolution <= 0) {
		throw std::runtime_error("grid_resolution must be positive");
	}
	if (max_distance <= 0) {
		throw std::runtime_error("max_dist must be positive");
	}

	auto box_size = (int)std::ceil(2. * max_distance / resolution + 1);
	// move all atoms to the nearest grid point;
	auto grid_coords = ((coords + max_distance + 0.) / resolution).round().to(torch::kInt);
	auto in_box = ((grid_coords >= 0) & (grid_coords < box_size)).all(1);  // remove atoms outside the box
	in_box = torch::nonzero(in_box).view(-1);
	// std::cout << in_box << std::endl;
	auto grid = torch::zeros({ box_size, box_size, box_size, features.size(1) }, torch::kFloat);
	grid_coords = grid_coords.index({ in_box });
	features = features.index({ in_box });
	for (auto i = 0; i < in_box.size(0); i++) {
		auto x = grid_coords[i][0];
		auto y = grid_coords[i][1];
		auto z = grid_coords[i][2];
		grid[x][y][z] += features[i];
	}
	return grid.unsqueeze(0);
}

void ModelSiter::grid_to_cluster(at::Tensor grid, at::Tensor centroid) {
	auto selected_points = grid.cpu().flatten().argsort(0, true).slice(0, None, top_points);
	
	// std::cout << selected_points << std::endl;
	
	std::vector<int64_t> current_points;
	std::unordered_map<int64_t, int64_t> remain_points;
	std::unordered_map<int64_t, int64_t> points_status;
	const int max_clusters = 16;

	auto dimension = torch::tensor({ grid.size(1) * grid.size(2), grid.size(2), (int64_t)1 }, at::kLong);
	auto dim_ptr = dimension.data_ptr<int64_t>();
	for (auto i = 0; i < selected_points.size(0); i++) {
		auto grid_id = selected_points[i].item<int64_t>();
		remain_points.insert({ grid_id, i });
		points_status.insert({ grid_id , -1 });
	}
	// Keep looping while there are uncolored atoms
	while (remain_points.size() > 0) {
		auto grid_id = remain_points.begin()->first;
		// Start searching from the first uncolored point, current_points should be empty at each iteration start
		current_points.push_back(grid_id);
		points_status.at(grid_id) = num_clusters;  // Color current point
		remain_points.erase(grid_id);

		// While there are still unvisited points in current cluster
		while (current_points.size() > 0) {
			grid_id = current_points[0];
			auto point_pos = torch::tensor({
				grid_id / dim_ptr[0],
				(grid_id % dim_ptr[0]) / dim_ptr[1],
				grid_id % dim_ptr[1] }, at::kLong);

			// Loop through 27 neighbors (skip 8 corners)
			auto dds = torch::ones({ 3, 3, 3 }, at::kLong).nonzero();
			for (auto i = 0; i < dds.size(0); i++) {
				auto dd = dds[i] - 1;
				// If at least one of ddx is 0, it's definitely not at corner
				if (torch::any(dd == 0).item<bool>()) {
					auto xyz = (point_pos + dd).clamp(0, grid.size(1) - 1);
					grid_id = xyz.dot(dimension).item<int64_t>();
					
					auto it = points_status.find(grid_id);
					// If neighbor exists and is not colored yet
					if (it != points_status.end() && it->second == -1) {
						points_status.at(grid_id) = num_clusters;
						remain_points.erase(grid_id);
						current_points.push_back(grid_id);
					}
				}
			}  // for 27
			current_points[0] = current_points[current_points.size() - 1];
			current_points.pop_back();
		}  // while current_point.size() > 0
		num_clusters++;
	}

	int selected_cluster = 0;
	for (int i = 0; i < num_clusters; i++) {
		// Count point numbers
		int num_points = 0;
		for (auto ite = points_status.begin(); ite != points_status.end(); ite++) {
			if (ite->second == i) {
				num_points++;
			}
		}
		if (num_points > 2 && selected_cluster < max_clusters) {
			std::vector<torch::Tensor> selected_points;
			for (auto ite = points_status.begin(); ite != points_status.end(); ite++) {
				if (ite->second == i) {
					auto grid_id = ite->first;
					auto point_pos = torch::tensor({
						grid_id / dim_ptr[0],
						grid_id % dim_ptr[0] / dim_ptr[1],
						grid_id % dim_ptr[1] }, at::kLong);
					selected_points.push_back(point_pos);
				}
			}
			auto point = torch::stack(selected_points) * resolution - max_distance + centroid;
			cluster.push_back(point);
			selected_cluster++;
		}
	}
	num_clusters = selected_cluster;
}

at::Tensor ModelSiter::points(int current) {
	auto pos = GridSiter::points(current);
	return pos;
}

Receptor* ModelSiter::extract(OpenBabel::OBMol* rec, at::Tensor pos, bool is_vina) {
	return GridSiter::extract(rec, pos, is_vina);
}
