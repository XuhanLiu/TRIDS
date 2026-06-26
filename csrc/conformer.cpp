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

#include <spdlog/spdlog.h>
#include "conformer.h"
#include "rotation.h"
#include "sampling.cuh"
#include "utility.h"

namespace F = torch::nn::functional;

void Dof::to(torch::TensorOptions options) {
	trans = trans.to(options);
	rots = rots.to(options);
	tors = tors.to(options);
}

void Dof::detach() {
	trans.detach_();
	rots.detach_();
	tors.detach_();
}

Dof Dof::clone() {
	Dof conf;
	// std::cout << trans << std::endl;
	conf.trans = trans.clone().contiguous();
	// std::cout << rots << std::endl;
	conf.rots = rots.clone().contiguous();
	// std::cout << tors << std::endl;
	conf.tors = tors.clone().contiguous();
	return conf;
}

void Dof::copy(Dof origin) {
	if (!trans.defined()) {
		trans = origin.trans.clone();
	} else {
		trans.copy_(origin.trans);
	}

	if (!rots.defined()) {
		rots = origin.rots.clone();
	} else {
		rots.copy_(origin.rots);
	}

	if (!tors.defined()) {
		tors = origin.tors.clone();
	} else {
		tors.copy_(origin.tors);
	}
}

void Dof::require_grad(bool flag) {
	trans.requires_grad_(flag);
	rots.requires_grad_(flag);
	tors.requires_grad_(flag);
}

void Dof::init(at::Tensor box_min, at::Tensor box_max, unsigned num_tors) {
	auto option = box_min.options();
	trans = torch::rand({ 3 }, option) * (box_max - box_min) * 0.75 + box_min;
	// std::cout << trans << std::endl;
	auto qt = F::normalize(torch::randn({ 4 }, option), F::NormalizeFuncOptions().dim(-1));
	rots = Rotation::quaternion_to_euler(qt);
	tors = torch::rand({ num_tors }, option) * 2 * PI - PI;

	// std::cout << rots << std::endl;
	// std::cout << euler_to_quaternion(rots) << std::endl;
	// std::cout << quaternion_to_matrix(qt) << std::endl;
	// std::cout << euler_to_matrix(rots);
	// std::cout << matrix_from_euler(rots);
	// std::cout << quaternion_to_euler(qt);

	// auto axis = F::normalize(torch::randn({ 3 }), F::NormalizeFuncOptions().dim(-1));
	// auto theta = torch::rand(1);
	// std::cout << rot_vec_to_matrix(axis, theta) << std::endl;;
	// std::cout << matrix_from_rot_vec(axis, theta) << std::endl;;
	// std::cout << rot_vec_to_quaternion(axis, theta) << std::endl;

	// auto qt2 = F::normalize(torch::randn({ 4 }), F::NormalizeFuncOptions().dim(-1));
	// std::cout << quaternion_increment(qt, qt2);
}

void Dof::mutate(at::Tensor box_min, at::Tensor box_max, float amplitude, RNG& rng) {
	int num_dof = tors.size(0) + 2;
	int which = randi(0, num_dof - 1, rng);
	assert(which >= 0 && which < num_dof);

	if (which == 0) {
		mutate_trans(box_min, box_max, amplitude, rng);
		return;
	}
	which--;
	if (which == 0) {
		mutate_rots(amplitude, rng);
	}
	which--;
	if (which < tors.size(0)) {
		mutate_tors(which, rng);
		return;
	}
}

void Dof::mutate_trans(at::Tensor box_min, at::Tensor box_max, float amplitude, RNG& rng) {
	bool in_box = false;
	at::Tensor trans_;
	int i = 10;
	while (!in_box && i > 0) {
		trans_ = amplitude * rand_sphere(rng).to(trans.options());
		auto current = trans + trans_;
		in_box = (current > box_min).logical_and(current < box_max).all().item<bool>();
		i--;
	}
	if (i == 0) {
		trans = torch::rand({ 3 }, trans.options()) * (box_max - box_min) + box_min;
	}
	trans += trans_;
}

void Dof::mutate_rots(float amplitude, RNG& rng) {
	// float gr = gyration_radius();
	auto theta = torch::tensor({ amplitude / PI / 2 }, rots.options());
	auto axis = rand_sphere(rng).to(rots.options());
	auto qt1 = Rotation::rot_vec_to_quaternion(axis, theta);
	auto qt2 = Rotation::euler_to_quaternion(rots);
	auto qt = Rotation::quaternion_increment(qt1, qt2);
	rots = Rotation::quaternion_to_euler(qt);
}

void Dof::mutate_tors(int which, RNG& rng) {
	tors[which] = randf(-PI, PI, rng);
}

void Dof::clear() {
	trans.fill_(0);
	rots.fill_(0);
	tors.fill_(0);
}

///////////////////////////////////////////////////////////////////////////////////

void Dofs::init(at::Tensor box_min, at::Tensor box_max, unsigned batch_size_, unsigned num_tors) {
	batch_size = batch_size_;
	auto option = box_min.options();
	trans = torch::rand({ batch_size, 1, 3 }, option) * (0.8 - 0.2) + 0.2;
	trans = trans * (box_max - box_min) + box_min;
	// std::cout << trans << std::endl;
	auto qt = F::normalize(torch::randn({ batch_size, 4 }, option), F::NormalizeFuncOptions().dim(-1));
	rots = Rotation::batched_quaternion_to_euler(qt);
	tors = torch::rand({ batch_size, num_tors }, option) * 2 * PI - PI;

	// std::cout << rots << std::endl;
	// std::cout << Rotation::batched_euler_to_quaternion(rots) << std::endl;
	// std::cout << Rotation::batched_quaternion_to_matrix(qt) << std::endl;
	// std::cout << Rotation::batched_euler_to_matrix(rots);
	// std::cout << Rotation::batched_matrix_from_euler(rots);
	// std::cout << Rotation::batched_quaternion_to_euler(qt);

	// auto axis = F::normalize(torch::randn({ batch_size, 3 }), F::NormalizeFuncOptions().dim(-1));
	// auto theta = torch::rand(batch_size);
	// std::cout << Rotation::batched_rot_vec_to_matrix(axis, theta) << std::endl;;
	// std::cout << Rotation::batched_matrix_from_rot_vec(axis, theta) << std::endl;;
	// std::cout << Rotation::batched_rot_vec_to_quaternion(axis, theta) << std::endl;

	// auto qt2 = F::normalize(torch::randn({ 4 }), F::NormalizeFuncOptions().dim(-1));
	// std::cout << Rotation::batched_quaternion_increment(qt, qt2) << std::endl;
}

void Dofs::mutate(at::Tensor box_min, at::Tensor box_max, RNG& rng) {
	// auto start = clock();
	int num_dof = tors.size(1) + 6;

	auto option = trans.options().dtype(torch::kLong);
	auto dof_ids = torch::randint(0, num_dof, { batch_size }, option);

	auto trans_id = torch::nonzero(dof_ids < 3).squeeze(1);
	if (trans_id.size(0) > 0) {
		auto dof_id = dof_ids.index_select(0, trans_id);
		auto rand_trans = torch::rand({ trans_id.size(0), 3 }, trans.options()) * (box_max - box_min) + box_min;
		auto rand_tran = rand_trans.gather(-1, dof_id.unsqueeze(1)).squeeze(1);
		trans.index_put_({ trans_id, 0, dof_id }, rand_tran);
	}

	auto rots_id = torch::nonzero((dof_ids >= 3) & (dof_ids < 6)).squeeze(1);
	if (rots_id.size(0) > 0) {
		auto dof_id = dof_ids.index_select(0, rots_id) - 3;
		auto qt = F::normalize(torch::randn({ rots_id.size(0), 4 }, rots.options()), F::NormalizeFuncOptions().dim(-1));
		auto rand_rots = Rotation::batched_quaternion_to_euler(qt);
		auto rand_rot = rand_rots.gather(1, dof_id.unsqueeze(1)).squeeze(1);
		rots.index_put_({ rots_id , dof_id }, rand_rot);
	}

	auto tors_id = torch::nonzero(dof_ids >= 6).squeeze(1);
	auto rand_tors = torch::rand({ tors_id.size(0) }, tors.options()) * 2 * PI - PI;
	if (tors_id.size(0) > 0) {
		auto dof_id = dof_ids.index_select(0, tors_id) - 6;
		auto rand_tors = torch::rand({ tors_id.size(0) }, tors.options()) * 2 * PI - PI;
		tors.index_put_({ tors_id, dof_id }, rand_tors);
	}
	// auto duration = static_cast<float>(std::clock() - start) / CLOCKS_PER_SEC;
	// std::cout << "Parallel: " << duration << std::endl;

	// start = clock();
	// for (auto i = 0; i < trans_id.size(0); i++) {
	// 	auto batch_id = trans_id[i].item<int>();
	// 	auto dof_id = dof_ids[batch_id].item<int>();
	// 	trans1[batch_id][0][dof_id] = rand_trans[i][dof_id];
	// }

	// for (auto i = 0; i < rots_id.size(0); i++) {
	// 	auto dof_id = dof_ids[rots_id[i]] - 3;
	// 	rots1[rots_id[i]][dof_id] = rand_rots[i][dof_id];
	// }

	// for (auto i = 0; i < tors_id.size(0); i++) {
	// 	auto dof_id = dof_ids[tors_id[i]] - 6;
	// 	tors1[tors_id[i]][dof_id] = rand_tors[i];
	// }

	// duration = static_cast<float>(std::clock() - start) / CLOCKS_PER_SEC;
	// std::cout << "For loop: " << duration << std::endl;
}

Dofs Dofs::clone() {
	Dofs dofs;
	// std::cout << trans << std::endl;
	dofs.trans = trans.clone().contiguous();
	// std::cout << rots << std::endl;
	dofs.rots = rots.clone().contiguous();
	// std::cout << tors << std::endl;
	dofs.tors = tors.clone().contiguous();
	dofs.batch_size = batch_size;
	return dofs;
}


Conformer::Conformer(OpenBabel::OBMol* lig_, int num_tasks, torch::TensorOptions options_)
	: lig(lig_), options(options_) {

	spdlog::info("Find rotational bond begin ...");
	auto start = clock();
	get_torsions(lig);
	auto duration = static_cast<float>(clock() - start) / CLOCKS_PER_SEC;
	spdlog::info("Find rotational bond end ({:.4f}s)", duration);

	spdlog::info("Initiate the coordinate of the ligand begin ...");
	start = clock();
	
	auto num_inits = num_tasks > 1 ? 1 : num_tasks;
	auto repeats = num_tasks / num_inits + (num_tasks % num_inits > 0 ? 1 : 0);
	init_pos = torch::empty({ num_inits, lig->NumAtoms(), 3 }, at::kDouble);
	
	int i = 0;
	if (lig->GetCoordinates() != nullptr) {
		lig_pos = at::from_blob(lig->GetCoordinates(), { lig->NumAtoms(), 3 }, at::kDouble).clone();
		init_pos[0] = lig_pos;
		i = 1;
	}
	
	OpenBabel::OBBuilder builder;
	for (; i < num_inits; i++) {
		bool is_built = builder.Build(*lig);
		if (!is_built) {
			throw runtime_error("Could not initiate the coordinate of the input ligand!");
		}
		init_pos[i] = at::from_blob(lig->GetCoordinates(), { lig->NumAtoms(), 3 }, at::kDouble).clone();
	}
	lig_pos = lig_pos - lig_pos[pivot];
	init_pos = init_pos - init_pos.select(1, pivot).unsqueeze(1);	
	init_pos = init_pos.expand({ repeats, -1, -1 }).to(options);
	if (init_pos.size(0) > num_tasks) {
		init_pos = init_pos.slice(0, 0, num_tasks);
	}
	pos = init_pos.clone();
	pos.requires_grad_(true);
	duration = static_cast<float>(clock() - start) / CLOCKS_PER_SEC;
	spdlog::info("Initiate the coordinate of the ligand end ({:.4f}s)", duration);
}

inline float Conformer::gyration_radius() {
	auto acc = (lig_pos * lig_pos).sum(-1).mean();
	return std::sqrt(acc.item<float>());
}

void Conformer::get_torsions(OpenBabel::OBMol* mol, const std::optional<vector<int> >& norotate) {
	const std::vector<int>* norot = norotate.has_value() ? &norotate.value() : nullptr;
	const auto bonds = OpenBabel::find_rotatable_bonds(mol, &pivot, norot);
	for (const auto& entry : bonds) {
		tor_bonds.push_back(entry.src);
		tor_bonds.push_back(entry.dst);
		std::vector<int64_t> mask_atoms(entry.rotating.begin(), entry.rotating.end());
		auto mask = torch::tensor(mask_atoms, torch::TensorOptions().dtype(torch::kLong).device(options.device()));
		tor_masks.push_back(mask);
	}
}

unsigned Conformer::num_tors() {
	return tor_masks.size();
}

void Conformer::update_coord(Dof& conf, int i) {
	pos[i] = init_pos[i];
	for (auto i = 0; i < tor_masks.size(); i++) {
		auto u = tor_bonds[i * 2];
		auto v = tor_bonds[i * 2 + 1];

		// convention : positive rotation if pointing inwards
		auto rot_vec = F::normalize(pos[u] - pos[v], F::NormalizeFuncOptions().dim(0));
		auto tor_mat = Rotation::rot_vec_to_matrix(rot_vec, conf.tors[i]);

		auto rotated_pos = pos.index_select(0, tor_masks[i]);
		auto updated_pos = (rotated_pos.reshape({ -1, 3 }) - pos[v]).mm(tor_mat.t()) + pos[v];
		pos[i] = pos.index_put({ tor_masks[i] }, updated_pos);
	}
	auto rot_mat = Rotation::matrix_from_euler(conf.rots);
	pos[i] = pos[i].mm(rot_mat.t()) + conf.trans;
}

void Conformer::batched_auto_udpate_coord(Dofs& dofs) {
	pos.copy_(init_pos);
	for (auto i = 0; i < tor_masks.size(); i++) {
		auto u = tor_bonds[i * 2];
		auto v = tor_bonds[i * 2 + 1];

		// convention : positive rotation if pointing inwards
		auto pos_u = pos.select(1, u);
		auto pos_v = pos.select(1, v);
		auto rot_vec = pos_u - pos_v;

		rot_vec = F::normalize(rot_vec, F::NormalizeFuncOptions().dim(-1));

		auto tor = dofs.tors.select(1, i);
		auto tor_mat = Rotation::batched_rot_vec_to_matrix(rot_vec, tor);

		auto rotated_pos = pos.index_select(1, tor_masks[i]);

		pos_v = pos_v.unsqueeze(1);
		auto updated_pos = (rotated_pos - pos_v).bmm(tor_mat.permute({ 0, 2, 1 })) + pos_v;
		// std::cout << dofs.tors[0][i] << std::endl;
		// std::cout << tor_mat[0] << std::endl;
		// std::cout << rot_vec[0] << std::endl;
		// std::cout << Rotation::matrix_from_rot_vec(rot_vec[0], dofs.tors[0][i]) << std::endl;
		// std::cout << Rotation::rot_vec_to_matrix(rot_vec[0], dofs.tors[0][i]) << std::endl;

		// std::cout << rotated_pos[0] - pos_u[0] << std::endl;
		// std::cout << (rotated_pos - pos_v)[0] << std::endl;
		// std::cout << updated_pos[0] - pos_u[0] << std::endl;
		// std::cout << (updated_pos - pos_v)[0] << std::endl;
		pos.index_put_({ torch::indexing::Slice(), tor_masks[i], torch::indexing::Slice() }, updated_pos);
	}

	auto rot_mat = Rotation::batched_matrix_from_euler(dofs.rots).permute({ 0, 2, 1 });
	pos = pos.bmm(rot_mat) + dofs.trans;
}

void Conformer::batched_cuda_update_coord(Dofs& dofs) {
	Sampling::apply(dofs.trans, dofs.rots, dofs.tors, this);
}

void Conformer::to(torch::TensorOptions options_) {
	options = options_;
	lig_pos = lig_pos.to(options);
	for (auto& tor_mask : tor_masks) {
		tor_mask = tor_mask.to(options.device());
	}
}