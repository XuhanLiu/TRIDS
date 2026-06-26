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
#include "scoring.cuh"
#include "kernel.cuh"


namespace F = torch::nn::functional;
using namespace torch::indexing;


at::Tensor TriFastScoring::forward(AutogradContext* ctx, at::Tensor pos,
								   TriScorer* scorer) {
	auto batch_size = pos.size(0);
	auto num_atoms = pos.size(1);

	auto score_and_grad = scorer->interpolate(pos);
	auto score = score_and_grad.select(-1, 0).sum(-1);
	auto grad = score_and_grad.slice(-1, 1);

	ctx->saved_data["grad"] = grad;

	// penalty for atom clashing within the ligand.
	auto inner_vec = pos.unsqueeze(-2) - pos.unsqueeze(-3);
	auto inner_dist = inner_vec.norm(2, -1, true).clip(FLT_EPSILON);
	auto inner_mask = (inner_dist < TRIDS_CLASH_CUTOFF) & scorer->inner_nbr.unsqueeze(-1);
	auto inner_penalty = inner_mask * torch::log(inner_dist / TRIDS_CLASH_CUTOFF);
	inner_penalty = inner_penalty.sum({ -3, -2, -1 });

	ctx->saved_data["inner_vec"] = inner_vec;
	ctx->saved_data["inner_mask"] = inner_mask;
	ctx->saved_data["inner_dist"] = inner_dist;

	return at::stack({score, inner_penalty * TRIDS_CLASH_WEIGHT});
}

tensor_list TriFastScoring::backward(AutogradContext* ctx, tensor_list grad_outputs) {
	auto inter_force = ctx->saved_data["grad"].toTensor();

	auto inner_mask = ctx->saved_data["inner_mask"].toTensor();
	auto inner_dist = ctx->saved_data["inner_dist"].toTensor();
	auto inner_vec = ctx->saved_data["inner_vec"].toTensor();

	inner_vec /= inner_dist;
	auto inner_pulse = -2 * (inner_vec * inner_mask / inner_dist).nansum(-2);

	auto grad = inter_force + inner_pulse;
	return { grad, at::Tensor() };
}

at::Tensor TriAccurateScoring::forward(AutogradContext* ctx, at::Tensor pos,
								       TriScorer* scorer) {
	auto score = at::zeros({ pos.size(0), pos.size(1), 2 }, pos.options());
	auto grad = at::zeros_like(pos, pos.options());
	auto numel = pos.size(0) * pos.size(1);
	auto score_ptr = reinterpret_cast<float2*>(score.data_ptr<float>());
	auto grad_ptr = reinterpret_cast<float3*>(grad.data_ptr<float>());
	auto resi_ptr = scorer->rec->ptr.data_ptr<int>();
	
	auto rec_pos_ptr = reinterpret_cast<float3*>(scorer->rec->pos.data_ptr<float>());
	auto rec_vdw_ptr = scorer->rec->vdw.data_ptr<float>();
	auto rec_hd_ptr = scorer->rec->hd.data_ptr<bool>();
	auto rec_ha_ptr = scorer->rec->ha.data_ptr<bool>();

	auto lig_pos_ptr = reinterpret_cast<float3*>(pos.data_ptr<float>());
	auto lig_vdw_ptr = scorer->lig_vdw.data_ptr<float>();
	auto lig_hd_ptr = scorer->lig_hd.data_ptr<bool>();
	auto lig_ha_ptr = scorer->lig_ha.data_ptr<bool>();
	
	auto inner_nbr_ptr = scorer->inner_nbr.data_ptr<bool>();
	auto mu_ptr = scorer->mu.data_ptr<float>();
	auto var_ptr = scorer->var.data_ptr<float>();
	auto rho_ptr = scorer->data.data_ptr<float>();
	if (pos.is_cuda()) {
		auto stream = at::cuda::getCurrentCUDAStream();
		accurate_rtm_fw_cuda<< < BLOCKS(numel), THREADS, 0, stream >> >(
			lig_pos_ptr, lig_vdw_ptr, lig_hd_ptr, lig_ha_ptr, inner_nbr_ptr, 
			rec_pos_ptr, rec_vdw_ptr, rec_hd_ptr, rec_ha_ptr, 
			resi_ptr, mu_ptr, var_ptr, rho_ptr, 
			score_ptr, grad_ptr, scorer->cutoff2, 
			numel, pos.size(1), scorer->rec->num_res, scorer->mu.size(-1)
		);

	} else {


		accurate_rtm_fw_cpu(
			lig_pos_ptr, lig_vdw_ptr, lig_hd_ptr, lig_ha_ptr, inner_nbr_ptr,
			rec_pos_ptr, rec_vdw_ptr, rec_hd_ptr, rec_ha_ptr,
			resi_ptr, mu_ptr, var_ptr, rho_ptr,
			score_ptr, grad_ptr, scorer->cutoff2,
			numel, pos.size(1), scorer->rec->num_res, scorer->mu.size(-1)
		);
		
	}
	score = score.sum(-2);
	ctx->saved_data["grad"] = grad;
	return score;
}

tensor_list TriAccurateScoring::backward(AutogradContext* ctx, tensor_list grad_outputs) {
	auto grad = ctx->saved_data["grad"].toTensor();
	return { grad, at::Tensor() };
}


////////////////// VinaFastScoring ////////////////////////////////////////////////
at::Tensor VinaFastScoring::forward(AutogradContext* ctx, at::Tensor pos,
									VinaScorer* scorer) {
	auto score = at::zeros({ pos.size(0), pos.size(1), 2 }, pos.options());
	auto grad = at::zeros_like(pos, pos.options());
	auto numel = pos.size(0) * pos.size(1);
	auto score_ptr = reinterpret_cast<float2*>(score.data_ptr<float>());
	auto grad_ptr = reinterpret_cast<float3*>(grad.data_ptr<float>());
	auto pos_ptr = reinterpret_cast<float3*>(pos.data_ptr<float>());
	auto inner_nbr_ptr = scorer->inner_nbr.data_ptr<bool>();
	auto lig_hd_ptr = scorer->lig_hd.data_ptr<bool>();
	auto lig_ha_ptr = scorer->lig_ha.data_ptr<bool>();
	auto lig_hydro_ptr = scorer->lig_hydro.data_ptr<bool>();
	auto lig_vdw_ptr = scorer->lig_vdw.data_ptr<float>();
	auto lig_atom_ptr = scorer->lig_atom.data_ptr<int>();
	auto box_min_ptr = scorer->rec->box_min.data_ptr<float>(); 
	auto box_max_ptr = scorer->rec->box_max.data_ptr<float>();
	if (pos.is_cuda()) {
		auto stream = at::cuda::getCurrentCUDAStream();
		fast_vina_fw_cuda << < BLOCKS(numel), THREADS, 0, stream >> > (
			numel, pos.size(1), inner_nbr_ptr, scorer->cutoff * scorer->cutoff,
			pos_ptr, lig_hd_ptr, lig_ha_ptr,
			lig_hydro_ptr, lig_vdw_ptr, lig_atom_ptr,
			grad_ptr, score_ptr, scorer->rec->grid->texObj_for_kernel, scorer->border_strength,
			box_min_ptr, box_max_ptr, scorer->rec->grid->grid_len_inv
			);
	} else {
		auto ff_ptr = reinterpret_cast<float4*>(scorer->rec->grid->potential.data_ptr<float>());

		fast_vina_fw_cpu(
			numel, pos.size(1), inner_nbr_ptr, scorer->cutoff * scorer->cutoff,
			pos_ptr, lig_hd_ptr, lig_ha_ptr,
			lig_hydro_ptr, lig_vdw_ptr, lig_atom_ptr,
			grad_ptr, score_ptr, ff_ptr, scorer->rec->grid->grid_dim,
			scorer->border_strength, box_min_ptr, box_max_ptr, scorer->rec->grid->grid_len_inv
		);
	}
	score = score.sum(-2);
	ctx->saved_data["grad"] = grad;
	return score;
}

tensor_list VinaFastScoring::backward(AutogradContext* ctx, tensor_list grad_outputs) {
	auto grad = ctx->saved_data["grad"].toTensor();
	return { grad, at::Tensor() };
}


////////////////// VinaAccurateScoring ////////////////////////////////////////////////
at::Tensor VinaAccurateScoring::forward(AutogradContext* ctx, at::Tensor pos,
										VinaScorer* scorer) {
	auto score = at::zeros({ pos.size(0), pos.size(1), 2 }, pos.options());
	auto grad = at::zeros_like(pos, pos.options());
	auto numel = pos.size(0) * pos.size(1);
	auto score_ptr = reinterpret_cast<float2*>(score.data_ptr<float>());
	auto grad_ptr = reinterpret_cast<float3*>(grad.data_ptr<float>());
	auto lig_pos_ptr = reinterpret_cast<float3*>(pos.data_ptr<float>());
	auto inner_nbr_ptr = scorer->inner_nbr.data_ptr<bool>();
	
	auto lig_hd_ptr = scorer->lig_hd.data_ptr<bool>();
	auto lig_ha_ptr = scorer->lig_ha.data_ptr<bool>();
	auto lig_hydro_ptr = scorer->lig_hydro.data_ptr<bool>();
	auto lig_vdw_ptr = scorer->lig_vdw.data_ptr<float>();
	
	auto rec_pos_ptr = reinterpret_cast<float3*>(scorer->rec->pos.data_ptr<float>());
	auto rec_hd_ptr = scorer->rec->hd.data_ptr<bool>();
	auto rec_ha_ptr = scorer->rec->ha.data_ptr<bool>();
	auto rec_hydro_ptr = scorer->rec->hydro.data_ptr<bool>();
	auto rec_vdw_ptr = scorer->rec->vdw.data_ptr<float>();
									
	if (pos.is_cuda()) {
		auto stream = at::cuda::getCurrentCUDAStream();
		accurate_vina_fw_cuda << < BLOCKS(numel), THREADS, 0, stream >> > (
			numel, inner_nbr_ptr, scorer->cutoff * scorer->cutoff,
			lig_pos_ptr, lig_hd_ptr, lig_ha_ptr, lig_hydro_ptr, lig_vdw_ptr,
			pos.size(1), scorer->rec->pos.size(0), grad_ptr, score_ptr, 
			rec_pos_ptr, rec_hd_ptr, rec_ha_ptr, rec_hydro_ptr, rec_vdw_ptr
		);
	} else {
		accurate_vina_fw_cpu(
			numel, inner_nbr_ptr, scorer->cutoff * scorer->cutoff,
			lig_pos_ptr, lig_hd_ptr, lig_ha_ptr, lig_hydro_ptr, lig_vdw_ptr,
			pos.size(1), scorer->rec->pos.size(0), grad_ptr, score_ptr,
			rec_pos_ptr, rec_hd_ptr, rec_ha_ptr, rec_hydro_ptr, rec_vdw_ptr
		);
	}
	score = score.sum(-2);
	ctx->saved_data["grad"] = grad;
	return score;
}

tensor_list VinaAccurateScoring::backward(AutogradContext* ctx, tensor_list grad_outputs) {
	auto grad = ctx->saved_data["grad"].toTensor();
	return { grad, at::Tensor() };
}