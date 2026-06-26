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

#include <cuda_runtime.h>

void accurate_rtm_fw_cpu(
	const float3* lig_pos, const float* lig_vdw, const bool* lig_hd, const bool* lig_ha, const bool* inner_nbr,
	const float3* rec_pos, const float* rec_vdw, const bool* rec_hd, const bool* rec_ha, const int* resi_id,
	const float* mu, const float* var, const float* rho, float2* score, float3* grad,
	float cutoff2, int numel, int num_atoms, int num_resis, int num_gauss
);

void accurate_vina_fw_cpu(
	int numel, const bool* inner_nbr, float cutoff2,
	const float3* lig_pos, const bool* lig_hd,
	const bool* lig_ha, const bool* lig_hydro, const float* lig_vdw,
	int num_lig_atoms, int num_rec_atoms, float3* frc, float2* energy,
	const float3* rec_pos, const bool* rec_hd,
	const bool* rec_ha, const bool* rec_hydro, const float* rec_vdw
);

void fast_vina_fw_cpu(
	int numel, int num_atoms, const bool* inner_nbr, float cutoff2,
	const float3* lig_pos, const bool* lig_hd, const bool* lig_ha,
	const bool* lig_hydro, const float* lig_vdw, const int* lig_atoms,
	float3* frc, float2* energy, const float4* potential, int3 grid_dim,
	float border_strength, const float* box_min, const float* box_max, float3 mesh_grid_len_inv
);

__global__ void accurate_rtm_fw_cuda(
	const float3* lig_pos, const float* lig_vdw, const bool* lig_hd, const bool* lig_ha, const bool* inner_nbr,
	const float3* rec_pos, const float* rec_vdw, const bool* rec_hd, const bool* rec_ha, const int* resi_id,
	const float* mu, const float* var, const float* rho, float2* score, float3* grad,
	float cutoff2, int numel, int num_atoms, int num_resis, int num_gauss
);

__global__ void accurate_vina_fw_cuda(
	int numel, const bool* inner_nbr, float cutoff2,
	const float3* lig_pos, const bool* lig_hd,
	const bool* lig_ha, const bool* lig_hydro, const float* lig_vdw,
	int num_lig_atoms, int num_rec_atoms, float3* frc, float2* energy,
	const float3* rec_pos, const bool* rec_hd,
	const bool* rec_ha, const bool* rec_hydro, const float* rec_vdw
);

__global__ void fast_vina_fw_cuda(
	int numel, int num_atoms, const bool* inner_nbr, float cutoff2,
	const float3* lig_pos, const bool* lig_hd, const bool* lig_ha,
	const bool* lig_hydro, const float* lig_vdw, const int* lig_atoms,
	float3* frc, float2* energy, const int64_t* meshs, float border_strenth,
	const float* box_min, const float* box_max, float3 mesh_grid_len_inv
);
