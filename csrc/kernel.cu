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

#include <cfloat>
#include <ATen/Parallel.h>
#include "kernel.cuh"
#include "constant.h"

__host__ __device__ inline float3 sub(float3 a, float3 b) {
	return { a.x - b.x, a.y - b.y, a.z - b.z };
}

__host__ __device__ inline float norm2(float3 dr) {
	return dr.x * dr.x + dr.y * dr.y + dr.z * dr.z;
}

__host__ __device__ inline void apply_force(float3& force, float3 dr, float frc_abs) {
	force.x -= frc_abs * dr.x;
	force.y -= frc_abs * dr.y;
	force.z -= frc_abs * dr.z;
}

__host__ __device__ inline void vina_accumulate_pair(
	float dist, float rij, float3 dr,
	bool hydro_mask, bool hbond_mask,
	float& energy, float3& force
) {
	float temp = GAUSS1 * expf(-GAUSS1_2 * dist * dist);
	energy += temp;
	float frc_abs = 2.f * GAUSS1_2 * temp * dist;

	const float dp = dist - GAUSS2_C;
	temp = GAUSS2 * expf(-GAUSS2_2 * dp * dp);
	energy += temp;
	frc_abs += 2.f * GAUSS2_2 * temp * dp;

	if (dist < 0) {
		temp = REPULSION * dist;
		energy += temp * dist;
		frc_abs -= 2.f * temp;
	}

	if (hydro_mask) {
		if (dist <= HYDRO_A) {
			energy += HYDRO * HYDRO_UA;
		} else if (dist >= HYDRO_B) {
			energy += HYDRO * HYDRO_UB;
		} else {
			energy += HYDRO * (HYDRO_RANGE * (dist - HYDRO_A) + HYDRO_UA);
			frc_abs -= HYDRO * HYDRO_RANGE;
		}
	}

	if (hbond_mask) {
		if (dist <= H_BOND_A) {
			energy += H_BOND * H_BOND_UA;
		} else if (dist >= H_BOND_B) {
			energy += H_BOND * H_BOND_UB;
		} else {
			energy += H_BOND * (H_BOND_RANGE * (dist - H_BOND_A) + H_BOND_UA);
			frc_abs -= H_BOND * H_BOND_RANGE;
		}
	}

	frc_abs /= (rij + FLT_EPSILON);
	apply_force(force, dr, frc_abs);
}

__host__ __device__ inline void accurate_rtm_compute(
	int i, int batch_i, int atom_i,
	const float3* lig_pos, const float* lig_vdw, const bool* lig_hd, const bool* lig_ha,
	const bool* inner_nbr, const float3* rec_pos, const float* rec_vdw,
	const bool* rec_hd, const bool* rec_ha, const int* resi_id,
	const float* mu, const float* var, const float* rho,
	float cutoff2, int num_atoms, int num_resis, int num_gauss,
	float2& local_score, float3& local_grad
) {
	const float3 my_pos = lig_pos[i];
	const float my_vdw = lig_vdw[atom_i];
	const bool my_hd = lig_hd[atom_i];
	const bool my_ha = lig_ha[atom_i];

	float rij, rij_hb2, rij_inv, rec_vdw_j, clash, temp;
	bool rec_hd_j, rec_ha_j;
	float3 dr, dr_hb, dr_curr;

	for (int resi_i = 0; resi_i < num_resis; resi_i++) {
		const int resi_start = resi_id[resi_i];
		const int resi_end = resi_id[resi_i + 1];
		rij = FLT_MAX;
		rij_hb2 = FLT_MAX;

		for (int k = resi_start; k < resi_end; k++) {
			const float3 rec_k = rec_pos[k];
			dr_curr = sub(my_pos, rec_k);
			const float rij_curr2 = norm2(dr_curr);
			if (rij_curr2 < rij) {
				rij = rij_curr2;
				dr = dr_curr;
				rec_vdw_j = rec_vdw[k];
				rec_hd_j = rec_hd[k];
				rec_ha_j = rec_ha[k];
			}

			if (rij_curr2 < rij_hb2 && ((my_ha & rec_hd[k]) | (my_hd & rec_ha[k]))) {
				rij_hb2 = rij_curr2;
				dr_hb = dr_curr;
			}
		}

		if (rij < cutoff2) {
			float frc_abs = 0.f;
			rij = sqrtf(rij);
			rij_inv = 1.f / fmaxf(FLT_EPSILON, rij);
			clash = my_vdw + rec_vdw_j;
			const float dp = rij - clash - GAUSS2_C;
			temp = expf(-GAUSS2_2 * dp * dp) * TRIDS_LONG_WEIGHT;
			local_score.y -= temp;
			frc_abs += 2.f * GAUSS2_2 * temp * dp;

			if (rij < TRIDS_MDN_CUTOFF) {
				if ((my_ha & rec_hd_j) | (my_hd & rec_ha_j)) {
					clash = TRIDS_HBOND_MIN;
				}
				if (rij < clash) {
					temp = TRIDS_CLASH_WEIGHT * logf(rij / clash);
					local_score.x -= temp;
					local_score.y -= temp;
					frc_abs -= TRIDS_CLASH_WEIGHT * rij_inv;
				}

				const int gi_base = atom_i * (num_gauss * num_resis) + resi_i * num_gauss;
				float score_mdn = 0.f;
				float frc_mdn = 0.f;

#ifdef __CUDA_ARCH__
#pragma unroll 10
#endif
				for (int gauss_i = 0; gauss_i < num_gauss; gauss_i++) {
					const int gi = gi_base + gauss_i;
					const float mu_val = mu[gi];
					const float var_val = var[gi];
					const float rho_val = rho[gi];
					const float diff = rij - mu_val;
					temp = expf(-diff * diff * var_val * 0.5f + rho_val);
					score_mdn -= temp;
					frc_mdn += temp * diff * var_val;
				}
				local_score.x += score_mdn;

				if (rij_hb2 < TRIDS_HBOND_MAX2 && TRIDS_HBOND_WEIGHT > 0) {
					rij = sqrtf(rij_hb2);
					const float diff = rij - TRIDS_HBOND_AVG;
					temp = expf(-TRIDS_HBOND_SIGMA2_INV * diff * diff * 0.5f) * TRIDS_HBOND_WEIGHT;
					local_score.y -= temp;
					const float frc_hbond = TRIDS_HBOND_SIGMA2_INV * diff * temp;
					apply_force(local_grad, dr_hb, -frc_hbond);
				} else {
					local_score.y += score_mdn;
					frc_abs += frc_mdn;
				}
			}
			frc_abs *= -rij_inv;
			apply_force(local_grad, dr, frc_abs);
		}
	}

	for (int atom_j = 0; atom_j < num_atoms; atom_j++) {
		if (!inner_nbr[atom_i * num_atoms + atom_j]) {
			continue;
		}
		const bool lig_hd_j = lig_hd[atom_j];
		const bool lig_ha_j = lig_ha[atom_j];

		if ((my_ha & lig_hd_j) | (my_hd & lig_ha_j)) {
			clash = TRIDS_HBOND_MIN;
		} else {
			clash = my_vdw + lig_vdw[atom_j];
		}

		const int j = batch_i * num_atoms + atom_j;
		const float3 lig_j = lig_pos[j];
		dr = sub(my_pos, lig_j);

		const float rij2 = norm2(dr);
		const float clash2 = clash * clash;
		if (rij2 < clash2) {
			rij_inv = rsqrtf(rij2);
			temp = TRIDS_CLASH_WEIGHT * logf(rij2 / clash2) * 0.5f;
			if (atom_j > atom_i) {
				local_score.x -= temp;
				local_score.y -= temp;
			}
			float frc = TRIDS_CLASH_WEIGHT * rij_inv;
			apply_force(local_grad, dr, frc);
		}
	}
}

__host__ __device__ inline void vina_inner_pairs(
	int batch_i, int atom_i, float3 my_pos,
	const bool* inner_nbr, float cutoff2,
	const float3* lig_pos, const bool* lig_hd, const bool* lig_ha,
	const bool* lig_hydro, const float* lig_vdw,
	int num_atoms, float my_vdw, bool my_hd, bool my_ha, bool my_hydro,
	float2& local_energy, float3& local_force
) {
	float rij;
	float3 dr;
	float energy_sink = 0.f;

	for (int atom_j = 0; atom_j < num_atoms; atom_j++) {
		if (!inner_nbr[atom_i * num_atoms + atom_j]) {
			continue;
		}
		const int j = batch_i * num_atoms + atom_j;
		const float3 lig_j = lig_pos[j];
		dr = sub(my_pos, lig_j);
		rij = norm2(dr);
		if (rij < cutoff2) {
			rij = sqrtf(rij);
			const float dist = rij - my_vdw - lig_vdw[atom_j];
			float& pair_energy = (atom_j > atom_i) ? local_energy.y : energy_sink;
			vina_accumulate_pair(
				dist, rij, dr,
				my_hydro & lig_hydro[atom_j],
				(my_hd & lig_ha[atom_j]) | (my_ha & lig_hd[atom_j]),
				pair_energy, local_force
			);
		}
	}
}

__host__ __device__ inline void accurate_vina_compute(
	int i, int batch_i, int atom_i,
	const bool* inner_nbr, float cutoff2,
	const float3* lig_pos, const bool* lig_hd, const bool* lig_ha,
	const bool* lig_hydro, const float* lig_vdw,
	int num_lig_atoms, int num_rec_atoms,
	const float3* rec_pos, const bool* rec_hd, const bool* rec_ha,
	const bool* rec_hydro, const float* rec_vdw,
	float2& local_energy, float3& local_force
) {
	const float3 my_pos = lig_pos[i];
	const bool my_hd = lig_hd[atom_i];
	const bool my_ha = lig_ha[atom_i];
	const bool my_hydro = lig_hydro[atom_i];
	const float my_vdw = lig_vdw[atom_i];

	float rij;
	float3 dr;

	for (int atom_j = 0; atom_j < num_rec_atoms; atom_j++) {
		const float3 rec_j = rec_pos[atom_j];
		dr = sub(my_pos, rec_j);
		rij = norm2(dr);
		if (rij < cutoff2) {
			const float rec_vdw_j = rec_vdw[atom_j];
			rij = sqrtf(rij);
			const float dist = rij - my_vdw - rec_vdw_j;
			vina_accumulate_pair(
				dist, rij, dr,
				my_hydro & rec_hydro[atom_j],
				(my_hd & rec_ha[atom_j]) | (my_ha & rec_hd[atom_j]),
				local_energy.x, local_force
			);
		}
	}

	vina_inner_pairs(
		batch_i, atom_i, my_pos,
		inner_nbr, cutoff2,
		lig_pos, lig_hd, lig_ha, lig_hydro, lig_vdw,
		num_lig_atoms, my_vdw, my_hd, my_ha, my_hydro,
		local_energy, local_force
	);
}

__host__ __device__ inline void fast_vina_border_penalty(
	float3 my_pos, float border_strength,
	const float* box_min, const float* box_max,
	float2& local_energy, float3& local_force
) {
	const float bmin_x = box_min[0];
	const float bmin_y = box_min[1];
	const float bmin_z = box_min[2];
	const float bmax_x = box_max[0];
	const float bmax_y = box_max[1];
	const float bmax_z = box_max[2];

	float3 dr;
	dr.x = fmaxf(0.f, bmin_x - my_pos.x);
	dr.y = fmaxf(0.f, bmin_y - my_pos.y);
	dr.z = fmaxf(0.f, bmin_z - my_pos.z);
	local_force.x += border_strength * dr.x;
	local_force.y += border_strength * dr.y;
	local_force.z += border_strength * dr.z;
	local_energy.x = 0.5f * border_strength * (dr.x * dr.x + dr.y * dr.y + dr.z * dr.z);

	dr.x = fmaxf(0.f, my_pos.x - bmax_x);
	dr.y = fmaxf(0.f, my_pos.y - bmax_y);
	dr.z = fmaxf(0.f, my_pos.z - bmax_z);
	local_force.x -= border_strength * dr.x;
	local_force.y -= border_strength * dr.y;
	local_force.z -= border_strength * dr.z;
	local_energy.x += 0.5f * border_strength * (dr.x * dr.x + dr.y * dr.y + dr.z * dr.z);
}

inline float4 sample_potential_cpu(const float4* grid, int3 dim, float x, float y, float z) {
	auto sample = [&](int ix, int iy, int iz) -> float4 {
		if (ix < 0 || iy < 0 || iz < 0 || ix >= dim.x || iy >= dim.y || iz >= dim.z) {
			return { 0.f, 0.f, 0.f, 1.e7f };
		}
		return grid[iz * dim.x * dim.y + iy * dim.x + ix];
	};

	const float px = x - 0.5f;
	const float py = y - 0.5f;
	const float pz = z - 0.5f;
	const int x0 = static_cast<int>(floorf(px));
	const int y0 = static_cast<int>(floorf(py));
	const int z0 = static_cast<int>(floorf(pz));
	const float tx = px - static_cast<float>(x0);
	const float ty = py - static_cast<float>(y0);
	const float tz = pz - static_cast<float>(z0);

	float4 result = { 0.f, 0.f, 0.f, 0.f };
	for (int dz = 0; dz <= 1; dz++) {
		for (int dy = 0; dy <= 1; dy++) {
			for (int dx = 0; dx <= 1; dx++) {
				const float wx = (dx == 0) ? (1.f - tx) : tx;
				const float wy = (dy == 0) ? (1.f - ty) : ty;
				const float wz = (dz == 0) ? (1.f - tz) : tz;
				const float w = wx * wy * wz;
				const float4 v = sample(x0 + dx, y0 + dy, z0 + dz);
				result.x += w * v.x;
				result.y += w * v.y;
				result.z += w * v.z;
				result.w += w * v.w;
			}
		}
	}
	return result;
}

void accurate_rtm_fw_cpu(
	const float3* lig_pos, const float* lig_vdw, const bool* lig_hd, const bool* lig_ha, const bool* inner_nbr,
	const float3* rec_pos, const float* rec_vdw, const bool* rec_hd, const bool* rec_ha, const int* resi_id,
	const float* mu, const float* var, const float* rho, float2* score, float3* grad,
	float cutoff2, int numel, int num_atoms, int num_resis, int num_gauss
) {
	at::parallel_for(0, numel, 1, [&](int64_t begin, int64_t end) {
		for (int64_t i = begin; i < end; ++i) {
			const int batch_i = static_cast<int>(i / num_atoms);
			const int atom_i = static_cast<int>(i % num_atoms);

			float2 local_score = { 0.f, 0.f };
			float3 local_grad = { 0.f, 0.f, 0.f };
			accurate_rtm_compute(
				static_cast<int>(i), batch_i, atom_i,
				lig_pos, lig_vdw, lig_hd, lig_ha, inner_nbr,
				rec_pos, rec_vdw, rec_hd, rec_ha, resi_id,
				mu, var, rho, cutoff2, num_atoms, num_resis, num_gauss,
				local_score, local_grad
			);

			score[i].x += local_score.x;
			score[i].y += local_score.y;
			grad[i].x += local_grad.x;
			grad[i].y += local_grad.y;
			grad[i].z += local_grad.z;
		}
	});
}

void accurate_vina_fw_cpu(
	int numel, const bool* inner_nbr, float cutoff2,
	const float3* lig_pos, const bool* lig_hd,
	const bool* lig_ha, const bool* lig_hydro, const float* lig_vdw,
	int num_lig_atoms, int num_rec_atoms, float3* frc, float2* energy,
	const float3* rec_pos, const bool* rec_hd,
	const bool* rec_ha, const bool* rec_hydro, const float* rec_vdw
) {
	at::parallel_for(0, numel, 1, [&](int64_t begin, int64_t end) {
		for (int64_t i = begin; i < end; ++i) {
			const int batch_i = static_cast<int>(i / num_lig_atoms);
			const int atom_i = static_cast<int>(i % num_lig_atoms);

			float2 local_energy = { 0.f, 0.f };
			float3 local_force = { 0.f, 0.f, 0.f };
			accurate_vina_compute(
				static_cast<int>(i), batch_i, atom_i,
				inner_nbr, cutoff2,
				lig_pos, lig_hd, lig_ha, lig_hydro, lig_vdw,
				num_lig_atoms, num_rec_atoms,
				rec_pos, rec_hd, rec_ha, rec_hydro, rec_vdw,
				local_energy, local_force
			);

			energy[i].x += local_energy.x;
			energy[i].y += local_energy.x + local_energy.y;
			frc[i].x += local_force.x;
			frc[i].y += local_force.y;
			frc[i].z += local_force.z;
		}
	});
}

void fast_vina_fw_cpu(
	int numel, int num_atoms, const bool* inner_nbr, float cutoff2,
	const float3* lig_pos, const bool* lig_hd, const bool* lig_ha,
	const bool* lig_hydro, const float* lig_vdw, const int* lig_atoms,
	float3* frc, float2* energy, const float4* potential, int3 grid_dim,
	float border_strength, const float* box_min, const float* box_max, float3 mesh_grid_len_inv
) {
	const int grid_numel = grid_dim.x * grid_dim.y * grid_dim.z;

	at::parallel_for(0, numel, 1, [&](int64_t begin, int64_t end) {
		for (int64_t i = begin; i < end; ++i) {
			const int batch_i = static_cast<int>(i / num_atoms);
			const int atom_i = static_cast<int>(i % num_atoms);

			const float3 my_pos = lig_pos[i];
			const bool my_hd = lig_hd[atom_i];
			const bool my_ha = lig_ha[atom_i];
			const bool my_hydro = lig_hydro[atom_i];
			const float my_vdw = lig_vdw[atom_i];

			float2 local_energy = { 0.f, 0.f };
			float3 local_force = { 0.f, 0.f, 0.f };

			const int lig_atom = lig_atoms[atom_i];
			if (lig_atom < HYDROGEN_ATOM_TYPE_SERIAL) {
				fast_vina_border_penalty(my_pos, border_strength, box_min, box_max, local_energy, local_force);

				const float bmin_x = box_min[0];
				const float bmin_y = box_min[1];
				const float bmin_z = box_min[2];
				float3 idx;
				idx.x = (my_pos.x - bmin_x) * mesh_grid_len_inv.x;
				idx.y = (my_pos.y - bmin_y) * mesh_grid_len_inv.y;
				idx.z = (my_pos.z - bmin_z) * mesh_grid_len_inv.z;
				const float4 ans = sample_potential_cpu(
					potential + lig_atom * grid_numel, grid_dim, idx.x + 0.5f, idx.y + 0.5f, idx.z + 0.5f
				);

				local_energy.x += ans.w;
				local_force.x += ans.x;
				local_force.y += ans.y;
				local_force.z += ans.z;
			}

			vina_inner_pairs(
				batch_i, atom_i, my_pos,
				inner_nbr, cutoff2,
				lig_pos, lig_hd, lig_ha, lig_hydro, lig_vdw,
				num_atoms, my_vdw, my_hd, my_ha, my_hydro,
				local_energy, local_force
			);

			energy[i].x = local_energy.x;
			energy[i].y = local_energy.x + local_energy.y;
			frc[i].x = local_force.x;
			frc[i].y = local_force.y;
			frc[i].z = local_force.z;
		}
	});
}

__global__ void accurate_rtm_fw_cuda(
	const float3* lig_pos, const float* lig_vdw, const bool* lig_hd, const bool* lig_ha, const bool* inner_nbr,
	const float3* rec_pos, const float* rec_vdw, const bool* rec_hd, const bool* rec_ha, const int* resi_id,
	const float* mu, const float* var, const float* rho, float2* score, float3* grad,
	const float cutoff2, const int numel, const int num_atoms, const int num_resis, const int num_gauss
) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= numel) {
		return;
	}

	const int batch_i = i / num_atoms;
	const int atom_i = i % num_atoms;

	float2 local_score = { 0.f, 0.f };
	float3 local_grad = { 0.f, 0.f, 0.f };
	accurate_rtm_compute(
		i, batch_i, atom_i,
		lig_pos, lig_vdw, lig_hd, lig_ha, inner_nbr,
		rec_pos, rec_vdw, rec_hd, rec_ha, resi_id,
		mu, var, rho, cutoff2, num_atoms, num_resis, num_gauss,
		local_score, local_grad
	);

	score[i].x += local_score.x;
	score[i].y += local_score.y;
	grad[i].x += local_grad.x;
	grad[i].y += local_grad.y;
	grad[i].z += local_grad.z;
}

__global__ void accurate_vina_fw_cuda(
	const int numel, const bool* inner_nbr, const float cutoff2,
	const float3* lig_pos, const bool* lig_hd,
	const bool* lig_ha, const bool* lig_hydro, const float* lig_vdw,
	const int num_lig_atoms, const int num_rec_atoms, float3* frc, float2* energy,
	const float3* rec_pos, const bool* rec_hd,
	const bool* rec_ha, const bool* rec_hydro, const float* rec_vdw
) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= numel) {
		return;
	}

	const int batch_i = i / num_lig_atoms;
	const int atom_i = i % num_lig_atoms;

	float2 local_energy = { 0.f, 0.f };
	float3 local_force = { 0.f, 0.f, 0.f };
	accurate_vina_compute(
		i, batch_i, atom_i,
		inner_nbr, cutoff2,
		lig_pos, lig_hd, lig_ha, lig_hydro, lig_vdw,
		num_lig_atoms, num_rec_atoms,
		rec_pos, rec_hd, rec_ha, rec_hydro, rec_vdw,
		local_energy, local_force
	);

	energy[i].x += local_energy.x;
	energy[i].y += local_energy.x + local_energy.y;
	frc[i].x += local_force.x;
	frc[i].y += local_force.y;
	frc[i].z += local_force.z;
}

__global__ void fast_vina_fw_cuda(
	const int numel, const int num_atoms, const bool* inner_nbr, const float cutoff2,
	const float3* lig_pos, const bool* lig_hd, const bool* lig_ha,
	const bool* lig_hydro, const float* lig_vdw, const int* lig_atoms,
	float3* frc, float2* energy, const int64_t* meshs, const float border_strenth,
	const float* box_min, const float* box_max, const float3 mesh_grid_len_inv
) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= numel) {
		return;
	}

	const int batch_i = i / num_atoms;
	const int atom_i = i % num_atoms;

	const float3 my_pos = lig_pos[i];
	const bool my_hd = lig_hd[atom_i];
	const bool my_ha = lig_ha[atom_i];
	const bool my_hydro = lig_hydro[atom_i];
	const float my_vdw = lig_vdw[atom_i];

	float2 local_energy = { 0.f, 0.f };
	float3 local_force = { 0.f, 0.f, 0.f };

	const int lig_atom = lig_atoms[atom_i];
	if (lig_atom < HYDROGEN_ATOM_TYPE_SERIAL) {
		fast_vina_border_penalty(my_pos, border_strenth, box_min, box_max, local_energy, local_force);

		const float bmin_x = box_min[0];
		const float bmin_y = box_min[1];
		const float bmin_z = box_min[2];
		float3 idx;
		idx.x = (my_pos.x - bmin_x) * mesh_grid_len_inv.x;
		idx.y = (my_pos.y - bmin_y) * mesh_grid_len_inv.y;
		idx.z = (my_pos.z - bmin_z) * mesh_grid_len_inv.z;
		const int64_t mesh = meshs[lig_atom];
		const float4 ans = tex3D<float4>(mesh, idx.x + 0.5f, idx.y + 0.5f, idx.z + 0.5f);

		local_energy.x += ans.w;
		local_force.x += ans.x;
		local_force.y += ans.y;
		local_force.z += ans.z;
	}

	vina_inner_pairs(
		batch_i, atom_i, my_pos,
		inner_nbr, cutoff2,
		lig_pos, lig_hd, lig_ha, lig_hydro, lig_vdw,
		num_atoms, my_vdw, my_hd, my_ha, my_hydro,
		local_energy, local_force
	);

	energy[i].x = local_energy.x;
	energy[i].y = local_energy.x + local_energy.y;
	frc[i].x = local_force.x;
	frc[i].y = local_force.y;
	frc[i].z = local_force.z;
}
