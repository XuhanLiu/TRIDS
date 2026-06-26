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

#include "grid.cuh"
#include "constant.h"

__global__ 
void vina_make_grid_cuda(
    float4* potential, const float3* rec_pos, const bool* rec_hd, const bool* rec_ha, const bool* rec_hydro, const float* rec_vdw, 
    const int num_atoms, const bool* lig_hd, const bool* lig_ha, const bool* lig_hydro, const float* lig_vdw, const int num_types, 
    const int grid_numel, const int3 grid_dim, const float* grid_min, const float3 grid_len, const float cutoff2,
	const float3 bucket_len_inv, const int3 bucket_dim, const int* bucket
) {
    // Iterate over each grid point of the interpolation mesh
    int3 grid_idx;
    int num_layers = grid_dim.x * grid_dim.y;
	int num_nbr_layers = bucket_dim.y * bucket_dim.z;
    float3 grid_crd;
	int3 temp_crd;
	int dx, dy, dz;
	int x, y, z;
	int grid_i = blockIdx.x * blockDim.x + threadIdx.x;
	if (grid_i < grid_numel) {
	// for (int grid_i = blockIdx.x * blockDim.x + threadIdx.x; grid_i < num_grids; grid_i += gridDim.x * blockDim.x){
        grid_idx.x = grid_i % grid_dim.x;
		grid_idx.y = (grid_i % num_layers) / grid_dim.x;
		grid_idx.z = grid_i / num_layers;

        // Real spatial coordinates of interpolation grid point grid_i
		grid_crd.x = grid_len.x * grid_idx.x + grid_min[0];
		grid_crd.y = grid_len.y * grid_idx.y + grid_min[1];
		grid_crd.z = grid_len.z * grid_idx.z + grid_min[2];

		// Bucket index for the protein spatial partition grid containing this interpolation grid coordinate
		temp_crd.x = bucket_len_inv.x * grid_crd.x;
		temp_crd.y = bucket_len_inv.y * grid_crd.y;
		temp_crd.z = bucket_len_inv.z * grid_crd.z;

		int3 temp_idx = { (int)temp_crd.x, (int)temp_crd.y, (int)temp_crd.z };
        // Search for protein atoms in 3x3x3 region of protein partition grid
		dx = 0, dy = 0, dz = 0;
		for (int bucket_i = 0; bucket_i < 27; bucket_i++) {
			z = (temp_idx.z + dz - 1);
			y = (temp_idx.y + dy - 1);
			x = (temp_idx.x + dx - 1);
			
			if (x >= 0 && x < bucket_dim.x && y >= 0 && y < bucket_dim.y && z >= 0 && z < bucket_dim.z) {
				int bucket_idx = (x * num_nbr_layers + y * bucket_dim.z + z) * MAX_NBR_ATOM;
				// Iterate over protein atoms in each neighbor bucket
				if ( bucket_idx >= bucket_dim.x * bucket_dim.y * bucket_dim.z * MAX_NBR_ATOM) {
					printf("bucket idx out of range: %d", bucket_idx);
				}
				int num_nbrs = bucket[bucket_idx + 0];
				if ( num_nbrs >= MAX_NBR_ATOM) {
					printf("Neighbor in bucket out of range: %d", bucket_idx);
				}
				for (int i = 1; i < num_nbrs + 1; i++) {
					int atom_i = bucket[bucket_idx + i];
					if (atom_i >= num_atoms) {
						printf("Atom out of range: %d", atom_i);
					}
					float3 dr = { rec_pos[atom_i].x - grid_crd.x,
								rec_pos[atom_i].y - grid_crd.y, 
								rec_pos[atom_i].z - grid_crd.z };
					float dis2 = dr.x * dr.x + dr.y * dr.y + dr.z * dr.z;

					if (dis2 < cutoff2) {
						dis2 = sqrtf(dis2);
						float dr_abs_inv = 1.f / fmaxf(dis2, FLT_EPSILON);
						dr.x *= dr_abs_inv;
						dr.y *= dr_abs_inv;
						dr.z *= dr_abs_inv;
						// Assuming grid_i interpolation grid has 18 possible types, need to calculate the effect of protein atoms on it
						for (int type_i = 0; type_i < num_types; type_i++) {
							float dist = dis2 - rec_vdw[atom_i] - lig_vdw[type_i];
							float temp_record;
							float energy = 0.f;
							float frc_abs = 0.f;
							temp_record = GAUSS1 * expf(-GAUSS1_2 * dist * dist);
							energy += temp_record;
							frc_abs = 2.f * GAUSS1_2 * temp_record * dist;

							float dp = dist - GAUSS2_C;
							temp_record = GAUSS2 * expf(-GAUSS2_2 * dp * dp);
							energy += temp_record;
							frc_abs += 2.f * GAUSS2_2 * temp_record * dp;

							temp_record = REPULSION * dist * signbit(dist);
							energy += temp_record * dist;
							frc_abs += -2.f * temp_record;

							if ((rec_hydro[atom_i] & lig_hydro[type_i])) {
								temp_record = 1.f * HYDRO;
								energy += temp_record * (HYDRO_UA * signbit(dist - HYDRO_A) + HYDRO_UB * signbit(HYDRO_B - dist) + (HYDRO_RANGE * (dist - HYDRO_A) + HYDRO_UA) * signbit(HYDRO_A - dist) * signbit(dist - HYDRO_B));
								frc_abs += -temp_record * HYDRO_RANGE * signbit(HYDRO_A - dist) * signbit(dist - HYDRO_B);
							}

							if (((rec_hd[atom_i] & lig_ha[type_i]) | (rec_ha[atom_i] & lig_hd[type_i]))) {
								temp_record = 1.f * H_BOND;
								energy += temp_record * (H_BOND_UA * signbit(dist - H_BOND_A) + H_BOND_UB * signbit(H_BOND_B - dist) + (H_BOND_RANGE * (dist - H_BOND_A) + H_BOND_UA) * signbit(H_BOND_A - dist) * signbit(dist - H_BOND_B));
								frc_abs += -temp_record * H_BOND_RANGE * signbit(H_BOND_A - dist) * signbit(dist - H_BOND_B);
							}
							if (grid_i > grid_numel * num_types) {
								printf("Grid out of range: %d, %d, %d", grid_idx.x, grid_idx.y, grid_idx.z);
							}
							auto pid = type_i * grid_numel + grid_i;
							if (pid < 0 || pid >= num_types * grid_numel) {
								printf("Potential out of range: %d, %d", type_i, grid_i);
							}
							atomicAdd(&potential[pid].w, energy);
							atomicAdd(&potential[pid].x, -frc_abs * dr.x);
							atomicAdd(&potential[pid].y, -frc_abs * dr.y);
							atomicAdd(&potential[pid].z, -frc_abs * dr.z);
						}
					}
				}
			}
			dx = dx + 1;
			dy = dy + (((2 - dx) >> 31) & 0x00000001);
			dx = dx & ((dx - 3) >> 31);
			dz = dz + (((2 - dy) >> 31) & 0x00000001);
			dy = dy & ((dy - 3) >> 31);
		}
    }
}


void vina_make_grid_cpu(
    float4* potential, const float3* rec_pos, const bool* rec_hd, const bool* rec_ha, const bool* rec_hydro, const float* rec_vdw, 
    const int num_atoms, const bool* lig_hd, const bool* lig_ha, const bool* lig_hydro, const float* lig_vdw, const int64_t num_types, 
    const int grid_numel, const int3 grid_dim, const float* grid_min, const float3 grid_len, const float cutoff2,
	const float3 bucket_len_inv, const int3 bucket_dim, const int* bucket
) {
    // Iterate over each grid point of the interpolation mesh
    int3 grid_idx;
    int num_layers = grid_dim.x * grid_dim.y;
	int num_nbr_layers = bucket_dim.y * bucket_dim.z;
    float3 grid_crd;
	int3 temp_crd;
	int dx, dy, dz;
	int x, y, z;
	
	for (int grid_i = 0; grid_i < grid_numel; grid_i++) {
        grid_idx.x = grid_i % grid_dim.x;
		grid_idx.y = (grid_i % num_layers) / grid_dim.x;
		grid_idx.z = grid_i / num_layers;

        // Real spatial coordinates of interpolation grid point grid_i
		grid_crd.x = grid_len.x * grid_idx.x + grid_min[0];
		grid_crd.y = grid_len.y * grid_idx.y + grid_min[1];
		grid_crd.z = grid_len.z * grid_idx.z + grid_min[2];

		// Bucket index for the protein spatial partition grid containing this interpolation grid coordinate
		temp_crd.x = bucket_len_inv.x * grid_crd.x;
		temp_crd.y = bucket_len_inv.y * grid_crd.y;
		temp_crd.z = bucket_len_inv.z * grid_crd.z;

		int3 temp_idx = { (int)temp_crd.x,(int)temp_crd.y ,(int)temp_crd.z };
        // Search for protein atoms in 3x3x3 region of protein partition grid
		dx = 0, dy = 0, dz = 0;
		for (int bucket_i = 0; bucket_i < 27; bucket_i++) {
			z = (temp_idx.z + dz - 1);
			y = (temp_idx.y + dy - 1);
			x = (temp_idx.x + dx - 1);
			
			if (x >= 0 && x < bucket_dim.x && y >= 0 && y < bucket_dim.y && z >= 0 && z < bucket_dim.z) {
				int bucket_idx = (x * num_nbr_layers + y * bucket_dim.z + z) * MAX_NBR_ATOM;
				// Iterate over protein atoms in each neighbor bucket
				for (int i = 1; i < bucket[bucket_idx + 0] + 1; i++) {
					int atom_i = bucket[bucket_idx + i];
					if (atom_i >= num_atoms) {
						printf("Atom out of range: %d", atom_i);
					}
					float3 dr = { rec_pos[atom_i].x - grid_crd.x,
								rec_pos[atom_i].y - grid_crd.y, 
								rec_pos[atom_i].z - grid_crd.z };
					float dis2 = dr.x * dr.x + dr.y * dr.y + dr.z * dr.z;

					if (dis2 < cutoff2) {
						dis2 = sqrtf(dis2);
						float dr_abs_inv = 1.f / fmaxf(dis2, FLT_EPSILON);
						dr.x *= dr_abs_inv;
						dr.y *= dr_abs_inv;
						dr.z *= dr_abs_inv;
						// For grid_i with 18 possible types, calculate the effect of protein atoms on each type
						for (int type_i = 0; type_i < num_types; type_i++) {
							float dist = dis2 - rec_vdw[atom_i] - lig_vdw[type_i];
							float temp_record;
							float energy = 0.f;
							float frc_abs = 0.f;
							temp_record = GAUSS1 * expf(-GAUSS1_2 * dist * dist);
							energy += temp_record;
							frc_abs = 2.f * GAUSS1_2 * temp_record * dist;

							float dp = dist - GAUSS2_C;
							temp_record = GAUSS2 * expf(-GAUSS2_2 * dp * dp);
							energy += temp_record;
							frc_abs += 2.f * GAUSS2_2 * temp_record * dp;

							temp_record = REPULSION * dist * signbit(dist);
							energy += temp_record * dist;
							frc_abs += -2.f * temp_record;

							if ((rec_hydro[atom_i] & lig_hydro[type_i])) {
								temp_record = 1.f * HYDRO;
								energy += temp_record * (HYDRO_UA * signbit(dist - HYDRO_A) + HYDRO_UB * signbit(HYDRO_B - dist) + (HYDRO_RANGE * (dist - HYDRO_A) + HYDRO_UA) * signbit(HYDRO_A - dist) * signbit(dist - HYDRO_B));
								frc_abs += -temp_record * HYDRO_RANGE * signbit(HYDRO_A - dist) * signbit(dist - HYDRO_B);
							}

							if (((rec_hd[atom_i] & lig_ha[type_i]) | (rec_ha[atom_i] & lig_hd[type_i]))) {
								temp_record = 1.f * H_BOND;
								energy += temp_record * (H_BOND_UA * signbit(dist - H_BOND_A) + H_BOND_UB * signbit(H_BOND_B - dist) + (H_BOND_RANGE * (dist - H_BOND_A) + H_BOND_UA) * signbit(H_BOND_A - dist) * signbit(dist - H_BOND_B));
								frc_abs += -temp_record * H_BOND_RANGE * signbit(H_BOND_A - dist) * signbit(dist - H_BOND_B);
							}
							if (grid_i > grid_numel * num_types) {
								printf("Grid out of range: %d, %d, %d", grid_idx.x, grid_idx.y, grid_idx.z);
							}
							potential[type_i * grid_numel + grid_i].w += energy;
							potential[type_i * grid_numel + grid_i].x += -frc_abs * dr.x;
							potential[type_i * grid_numel + grid_i].y += -frc_abs * dr.y;
							potential[type_i * grid_numel + grid_i].z += -frc_abs * dr.z;
						}
					}
				}
			}
			dx = dx + 1;
			dy = dy + (((2 - dx) >> 31) & 0x00000001);
			dx = dx & ((dx - 3) >> 31);
			dz = dz + (((2 - dy) >> 31) & 0x00000001);
			dy = dy & ((dy - 3) >> 31);
		}
    }
}


Grid::Grid(int num_grids_, float cutoff_, at::TensorOptions options_) 
	: num_grids(num_grids_), cutoff(cutoff_), options(options_) {
	
	auto bool_opt = at::TensorOptions().device(options.device()).dtype(at::kByte);
	base_ha = at::tensor({ 0, 0, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0 }, bool_opt).to(at::kBool);
	base_hd = at::tensor({0, 0, 0, 1, 0, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0 }, bool_opt).to(at::kBool);
	base_hydro = at::tensor({1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0 }, bool_opt).to(at::kBool);
	base_vdw = at::tensor({ 1.9, 1.9, 1.8, 1.8, 1.8, 1.8, 1.7, 1.7, 1.7, 1.7, 2.0, 2.1, 1.5, 1.8, 2.0, 2.2, 1.2, -100.f }, options);
	
	box_len = { 30.f, 30.f, 30.f };
	grid_dim.x = num_grids;
	grid_dim.y = num_grids;
	grid_dim.z = num_grids;
	layer_numbers = grid_dim.x * grid_dim.y;
	grid_numbers = layer_numbers * grid_dim.z;
	Cuda_Texture_Initial();
}

void Grid::Cuda_Texture_Initial() {
	potential = at::zeros({18, 100, 100, 100, 4}, options);
	if (!options.device().is_cuda()) {
		return;
	}
	auto potential_ptr = reinterpret_cast<float4*>(potential.data_ptr<float>());
	// cudaMalloc((void**)&potential, sizeof(float4) * grid_numbers * num_types);
	cudaMalloc((void**)&texObj_for_kernel, sizeof(int64_t) * num_types);
	cudaArray_potential.resize(num_types);
	copyParams_potential.resize(num_types);
	texObj_potential.resize(num_types);

	cudaChannelFormatDesc channelDesc =
		cudaCreateChannelDesc(32, 32, 32, 32,
			cudaChannelFormatKindFloat);
	cudaExtent cuEx;
	cuEx.depth = grid_dim.z;
	cuEx.height = grid_dim.y;
	cuEx.width = grid_dim.x;
	cudaMemcpy3DParms temp_cudaMemcpy3DParms = { 0 };

	cudaTextureDesc texDesc;
	memset(&texDesc, 0, sizeof(texDesc));
	texDesc.addressMode[0] = cudaAddressModeBorder;
	texDesc.addressMode[1] = cudaAddressModeBorder;
	texDesc.addressMode[2] = cudaAddressModeBorder;
	texDesc.filterMode = cudaFilterModeLinear;// GPU built-in trilinear interpolation, works well, no need for manual implementation
	texDesc.readMode = cudaReadModeElementType;
	texDesc.borderColor[0] = 0.f;
	texDesc.borderColor[1] = 0.f;
	texDesc.borderColor[2] = 0.f;
	texDesc.borderColor[3] = 10.e6f;// Create high energy region at interpolation boundary
	texDesc.normalizedCoords = 0;

	cudaResourceDesc resDesc;
	memset(&resDesc, 0, sizeof(resDesc));
	resDesc.resType = cudaResourceTypeArray;
	for (int i = 0; i < num_types; i++) {
		cudaMalloc3DArray(&cudaArray_potential[i], &channelDesc, cuEx);

		float4* temp_ptr = &potential_ptr[i * grid_numbers];
		temp_cudaMemcpy3DParms.srcPtr = make_cudaPitchedPtr((void*)temp_ptr, cuEx.width * sizeof(float4), cuEx.width, cuEx.height);
		temp_cudaMemcpy3DParms.dstArray = cudaArray_potential[i];
		temp_cudaMemcpy3DParms.extent = cuEx;
		temp_cudaMemcpy3DParms.kind = cudaMemcpyDeviceToDevice;

		copyParams_potential[i] = temp_cudaMemcpy3DParms;

		resDesc.res.array.array = cudaArray_potential[i];
		cudaCreateTextureObject(&texObj_potential[i], &resDesc, &texDesc, NULL);
	}
	cudaMemcpy(texObj_for_kernel, &texObj_potential[0], sizeof(int64_t) * num_types, cudaMemcpyHostToDevice);
}

void Grid::Copy_Potential_To_Texture(){
	for (int i = 0; i < num_types; i++){
		cudaMemcpy3D(&copyParams_potential[i]);
	}
}

void Grid::make_grid(at::Tensor rec_pos, at::Tensor rec_hd, at::Tensor rec_ha, 
					 at::Tensor rec_hydro, at::Tensor rec_vdw, torch::Tensor box_min, Bucket& bucket) {
	potential.zero_();

	auto rec_crd = rec_pos - bucket.offset;
	auto grid_min = box_min - bucket.offset;
	auto neighbors = bucket.neighbors;
	grid_len = { box_len.x / (grid_dim.x - 1) , box_len.y / (grid_dim.y - 1) , box_len.z / (grid_dim.z - 1) };
	grid_len_inv = { 1.f / grid_len.x, 1.f / grid_len.y, 1.f / grid_len.z };

	float4* ff_ptr = reinterpret_cast<float4*>(potential.data_ptr<float>());
	float3* rec_crd_ptr = reinterpret_cast<float3*>(rec_crd.data_ptr<float>());
	auto rec_hd_ptr = rec_hd.data_ptr<bool>();
	auto rec_ha_ptr = rec_ha.data_ptr<bool>();
	auto rec_hydro_ptr = rec_hydro.data_ptr<bool>();
	auto rec_vdw_ptr = rec_vdw.data_ptr<float>();
	auto base_hd_ptr = base_hd.data_ptr<bool>();
	auto base_ha_ptr = base_ha.data_ptr<bool>();
	auto base_hydro_ptr = base_hydro.data_ptr<bool>();
	auto base_vdw_ptr = base_vdw.data_ptr<float>();

	int grid_numel = grid_dim.x * grid_dim.y * grid_dim.z;
	auto grid_min_ptr = grid_min.data_ptr<float>();
	auto bucket_ptr = neighbors.data_ptr<int>();

	// std::cout << grid_min << std::endl;
	if (rec_pos.is_cuda()) {
		// auto stream = at::cuda::getCurrentCUDAStream();
		vina_make_grid_cuda<< < BLOCKS(grid_numel), THREADS >> >(
			ff_ptr, rec_crd_ptr, rec_hd_ptr, rec_ha_ptr, rec_hydro_ptr, rec_vdw_ptr, 
			rec_pos.size(0), base_hd_ptr, base_ha_ptr, base_hydro_ptr, base_vdw_ptr, num_types, 
			grid_numel, grid_dim, grid_min_ptr, grid_len, bucket.cutoff2,
			bucket.bucket_len_inv, bucket.bucket_dim, bucket_ptr
		);
	} else { 
		vina_make_grid_cpu(
			ff_ptr, rec_crd_ptr, rec_hd_ptr, rec_ha_ptr, rec_hydro_ptr, rec_vdw_ptr, 
			rec_pos.size(0), base_hd_ptr, base_ha_ptr, base_hydro_ptr, base_vdw_ptr, num_types, 
			grid_numel, grid_dim, grid_min_ptr, grid_len, bucket.cutoff2,
			bucket.bucket_len_inv, bucket.bucket_dim, bucket_ptr
		);
	}

    if (options.device().is_cuda()) {
        Copy_Potential_To_Texture();
    }
}