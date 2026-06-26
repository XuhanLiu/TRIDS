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

#include "surface.cuh"


__global__ void Calculate_Origin_Grid_Occupation_Device (
	const int atom_numbers, const float3* crd, const float* radius,
	const int num_grids, const int num_layers, const int3 grid_dimension, const float grid_len, const float grid_len_inv,
	const int extending_numbers, int* grid_occupation
) {
	const int total_extending_numbers_minus_one = extending_numbers * 2;
	const int total_extending_numbers = extending_numbers * 2 + 1;
	const int total_extending_grid_numbers = total_extending_numbers * total_extending_numbers * total_extending_numbers;
	// Loop over each atom and add its contribution to the grid
	for (int atom_i = blockIdx.x * blockDim.x + threadIdx.x; atom_i < atom_numbers; atom_i += gridDim.x * blockDim.x) {
		float3 atom_crd = crd[atom_i];
		float atom_radius2 = radius[atom_i] * radius[atom_i];
		float3 atom_fraction_crd = { atom_crd.x * grid_len_inv,atom_crd.y * grid_len_inv,atom_crd.z * grid_len_inv };
		int3 atom_in_grid_serial = { (int)atom_fraction_crd.x, (int)atom_fraction_crd.y , (int)atom_fraction_crd.z };

		int dx = 0, dy = 0, dz = 0;
		int x, y, z;

		// Note: ensure traversal stays within box boundaries
		for (int neighbor_grid_i = 0; neighbor_grid_i < total_extending_grid_numbers; neighbor_grid_i++) {
			z = (atom_in_grid_serial.z + dz - extending_numbers);
			y = (atom_in_grid_serial.y + dy - extending_numbers);
			x = (atom_in_grid_serial.x + dx - extending_numbers);

			//do sth
			float3 grid_crd = { grid_len * x,grid_len * y,grid_len * z };
			float3 dr = { grid_crd.x - atom_crd.x,grid_crd.y - atom_crd.y ,grid_crd.z - atom_crd.z };
			float dr2 = dr.x * dr.x + dr.y * dr.y + dr.z * dr.z;
			if (dr2 < atom_radius2) {
				int neighbor_grid_serial = z * num_layers + y * grid_dimension.x + x;
				grid_occupation[neighbor_grid_serial] = 1;
			}
			dx = dx + 1;
			dy = dy + (((total_extending_numbers_minus_one - dx) >> 31) & 0x00000001);
			dx = dx & ((dx - total_extending_numbers) >> 31);
			dz = dz + (((total_extending_numbers_minus_one - dy) >> 31) & 0x00000001);
			dy = dy & ((dy - total_extending_numbers) >> 31);
		}
	}
}

__global__ void Calculate_Atom_Near_Surface_Device (
	const int atom_numbers, const float3* crd, const float* radius, int* is_near_surface,
	const int num_grids, const int num_layers, const int3 grid_dimension, const float grid_len, const float grid_len_inv,
	const int extending_numbers, const int* surface
) {
	const int total_extending_numbers_minus_one = extending_numbers * 2;
	const int total_extending_numbers = extending_numbers * 2 + 1;
	const int total_extending_grid_numbers = total_extending_numbers * total_extending_numbers * total_extending_numbers;
	// Loop over each atom and add its contribution to the grid
	for (int atom_i = blockIdx.x * blockDim.x + threadIdx.x; atom_i < atom_numbers; atom_i += gridDim.x * blockDim.x) {
		float3 atom_crd = crd[atom_i];
		float atom_radius2 = radius[atom_i] * radius[atom_i];
		float3 atom_fraction_crd = { atom_crd.x * grid_len_inv,atom_crd.y * grid_len_inv,atom_crd.z * grid_len_inv };
		int3 atom_in_grid_serial = { (int)atom_fraction_crd.x, (int)atom_fraction_crd.y, (int)atom_fraction_crd.z };

		int dx = 0, dy = 0, dz = 0;
		int x, y, z;

		int temp_near_surface = 0;
		// Note: ensure traversal stays within box boundaries
		for (int neighbor_grid_i = 0; neighbor_grid_i < total_extending_grid_numbers; neighbor_grid_i++) {
			z = (atom_in_grid_serial.z + dz - extending_numbers);
			y = (atom_in_grid_serial.y + dy - extending_numbers);
			x = (atom_in_grid_serial.x + dx - extending_numbers);

			//do sth
			float3 grid_crd = { grid_len * x,grid_len * y,grid_len * z };
			float3 dr = { grid_crd.x - atom_crd.x,grid_crd.y - atom_crd.y ,grid_crd.z - atom_crd.z };
			float dr2 = dr.x * dr.x + dr.y * dr.y + dr.z * dr.z;
			if (dr2 < atom_radius2) {
				int neighbor_grid_serial = z * num_layers + y * grid_dimension.x + x;
				if (surface[neighbor_grid_serial] == 1) {
					temp_near_surface = 1;
					break;
				}
			}
			dx = dx + 1;
			dy = dy + (((total_extending_numbers_minus_one - dx) >> 31) & 0x00000001);
			dx = dx & ((dx - total_extending_numbers) >> 31);
			dz = dz + (((total_extending_numbers_minus_one - dy) >> 31) & 0x00000001);
			dy = dy & ((dy - total_extending_numbers) >> 31);
		}
		is_near_surface[atom_i] = temp_near_surface;
	}
}

__global__ void Smooth_Grid_Occupation_Device
(
	const int num_grids, const int num_layers, const int3 grid_dimension, const int3 grid_dimension_minus_one,
	const int* origin_grid_occupation, int* smoothed_grid_occupation
) {
	int3 grid_3d_serial;
	for (int grid_i = blockIdx.x * blockDim.x + threadIdx.x; grid_i < num_grids; grid_i = grid_i + gridDim.x * blockDim.x) {
		grid_3d_serial.x = grid_i % grid_dimension.x;
		grid_3d_serial.y = (grid_i % num_layers) / grid_dimension.x;
		grid_3d_serial.z = grid_i / num_layers;
		int dx = 0, dy = 0, dz = 0;
		int x_, y_, z_, x, y, z, ddx, ddy, ddz, temp_x, temp_y, temp_z;
		int temp_occupation = 0;
		for (int neighbor_grid_i = 0; neighbor_grid_i < 13; neighbor_grid_i = neighbor_grid_i + 1) { // Keep only half-symmetric mapping
			ddx = dx - 1;
			ddy = dy - 1;
			ddz = dz - 1;

			z_ = (grid_3d_serial.z + ddz);
			y_ = (grid_3d_serial.y + ddy);
			x_ = (grid_3d_serial.x + ddx);// One of the 13 grids

			z = (grid_3d_serial.z - ddz);
			y = (grid_3d_serial.y - ddy);
			x = (grid_3d_serial.x - ddx);// The symmetric one about center grid

			x = x - (x & ((x) >> 31));// If x<0 then x=0, if x>=0 then x=x
			temp_x = (grid_dimension_minus_one.x - x);
			x = x + (temp_x & (temp_x >> 31));// If x<grid_dimension.x then x=x, if x>=grid_dimension.x then x=grid_dimension_minus_one.x

			y = y - (y & ((y) >> 31));
			temp_y = (grid_dimension_minus_one.y - y);
			y = y + (temp_y & (temp_y >> 31));

			z = z - (z & ((z) >> 31));
			temp_z = (grid_dimension_minus_one.z - z);
			z = z + (temp_z & (temp_z >> 31));

			x_ = x_ - (x_ & ((x_) >> 31));
			temp_x = (grid_dimension_minus_one.x - x_);
			x_ = x_ + (temp_x & (temp_x >> 31));

			y_ = y_ - (y_ & ((y_) >> 31));
			temp_y = (grid_dimension_minus_one.y - y_);
			y_ = y_ + (temp_y & (temp_y >> 31));

			z_ = z_ - (z_ & ((z_) >> 31));
			temp_z = (grid_dimension_minus_one.z - z_);
			z_ = z_ + (temp_z & (temp_z >> 31));

			// Can optimize with max min
			int grid_j = z * num_layers + y * grid_dimension.x + x;
			int grid_k = z_ * num_layers + y_ * grid_dimension.x + x_;
			temp_occupation = (temp_occupation | (origin_grid_occupation[grid_j] & origin_grid_occupation[grid_k]));

			dx = dx + 1;
			dy = dy + (((2 - dx) >> 31) & 0x00000001);
			dx = dx & ((dx - 3) >> 31);
			dz = dz + (((2 - dy) >> 31) & 0x00000001);
			dy = dy & ((dy - 3) >> 31);
		}
		smoothed_grid_occupation[grid_i] = temp_occupation | origin_grid_occupation[grid_i];
	}
}
__global__ void Build_Surface_Device (
	const int num_grids, const int num_layers, const int3 grid_dimension, const int3 grid_dimension_minus_one,
	const int* origin_grid_occupation, int* surface
) {
	int3 grid_3d_serial;
	for (int grid_i = blockIdx.x * blockDim.x + threadIdx.x; grid_i < num_grids; grid_i += gridDim.x * blockDim.x) {
		grid_3d_serial.x = grid_i % grid_dimension.x;
		grid_3d_serial.y = (grid_i % num_layers) / grid_dimension.x;
		grid_3d_serial.z = grid_i / num_layers;

		int temp_occupation = 1;
		// If all 6 adjacent cells are 1, self is 0, otherwise use original origin_grid_occupation
		int grid_j, x, y, z, temp_z, temp_y, temp_x;
		x = grid_3d_serial.x;
		y = grid_3d_serial.y;
		z = grid_3d_serial.z - 1;
		z = z - (z & ((z) >> 31));

		grid_j = z * num_layers + y * grid_dimension.x + x;
		temp_occupation = temp_occupation & origin_grid_occupation[grid_j];

		x = grid_3d_serial.x;
		y = grid_3d_serial.y - 1;
		z = grid_3d_serial.z;
		y = y - (y & ((y) >> 31));

		grid_j = z * num_layers + y * grid_dimension.x + x;
		temp_occupation = temp_occupation & origin_grid_occupation[grid_j];

		x = grid_3d_serial.x - 1;
		y = grid_3d_serial.y;
		z = grid_3d_serial.z;
		x = x - (x & ((x) >> 31));

		grid_j = z * num_layers + y * grid_dimension.x + x;
		temp_occupation = temp_occupation & origin_grid_occupation[grid_j];

		//+1
		x = grid_3d_serial.x + 1;
		y = grid_3d_serial.y;
		z = grid_3d_serial.z;
		temp_x = (grid_dimension_minus_one.x - x);
		x = x + (temp_x & (temp_x >> 31));

		grid_j = z * num_layers + y * grid_dimension.x + x;
		temp_occupation = temp_occupation & origin_grid_occupation[grid_j];

		x = grid_3d_serial.x;
		y = grid_3d_serial.y + 1;
		z = grid_3d_serial.z;
		temp_y = (grid_dimension_minus_one.y - y);
		y = y + (temp_y & (temp_y >> 31));

		grid_j = z * num_layers + y * grid_dimension.x + x;
		temp_occupation = temp_occupation & origin_grid_occupation[grid_j];

		x = grid_3d_serial.x;
		y = grid_3d_serial.y;
		z = grid_3d_serial.z + 1;
		temp_z = (grid_dimension_minus_one.z - z);
		z = z + (temp_z & (temp_z >> 31));

		grid_j = z * num_layers + y * grid_dimension.x + x;
		temp_occupation = temp_occupation & origin_grid_occupation[grid_j];

		// If surrounded, temp_occupation=1, otherwise 0
		surface[grid_i] = origin_grid_occupation[grid_i] & (1 - temp_occupation);  // Only 1 when origin_grid_occupation[grid_i]=1 and temp_occupation=0
	}
}


Surface::Surface(at::Tensor pos, const int* atomicnum) {
	grid_len_inv = 1.f / grid_len;
	this->atom_crd.clear();
	this->atomic_number.clear();
	auto num_atoms = pos.size(0);
	
	std::vector<float3> atom_crd;
	atom_crd.resize(num_atoms);
	for (int i = 0; i < num_atoms; i = i + 1) {
		atom_crd[i].x = pos[i][0].item<float>();
		atom_crd[i].y = pos[i][1].item<float>();
		atom_crd[i].z = pos[i][2].item<float>();
	}	
	float3 inner_min = { 100000.f,100000.f,100000.f, }, inner_max = { -100000.f,-100000.f,-100000.f, };
	for (int i = 0; i < num_atoms; i = i + 1) {
		inner_min.x = fminf(inner_min.x, atom_crd[i].x);
		inner_min.y = fminf(inner_min.y, atom_crd[i].y);
		inner_min.z = fminf(inner_min.z, atom_crd[i].z);

		inner_max.x = fmaxf(inner_max.x, atom_crd[i].x);
		inner_max.y = fmaxf(inner_max.y, atom_crd[i].y);
		inner_max.z = fmaxf(inner_max.z, atom_crd[i].z);
	}
	move_vec = { -inner_min.x + skin ,-inner_min.y + skin ,-inner_min.z + skin };
	for (int i = 0; i < num_atoms; i = i + 1) {
		this->atom_crd.push_back({ atom_crd[i].x + move_vec.x ,atom_crd[i].y + move_vec.y , atom_crd[i].z + move_vec.z });
		this->atomic_number.push_back(atomicnum[i]);
	}
	float3 protein_size;
	protein_size.x = inner_max.x - inner_min.x + 2.f * skin;
	protein_size.y = inner_max.y - inner_min.y + 2.f * skin;
	protein_size.z = inner_max.z - inner_min.z + 2.f * skin;

	extending_numbers = ceilf(skin * grid_len_inv) - 1;

	grid_dimension.x = ceilf(protein_size.x / grid_len);
	grid_dimension.y = ceilf(protein_size.y / grid_len);
	grid_dimension.z = ceilf(protein_size.z / grid_len);
	grid_dimension_minus_one.x = grid_dimension.x - 1;
	grid_dimension_minus_one.y = grid_dimension.y - 1;
	grid_dimension_minus_one.z = grid_dimension.z - 1;


	num_layers = grid_dimension.x * grid_dimension.y;
	num_grids = num_layers * grid_dimension.z;

	box_length.x = grid_len * grid_dimension.x;
	box_length.y = grid_len * grid_dimension.y;
	box_length.z = grid_len * grid_dimension.z;

	// Atom radius, from Siyuan Jiang's code
	atom_radius.clear();
	for (int i = 0; i < num_atoms; i = i + 1) {
		if (atomicnum[i] == 1) {
			atom_radius.push_back(0.f);
		} else if (atomicnum[i] == 6) {
			atom_radius.push_back(1.7f);
		} else if (atomicnum[i] == 7) {
			atom_radius.push_back(1.6f);
		} else if (atomicnum[i] == 8) {
			atom_radius.push_back(1.55f);
		} else if (atomicnum[i] == 16) {
			atom_radius.push_back(1.8f);
		} else if (atomicnum[i] == 15) {
			atom_radius.push_back(1.95f);
		} else if (atomicnum[i] == 17) {
			atom_radius.push_back(1.8f);
		} else if (atomicnum[i] == 9) {
			atom_radius.push_back(1.5f);
		} else if (atomicnum[i] == 35) {
			atom_radius.push_back(1.9f);
		} else if (atomicnum[i] == 53) {
			atom_radius.push_back(2.1f);
		} else if (atomicnum[i] == 34) {
			atom_radius.push_back(1.9f);
		} else if (atomicnum[i] == 35) {
			atom_radius.push_back(2.05f);
		} else if (atomicnum[i] == 5) {
			atom_radius.push_back(1.8f);
		} else if (atomicnum[i] == 20) {
			atom_radius.push_back(2.4f);
		} else if (atomicnum[i] == 12) {
			atom_radius.push_back(2.2f);
		} else if (atomicnum[i] == 30) {
			atom_radius.push_back(2.1f);
		} else if (atomicnum[i] == 100) {
			atom_radius.push_back(2.05f);
		}
		atom_radius[i] *= 1.f;  // For global radius adjustment
	}

	//malloc
	cudaMalloc((void**)&d_atom_crd, sizeof(float3) * num_atoms);
	cudaMalloc((void**)&d_atom_radius, sizeof(float) * num_atoms);
	cudaMalloc((void**)&d_atom_is_near_surface, sizeof(int) * num_atoms);
	cudaMalloc((void**)&d_origin_grid_occupation, sizeof(int) * num_grids);
	cudaMalloc((void**)&d_smoothed_grid_occupation, sizeof(int) * num_grids);

	//memcpy
	cudaMemcpy(d_atom_crd, &this->atom_crd[0], sizeof(float3) * num_atoms, cudaMemcpyHostToDevice);
	cudaMemcpy(d_atom_radius, &this->atom_radius[0], sizeof(float) * num_atoms, cudaMemcpyHostToDevice);
	cudaMemset(d_origin_grid_occupation, 0, sizeof(int) * num_grids);
	cudaMemset(d_smoothed_grid_occupation, 0, sizeof(int) * num_grids);

	//run
	Calculate_Origin_Grid_Occupation_Device << <64, 64 >> > (
		num_atoms, d_atom_crd, d_atom_radius,
		num_grids, num_layers, grid_dimension, grid_len, grid_len_inv,
		extending_numbers, d_origin_grid_occupation
	);

	Smooth_Grid_Occupation_Device << <64, 64 >> > (
		num_grids, num_layers, grid_dimension, grid_dimension_minus_one,
		d_origin_grid_occupation, d_smoothed_grid_occupation
	);

	Smooth_Grid_Occupation_Device << <64, 64 >> > (
		num_grids, num_layers, grid_dimension, grid_dimension_minus_one,
		d_smoothed_grid_occupation, d_origin_grid_occupation
	);  // Run twice, after which d_origin_grid_occupation contains the twice-smoothed result

	Build_Surface_Device << <64, 64 >> > (
		num_grids, num_layers, grid_dimension, grid_dimension_minus_one,
		d_origin_grid_occupation, d_smoothed_grid_occupation
	);  // Now d_smoothed_grid_occupation records the surface grids
	
	Calculate_Atom_Near_Surface_Device << <64, 64 >> >(
		num_atoms, d_atom_crd, d_atom_radius, d_atom_is_near_surface,
		num_grids, num_layers, grid_dimension, grid_len, grid_len_inv,
		extending_numbers, d_smoothed_grid_occupation
	);

	atom_is_near_surface.resize(num_atoms);
	cudaMemcpy(&atom_is_near_surface[0], d_atom_is_near_surface, sizeof(int) * num_atoms, cudaMemcpyDeviceToHost);
}