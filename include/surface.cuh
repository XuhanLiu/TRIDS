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
#include <torch/torch.h>
#include <vector>

// Corresponds to Rigid_Protein.cuh location, also needs to link common.o during compilation
class Surface {

public:
	float skin = 3.f;  // Extend outward by skin length in all directions to ensure convolution and atom radius don't exceed boundary
	float3 move_vec;   // Translation vector relative to input atom coordinates (see Rigid_Protein.cuh)

	std::vector<float> atom_radius;  // Provided by Initial function parameters
	std::vector<float3> atom_crd;
	std::vector<int> atomic_number;
	std::vector<int> atom_is_near_surface;  // Final stored result: 1 means at boundary, 0 means not

	// GPU-related
	float3* d_atom_crd = nullptr;
	float* d_atom_radius = nullptr;
	int* d_atom_is_near_surface = nullptr;

	int extending_numbers;  // Related to skin and atom radius (larger = slower, but not noticeable for single GPU run)
					        // Used to ensure each atom can traverse grid points covered by its radius
	// Basic grid parameters, requires cubic grid, so grid_length is float not float3
	int3 grid_dimension;
	int3 grid_dimension_minus_one;
	int num_layers;
	int num_grids;
	float grid_len_inv;
	float grid_len = 2.f;
	float3 box_length;  // Generally slightly larger than protein (determined by skin and protein size)

	int* d_origin_grid_occupation = nullptr;  // 0/1 3D grid, records original protein occupation
	int* d_smoothed_grid_occupation = nullptr;

	Surface() = default;

	~Surface() {
		if (d_atom_crd) cudaFree(d_atom_crd);
		if (d_atom_radius) cudaFree(d_atom_radius);
		if (d_atom_is_near_surface) cudaFree(d_atom_is_near_surface);
		if (d_origin_grid_occupation) cudaFree(d_origin_grid_occupation);
		if (d_smoothed_grid_occupation) cudaFree(d_smoothed_grid_occupation);
	}

	Surface(at::Tensor pos, const int* atomic_number);
};
