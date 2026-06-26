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

#include<cuda_runtime.h>
#include <vector>
#include <torch/torch.h>
#include "bucket.cuh"

#define THREADS 256
#define BLOCKS(B) (B + THREADS - 1) / THREADS
class Grid {
    void Copy_Potential_To_Texture();

public:
	// Only consider cubic box, mainly ensure cubic box is always larger than restricted region
	float cutoff = 8.f;
	float3 grid_min;     // Position of min grid point in real space; not necessarily same as box_min in search, should be wider to prevent molecule from leaving grid region (theoretically can partially leave box during random sampling)
	float3 box_len;      // 3D size of complete interpolation space (all three components should be equal)
	float3 grid_len;     // Unit grid size
	float3 grid_len_inv;
	int3 grid_dim;       // Number of grids; since this is not periodic boundary grid, actual grid count is one less than interpolation points (per direction)
	int layer_numbers = 0;
	int grid_numbers = 0;  // Actually grid point count, not grid count: if point count is a*a*a, grid count is only (a-1)*(a-1)*(a-1)

    // Total 18 atom types, need 18 sets of grids to store energy different atoms may experience at same position
    int num_types = 18;
    int num_grids;
	at::TensorOptions options;
	at::Tensor base_hd;
	at::Tensor base_ha;
	at::Tensor base_hydro;
	at::Tensor base_vdw;

	Grid() = default;
	// Initialize with grid point count per side of restricted search space and cutoff radius for scoring
	// Since each system and site needs updating, this init only handles basic memory allocation etc.
	// grid_numbers_in_one_dimension specifies interpolation grid point count in each direction
	Grid(int grid_numbers_in_one_dimemsion_, float cutoff_, at::TensorOptions options_ = {});

	~Grid() {
		if (!options.device().is_cuda()) {
			return;
		}
		for (auto& tex : texObj_potential) {
			cudaDestroyTextureObject(tex);
		}
		for (auto& arr : cudaArray_potential) {
			cudaFreeArray(arr);
		}
		if (texObj_for_kernel) {
			cudaFree(texObj_for_kernel);
		}
	};

    at::Tensor potential;  // Currently using separate interpolation for energy and force

    // GPU texture memory related variables
    // Actually a double pointer, pointing to 18 type interpolation grid addresses texObj_potential
    int64_t* texObj_for_kernel = nullptr; 
    // Actual storage location for interpolation grid points
    std::vector<cudaArray*> cudaArray_potential;
    // Parameters used during memory copy
    std::vector<cudaMemcpy3DParms> copyParams_potential;
    // Actually a int, so a separate texture index array is created for use in actual kernel function
    std::vector<cudaTextureObject_t> texObj_potential;

    // Given actual coordinate min point box_min of restricted search space, box_length of search space box,
    // total protein atom count atom_numbers, protein atom info d_protein (protein atom info on GPU: coordinates, type, radius...),
    // spatial partition of protein atom coordinates stored in neighbor_grid_bucket
    // Calculate protein grid interpolation force field
    void make_grid(at::Tensor rec_pos, at::Tensor rec_hd, at::Tensor rec_ha, 
                   at::Tensor rec_hydro, at::Tensor rec_vdw, 
                   at::Tensor box_min, Bucket& bucket);

    void Cuda_Texture_Initial();  // Initialize texture memory

};