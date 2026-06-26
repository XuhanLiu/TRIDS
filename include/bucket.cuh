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

#define MAX_NBR_ATOM 64

// Spatial grid partition for protein atoms; neighbor list can be obtained from the grid at different positions
class Bucket {
public:
	at::Tensor neighbors;
	int3 bucket_dim;
	float3 box_len;
	float3 bucket_len;
	float3 bucket_len_inv;
	at::Tensor offset;
	float skin = 0.f;    // Extend a certain distance beyond cutoff
	float skin2 = 0.f;
	float cutoff = 0.f;
	float cutoff2 = 0.f;
	int num_buckets = 0;  // Total size of 3D array Nx*Ny*Nz
	int num_layers = 0;   // Number per layer Nx*Ny

	// Only allow initialization once, cannot reinitialize (reasonable requirement)
	// Ensure box_len is large enough to fully contain the entire protein
	// Note: distinct from interpolation grid; this box is for building the protein grid neighbor list
	Bucket(float3 box_len, float cutoff, float skin);

	// Place each protein atom into its corresponding spatial grid
	// Afterward, nearby protein atom indices can be directly extracted from the grid
	void put_atom_into_bucket(at::Tensor crd);

	// Clear atom records in the grid
	void clear();
};
