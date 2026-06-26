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

#include <openbabel/mol.h>
#include <torch/script.h>
#include "receptor.h"

class Siter {

public:
	std::vector<Receptor*> pockets;

	float cutoff;

	int64_t num_clusters = 0;

	torch::TensorOptions options;

	Siter(torch::TensorOptions options_ = {}, float cutoff_ = 10) : cutoff(cutoff_), options(options_) {}

	Siter() = default;

	virtual ~Siter();

	virtual at::Tensor points(int current) = 0;

	virtual Receptor* extract(OpenBabel::OBMol* rec, at::Tensor points, bool is_vina = false) = 0;

	virtual std::tuple<at::Tensor, at::Tensor> get_box(at::Tensor points);

	virtual void init(OpenBabel::OBMol* rec, bool is_vina = false);
};


class RefSiter : public Siter {
	std::vector<OpenBabel::OBMol*> refs;

public:
	RefSiter(const std::vector<OpenBabel::OBMol*>& refs_, torch::TensorOptions options_ = {}, float cutoff_ = 8)
		: Siter(options_, cutoff_), refs(refs_) {
		num_clusters = refs_.size();
	}

	virtual ~RefSiter();

	virtual at::Tensor points(int current);

	virtual Receptor* extract(OpenBabel::OBMol* rec, at::Tensor points, bool is_vina = false);
};

class GridSiter : public Siter {
public:
	float resolution = 2;

	float max_distance = 35;

	std::vector<torch::Tensor>cluster;

	GridSiter(torch::TensorOptions options_ = {}, float cutoff_ = 8) : Siter(options_, cutoff_) {}

	GridSiter(std::vector<torch::Tensor>cluster_, torch::TensorOptions options_ = {}, float cutoff_ = 8)
		: Siter(options_, cutoff_), cluster(cluster_) {
		num_clusters = cluster.size();
	}

	virtual at::Tensor points(int current);

	virtual Receptor* extract(OpenBabel::OBMol* rec, at::Tensor points, bool is_vina = false);
};


class ModelSiter : public GridSiter {
	torch::jit::Module model;

	int top_points = 200;

public:

	ModelSiter(torch::jit::Module model_, OpenBabel::OBMol* rec, torch::TensorOptions options_ = {}, float cutoff_ = 8);

	at::Tensor make_grid(at::Tensor coords, at::Tensor features);

	void grid_to_cluster(at::Tensor grid, at::Tensor centroid);

	virtual at::Tensor points(int current);

	virtual Receptor* extract(OpenBabel::OBMol* rec, at::Tensor points, bool is_vina = false);
};