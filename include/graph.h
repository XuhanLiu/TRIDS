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

#include <torch/torch.h>

using namespace std;

class Batch;
struct Graph {
public:
    at::Tensor node_feats;
    at::Tensor edge_feats;
    at::Tensor edge_index;
    // at::Tensor pos;
    size_t num_nodes;
    size_t num_edges;

    Graph() = default;

    Graph(at::Tensor node_feats_, at::Tensor edge_feats_, at::Tensor edge_index_/*, at::Tensor pos_*/);

    Batch pack(vector<Graph>& graphs);

    void to(torch::TensorOptions options);

    void cuda();

    void cpu();

    void load(const string& path);

    void save(const string& path);
};

struct Batch : public Graph {
public:
    at::Tensor batch;
    at::Tensor offset;

    Batch() = default;

    Batch(at::Tensor node_feats_, at::Tensor edge_feats_, at::Tensor edge_index_,
          /*at::Tensor pos_,*/ at::Tensor batch, at::Tensor offset);

    vector<Graph> unpack();

    Graph operator[](int index);

    void to(torch::TensorOptions options);

    void cuda();

    void cpu();
};
