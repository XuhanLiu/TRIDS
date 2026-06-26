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

#include "graph.h"

Graph::Graph(at::Tensor node_feats_, at::Tensor edge_feats_, at::Tensor edge_index_ /*, at::Tensor pos_*/) {

    // assert(node_feats.size(0) == pos.size(0));
    assert(edge_index_.size(0) == 2);
    assert(edge_index_.size(1) == edge_feats_.size(0));
    assert(edge_index_.max().item<int64_t>() < edge_feats_.size(0));
    num_nodes = node_feats_.size(0);
    num_edges = edge_feats_.size(0);
    node_feats = node_feats_.to(at::kFloat);
    edge_feats = edge_feats_.to(at::kFloat);
    // std::cout << edge_index_[0][0] << std::endl;
    edge_index = edge_index_.to(at::kLong);
    // std::cout << edge_index[0][0] << std::endl;
    // pos = pos_.to(at::kFloat);
}

Batch Graph::pack(vector<Graph>& graphs) {
    vector<at::Tensor> node_feat_list;
    vector<at::Tensor> edge_feat_list;
    vector<at::Tensor> edge_idx_list;
    vector<at::Tensor> pos_list;
    vector<size_t> batch_list;

    int offset_ = 0;
    size_t index = 0;
    vector<size_t> offset_list{ graphs.size() };
    for (auto& graph : graphs) {
        offset_list.push_back(offset_);
        node_feat_list.push_back(graph.node_feats);
        edge_feat_list.push_back(graph.edge_feats);
        edge_idx_list.push_back(graph.edge_index + offset_);
        // pos_list.push_back(graph.pos);
        for (auto i = 0u; i < graph.num_edges; i++) {
            batch_list.push_back(index);
        }
        offset_ += graph.num_nodes;
        index++;
    }

    offset_list.push_back(offset_);
    auto pos = at::cat(pos_list);
    auto batch = at::from_blob(batch_list.data(), { (int)batch_list.size() }).clone();
    auto offset = at::from_blob(offset_list.data(), { (int)offset_list.size() }).clone();
    Batch bg(at::cat(node_feat_list), at::cat(edge_feat_list),
             at::cat(edge_idx_list, 1), /*pos,*/ batch, offset);
    return bg;
}

void Graph::to(torch::TensorOptions options) {
    node_feats = node_feats.to(options);
    edge_feats = edge_feats.to(options);
    edge_index = edge_index.to(options.device());
    // pos = pos.to(options);
}

void Graph::cuda() {
    node_feats = node_feats.cuda();
    edge_feats = edge_feats.cuda();
    edge_index = edge_index.cuda();
    // pos = pos.cuda();
}

void Graph::cpu() {
    node_feats = node_feats.cpu();
    edge_feats = edge_feats.cpu();
    edge_index = edge_index.cpu();
    // pos = pos.cpu();
}

void Graph::load(const string& path) {
    vector<at::Tensor> tensor_list;
    torch::load(tensor_list, path);
    node_feats = tensor_list[0];
    edge_feats = tensor_list[1];
    edge_index = tensor_list[2];
    // pos = tensor_list[3];
}

void Graph::save(const string& path) {
    vector<at::Tensor> tensor_list{ node_feats, edge_feats, edge_index, /*pos*/ };
    torch::save(tensor_list, path);
}

Batch::Batch(at::Tensor node_feats_, at::Tensor edge_feats_, at::Tensor edge_index_,
             /*at::Tensor pos_,*/ at::Tensor batch_, at::Tensor offset_)
    : Graph(node_feats_, edge_feats_, edge_index_ /*, pos_*/), batch(batch_), offset(offset_) {
    // assert(node_feats.size(0) == pos.size(0));
    assert(edge_index.size(0) == 2);
    assert(edge_index.size(1) == edge_feats.size(0));
    num_nodes = node_feats.size(0);
    num_edges = edge_feats.size(0);
}

vector<Graph> Batch::unpack() {
    auto graph_size = static_cast<size_t>(batch[-1].item().toInt() + 1);
    vector<Graph> graphs;
    graphs.reserve(graph_size);
    for (size_t i = 0; i < graph_size; i++) {
        graphs.push_back((*this)[static_cast<int>(i)]);
    }
    return graphs;
}

Graph Batch::operator[](int index) {
    auto num_graphs = static_cast<int>(batch[-1].item().toInt() + 1);
    assert(index < num_graphs);
    if (index < 0) {
        index = num_graphs - index;
    }
    auto offset_0 = offset[index].item().toInt();
    auto offset_1 = offset[index + 1].item().toInt();
    auto mask = at::argwhere(batch == index);
    auto edge_index_ = edge_index.index_select(1, mask) - offset_0;
    auto edge_feat_ = edge_feats.index_select(0, mask);
    auto node_feat_ = node_feats.index({at::indexing::Slice(offset_0, offset_1)});

    // auto pos_ = pos.index({at::indexing::Slice(offset_0, offset_1)});
    auto graph = Graph(node_feat_, edge_feat_, edge_index_ /*, /pos_*/);
    return graph;
}

void Batch::to(torch::TensorOptions options) {
    Graph::to(options);
    batch = batch.to(options);
    offset = offset.to(options);
}

void Batch::cuda() {
    Graph::cuda();
    batch = batch.cuda();
    offset = offset.cpu();
}

void Batch::cpu() {
    Graph::cpu();
    batch = batch.cpu();
    offset = offset.cpu();
}

