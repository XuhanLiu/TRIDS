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
#include <torch/nn.h>
#include "graph.h"

using namespace std;
using namespace torch::indexing;

class MultiHeadAttentionImpl : public torch::nn::Module {
public:
    int num_heads;
    int num_output_feats;
    torch::nn::Linear Q{ nullptr };
    torch::nn::Linear K{ nullptr };
    torch::nn::Linear V{ nullptr };
    torch::nn::Linear edge_feats_projection{ nullptr };

    MultiHeadAttentionImpl() = default;

    MultiHeadAttentionImpl(int num_input_feats, int num_output_feats_,
                           int num_heads_, bool using_bias = false, bool update_edge_feats = true);

    std::pair<torch::Tensor, torch::Tensor> forward(at::Tensor node_feats, at::Tensor edge_feats, at::Tensor edge_index);

}; TORCH_MODULE(MultiHeadAttention);

class GraphTransformerModuleImpl : public torch::nn::Module {

public:
    bool residual = true;
    int num_output_feats;
    torch::nn::Dropout dropout{ nullptr };

    // torch::nn::LayerNorm layer_norm1_node_feats{ nullptr };
    // torch::nn::LayerNorm layer_norm1_edge_feats{ nullptr };
    // torch::nn::LayerNorm layer_norm2_node_feats{ nullptr };
    // torch::nn::LayerNorm layer_norm2_edge_feats{ nullptr };

    torch::nn::BatchNorm1d batch_norm1_node_feats{ nullptr };
    torch::nn::BatchNorm1d batch_norm1_edge_feats{ nullptr };
    torch::nn::BatchNorm1d batch_norm2_node_feats{ nullptr };
    torch::nn::BatchNorm1d batch_norm2_edge_feats{ nullptr };

    torch::nn::Linear O_node_feats{ nullptr };
    torch::nn::Linear O_edge_feats{ nullptr };
    MultiHeadAttention mha_module{ nullptr };
    torch::nn::Sequential node_feats_MLP{ nullptr };
    torch::nn::Sequential edge_feats_MLP{ nullptr };

    GraphTransformerModuleImpl() = default;

    GraphTransformerModuleImpl(int num_hidden_channels, int num_heads,
                               float dropout_, bool residual_);

    std::pair<torch::Tensor, torch::Tensor> forward(at::Tensor node_feats, at::Tensor edge_feats, at::Tensor edge_index);

}; TORCH_MODULE(GraphTransformerModule);

class FinalTransformerModuleImpl : public torch::nn::Module {
public:
    bool residual = true;
    int num_output_feats;
    torch::nn::Dropout dropout{ nullptr };
    // torch::nn::LayerNorm layer_norm1_node_feats{ nullptr };
    // torch::nn::LayerNorm layer_norm2_node_feats{ nullptr };

    torch::nn::BatchNorm1d batch_norm1_node_feats{ nullptr };
    torch::nn::BatchNorm1d batch_norm1_edge_feats{ nullptr };
    torch::nn::BatchNorm1d batch_norm2_node_feats{ nullptr };

    torch::nn::Linear O_node_feats{ nullptr };
    MultiHeadAttention mha_module{ nullptr };
    torch::nn::Sequential node_feats_MLP{ nullptr };

    FinalTransformerModuleImpl() = default;

    FinalTransformerModuleImpl(int num_hidden_channels, int num_heads,
                               float dropout_, bool residual_);

    at::Tensor forward(at::Tensor node_feats, at::Tensor edge_feats, at::Tensor edge_index);

}; TORCH_MODULE(FinalTransformerModule);

class GraphTransformerImpl : public torch::nn::Module {
public:
    torch::nn::Linear node_encoder{ nullptr };
    torch::nn::Linear edge_encoder{ nullptr };
    torch::nn::ModuleList gt_block{ nullptr };
    FinalTransformerModule final{ nullptr };

    GraphTransformerImpl() = default;

    GraphTransformerImpl(int in_channels,
                         int edge_features = 10,
                         int num_hidden_channels = 128,
                         bool residual = true,
                         int num_heads = 4,
                         float dropout_rate = 0.1,
                         int num_layers = 6);

    at::Tensor forward(at::Tensor node_feats, at::Tensor edge_feats, at::Tensor edge_index);

}; TORCH_MODULE(GraphTransformer);

class TriScoreImpl : public torch::nn::Module {
public:
    int in_channels;
    int hidden_dim;
    int n_gaussians;
    torch::nn::Linear z_pi{ nullptr };
    torch::nn::Linear z_sigma{ nullptr };
    torch::nn::Linear z_mu{ nullptr };
    torch::nn::Sequential MLP{ nullptr };
    torch::nn::Linear atom_types{ nullptr };
    torch::nn::Linear bond_types{ nullptr };
    GraphTransformer ligand_model{ nullptr };
    GraphTransformer target_model{ nullptr };

    TriScoreImpl(GraphTransformer ligand_, GraphTransformer target_,
                 int in_channels_ = 128, int hidden_dim_ = 128, int n_gaussians_ = 10, float dropout_ = 0.15);

    std::pair<torch::Tensor, torch::Tensor> forward(Batch& lig, Batch& rec);

    at::Tensor single(Graph& lig, Graph& rec);

    at::Tensor scoring(at::Tensor rho, at::Tensor mu, at::Tensor sigma, at::Tensor mask,
                       at::Tensor lig_pos, at::Tensor rec_pos, std::optional<float> cutoff);

}; TORCH_MODULE(TriScore);