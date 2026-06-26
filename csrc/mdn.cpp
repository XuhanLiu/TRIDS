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

#include "mdn.h"

inline std::pair<torch::Tensor, torch::Tensor> to_dense_batch(at::Tensor x,
                                                              at::Tensor batch) {
    auto batch_size = batch.max().item<int>() + 1;
    auto num_nodes = batch.new_zeros({ batch_size });
    num_nodes.scatter_add_(0, batch, batch.new_ones(x.size(0)));

    auto cum_nodes = batch.new_zeros({ batch_size + 1 });
    cum_nodes.slice(0, 1, None) = num_nodes.cumsum(0);

    int max_num_nodes = num_nodes.max().item<int>();

    auto tmp = torch::arange(batch.size(0), x.options()) - cum_nodes.index_select(0, batch);
    auto idx = tmp + (batch * max_num_nodes);

    auto out = x.new_zeros({ batch_size * max_num_nodes, x.size(1) });
    out.index_fill_(0, idx, x);
    out = out.view({ batch_size, max_num_nodes, x.size(1) });

    auto mask = torch::zeros({ batch_size * max_num_nodes }, x.options()).to(torch::kBool);
    mask.index_fill_(0, idx, 1);
    mask = mask.view({ batch_size, max_num_nodes });

    return std::make_pair(out, mask);
}

inline at::Tensor broadcast(at::Tensor src, at::Tensor other, int dim) {
    if (dim < 0) {
        dim = other.dim() + dim;
    }
    if (src.dim() == 1) {
        for (auto i = 0; i < dim; i++) {
            src = src.unsqueeze(0);
        }
    }
    for (auto i = src.dim(); i < other.dim(); i++) {
        src = src.unsqueeze(-1);
    }
    src = src.expand(other.sizes());
    return src;
}

MultiHeadAttentionImpl::MultiHeadAttentionImpl(int num_input_feats, int num_output_feats_,
                                               int num_heads_, bool using_bias, bool update_edge_feats)
    : num_heads(num_heads_), num_output_feats(num_output_feats_) {

    auto options = torch::nn::LinearOptions(num_input_feats, num_output_feats * num_heads).bias(using_bias);
    Q = register_module("Q", torch::nn::Linear(options));
    K = register_module("K", torch::nn::Linear(options));
    V = register_module("V", torch::nn::Linear(options));
    edge_feats_projection = register_module("edge_feats_projection", torch::nn::Linear(options));
}

std::pair<torch::Tensor, torch::Tensor> MultiHeadAttentionImpl::forward(at::Tensor node_feats, at::Tensor edge_feats, at::Tensor edge_index) {
    auto node_feats_q = this->Q(node_feats).view({ -1, num_heads, num_output_feats });
    auto node_feats_k = this->K(node_feats).view({ -1, num_heads, num_output_feats });
    auto node_feats_v = this->V(node_feats).view({ -1, num_heads, num_output_feats });
    auto edge_feats_prj = this->edge_feats_projection(edge_feats).view({ -1, num_heads, num_output_feats });
    // wV, z, e_out = this->propagate_attention(edge_index, node_feats_q, node_feats_k, node_feats_v, edge_feats_projection)
    auto row = edge_index[0], col = edge_index[1];
    // Compute attention scores
    auto alpha = node_feats_k.index_select(0, row) * node_feats_q.index_select(0, col);
    // Scale and clip attention scores
    alpha = (alpha / std::sqrt(num_output_feats)).clamp(-5.0, 5.0);
    // Use available edge features to modify the attention scores
    alpha *= edge_feats_prj;
    // Apply softmax to attention scores, followed by clipping	
    auto alphax = torch::exp((alpha.sum(-1, true)).clamp(-5.0, 5.0));

    // Send weighted values to target nodes
    auto wV = torch::zeros_like(node_feats_q);
    auto src = node_feats_v.index_select(0, row) * alphax;
    auto index = broadcast(col, src, 0);
    wV.scatter_add_(0, index, src);

    auto z = node_feats.new_zeros({ node_feats_q.size(0), alphax.size(1), alphax.size(2) });
    index = broadcast(col, alphax, 0);
    z.scatter_add_(0, index, alphax);

    auto h_out = wV / (z + torch::full_like(z, 1e-6));
    return std::make_pair(h_out, alpha);
}

//////////////////////////////////////////////////////////////////////

GraphTransformerModuleImpl::GraphTransformerModuleImpl(int num_hidden_channels, int num_heads,
                                                       float dropout_, bool residual_)
    : residual(residual_), num_output_feats(num_hidden_channels) {
    auto batch_norm_opt = torch::nn::BatchNormOptions({ num_output_feats });
    batch_norm1_node_feats = register_module("batch_norm1_node_feats", torch::nn::BatchNorm1d(batch_norm_opt));
    batch_norm1_edge_feats = register_module("batch_norm1_edge_feats", torch::nn::BatchNorm1d(batch_norm_opt));
    batch_norm2_node_feats = register_module("batch_norm2_node_feats", torch::nn::BatchNorm1d(batch_norm_opt));
    batch_norm2_edge_feats = register_module("batch_norm2_edge_feats", torch::nn::BatchNorm1d(batch_norm_opt));

    mha_module = register_module("mha_module", MultiHeadAttention(num_hidden_channels,
                                                                  num_output_feats / num_heads,
                                                                  num_heads,
                                                                  num_hidden_channels != num_output_feats,
                                                                  true));
    O_node_feats = register_module("O_node_feats", torch::nn::Linear(num_output_feats, num_output_feats));
    O_edge_feats = register_module("O_edge_feats", torch::nn::Linear(num_output_feats, num_output_feats));
    dropout = torch::nn::Dropout(dropout_);
    auto in_options = torch::nn::LinearOptions(num_output_feats, num_output_feats * 2).bias(false);
    auto out_options = torch::nn::LinearOptions(num_output_feats * 2, num_output_feats * 2).bias(false);
    node_feats_MLP = register_module("node_feats_MLP", torch::nn::Sequential(
        torch::nn::Linear(in_options),
        torch::nn::SiLU(),
        dropout,
        torch::nn::Linear(out_options)
    ));
    edge_feats_MLP = register_module("edge_feats_MLP", torch::nn::Sequential(
        torch::nn::Linear(in_options),
        torch::nn::SiLU(),
        dropout,
        torch::nn::Linear(out_options)
    ));
}

std::pair<torch::Tensor, torch::Tensor> GraphTransformerModuleImpl::forward(at::Tensor node_feats, at::Tensor edge_feats, at::Tensor edge_index) {
    // Perform a forward pass of geometric attention using a multi-head attention (MHA) module.
    auto node_feats_in1 = node_feats;  // Cache node representations for first residual connection
    auto edge_feats_in1 = edge_feats;  // Cache edge representations for first residual connection

    // std::cout << batch_norm1_node_feats->weight << std::endl;
    // std::cout << batch_norm1_node_feats->bias << std::endl;
    node_feats = batch_norm1_node_feats->forward(node_feats);
    // std::cout << node_feats << std::endl;
    edge_feats = batch_norm1_edge_feats->forward(edge_feats);
    auto mha_out = mha_module->forward(node_feats, edge_feats, edge_index);
    auto node_attn_out = mha_out.first;
    auto edge_attn_out = mha_out.second;

    node_feats = node_attn_out.view({ -1, num_output_feats });
    edge_feats = edge_attn_out.view({ -1, num_output_feats });

    node_feats = dropout(node_feats);
    edge_feats = dropout(edge_feats);

    node_feats = O_node_feats(node_feats);
    edge_feats = O_edge_feats(edge_feats);

    if (residual) {
        node_feats += node_feats_in1;
        edge_feats += edge_feats_in1;
    }

    node_feats_in1 = node_feats;
    edge_feats_in1 = edge_feats;

    node_feats = batch_norm2_node_feats(node_feats);
    edge_feats = batch_norm2_edge_feats(edge_feats);
    node_feats = node_feats_MLP->forward(node_feats);
    edge_feats = edge_feats_MLP->forward(edge_feats);

    if (residual) {
        node_feats += node_feats_in1;
        edge_feats += edge_feats_in1;
    }

    return std::make_pair(node_feats, edge_feats);
}

/////////////////////////////////////////////////////////

FinalTransformerModuleImpl::FinalTransformerModuleImpl(int num_hidden_channels, int num_heads,
                                                       float dropout_, bool residual_)
    : residual(residual_), num_output_feats(num_hidden_channels) {
    auto batch_norm_opt = torch::nn::BatchNormOptions({ num_output_feats });
    batch_norm1_node_feats = register_module("batch_norm1_node_feats", torch::nn::BatchNorm1d(batch_norm_opt));
    batch_norm1_edge_feats = register_module("batch_norm1_edge_feats", torch::nn::BatchNorm1d(batch_norm_opt));
    batch_norm2_node_feats = register_module("batch_norm2_node_feats", torch::nn::BatchNorm1d(batch_norm_opt));

    mha_module = register_module("mha_module", MultiHeadAttention(num_hidden_channels, num_output_feats / num_heads,
                                                                  num_heads, num_hidden_channels != num_output_feats, true));
    O_node_feats = register_module("O_node_feats", torch::nn::Linear(num_output_feats, num_output_feats));
    dropout = torch::nn::Dropout(dropout_);
    auto in_options = torch::nn::LinearOptions(num_output_feats, num_output_feats * 2).bias(false);
    auto out_options = torch::nn::LinearOptions(num_output_feats * 2, num_output_feats).bias(false);
    node_feats_MLP = register_module("node_feats_MLP", torch::nn::Sequential(
        torch::nn::Linear(in_options),
        torch::nn::SiLU(),
        dropout,
        torch::nn::Linear(out_options)
    ));
}

at::Tensor FinalTransformerModuleImpl::forward(at::Tensor node_feats, at::Tensor edge_feats, at::Tensor edge_index) {
    // Perform a forward pass of geometric attention using a multi-head attention (MHA) module.
    auto node_feats_in1 = node_feats;  // Cache node representations for first residual connection
    node_feats = batch_norm1_node_feats(node_feats);
    edge_feats = batch_norm1_edge_feats(edge_feats);

    auto mha_out = mha_module->forward(node_feats, edge_feats, edge_index);
    auto node_attn_out = mha_out.first;
    node_feats = node_attn_out.view({ -1, num_output_feats });
    // std::cout << node_feats;
    node_feats = dropout(node_feats);

    node_feats = O_node_feats(node_feats);

    if (residual) {
        node_feats += node_feats_in1;
    }

    node_feats_in1 = node_feats;
    node_feats = batch_norm2_node_feats(node_feats);
    node_feats = node_feats_MLP->forward(node_feats);

    if (residual) {
        node_feats += node_feats_in1;
    }
    return node_feats;
}

////////////////////////////////////////////////

GraphTransformerImpl::GraphTransformerImpl(int in_channels,
                                           int edge_features,
                                           int num_hidden_channels,
                                           bool residual,
                                           int num_heads,
                                           float dropout_rate,
                                           int num_layers) {
    node_encoder = register_module("node_encoder", torch::nn::Linear(in_channels, num_hidden_channels));
    edge_encoder = register_module("edge_encoder", torch::nn::Linear(edge_features, num_hidden_channels));
    assert(num_layers > 1);

    gt_block = torch::nn::ModuleList();
    for (auto i = 0; i < num_layers - 1; i++) {
        auto gt = GraphTransformerModule(num_hidden_channels,
                                         num_heads, dropout_rate, residual);
        gt_block->push_back(gt);
    }
    gt_block->push_back(FinalTransformerModule(num_hidden_channels,
                                               num_heads, dropout_rate, residual));
    register_module("gt_block", gt_block);
}

at::Tensor GraphTransformerImpl::forward(at::Tensor node_feats, at::Tensor edge_feats, at::Tensor edge_index) {
    node_feats = node_encoder->forward(node_feats);
    edge_feats = edge_encoder->forward(edge_feats);
    // std::cout << node_feats << std::endl;
    int num_layers = 0;
    for (auto gt_layer : *gt_block) {
        if (num_layers != gt_block->size() - 1) {
            auto layer = gt_layer->as<GraphTransformerModule>();
            auto gt_out = layer->forward(node_feats, edge_feats, edge_index);
            node_feats = gt_out.first;
            edge_feats = gt_out.second;
        } else {
            auto layer = gt_layer->as<FinalTransformerModule>();
            node_feats = layer->forward(node_feats, edge_feats, edge_index);
        }
        num_layers++;
    }

    return node_feats;
}

//////////////////////////////////////////////

TriScoreImpl::TriScoreImpl(GraphTransformer ligand_model_, GraphTransformer target_model_,
                           int in_channels_, int hidden_dim_, int n_gaussians_, float dropout_)
    : ligand_model(register_module("ligand_model", ligand_model_)),
    target_model(register_module("target_model", target_model_)), in_channels(in_channels_),
    hidden_dim(hidden_dim_), n_gaussians(n_gaussians_) {

    z_pi = register_module("z_pi", torch::nn::Linear(hidden_dim, n_gaussians));
    z_sigma = register_module("z_sigma", torch::nn::Linear(hidden_dim, n_gaussians));
    z_mu = register_module("z_mu", torch::nn::Linear(hidden_dim, n_gaussians));
    atom_types = register_module("atom_types", torch::nn::Linear(in_channels, 17));
    bond_types = register_module("bond_types", torch::nn::Linear(in_channels * 2, 4));
    MLP = register_module(
        "MLP", torch::nn::Sequential(torch::nn::Linear(in_channels * 2, hidden_dim),
                                     torch::nn::BatchNorm1d(hidden_dim),
                                     torch::nn::ELU(),
                                     torch::nn::Dropout(dropout_))
    );
}

std::pair<torch::Tensor, torch::Tensor> TriScoreImpl::forward(Batch& lig, Batch& rec) {
    auto h_l = ligand_model->forward(lig.node_feats, lig.edge_feats, lig.edge_index);
    auto h_t = target_model->forward(rec.node_feats, rec.edge_feats, rec.edge_index);

    auto h_l_out = to_dense_batch(h_l, lig.batch);
    auto h_l_x = h_l_out.first;
    auto l_mask = h_l_out.second;
    auto h_t_out = to_dense_batch(h_t, rec.batch);
    auto h_t_x = h_t_out.first;
    auto t_mask = h_t_out.second;

    auto B = h_l_x.size(0);
    auto N_l = h_l_x.size(1);
    auto N_t = h_t_x.size(1);

    // Combine and mask;
    h_l_x = h_l_x.unsqueeze(-2).expand({ B, N_l, N_t, -1 }); // [B, N_l, N_t, C_out];
    h_t_x = h_t_x.unsqueeze(-3).expand({ B, N_l, N_t, -1 }); // [B, N_l, N_t, C_out];

    auto C = torch::cat((h_l_x, h_t_x), -1);
    auto C_mask = torch::logical_and(l_mask.view({ B, N_l, 1 }), t_mask.view({ B, 1, N_t }));
    C = C.masked_select(C_mask);
    C = MLP->forward(C);

    // Outputs
    auto rho = torch::softmax(z_pi(C), -1);
    auto sigma = torch::elu(z_sigma(C)) + 1.1;
    auto mu = torch::elu(z_mu(C)) + 1;
    auto output = torch::stack({ rho, sigma, mu }, 0);
    return std::make_pair(output, C_mask);
}

at::Tensor TriScoreImpl::single(Graph& lig, Graph& rec) {
    auto h_l = ligand_model->forward(lig.node_feats, lig.edge_feats, lig.edge_index);
    auto h_t = target_model->forward(rec.node_feats, rec.edge_feats, rec.edge_index);
    // std::cout << h_l << std::endl;
    auto N_l = h_l.size(0);
    auto N_t = h_t.size(0);

    // Combine and mask;
    h_l = h_l.unsqueeze(1).expand({ N_l, N_t, -1 }); // [N_l, N_t, C_out];
    h_t = h_t.unsqueeze(0).expand({ N_l, N_t, -1 }); // [N_l, N_t, C_out];

    auto mix = torch::cat({ h_l, h_t }, -1).view({ N_l * N_t, -1 });
    auto C = MLP->forward(mix);

    // Outputs
    auto rho = torch::softmax(z_pi(C), -1);
    auto sigma = torch::elu(z_sigma(C)) + 1.1;
    auto mu = torch::elu(z_mu(C)) + 1;
    auto output = torch::stack({ rho, mu, sigma }, 0);
    return output;
}

////////////////////////////////////////////