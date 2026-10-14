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

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/extension.h>
#include <openbabel/mol.h>

#include <cstdint>

#include "scorer.h"
#include "vina_scorer.h"
#include "engine.h"
#include "conformer.h"
#include "feat.h"
#include "siter.h"
#include "cluster.h"
#include "rand.h"
#include "constant.h"
#include "log.h"

namespace py = pybind11;


torch::TensorOptions make_options(const std::string& device) {
    return at::TensorOptions().device(device).dtype(torch::kFloat);
}

std::vector<std::shared_ptr<Receptor>> take_pockets(Siter& siter) {
    std::vector<std::shared_ptr<Receptor>> results;
    results.reserve(siter.pockets.size());
    for (auto* r : siter.pockets) {
        results.emplace_back(r);
    }
    siter.pockets.clear();
    return results;
}

OpenBabel::OBMol* mol_from_ptr(std::uintptr_t ptr) {
    if (ptr == 0) {
        throw std::runtime_error("Null molecule pointer");
    }
    return reinterpret_cast<OpenBabel::OBMol*>(ptr);
}

std::shared_ptr<Receptor> make_pocket(
    OpenBabel::OBMol* receptor,
    torch::Tensor ref_coords,
    float radius,
    bool only_vina,
    const std::string& device
) {
    OPTIONS = make_options(device);
    RefSiter siter({}, radius);
    auto pos = ref_coords.to(torch::kDouble).cpu().contiguous();
    auto* raw = siter.extract(receptor, pos, only_vina);
    if (raw->num_res < 3) {
        delete raw;
        throw std::runtime_error("Failed to extract binding pocket");
    }
    std::tie(raw->box_min, raw->box_max) = siter.get_box(pos);
    return std::shared_ptr<Receptor>(raw);
}

std::vector<std::shared_ptr<Receptor>> predict_pockets(
    OpenBabel::OBMol* receptor,
    const std::string& model_path,
    float cutoff,
    bool only_vina,
    const std::string& device
) {
    OPTIONS = make_options(device);
    auto site_model = torch::jit::load(model_path, OPTIONS.device());
    site_model.eval();
    ModelSiter model_siter(site_model, receptor, cutoff);
    model_siter.init(receptor, only_vina);
    return take_pockets(model_siter);
}

std::vector<std::shared_ptr<Receptor>> load_pockets(
    OpenBabel::OBMol* receptor,
    const std::string& path,
    float cutoff,
    bool only_vina,
    const std::string& device
) {
    OPTIONS = make_options(device);
    std::vector<torch::Tensor> clusters;
    torch::load(clusters, path);
    GridSiter siter(clusters, cutoff);
    siter.init(receptor, only_vina);
    return take_pockets(siter);
}

std::pair<std::vector<torch::Tensor>, torch::Tensor> cluster_poses(
    torch::Tensor coords,
    torch::Tensor scores,
    float rmsd,
    int top_n
) {
    Cluster cluster;
    cluster.sort_structures(coords, scores, rmsd, top_n);
    std::vector<torch::Tensor> coords_list;
    coords_list.reserve(cluster.selected_numbers);
    for (int i = 0; i < cluster.selected_numbers; i++) {
        coords_list.push_back(cluster.selected_coord[i].clone());
    }
    return {
        coords_list,
        cluster.selected_score.slice(0, 0, cluster.selected_numbers).clone()
    };
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "TRIDS: Deep Learning-based Molecular Docking - C++ Core Module";

    m.def("set_verbose", &OpenBabel::set_verbose, py::arg("verbose"),
          R"doc(Set C++ log verbosity. 0=error, 1=warn, 2=info. Default is 0.)doc");
    m.def("get_verbose", &OpenBabel::get_verbose,
          R"doc(Return current C++ log verbosity level.)doc");
    m.def("set_seed", &set_seed, py::arg("seed"));
    m.def("cluster_poses", &cluster_poses,
          py::arg("coords"), py::arg("scores"), py::arg("rmsd") = 0.5f, py::arg("top_n"));
    OpenBabel::set_verbose(0);

    py::class_<Graph>(m, "_Graph")
        .def_readonly("node_feats", &Graph::node_feats)
        .def_readonly("edge_feats", &Graph::edge_feats)
        .def_readonly("edge_index", &Graph::edge_index);

    m.def("mol_to_graph", [](std::uintptr_t mol) {
        return Feature::obmol_to_graph(mol_from_ptr(mol));
    }, py::arg("mol"));
    m.def("extract_pocket", [](std::uintptr_t mol, torch::Tensor ref_coords, float radius,
                               bool only_vina, const std::string& device) {
        return make_pocket(mol_from_ptr(mol), ref_coords, radius, only_vina, device);
    }, py::arg("mol"), py::arg("ref_coords"), py::arg("radius") = TRIDS_POCKET_CUTOFF,
       py::arg("only_vina") = false, py::arg("device") = "cuda:0");
    m.def("predict_pockets", [](std::uintptr_t mol, const std::string& model_path, float cutoff,
                                bool only_vina, const std::string& device) {
        return predict_pockets(mol_from_ptr(mol), model_path, cutoff, only_vina, device);
    }, py::arg("mol"), py::arg("model_path"), py::arg("cutoff") = TRIDS_POCKET_CUTOFF,
       py::arg("only_vina") = false, py::arg("device") = "cuda:0");
    m.def("load_pockets", [](std::uintptr_t mol, const std::string& path, float cutoff,
                             bool only_vina, const std::string& device) {
        return load_pockets(mol_from_ptr(mol), path, cutoff, only_vina, device);
    }, py::arg("mol"), py::arg("path"), py::arg("cutoff") = TRIDS_POCKET_CUTOFF,
       py::arg("only_vina") = false, py::arg("device") = "cuda:0");

    py::class_<Receptor, std::shared_ptr<Receptor>>(m, "_Receptor")
        .def_readonly("box_min", &Receptor::box_min)
        .def_readonly("box_max", &Receptor::box_max)
        .def_property_readonly("num_atoms",
            [](const Receptor& rec) { return rec.pos.size(0); })
        .def_property_readonly("num_residues",
            [](const Receptor& rec) { return rec.num_res; })
        .def_property_readonly("center",
            [](const Receptor& rec) { return (rec.box_min + rec.box_max) / 2; })
        .def("set_box", [](std::shared_ptr<Receptor> receptor, torch::Tensor ref_coords) {
            RefSiter siter({});
            auto pos = ref_coords.to(torch::kDouble).cpu().contiguous();
            std::tie(receptor->box_min, receptor->box_max) = siter.get_box(pos);
        })
        .def("to_graph", [](const Receptor& rec) {
            const auto& g = rec.graph;
            return std::make_tuple(g.node_feats.cpu(), g.edge_feats.cpu(), g.edge_index.cpu());
        });

    py::class_<Scorer, std::shared_ptr<Scorer>>(m, "_Scorer")
        .def("set_pocket", [](std::shared_ptr<Scorer> scorer, std::shared_ptr<Receptor> receptor) {
            receptor->to(OPTIONS);
            scorer->set_pocket(*receptor);
        })
        .def("set_ligand", [](std::shared_ptr<Scorer> scorer, std::uintptr_t mol) {
            scorer->set_ligand(mol_from_ptr(mol));
        }, py::arg("mol"))
        .def("profile", &Scorer::profiling)
        .def("score", [](std::shared_ptr<Scorer> scorer, torch::Tensor coords) {
            at::NoGradGuard guard;
            return scorer->scoring(coords.to(OPTIONS))
                / (1 + scorer->num_tors * scorer->omega);
        });

    py::class_<TriScorer, Scorer, std::shared_ptr<TriScorer>>(m, "_TriScorer")
        .def(py::init([](const std::string& model_path, const std::string& device,
                         float cutoff, float beta) {
            OPTIONS = make_options(device);
            auto model = TriScore(GraphTransformer(42, 9), GraphTransformer(41, 5));
            model->eval();
            torch::load(model, model_path, OPTIONS.device());
            return std::make_shared<TriScorer>(model, cutoff, beta);
        }), py::arg("model_path"), py::arg("device") = "cuda:0",
           py::arg("cutoff") = 10.0f, py::arg("beta") = 0.069314718f);

    py::class_<VinaScorer, Scorer, std::shared_ptr<VinaScorer>>(m, "_VinaScorer")
        .def(py::init([](const std::string& device, float cutoff, float beta) {
            OPTIONS = make_options(device);
            return std::make_shared<VinaScorer>(cutoff, beta);
        }), py::arg("device") = "cuda:0", py::arg("cutoff") = 8.0f,
           py::arg("beta") = 0.838718f);

    py::class_<Conformer, std::shared_ptr<Conformer>>(m, "_Conformer")
        .def(py::init([](std::uintptr_t ligand, int streams, const std::string& device) {
            OPTIONS = make_options(device);
            return std::make_shared<Conformer>(mol_from_ptr(ligand), streams);
        }), py::arg("ligand"), py::arg("streams"), py::arg("device") = "cuda:0");

    py::class_<Engine>(m, "_Engine")
        .def(py::init([](std::shared_ptr<Scorer> scorer,
                         int streams, int depth, int top_n, const std::string& device) {
            OPTIONS = make_options(device);
            return Engine(scorer, streams, depth, top_n);
        }), py::arg("scorer"),
           py::arg("streams") = 1024, py::arg("depth") = 8,
           py::arg("top_n") = 1, py::arg("device") = "cuda:0")
        .def("scoring", &Engine::scoring)
        .def("docking", [](Engine& engine, std::shared_ptr<Conformer> conformer,
                          std::shared_ptr<Scorer> screener) {
            torch::Tensor best_coords;
            torch::Tensor best_scores;
            {
                py::gil_scoped_release release;
                std::tie(best_coords, best_scores) = engine.docking(*conformer);
                if (screener) {
                    at::NoGradGuard guard;
                    best_scores = screener->batched_auto_scoring(best_coords.to(OPTIONS));
                }
            }
            return std::make_pair(best_coords, best_scores);
        }, py::arg("conformer"), py::arg("screener") = nullptr);
}
