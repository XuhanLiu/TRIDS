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
#include <openbabel/obconversion.h>
#include <openbabel/mol.h>
#include <openbabel/atom.h>
#include <openbabel/obiter.h>
#include <openbabel/builder.h>

#include <filesystem>

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
    return torch::TensorOptions().device(device).dtype(torch::kFloat);
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

OpenBabel::OBMol* load_molecule(const std::string& path) {
    OpenBabel::OBConversion conv;
    auto suffix = std::filesystem::path(path).extension().string();
    if (!suffix.empty() && suffix[0] == '.') {
        suffix = suffix.substr(1);
    }
    conv.SetInFormat(suffix.c_str());

    auto mol = new OpenBabel::OBMol();
    if (!conv.ReadFile(mol, path)) {
        delete mol;
        throw std::runtime_error("Failed to load molecule from: " + path);
    }
    return mol;
}

OpenBabel::OBMol* mol_from_smiles(const std::string& smiles, bool gen_3d) {
    OpenBabel::OBConversion conv;
    conv.SetInFormat("smi");

    auto mol = new OpenBabel::OBMol();
    if (!conv.ReadString(mol, smiles)) {
        delete mol;
        throw std::runtime_error("Failed to parse SMILES: " + smiles);
    }

    if (gen_3d) {
        OpenBabel::OBBuilder builder;
        if (!builder.Build(*mol)) {
            delete mol;
            throw std::runtime_error("Could not initiate the coordinate of the input ligand!");
        }
    }
    return mol;
}

std::shared_ptr<Receptor> make_pocket(
    OpenBabel::OBMol* receptor,
    torch::Tensor ref_coords,
    float radius,
    bool is_vina,
    const std::string& device
) {
    auto options = make_options(device);
    RefSiter siter({}, options, radius);
    auto pos = ref_coords.to(torch::kDouble).cpu().contiguous();
    auto* raw = siter.extract(receptor, pos, is_vina);
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
    bool is_vina,
    const std::string& device
) {
    auto options = make_options(device);
    auto site_model = torch::jit::load(model_path, options.device());
    site_model.eval();
    ModelSiter model_siter(site_model, receptor, options, cutoff);
    model_siter.init(receptor, is_vina);
    return take_pockets(model_siter);
}


PYBIND11_MODULE(_core, m) {
    m.doc() = "TRIDS: Deep Learning-based Molecular Docking - C++ Core Module";

    m.def("set_verbose", &OpenBabel::set_verbose, py::arg("verbose"),
          R"doc(Set C++ log verbosity. 0=error, 1=warn, 2=info. Default is 0.)doc");
    m.def("get_verbose", &OpenBabel::get_verbose,
          R"doc(Return current C++ log verbosity level.)doc");
    OpenBabel::set_verbose(0);

    py::class_<Graph>(m, "_Graph")
        .def_readonly("node_feats", &Graph::node_feats)
        .def_readonly("edge_feats", &Graph::edge_feats)
        .def_readonly("edge_index", &Graph::edge_index);

    py::class_<OpenBabel::OBMol>(m, "_OBMol")
        .def_static("load", &load_molecule, py::return_value_policy::take_ownership)
        .def_static("from_smiles", &mol_from_smiles,
                    py::return_value_policy::take_ownership,
                    py::arg("smiles"), py::arg("gen_3d") = true)
        .def_property_readonly("num_atoms",
            [](OpenBabel::OBMol* mol) { return mol->NumAtoms(); })
        .def_property_readonly("num_bonds",
            [](OpenBabel::OBMol* mol) { return mol->NumBonds(); })
        .def_property_readonly("num_residues",
            [](OpenBabel::OBMol* mol) { return mol->NumResidues(); })
        .def_property("title",
            [](OpenBabel::OBMol* mol) { return std::string(mol->GetTitle()); },
            [](OpenBabel::OBMol* mol, const std::string& title) {
                std::string copy = title;
                mol->SetTitle(copy);
            })
        .def_property("coordinates",
            [](OpenBabel::OBMol* mol) {
                auto coords = torch::empty({mol->NumAtoms(), 3}, torch::kFloat);
                auto accessor = coords.accessor<float, 2>();
                FOR_ATOMS_OF_MOL(atom, mol) {
                    int idx = atom->GetIndex();
                    accessor[idx][0] = atom->x();
                    accessor[idx][1] = atom->y();
                    accessor[idx][2] = atom->z();
                }
                return coords;
            },
            [](OpenBabel::OBMol* mol, torch::Tensor coords) {
                auto cpu_coords = coords.cpu();
                auto accessor = cpu_coords.accessor<float, 2>();
                FOR_ATOMS_OF_MOL(atom, mol) {
                    int idx = atom->GetIndex();
                    atom->SetVector(accessor[idx][0], accessor[idx][1], accessor[idx][2]);
                }
            })
        .def("copy", [](OpenBabel::OBMol* mol) {
            return new OpenBabel::OBMol(*mol);
        }, py::return_value_policy::take_ownership)
        .def("write", [](OpenBabel::OBMol* mol, const std::string& path, const std::string& format) {
            OpenBabel::OBConversion conv;
            conv.SetOutFormat(format.c_str());
            std::ofstream ofs(path);
            conv.Write(mol, &ofs);
        }, py::arg("path"), py::arg("format"))
        .def("delete_hydrogens", [](OpenBabel::OBMol* mol) { mol->DeleteHydrogens(); })
        .def("to_graph", [](OpenBabel::OBMol* mol) { return Feature::obmol_to_graph(mol); })
        .def("extract_pocket", &make_pocket,
             py::arg("ref_coords"), py::arg("radius") = TRIDS_POCKET_CUTOFF,
             py::arg("is_vina") = false, py::arg("device") = "cuda:0")
        .def("predict_pockets", &predict_pockets,
             py::arg("model_path"), py::arg("cutoff") = TRIDS_POCKET_CUTOFF,
             py::arg("is_vina") = false, py::arg("device") = "cuda:0");

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
            RefSiter siter({}, receptor->options);
            auto pos = ref_coords.to(torch::kDouble).cpu().contiguous();
            std::tie(receptor->box_min, receptor->box_max) = siter.get_box(pos);
        })
        .def("to_graph", [](const Receptor& rec) {
            const auto& g = rec.graph;
            return std::make_tuple(g.node_feats.cpu(), g.edge_feats.cpu(), g.edge_index.cpu());
        });

    py::class_<Scorer, std::shared_ptr<Scorer>>(m, "_Scorer")
        .def("set_pocket", [](std::shared_ptr<Scorer> scorer, std::shared_ptr<Receptor> receptor) {
            receptor->to(scorer->options);
            scorer->set_pocket(*receptor);
        })
        .def("set_ligand", &Scorer::set_ligand)
        .def("profile", &Scorer::profiling)
        .def("score", [](std::shared_ptr<Scorer> scorer, torch::Tensor coords) {
            at::NoGradGuard guard;
            return scorer->scoring(coords.to(scorer->options))
                / (1 + scorer->num_tors * TRIDS_OMEGA);
        });

    py::class_<TriScorer, Scorer, std::shared_ptr<TriScorer>>(m, "_TriScorer")
        .def(py::init([](const std::string& model_path, const std::string& device,
                         float cutoff, float beta) {
            auto options = make_options(device);
            auto model = TriScore(GraphTransformer(42, 9), GraphTransformer(41, 5));
            model->eval();
            torch::load(model, model_path, options.device());
            return std::make_shared<TriScorer>(model, options, cutoff, beta);
        }), py::arg("model_path"), py::arg("device") = "cuda:0",
           py::arg("cutoff") = 10.0f, py::arg("beta") = 0.069314718f);

    py::class_<VinaScorer, Scorer, std::shared_ptr<VinaScorer>>(m, "_VinaScorer")
        .def(py::init([](const std::string& device, float cutoff, float beta) {
            return std::make_shared<VinaScorer>(make_options(device), cutoff, beta);
        }), py::arg("device") = "cuda:0", py::arg("cutoff") = 8.0f,
           py::arg("beta") = 0.069314718f);

    py::class_<Conformer, std::shared_ptr<Conformer>>(m, "_Conformer")
        .def(py::init([](OpenBabel::OBMol* ligand, int streams, const std::string& device) {
            return std::make_shared<Conformer>(ligand, streams, make_options(device));
        }), py::arg("ligand"), py::arg("streams"), py::arg("device") = "cuda:0");

    py::class_<Engine>(m, "_Engine")
        .def(py::init([](std::shared_ptr<Scorer> scorer,
                         int streams, int depth, int top_n, const std::string& device) {
            return Engine(scorer, streams, depth, top_n, make_options(device));
        }), py::arg("scorer"),
           py::arg("streams") = 256, py::arg("depth") = 32,
           py::arg("top_n") = 1, py::arg("device") = "cuda:0")
        .def("scoring", &Engine::scoring)
        .def("docking", [](Engine& engine, std::shared_ptr<Conformer> conformer) {
            RNG rng(auto_seed());
            torch::Tensor best_coords;
            torch::Tensor best_scores;
            {
                py::gil_scoped_release release;
                std::tie(best_coords, best_scores) = engine.docking(*conformer, rng);
            }
            Cluster cluster;
            cluster.sort_structures(best_coords, best_scores, 0.5f, static_cast<int>(engine.topn));
            std::vector<torch::Tensor> coords_list;
            coords_list.reserve(cluster.selected_numbers);
            for (int i = 0; i < cluster.selected_numbers; i++) {
                coords_list.push_back(cluster.selected_coord[i].clone());
            }
            return std::make_pair(
                coords_list,
                cluster.selected_score.slice(0, 0, cluster.selected_numbers).clone()
            );
        }, py::arg("conformer"));
}
