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

#include <openbabel/obconversion.h>
#include <cuda_runtime.h>
#include <c10/cuda/CUDACachingAllocator.h>
#include <torch/script.h>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <spdlog/spdlog.h>
#include <CLI/CLI.hpp>
#include "conformer.h"
#include "scorer.h"
#include "siter.h"
#include "vina_scorer.h"
#include "error.h"
#include "engine.h"
#include "constant.h"
#include "cluster.h"
#include "log.h"

#if defined(_WIN32)
#define strncasecmp _strnicmp
#endif

#define SAMPLE_MODEL "libtrids_sampling.pt"
#define SCORE_MODEL "libtrids_scoring.pt"
#define POCKET_MODEL "libtrids_site_binding.pt"

namespace fs = std::filesystem; 

// Find lib path (cross-platform, searches PATH if needed)
fs::path find_lib_path(const char* argv0) {
    // Check ../lib relative to argv0 (typical layout: bin/trids, lib/...)
    auto local_lib = fs::path(argv0).parent_path() / ".." / "lib";
    if (fs::exists(local_lib)) {
        return fs::weakly_canonical(local_lib);
    }
    
    // Search PATH (: on Unix, ; on Windows)
#ifdef _WIN32
    constexpr char sep = ';';
#else
    constexpr char sep = ':';
#endif
    
    if (auto env = std::getenv("PATH")) {
        std::istringstream ss(env);
        for (std::string dir; std::getline(ss, dir, sep); ) {
            if (auto f = fs::path(dir); fs::exists(f)) {
                auto lib = fs::canonical(f) / ".." / "lib";
                if (fs::exists(lib)) return lib;
            }
        }
    }

    throw std::runtime_error("Error: Library path not found!");
}

int main(int argc, char* argv[]) {
    auto total_start = clock();
    const std::string version = "1.0.0";
    const std::string version_string = "TRIDS: Built " __DATE__ ". Version: " + version;
    const std::string error_message = "\n\n\
Please report this error at https://github.com/trids/trids/issues\n"
"Please remember to include the following in your problem report:\n\
	* the EXACT error message,\n\
	* your version of the program,\n\
	* the type of computer system you are running it on,\n\
	* all command line options,\n\
	* configuration file (if used),\n\
	* ligand file as provided to TRIDS,\n\
	* receptor file as provided to TRIDS,\n\
	* output file (if any),\n\
	* random seed the program used (this is printed when the program starts).\n\
\n\
Thank you!\n";

    try {
        std::string lig_path, rec_path, ref_path;
        std::string out_path;
        int cpu, cuda, seed = -1, num_tasks, max_depth, topn, verbose;
        bool score_only = false, is_vina = false, is_hts = false;
        
        // Determine the library path for model files
        auto base_lib = find_lib_path(argv[0]);
        std::string lib_path = base_lib.string();
        
        // Check if lib/ contains a directory starting with "trids"
        if (!fs::exists(base_lib / POCKET_MODEL) || !fs::exists(base_lib / SAMPLE_MODEL)) {
            lib_path = (base_lib / ("trids" + version)).string();
            if (lib_path.empty()) {
                throw std::runtime_error("Error: Library path not found!");
            }
        } 
        
        // Begin: Parse the input parameters
        CLI::App app{ "TRIDS" };
        app.description(version_string);
        app.get_formatter()->column_width(40);
        app.get_formatter()->label("FLOAT", "<float>");
        app.get_formatter()->label("INT", "<int>");
        app.get_formatter()->label("UINT", "<uint>");
        app.get_formatter()->label("TEXT", "<string>");
        app.get_formatter()->label("{false}", "");
        app.footer("Reference: \n\n\tLiu X., Zhang H. et al. TRIDS: An unified molecular docking framework integrated "
                "with deep learning-based site binding, sampling and scoring. Preprint.");
        // desc.add_options("Input")
        app.add_option("-r,--receptor", rec_path, "Rigid part of the receptor [REQUIRED]")
            ->required()
            ->option_text("<pdb>")->check(CLI::ExistingFile);
        
            app.add_option("-l,--ligand", lig_path, "Ligand [REQUIRED]")
            ->required()
            ->option_text("<smi, mol2, sdf, pdb>")->check(CLI::ExistingFile);

        app.add_option("-k,--pocket", ref_path, "Reference profile for Binding pocket identification! [REQUIRED]")
            ->default_val(lib_path + "/" + POCKET_MODEL)
            ->option_text("<pt, pth, mol2, sdf, pdb>")->check(CLI::ExistingFile);

        app.add_flag("--hts", is_hts, "High throughput screening mode with higher virtual screening accuracy but lower binding pose accuracy.");

        app.add_flag("--vina", is_vina, "Conformation sampling with Vina forcefiled");

        // desc.add_options("Output")
        app.add_option("-o,--out", out_path, "File path for outputing docking results, the format of molecules is based on file extension");
            
        app.add_option("-e,--stream", num_tasks, "Max number of sampling for Monte carlo research")
            ->option_text("<uint> [256]")
            ->default_val(1024)->check(CLI::PositiveNumber)->check(CLI::Range(1, INT_MAX));

        app.add_option("-d,--depth", max_depth, "Max depths for Monte carlo research")
            ->option_text("<uint> [32]")
            ->default_val(8)->check(CLI::PositiveNumber)->check(CLI::Range(1, INT_MAX));

        app.add_option("-t,--top", topn, "Record Number of N best conformers in output")
        ->option_text("<smi, mol2, sdf, pdb>")    
        ->default_val(1)->check(CLI::PositiveNumber)->check(CLI::Range(1, INT_MAX));

        app.add_option("--seed", seed, "User defined random seed")
            ->option_text("<int> [0]")
            ->check(CLI::PositiveNumber)->check(CLI::Range(0, INT_MAX));

        app.add_option("-c,--cpu", cpu, "Number of CPU cores")
            ->option_text("<uint> [1]")
            ->default_val(1)->check(CLI::PositiveNumber)->check(CLI::Range(1, INT_MAX));

        app.add_option("-g,--cuda", cuda, "Index of Nvidia CUDA Device. If set to -1, no CUDA device will be used")
            ->option_text("<int> [0]")
            ->default_val(0)->check(CLI::Range(-1, INT_MAX));

        app.add_flag("--score_only", score_only, "Only calculating the score of given conformation without sampling");

        app.add_option("-v,--verbose", verbose, "Verbose mode, print more information, 0: Error, 1: Warning, 2: Info")
            ->option_text("<0, 1, 2> [0]")
            ->default_val(0)->check(CLI::Range(0, 2));

        // desc.add_options("Configuration file (optional)")
        app.set_config("--config");

        CLI11_PARSE(app, argc, argv);

        /////////////////// End: Parse the input parameters ...////////////////////////////////////    
        OpenBabel::set_verbose(verbose);
        
        spdlog::info("CUDA support: {}", torch::cuda::is_available() ? "Yes" : "No");
        spdlog::info("CUDNN support: {}", torch::cuda::cudnn_is_available() ? "Yes" : "No");

        if (seed < 0) {
            seed = auto_seed();
        }
        RNG rng = RNG(seed);
        torch::manual_seed(seed);

        auto dev_id = cuda < 0 ? "cpu" : "cuda:" + to_string(cuda);
        if (cuda >= 0) {
            cudaSetDevice(cuda);
        }
        torch::set_num_threads(cpu);
        torch::InferenceMode(true);
        torch::jit::setGraphExecutorOptimize(false);
        auto options = torch::TensorOptions().device(dev_id).dtype(torch::kFloat);

        OpenBabel::OBConversion conv;
        OpenBabel::OBMol* lig = new OpenBabel::OBMol();
        OpenBabel::OBMol* rec = new OpenBabel::OBMol();
        std::vector<OpenBabel::OBMol*> refs;

        spdlog::info("Parsing input receptor file begin: {}", rec_path);
        auto start = clock();
        auto suffix = std::filesystem::path(rec_path).extension().string();
        conv.SetInFormat(suffix.c_str());
        conv.SetOutFormat("sdf");

        conv.ReadFile(rec, rec_path); // Note: rec should not be nullptr;
        
        rec->DeleteHydrogens();
        if (is_vina) {
            for (auto it = rec->EndResidues() - 1; it >= rec->BeginResidues(); it--) {
                auto res = *it;
                if (res->GetName() == "HOH") {
                    rec->DeleteResidue(res);
                }
            }
        }

        auto duration = static_cast<float>(clock() - start) / CLOCKS_PER_SEC;
        spdlog::info("Parsing input receptor file end ({:.4f}s)", duration);

        spdlog::info("Construct binding site model begin ...");
        start = clock();
        shared_ptr<Siter> siter;
        auto ref_suffex = std::filesystem::path(ref_path).extension().string();
        if (ref_suffex == ".pt" && (!score_only)) {
            spdlog::info("Determing binding pocket with AI model ...");
            auto site_model = torch::jit::load(ref_path, options.device());
            site_model.eval();
            siter = std::make_shared<ModelSiter>(site_model, rec, options);
        } else if (ref_suffex == ".dat") {
            spdlog::info("Determing binding pocket with the profile ...");
            std::vector<torch::Tensor> clusters;
            torch::load(clusters, ref_path);
            siter = std::make_shared<GridSiter>(clusters, options);
        } else {
            spdlog::info("Parsing reference ligand for determining bingding pocket: {}", ref_path);
            suffix = ref_path.substr(ref_path.find_last_of('.') + 1);
            conv.SetInFormat(suffix.c_str());

            auto ref = new OpenBabel::OBMol();
            auto has_ref = conv.ReadFile(ref, ref_path);
            while (has_ref) {
                if (ref->NumAtoms() > 0) {
                    ref->DeleteHydrogens();
                    refs.push_back(ref);
                } else {
                    delete ref;
                }
                ref = new OpenBabel::OBMol();
                has_ref = conv.Read(ref);
            }
            siter = std::make_shared<RefSiter>(refs, options);
        }

        siter->init(rec, is_vina); 

        duration = static_cast<float>(clock() - start) / CLOCKS_PER_SEC;
        spdlog::info("Construct binding site model end ({:.4f}s)", duration);

        start = clock();
        spdlog::info("Initiating scoring functions begin ...");
        shared_ptr<Scorer> scorer, screener;
        
        // Load TriScore unless Vina-only (vina without HTS, including score_only).
        const bool load_model = !is_vina || (is_hts && !score_only);
        if (load_model) {
            start = clock();
            spdlog::info("Construct Pytorch JIT-based scorer begin ...");
            
            auto model_path = is_hts ? lib_path + "/" + SCORE_MODEL : lib_path + "/" + SAMPLE_MODEL;
        
            auto lig_model = GraphTransformer(42, 9);
            auto rec_model = GraphTransformer(41, 5);
            auto model = TriScore(lig_model, rec_model);
            // Load weights directly to target device (avoid CPU->GPU copy overhead)
            torch::load(model, model_path, options.device());
            model->eval();

            duration = static_cast<float>(clock() - start) / CLOCKS_PER_SEC;
            spdlog::info("Construct Pytorch JIT-based scorer end ({:.4f}s)", duration);
            screener = std::make_shared<TriScorer>(model, options);
            if (is_vina) {
                spdlog::info("Construct Vina scorer begin ...");
                scorer = std::make_shared<VinaScorer>(options);
            } else {
                scorer = screener;
            }
        } else {
            start = clock();
            spdlog::info("Construct Vina scorer begin ...");
            scorer = std::make_shared<VinaScorer>(options);
            screener = scorer;
            duration = static_cast<float>(clock() - start) / CLOCKS_PER_SEC;
            spdlog::info("Construct Vina scorer end ({:.4f}s)", duration);
        }

        duration = static_cast<float>(clock() - start) / CLOCKS_PER_SEC;
        spdlog::info("Initiating scoring functions end ({:.4f}s)", duration);

        spdlog::info("Parsing input ligand file begin: {}", lig_path);
        start = clock();

        suffix = std::filesystem::path(lig_path).extension().string();
        conv.SetInFormat(suffix.c_str());
        bool has_mol = conv.ReadFile(lig, lig_path);

        duration = static_cast<float>(clock() - start) / CLOCKS_PER_SEC;
        spdlog::info("Parsing input ligand file end ({:.4f}s)", duration);
        std::shared_ptr<Engine> engine;

        engine = std::make_shared<Engine>(scorer, num_tasks, max_depth, topn, options);
        stringstream out;
        int i = 0;
        while (has_mol) {
            if (lig == nullptr) {
                continue;
            }
            auto title = std::string(lig->GetTitle());
            if (title.empty()) {
                title = "TRIDS_" + std::to_string(i);
            }
            auto iter_start = clock();
            spdlog::info("Docking Ligand {} begin ...", title);
            start = clock();
            spdlog::info("Scoring function receives ligand begin ...");
            lig->DeleteHydrogens();

            engine->scorer->set_ligand(lig);
            duration = static_cast<float>(clock() - start) / CLOCKS_PER_SEC;
            spdlog::info("Scoring function receives ligand end ({:.4f}s)", duration);

            if (score_only) {
                start = clock();
                spdlog::info("Running scoring function begin ...");
                spdlog::info("Extract binding pocket begin ...");
                
                auto lig_pos = torch::from_blob(lig->GetCoordinates(), { lig->NumAtoms(), 3 }, torch::kDouble).clone();
                if (refs.empty()) {
                    auto pocket = new Receptor(rec, lig_pos, is_vina, siter->cutoff, options);
                    auto [box_min, box_max] = siter->get_box(lig_pos);
                    pocket->box_min = box_min;
                    pocket->box_max = box_max;
                    if (siter->pockets.empty()) {
                        siter->pockets.push_back(pocket);
                    } else {
                        delete siter->pockets[0];
                        siter->pockets[0] = pocket;
                    }
                }
                engine->scorer->set_pocket(*siter->pockets[0]);
                engine->scorer->profiling();
                duration = static_cast<float>(clock() - start) / CLOCKS_PER_SEC;
                spdlog::info("Extract binding pocket end ({:.4f}s)", duration);
                
                auto score = engine->scoring(lig_pos.to(options)).item<float>();
                out << title << "\t" << score << std::endl;
                //out << title << "\t" << score << "\t" << scorer->num_tors << "\t" << siter->pockets[0]->num_res << std::endl;
                duration = static_cast<float>(clock() - start) / CLOCKS_PER_SEC;
                spdlog::info("Running scoring function end ({:.4f}s)", duration);
            } else {
                if(scorer != screener) {
                    screener->set_ligand(lig);
                }
                std::vector<at::Tensor> coord_list;
                std::vector<at::Tensor> score_list;

                for (auto& pocket : siter->pockets) {
                    engine->scorer->set_pocket(*pocket);
                    engine->scorer->profiling();
                    if(scorer != screener) {
                        screener->set_pocket(*pocket);
                        screener->profiling();
                    }
                    
                    start = clock();
                    spdlog::info("Initiating sampling function begin ...");
                    auto conformer = std::make_shared<Conformer>(lig, num_tasks, options);

                    duration = static_cast<float>(clock() - start) / CLOCKS_PER_SEC;
                    spdlog::info("Initiating sampling function end ({:.4f}s)", duration);

                    start = clock();
                    spdlog::info("Sampling the best conformation begin ...");
                    auto best_out = engine->docking(*conformer, rng);
                    auto [best_coord, best_score] = best_out;
                    if (screener != scorer) {
                        at::NoGradGuard guard;
                        best_score = screener->batched_auto_scoring(best_coord);
                    }
                    duration = static_cast<float>(clock() - start) / CLOCKS_PER_SEC;
                    spdlog::info("Sampling the best conformation end ({:.4f}s)", duration);

                    coord_list.push_back(best_coord);
                    score_list.push_back(best_score);
                    if (cuda >= 0) {
                        c10::cuda::CUDACachingAllocator::emptyCache();
                    }
                }

                start = std::clock();
                spdlog::info("Screening top {} candidates begin ...", topn);
                // use sceener calculating the final score for virtual screening
                auto coords = at::cat(coord_list);
                auto scores = at::cat(score_list);

                Cluster cluster;
                cluster.sort_structures(coords, scores, 0.5f, topn);
                
                duration = static_cast<float>(std::clock() - start) / CLOCKS_PER_SEC;
                spdlog::info("Screening top {} candidates end ({:.4f}s)", topn, duration);
                
                for (auto i = 0; i < topn; i++) {
                    Result result(cluster.selected_coord[i], cluster.selected_score[i].item<float>());
                    result.updateMol(lig);
                    auto result_title = title + "_Model_" + std::to_string(i + 1);
                    lig->SetTitle(result_title.c_str());
                    conv.Write(lig, &out);
                }
                coord_list.clear();
            }
            
            auto iter_duration = static_cast<float>(clock() - iter_start) / CLOCKS_PER_SEC;
            spdlog::info("Docking Ligand {} end ({:.4f}s)", title, iter_duration);
            lig->SetTitle("");
            lig->Clear();
            has_mol = conv.Read(lig);
            i++;
        } 

        if (out_path != "") {
            std::ofstream fout(out_path);
            fout << out.str() << std::endl;
            fout.close();
        } else {
            std::cout << out.str() << std::endl;  // Keep this for actual output data
        }
        
        scorer.reset();
        screener.reset();
        // refs and pockets will be deleted in RefSiter destructor, no need to delete here
        siter.reset();
        delete lig;
        delete rec;
        auto total_duration = static_cast<float>(clock() - total_start) / CLOCKS_PER_SEC;
        spdlog::info("Finished ({:.4f}s)!", total_duration);
        return 0;

    } catch (runtime_error e) {
        cerr << "\nRuntime Error\n";
        cerr << e.what() << "\n";
        size_t free_byte = 0, total_byte = 0;
        cudaMemGetInfo(&free_byte, &total_byte);

        double free_db = (double)free_byte;
        double total_db = (double)total_byte;
        double used_db = total_db - free_db;
        printf("GPU memory usage: used = %f, free = %f MB, total = %f MB\n", used_db / 1024.0 / 1024.0,
               free_db / 1024.0 / 1024.0, total_db / 1024.0 / 1024.0);
        return 1;
    } catch (exception& e) { // Errors that shouldn't happen:
        cerr << "\n\nAn error occurred: " << e.what() << ". " << error_message;
        return 1;
    }
}