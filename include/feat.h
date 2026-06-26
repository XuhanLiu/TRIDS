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
#include <openbabel/mol.h>
#include <openbabel/atom.h>
#include <openbabel/bond.h>
#include <openbabel/residue.h>
#include <openbabel/obconversion.h>
#include <openbabel/obiter.h>
#include <openbabel/stereo/stereo.h>
#include <openbabel/stereo/cistrans.h>
#include <openbabel/stereo/tetrahedral.h>
#include <iostream>
#include "graph.h"

using namespace std;

namespace Feature {
    const vector<unsigned> DEGREE{ 1, 2, 3, 4, 5, 6, 0 };

    const vector<int> CHARGE{ -5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 10000000};
    
    const vector<string> SYMBOL{
        "C", "N", "O", "S", "F", "P", "Cl",
        "Br", "I", "B", "Si", "Fe", "Zn",
        "Cu", "Mn", "Mo", "other" };

    const vector<unsigned> HYB{ 1, 2, 3, 4, 5, 6, 0 };

    const vector<unsigned char> NUM_H{ 0, 1, 2, 3, 4 };
    
    const  vector<unsigned> BOND_ORDER{ 1, 2, 3, 0 };

    const vector<string> METAL{ "LI","NA","K","RB","CS","MG","TL","CU","AG","BE","NI",
                                "PT", "ZN", "CO","PD","AG","CR","FE","V","MN","HG","GA",
                                "CD","YB","CA","SN","PB","EU","SR","SM","BA","RA","AL",
                                "IN","TL","Y","LA","CE","PR","ND","GD","TB","DY","ER",
                                "TM","LU","HF","ZR","CE","U","PU","TH" };

    const vector<string> RES3{ "GLY", "ALA", "VAL", "LEU", "ILE", "PRO", "PHE", "TYR",
                                "TRP", "SER", "THR", "CYS", "MET", "ASN", "GLN", "ASP",
                                "GLU", "LYS", "ARG", "HIS", "MSE", "CSO", "PTR", "TPO",
                                "KCX", "CSD", "SEP", "MLY", "PCA", "LLP", "M", "X" };

    Graph obmol_to_graph(OpenBabel::OBMol* mol);

    // Graph& rdmol_to_graph(OpenBabel::OBMol* lig);

    string obtain_resname(OpenBabel::OBResidue* res);

    at::Tensor obtain_self_dist(OpenBabel::OBResidue* res);

    struct Coord {
        at::Tensor coords;
        at::Tensor ca_crd;
        at::Tensor ce_crd;
    };

    Coord coordinate(OpenBabel::OBResidue* res);
    
    float check_connect(OpenBabel::OBResidue* res1, OpenBabel::OBResidue* res2);
    
    Graph pdb_to_graph(OpenBabel::OBMol* mol, float cutoff = 10);

    Graph pdb_to_graph(OpenBabel::OBMol* mol, const std::vector<OpenBabel::OBResidue*>& residues, float cutoff = 10);
}
