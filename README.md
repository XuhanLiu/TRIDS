# TRIDS

#### Introduction
TriDS: AI-native molecular docking framework unified with binding site identification, conformational sampling and scoring

#### Architectures
TriDS is implemented with PyTorch C++ (LibTorch). It combined CUDA graph-operator merge method to accelerate the program dramatically.


#### Installation

1. The compiled program "trids" is in the ./bin file.


2. To run the compiled "trids", some dependent packages need to be installed:
# Nvidia CUDA
>>> conda install cuda==11.8.0 -c nvidia/label/cuda-11.8.0

# OpenBabel
>>> conda install openbabel==3.1.1

# LibTorch
>>> wget http://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.7.1%2Bcu118.zip
>>> unzip libtorch-cxx11-abi-shared-with-deps-2.7.1+cu118.zip
>>> mv libtorch ~/

# PS: If meet any error like "error while loading shared libraries: *** cannot open shared object file: No such file or directory" when you run the compiled "trids", please run "locate ***" to find the related libfile path. And then add the related path to the "LD_LIBRARY_PATH".

 

#### Usage

Usage: ./trids -r receptor_path -l ligand_path -k refrence_ligand_path [OPTIONS]

Options:
  -h,--help                             Print this help message and exit
  -r,--receptor <pdb>                   Rigid part of the receptor [REQUIRED]
  -l,--ligand <smi, mol2, sdf, pdb>     Ligand [REQUIRED]
  -k,--pocket <pt, pth, mol2, sdf, pdb> Reference profile for Binding pocket identification! [REQUIRED]
  -m,--model <string>                   Torch-based JIT model file path for conformation sampling and scoring
                                        [Default: ../ckpt/docking.pth.pt]
  -o,--out <string>                     Output file name, format taken from file extension
  -e,--exhaust INT:POSITIVE [384]       Max number of loops for Monte carlo research
  -d,--depth INT:POSITIVE [5]           Max depths for Monte carlo research
  -t,--top INT:POSITIVE [1]             Record Number of N best conformers in output
  --seed INT:POSITIVE                   User defined random seed
  -c,--cpu INT:POSITIVE [1]             Number of CPU cores
  -g,--cuda <Int> [0]                   Index of Nvidia CUDA Device ID. If set to -1, no CUDA device will be used
  --score_only                          Only calculating the score of given conformation without sampling
  --log <string>                        optionally, writing log file
  --config                              Read an ini file


#### Aknowledgements

1. Shenzhen Bay Laboratory
2. Peking University
3. Changping Laboratory
4. Westlake University


#### References

1.  TriDS [https://arxiv.org/abs/2510.24186]
