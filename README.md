TriDS Release Notes
============
#### TriDS: AI-native molecular docking framework unified with binding site identification, conformational sampling and scoring

By Xuhan Liu & Hong Zhang, on Octobor 26th 2025

Please see the LICENSE file for the license terms for the software. Basically it's free to academic users. If you do wish to sell the software or use it in a commercial product, then please contact us:

   [Xuhan Liu](mailto:xuhanliu@hotmail.com) (First Author): xuhanliu@hotmail.com 

   [Hong Zhang](mailto:zhangh@cpl.ac.cn) (Correspondent Author): zhangh@cpl.ac.cn 


### Introduction
----------------
Molecular docking is a cornerstone of drug discovery to unveil the mechanism of ligand-receptor interactions. With the recent development of deep learning in the field of artificial intelligence, innovative methods were developed for molecular docking. However, the mainstream docking programs adopt a docking-then-rescoring streamline to increase the docking accuracy, which make the virtual screening process cumbersome. Moreover, there still lacks a unified framework to integrate binding site identification, conformational sampling and scoring, in a user-friendly manner. In our previous work of DSDP and its subsequent flexible version, we have demonstrated the effectiveness of guiding conformational sampling with the gradient of analytic scoring function. As the third generation of DSDP, here we expanded the similar strategy to ML-based differentiable scoring model to device a novel docking method named TriDS under the mainstream AI training framework, which unifies the sampling and scoring steps. To be user-friendly, TriDS also integrates ML-based model for binding site prediction and has compatibility with multiple input file formats. We show here that gradients of a suitable ML-based scoring function can lead to excellent docking accuracy in the benchmark datasets, especially for large ligands. Moreover, TriDS is implemented with enhanced computational efficiency in terms of both running speed and GPU memory requirement.

### Architectures
------------------
TriDS is implemented with PyTorch C++ (LibTorch). It combined CUDA graph-operator merge method to accelerate the program dramatically. General speaking, any differentiable ML-based scoring function could be compatible with this framework for conformational sampling.


### Installation
-------------
To run the compiled **TriDS**, some dependent packages need to be installed. If you are not the **root** user, you could create an virtual environment with **Conda**, the instruction are as follows: 

**1. Create a new environment named trids in Conda:**
      
      $ conda create -n trids python==3.9
      
      $ conda create trids
   
**1. [Nvidia CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) (version >= 11.8)**

      $ conda install cuda==11.8.0 -c nvidia/label/cuda-11.8.0

**2. [OpenBabel](https://openbabel.org) (version >= 3.0.0)**

      $ conda install openbabel==3.1.1

**3. [LibTorch](https://www.pytorch.org) (version >=2.0)**
      
      $ wget http://download.pytorch.org/libtorch/cu118/libtorch-cxx11-abi-shared-with-deps-2.7.1%2Bcu118.zip
      
      $ unzip libtorch-cxx11-abi-shared-with-deps-2.7.1+cu118.zip

      $ mv libtorch ~/

#### Note
----------

1. The compiled program "trids" is in the ./bin file.

2. The version of LibTorch should be consistent with the version of CUDA-ToolKit (Default: 11.8).

3. If meet any error like "error while loading shared libraries: cannot open shared object file: No such file or directory" when you run the compiled "trids", please run **locate** to find the related libfile path. And then add the related path to the **LD_LIBRARY_PATH**.

 

### Usage
--------
#### Command line: 

    $ trids -r receptor_path -l ligand_path -k refrence_ligand_path [OPTIONS]

#### Options:

    -h,--help                             Print this help message and exit
  
    -r,--receptor <pdb>                   Rigid part of the receptor [REQUIRED]
  
    -l,--ligand <smi, mol2, sdf, pdb>     Ligand [REQUIRED]
  
    -k,--pocket <pt, pth, mol2, sdf, pdb> Reference profile for Binding pocket identification! 
                                          [Default: ../ckpt/site_binding.pt]
  
    -m,--model <string>                   Torch-based JIT model file path for conformation sampling and scoring
                                          [Default: ../ckpt/sampling.pt]
  
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


### Aknowledgements
------------

1. Shenzhen Bay Laboratory
2. Westlake University
3. Changping Laboratory
4. Peking University

### References
--------------
1. [Liu X, Bonghua Zhang, Hong Zhang, Yi Qin Gao. TriDS: AI-native molecular docking framework unified with binding site identification, conformational sampling and scoring. (2025). preprint](https://arxiv.org/abs/2510.24186)

2. [YuPeng Huang, Hong Zhang, Siyuan Jiang, Dajiong Yue, Xiaohan Lin, Jun Zhang, and Yi Qin Gao. DSDP: A Blind Docking Strategy Accelerated by GPUs. (2023) Journal of Chemical Information and Modeling](https://pubs.acs.org/doi/10.1021/acs.jcim.3c00519)

3. [Chengwei Dong, Yu-Peng Huang, Xiaohan Lin, Hong Zhang, and Yi Qin Gao. DSDPFlex: Flexible-Receptor Docking with GPU Acceleration. (2024) Journal of Chemical Information and Modeling](https://doi.org/10.1021/acs.jcim.4c01715)
