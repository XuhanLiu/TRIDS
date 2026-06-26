"""
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

Molecular docking engine for TRIDS.
"""

from typing import Optional, Union, List
from dataclasses import dataclass
import torch
from .molecule import Molecule, Pocket
from .scorer import TriScorer, VinaScorer
from .utils import get_device
from . import _core, require_core


@dataclass
class DockingResult:
    molecule: Molecule
    score: float
    rank: int = 0
    rmsd: Optional[float] = None

    def __repr__(self) -> str:
        rmsd_str = f", rmsd={self.rmsd:.2f}" if self.rmsd is not None else ""
        return f"DockingResult(rank={self.rank}, score={self.score:.2f}{rmsd_str})"


class Engine:
    """Molecular docking engine."""

    def __init__(
        self,
        scorer: Union[TriScorer, VinaScorer],
        streams: int = 256,
        depth: int = 32,
        top_n: int = 1,
    ):
        require_core()
        self.scorer = scorer
        self.streams = streams
        self.depth = depth
        self.top_n = top_n
        device = get_device()
        self._engine = _core._Engine(
            scorer._scorer, streams, depth, top_n, device
        )

    def dock(
        self,
        ligand: Molecule,
        pocket: Pocket,
    ) -> List[DockingResult]:
        self.scorer.set_pocket(pocket)
        self.scorer.set_ligand(ligand)
        self.scorer.profile()

        conformer = _core._Conformer(ligand._obmol, self.streams, get_device())
        coords_list, scores = self._engine.docking(conformer)

        if isinstance(scores, torch.Tensor):
            if scores.dim() == 2:
                scores = scores[:, 0]
            scores = scores.detach().cpu().tolist()

        results = []
        for i, (coords, score) in enumerate(zip(coords_list, scores)):
            mol = ligand.copy()
            mol.coordinates = coords
            mol.title = f"{ligand.title}_pose_{i + 1}"
            results.append(DockingResult(molecule=mol, score=float(score), rank=i + 1))
        return results

    def dock_batch(
        self,
        ligands: List[Molecule],
        pocket: Pocket,
    ) -> List[List[DockingResult]]:
        return [self.dock(lig, pocket) for lig in ligands]

    def __repr__(self) -> str:
        return (
            f"Engine(scorer={type(self.scorer).__name__}, "
            f"streams={self.streams}, depth={self.depth}, top_n={self.top_n})"
        )
