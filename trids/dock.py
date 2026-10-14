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
from .scorer import BaseScorer, TriScorer, VinaScorer
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


def cluster_poses(
    coords: torch.Tensor,
    scores: torch.Tensor,
    top_n: int,
    rmsd: float = 0.5,
):
    require_core()
    return _core.cluster_poses(coords, scores, rmsd, top_n)


def poses_from_cluster(
    ligand: Molecule,
    coords: torch.Tensor,
    scores: torch.Tensor,
    top_n: int,
    rmsd: float = 0.5,
) -> List[DockingResult]:
    coords_list, selected = cluster_poses(coords, scores, top_n, rmsd)
    if isinstance(selected, torch.Tensor):
        selected = selected.detach().cpu().tolist()
    title = ligand.title or "TRIDS"
    results = []
    for i, (pose, score) in enumerate(zip(coords_list, selected)):
        mol = ligand.copy()
        mol.coordinates = pose
        mol.title = f"{title}_Model_{i + 1}"
        results.append(DockingResult(molecule=mol, score=float(score), rank=i + 1))
    return results


class Engine:
    """Molecular docking engine."""

    def __init__(
        self,
        scorer: Union[TriScorer, VinaScorer],
        streams: int = 1024,
        depth: int = 8,
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

    def docking(
        self,
        ligand: Molecule,
        pocket: Pocket,
        screener: Optional[BaseScorer] = None,
        ligand_ready: bool = False,
    ):
        """Dock one pocket and return every stream's best pose."""
        self.scorer.set_pocket(pocket)
        if not ligand_ready:
            self.scorer.set_ligand(ligand)
        self.scorer.profile()
        if screener is not None:
            screener.set_pocket(pocket)
            if not ligand_ready:
                screener.set_ligand(ligand)
            screener.profile()

        conformer = _core._Conformer(int(ligand._obmol.this), self.streams, get_device())
        screener_cpp = None if screener is None else screener._scorer
        return self._engine.docking(conformer, screener_cpp)

    def dock(
        self,
        ligand: Molecule,
        pocket: Pocket,
        screener: Optional[BaseScorer] = None,
    ) -> List[DockingResult]:
        coords, scores = self.docking(ligand, pocket, screener)
        return poses_from_cluster(ligand, coords, scores, self.top_n)

    def dock_batch(
        self,
        ligands: List[Molecule],
        pocket: Pocket,
    ) -> List[List[DockingResult]]:
        return [poses_from_cluster(lig, *self.docking(lig, pocket), self.top_n) for lig in ligands]

    def __repr__(self) -> str:
        return (
            f"Engine(scorer={type(self.scorer).__name__}, "
            f"streams={self.streams}, depth={self.depth}, top_n={self.top_n})"
        )
