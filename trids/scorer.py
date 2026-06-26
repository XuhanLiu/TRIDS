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

Scoring functions for TRIDS.
"""

from pathlib import Path
import torch

from .molecule import Molecule, Pocket
from .utils import get_device
from . import _core, require_core


class BaseScorer:
    """Base class for scoring functions."""

    def __init__(self, cutoff: float = 8.0):
        self.device = get_device()
        self.cutoff = cutoff
        self._scorer = None
        self._pocket = None

    def set_pocket(self, pocket: Pocket):
        require_core()
        self._pocket = pocket
        self._scorer.set_pocket(pocket)

    def set_ligand(self, ligand: Molecule):
        require_core()
        self._scorer.set_ligand(ligand._obmol)

    def set_box(self, ref_coords: torch.Tensor):
        require_core()
        if self._pocket is None:
            raise RuntimeError("Pocket not set")
        if not isinstance(ref_coords, torch.Tensor):
            ref_coords = torch.tensor(ref_coords, dtype=torch.float32)
        self._pocket.set_box(ref_coords.float())

    def profile(self):
        require_core()
        self._scorer.profile()

    def score(self, coordinates: torch.Tensor) -> float:
        coords = coordinates.to(get_device())
        if coords.dim() == 3:
            if coords.size(0) != 1:
                raise ValueError("score() expects a single conformation [N, 3]")
            coords = coords.squeeze(0)
        return self._scorer.score(coords).item()

class TriScorer(BaseScorer):
    """Graph Transformer-based TRI scoring function."""

    def __init__(
        self,
        model_path: str = None,
        cutoff: float = 10.0,
        beta: float = 0.069314718,
    ):
        require_core()
        device = get_device()
        super().__init__(cutoff)
        if model_path is None:
            from . import get_model_path
            model_path = get_model_path("libtrids_sampling.pt")
        self._scorer = _core._TriScorer(
            str(Path(model_path).resolve()), device, cutoff, beta
        )

    clash = 3.0
    clash_weight = 10.0

    def __repr__(self) -> str:
        return f"TriScorer(device='{self.device}', cutoff={self.cutoff})"


class VinaScorer(BaseScorer):
    """AutoDock Vina scoring function."""

    def __init__(
        self,
        cutoff: float = 8.0,
        beta: float = 0.069314718,
    ):
        require_core()
        device = get_device()
        super().__init__(cutoff)
        self._scorer = _core._VinaScorer(device, cutoff, beta)

    def __repr__(self) -> str:
        return f"VinaScorer(device='{self.device}', cutoff={self.cutoff})"
