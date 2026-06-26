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

Molecule handling classes for TRIDS.
"""

from typing import Union, Tuple, List
from pathlib import Path
import torch

from . import _HAS_CORE, _core, require_core
from .utils import get_device

Pocket = _core._Receptor if _HAS_CORE else None


class Molecule:
    """Molecule wrapper around OpenBabel OBMol."""

    def __init__(self, obmol=None):
        self._obmol = obmol

    @classmethod
    def from_file(cls, path: str) -> "Molecule":
        require_core()
        return cls(_core._OBMol.load(str(Path(path).resolve())))

    @classmethod
    def from_smiles(cls, smiles: str, gen_3d: bool = True) -> "Molecule":
        require_core()
        return cls(_core._OBMol.from_smiles(smiles, gen_3d))

    @property
    def num_atoms(self) -> int:
        return 0 if self._obmol is None else self._obmol.num_atoms

    @property
    def num_bonds(self) -> int:
        return 0 if self._obmol is None else self._obmol.num_bonds

    @property
    def coordinates(self) -> torch.Tensor:
        return torch.empty(0, 3) if self._obmol is None else self._obmol.coordinates

    @coordinates.setter
    def coordinates(self, coords: torch.Tensor):
        if self._obmol is None:
            raise ValueError("Molecule is empty")
        self._obmol.coordinates = coords

    @property
    def title(self) -> str:
        return "" if self._obmol is None else self._obmol.title

    @title.setter
    def title(self, name: str):
        if self._obmol is not None:
            self._obmol.title = name

    def to_graph(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if self._obmol is None:
            raise ValueError("Molecule is empty")
        g = self._obmol.to_graph()
        return g.node_feats, g.edge_feats, g.edge_index

    def to_sdf(self, path: str):
        self._check_obmol().write(str(path), "sdf")

    def to_pdb(self, path: str):
        self._check_obmol().write(str(path), "pdb")

    def delete_hydrogens(self):
        if self._obmol is not None:
            self._obmol.delete_hydrogens()

    def copy(self) -> "Molecule":
        return Molecule() if self._obmol is None else Molecule(self._obmol.copy())

    def _check_obmol(self):
        if self._obmol is None:
            raise ValueError("Molecule is empty")
        return self._obmol

    def __repr__(self) -> str:
        return f"Molecule(atoms={self.num_atoms}, bonds={self.num_bonds}, title='{self.title}')"


class Receptor(Molecule):
    """Receptor protein."""

    @classmethod
    def from_pdb(cls, path: str) -> "Receptor":
        require_core()
        return cls(_core._OBMol.load(str(Path(path).resolve())))

    from_file = from_pdb

    @property
    def num_residues(self) -> int:
        return 0 if self._obmol is None else self._obmol.num_residues

    def extract_pocket(
        self,
        reference: Union[Molecule, torch.Tensor],
        radius: float = 8.0,
        is_vina: bool = False,
    ) -> Pocket:
        obmol = self._check_obmol()
        if isinstance(reference, Molecule):
            ref_coords = reference.coordinates
        else:
            ref_coords = reference.float() if isinstance(reference, torch.Tensor) else torch.tensor(
                reference, dtype=torch.float32
            )
        return obmol.extract_pocket(ref_coords, radius, is_vina, get_device())

    def predict_pockets(
        self,
        model_path: str = None,
        cutoff: float = 8.0,
        is_vina: bool = False,
    ) -> List[Pocket]:
        obmol = self._check_obmol()
        require_core()
        if model_path is None:
            from . import get_model_path
            model_path = get_model_path("libtrids_site_binding.pt")
        return obmol.predict_pockets(
            str(Path(model_path).resolve()), cutoff, is_vina, get_device()
        )

    def __repr__(self) -> str:
        return f"Receptor(residues={self.num_residues}, atoms={self.num_atoms})"
