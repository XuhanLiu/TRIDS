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
from openbabel import openbabel as ob
import torch

from . import _HAS_CORE, _core, require_core
from .utils import get_device

Pocket = _core._Receptor if _HAS_CORE else None


def _format_from_path(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix.startswith("."):
        suffix = suffix[1:]
    return suffix


def _open_conversion(path: str):
    conv = ob.OBConversion()
    conv.SetInFormat(_format_from_path(path))
    return conv


def _mol_ptr(mol) -> int:
    if mol is None:
        raise ValueError("Molecule is empty")
    return int(mol.this)


def _read_first(path: str):
    conv = _open_conversion(path)
    mol = ob.OBMol()
    if not conv.ReadFile(mol, path):
        raise RuntimeError(f"Failed to load molecule from: {path}")
    return mol


def _read_all(path: str):
    conv = _open_conversion(path)
    records = []
    mol = ob.OBMol()
    ok = conv.ReadFile(mol, path)
    while ok:
        if mol.NumAtoms() > 0:
            records.append(mol)
        mol = ob.OBMol()
        ok = conv.Read(mol)
    if not records:
        raise RuntimeError(f"Failed to load molecule from: {path}")
    return records


class Molecule:
    """Molecule wrapper around OpenBabel OBMol."""

    def __init__(self, obmol=None):
        self._obmol = obmol

    @classmethod
    def from_file(cls, path: str) -> "Molecule":
        path = str(Path(path).resolve())
        return cls(_read_first(path))

    @classmethod
    def read_all(cls, path: str) -> List["Molecule"]:
        path = str(Path(path).resolve())
        return [cls(mol) for mol in _read_all(path)]

    @classmethod
    def from_smiles(cls, smiles: str, gen_3d: bool = True) -> "Molecule":
        conv = _open_conversion(smiles)
        conv.SetInFormat("smi")
        mol = ob.OBMol()
        if not conv.ReadString(mol, smiles):
            raise RuntimeError(f"Failed to parse SMILES: {smiles}")
        if gen_3d and not ob.OBBuilder().Build(mol):
            raise RuntimeError("Could not initiate the coordinate of the input ligand!")
        return cls(mol)

    @property
    def num_atoms(self) -> int:
        return 0 if self._obmol is None else self._obmol.NumAtoms()

    @property
    def num_bonds(self) -> int:
        return 0 if self._obmol is None else self._obmol.NumBonds()

    @property
    def coordinates(self) -> torch.Tensor:
        if self._obmol is None:
            return torch.empty(0, 3)
        coords = torch.empty(self._obmol.NumAtoms(), 3)
        for atom in ob.OBMolAtomIter(self._obmol):
            idx = atom.GetIndex()
            coords[idx, 0] = atom.GetX()
            coords[idx, 1] = atom.GetY()
            coords[idx, 2] = atom.GetZ()
        return coords

    @coordinates.setter
    def coordinates(self, coords: torch.Tensor):
        mol = self._check_obmol()
        cpu_coords = coords.detach().cpu()
        for atom in ob.OBMolAtomIter(mol):
            idx = atom.GetIndex()
            atom.SetVector(
                float(cpu_coords[idx, 0]),
                float(cpu_coords[idx, 1]),
                float(cpu_coords[idx, 2]),
            )

    @property
    def title(self) -> str:
        return "" if self._obmol is None else self._obmol.GetTitle()

    @title.setter
    def title(self, name: str):
        if self._obmol is not None:
            self._obmol.SetTitle(name)

    def to_graph(self) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        require_core()
        g = _core.mol_to_graph(_mol_ptr(self._obmol))
        return g.node_feats, g.edge_feats, g.edge_index

    def _write(self, path: str, fmt: str):
        conv = _open_conversion(path)
        conv.SetOutFormat(fmt)
        conv.WriteFile(self._check_obmol(), str(path))

    def to_sdf(self, path: str):
        self._write(path, "sdf")

    def to_pdb(self, path: str):
        self._write(path, "pdb")

    def delete_hydrogens(self):
        if self._obmol is not None:
            self._obmol.DeleteHydrogens()

    def remove_water(self):
        mol = self._obmol
        if mol is None or mol.NumResidues() == 0:
            return
        waters = [res for res in ob.OBResidueIter(mol) if res.GetName() == "HOH"]
        for res in waters:
            mol.DeleteResidue(res)

    def copy(self) -> "Molecule":
        if self._obmol is None:
            return Molecule()
        return Molecule(ob.OBMol(self._obmol))

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
        return cls.from_file(path)

    @property
    def num_residues(self) -> int:
        return 0 if self._obmol is None else self._obmol.NumResidues()

    def extract_pocket(
        self,
        reference: Union[Molecule, torch.Tensor],
        radius: float = 8.0,
        only_vina: bool = False,
    ) -> Pocket:
        obmol = self._check_obmol()
        if isinstance(reference, Molecule):
            ref_coords = reference.coordinates
        else:
            ref_coords = reference.float() if isinstance(reference, torch.Tensor) else torch.tensor(
                reference, dtype=torch.float32
            )
        require_core()
        return _core.extract_pocket(_mol_ptr(obmol), ref_coords, radius, only_vina, get_device())

    def predict_pockets(
        self,
        model_path: str = None,
        cutoff: float = 8.0,
        only_vina: bool = False,
    ) -> List[Pocket]:
        obmol = self._check_obmol()
        require_core()
        if model_path is None:
            from . import get_model_path
            model_path = get_model_path("libtrids_site_binding.pt")
        return _core.predict_pockets(
            _mol_ptr(obmol), str(Path(model_path).resolve()), cutoff, only_vina, get_device()
        )

    def load_pockets(
        self,
        path: str,
        cutoff: float = 8.0,
        only_vina: bool = False,
    ) -> List[Pocket]:
        obmol = self._check_obmol()
        require_core()
        return _core.load_pockets(
            _mol_ptr(obmol), str(Path(path).resolve()), cutoff, only_vina, get_device()
        )

    def __repr__(self) -> str:
        return f"Receptor(residues={self.num_residues}, atoms={self.num_atoms})"
