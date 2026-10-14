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

High-level API for TRIDS molecular docking.

Example:
    >>> import trids
    >>> results = trids.docking(
    ...     receptor="protein.pdb",
    ...     ligand="ligand.sdf",
    ...     reference="ref_ligand.sdf"
    ... )
    >>> print(f"Best score: {results[0].score}")
"""

from pathlib import Path
from typing import Optional, List, Union

import torch

from .molecule import Molecule, Receptor, Pocket
from .scorer import TriScorer, VinaScorer
from .dock import Engine, DockingResult, poses_from_cluster
from .utils import get_device, log

ReceptorInput = Union[str, Pocket]


def _reference_kind(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix == ".pt":
        return "model"
    if suffix == ".dat":
        return "grid"
    return "ligand"


def _load_receptor(path: str, use_vina: bool) -> Receptor:
    if not path:
        raise ValueError("receptor is required")
    log(f"# Loading receptor: {path}")
    rec = Receptor.from_pdb(path)
    if use_vina:
        rec.remove_water()
    return rec


def _tri_model(model: Optional[str], hts: bool) -> str:
    if model is None:
        from . import get_model_path
        name = "libtrids_scoring.pt" if hts else "libtrids_sampling.pt"
        model = str(get_model_path(name))
    return model


def _make_scorers(model: Optional[str], use_vina: bool, hts: bool, score_only: bool):
    """Match main.cpp: HTS loads the scoring model; Vina HTS rescores with it."""
    load_model = (not use_vina) or (hts and not score_only)
    tri = None
    if load_model:
        model = _tri_model(model, hts)
        log(f"# Using TRI scoring function: {model}")
        tri = TriScorer(model_path=model)
    if use_vina:
        log("# Using Vina scoring function")
        scorer = VinaScorer()
        screener = tri if hts and not score_only else None
        return scorer, screener
    return tri, None


def _docking_pockets(
    receptor: ReceptorInput,
    reference: Optional[str],
    radius: float,
    use_vina: bool,
    hts: bool = False,
) -> List[Pocket]:
    if not isinstance(receptor, str):
        log("# Using pre-defined pocket")
        return [receptor]

    only_vina = use_vina and not hts
    rec = _load_receptor(receptor, use_vina)
    if not reference:
        from . import get_model_path
        reference = str(get_model_path("libtrids_site_binding.pt"))
    kind = _reference_kind(reference)
    if kind == "model":
        log(f"# Predicting pockets: {reference}")
        pockets = rec.predict_pockets(reference, cutoff=radius, only_vina=only_vina)
    elif kind == "grid":
        log(f"# Loading pocket profile: {reference}")
        pockets = rec.load_pockets(reference, cutoff=radius, only_vina=only_vina)
    else:
        log(f"# Extracting pockets from reference: {reference}")
        pockets = []
        for ref in Molecule.read_all(reference):
            ref.delete_hydrogens()
            pockets.append(rec.extract_pocket(ref, radius=radius, only_vina=only_vina))
    if not pockets:
        raise RuntimeError("Failed to extract binding pocket")
    return pockets


def docking(
    receptor: ReceptorInput,
    ligand: str,
    reference: Optional[str] = None,
    model: Optional[str] = None,
    top_n: int = 1,
    streams: int = 1024,
    depth: int = 8,
    use_vina: bool = False,
    radius: float = 8.0,
    hts: bool = False,
    verbose: int = 0,
) -> List[DockingResult]:
    """
    One-step molecular docking function.

    Args:
        receptor: Receptor PDB file path, or a pre-defined Pocket object
        ligand: Ligand file path (supports sdf, mol2, pdb, smi). Every record is docked.
        reference: Pocket source. Omitted uses libtrids_site_binding.pt.
            .pt predicts sites, .dat loads a grid profile, and a ligand file
            makes one pocket per reference molecule.
        model: TRI weights. Default is libtrids_sampling.pt, or libtrids_scoring.pt when hts is set.
        top_n: Number of top conformations kept after clustering
        streams: Parallel Monte Carlo walkers (default 1024)
        depth: Monte Carlo search depth (default 8)
        use_vina: Sample with the Vina scoring function
        radius: Pocket extraction radius (Angstrom)
        hts: Rescore Vina samples with libtrids_scoring.pt. Without Vina, sample with that model.
        verbose: Log level, 0=quiet, 1=warn, 2=info (default: 0)

    Returns:
        List[DockingResult]: Docking results list

    Example:
        >>> import trids
        >>> results = trids.docking(
        ...     receptor="protein.pdb",
        ...     ligand="ligand.sdf",
        ...     reference="ref_ligand.sdf",
        ...     top_n=5
        ... )
        >>> print(f"Best score: {results[0].score:.2f}")
        >>> results[0].molecule.to_sdf("docked.sdf")
    """
    from . import set_verbose, get_verbose

    prev_verbose = get_verbose()
    set_verbose(verbose)
    try:
        return _docking_impl(
            receptor, ligand, reference, model, top_n,
            streams, depth, use_vina, radius, hts,
        )
    finally:
        set_verbose(prev_verbose)


def _docking_impl(
    receptor: ReceptorInput,
    ligand: str,
    reference: Optional[str] = None,
    model: Optional[str] = None,
    top_n: int = 1,
    streams: int = 1024,
    depth: int = 8,
    use_vina: bool = False,
    radius: float = 8.0,
    hts: bool = False,
) -> List[DockingResult]:
    pockets = _docking_pockets(receptor, reference, radius, use_vina, hts)
    for pocket in pockets:
        log(f"# Pocket: {pocket.num_residues} residues, {pocket.num_atoms} atoms")

    scorer, screener = _make_scorers(model, use_vina, hts, score_only=False)
    engine = Engine(scorer=scorer, streams=streams, depth=depth, top_n=top_n)

    log(f"# Loading ligand: {ligand}")
    results: List[DockingResult] = []
    for index, lig in enumerate(Molecule.read_all(ligand)):
        if not lig.title:
            lig.title = f"TRIDS_{index}"
        log(f"# Docking {lig.title} into {len(pockets)} pocket(s)...")
        coord_list = []
        score_list = []
        for pocket_i, pocket in enumerate(pockets):
            coords, scores = engine.docking(
                lig, pocket, screener, ligand_ready=pocket_i > 0,
            )
            coord_list.append(coords)
            score_list.append(scores)
            if get_device().startswith("cuda") and torch.cuda.is_available():
                torch.cuda.empty_cache()
        poses = poses_from_cluster(
            lig, torch.cat(coord_list, 0), torch.cat(score_list, 0), top_n,
        )
        results.extend(poses)

    log(f"# Docking completed. Top {len(results)} results:")
    for result in results:
        log(f"#   Rank {result.rank}: score = {result.score:.2f}")
    return results


def _scoring_impl(
    receptor: ReceptorInput,
    ligand: str,
    reference: Optional[str] = None,
    model: Optional[str] = None,
    use_vina: bool = False,
    radius: float = 8.0,
    hts: bool = False,
) -> Union[float, List[float]]:
    """
    Calculate score only (without docking sampling).

    A reference ligand fixes the pocket. A site model or grid profile is not
    used: the pocket is cut around the ligand coordinates, matching score_only.
    """
    scorer, _screener = _make_scorers(model, use_vina, hts, score_only=True)
    ligands = Molecule.read_all(ligand)
    fixed = None
    rec = None
    if isinstance(receptor, str):
        rec = _load_receptor(receptor, use_vina)
        if reference and _reference_kind(reference) == "ligand":
            ref = Molecule.read_all(reference)[0]
            ref.delete_hydrogens()
            fixed = rec.extract_pocket(ref, radius=radius, only_vina=use_vina)
    else:
        fixed = receptor
    scores = []
    for lig in ligands:
        pocket = fixed
        if pocket is None:
            pocket = rec.extract_pocket(lig, radius=radius, only_vina=use_vina)
        scorer.set_pocket(pocket)
        scorer.set_ligand(lig)
        scorer.profile()
        scores.append(scorer.score(lig.coordinates))
    if len(scores) == 1:
        return scores[0]
    return scores


def scoring(
    receptor: ReceptorInput,
    ligand: str,
    reference: Optional[str] = None,
    model: Optional[str] = None,
    use_vina: bool = False,
    radius: float = 8.0,
    hts: bool = False,
    verbose: int = 0,
) -> Union[float, List[float]]:
    """
    Calculate score only (without docking sampling).

    Args:
        receptor: Receptor PDB file path, or a pre-defined Pocket object
        ligand: Ligand file path. One score is returned per record.
        reference: Reference ligand file. A .pt or .dat path is ignored and the
            pocket is taken from the ligand coordinates.
        model: TRI weights. Default is libtrids_sampling.pt, or libtrids_scoring.pt when hts is set.
        use_vina: Whether to use the Vina scoring function
        radius: Pocket extraction radius
        hts: With Vina this flag is ignored. Without Vina, load libtrids_scoring.pt.
        verbose: Log level, 0=quiet, 1=warn, 2=info (default: 0)

    Returns:
        float: Score of a single-record file. A list when the file has several records.
    """
    from . import set_verbose, get_verbose

    prev_verbose = get_verbose()
    set_verbose(verbose)
    try:
        return _scoring_impl(
            receptor, ligand, reference, model, use_vina, radius, hts,
        )
    finally:
        set_verbose(prev_verbose)

