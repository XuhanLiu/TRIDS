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

from typing import Optional, List, Union

from .molecule import Molecule, Receptor, Pocket
from .scorer import TriScorer, VinaScorer
from .dock import Engine, DockingResult
from .utils import log

ReceptorInput = Union[str, Pocket]


def _resolve_pocket(
    receptor: ReceptorInput,
    lig: Molecule,
    reference: Optional[str],
    radius: float,
    use_vina: bool,
) -> Pocket:
    if isinstance(receptor, str):
        if not receptor:
            raise ValueError("receptor is required")
        log(f"# Loading receptor: {receptor}")
        rec = Receptor.from_pdb(receptor)
        rec.delete_hydrogens()
        if reference is not None:
            log(f"# Extracting pocket from reference: {reference}")
            ref = Molecule.from_file(reference)
            ref.delete_hydrogens()
            return rec.extract_pocket(ref, radius=radius, is_vina=use_vina)
        log("# Extracting pocket from ligand position")
        return rec.extract_pocket(lig, radius=radius, is_vina=use_vina)

    log("# Using pre-defined pocket")
    return receptor


def docking(
    receptor: ReceptorInput,
    ligand: str,
    reference: Optional[str] = None,
    model: Optional[str] = None,
    top_n: int = 1,
    streams: int = 256,
    depth: int = 32,
    use_vina: bool = False,
    radius: float = 8.0,
    verbose: int = 0,
) -> List[DockingResult]:
    """
    One-step molecular docking function
    
    Args:
        receptor: Receptor PDB file path, or a pre-defined Pocket object
        ligand: Ligand file path (supports sdf, mol2, pdb, smi)
        reference: Reference ligand file path for defining binding pocket (optional)
        model: TRI model weights path (default: lib/libtrids_sampling.pt)
        top_n: Number of top conformations to return
        streams: Number of parallel sampling tasks
        depth: Monte Carlo search depth
        use_vina: Whether to use Vina scoring function
        radius: Pocket extraction radius (Angstrom)
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
            streams, depth, use_vina, radius,
        )
    finally:
        set_verbose(prev_verbose)


def _docking_impl(
    receptor: ReceptorInput,
    ligand: str,
    reference: Optional[str] = None,
    model: Optional[str] = None,
    top_n: int = 1,
    streams: int = 256,
    depth: int = 32,
    use_vina: bool = False,
    radius: float = 8.0,
) -> List[DockingResult]:
    log(f"# Loading ligand: {ligand}")
    lig = Molecule.from_file(ligand)
    lig.delete_hydrogens()

    pocket = _resolve_pocket(receptor, lig, reference, radius, use_vina)
    log(f"# Pocket: {pocket.num_residues} residues, {pocket.num_atoms} atoms")
    
    # Create scoring function
    if use_vina:
        log("# Using Vina scoring function")
        scorer = VinaScorer()
    else:
        if model is None:
            from . import get_model_path
            model = str(get_model_path("libtrids_sampling.pt"))
        
        log(f"# Using TRI scoring function: {model}")
        scorer = TriScorer(model_path=model)
    
    # Create docking engine
    engine = Engine(
        scorer=scorer,
        streams=streams,
        depth=depth,
        top_n=top_n,
    )
    
    log(f"# Docking {lig.title or 'ligand'} into pocket...")
    results = engine.dock(lig, pocket)
    
    # Output results
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
) -> float:
    """
    Calculate score only (without docking sampling)
    
    Args:
        receptor: Receptor PDB file path, or a pre-defined Pocket object
        ligand: Ligand file path
        reference: Reference ligand file path
        model: Model weights path
        use_vina: Whether to use Vina scoring function
        radius: Pocket extraction radius
        
    Returns:
        float: Docking score
    """
    lig = Molecule.from_file(ligand)
    lig.delete_hydrogens()
    pocket = _resolve_pocket(receptor, lig, reference, radius, use_vina)
    # Create scoring function
    if use_vina:
        scorer = VinaScorer()
    else:
        if model is None:
            from . import get_model_path
            model = str(get_model_path("libtrids_sampling.pt"))
        scorer = TriScorer(model_path=model)
    
    # Set up and score
    scorer.set_pocket(pocket)
    scorer.set_ligand(lig)
    scorer.profile()
    
    score = scorer.score(lig.coordinates)
    return score


def scoring(
    receptor: ReceptorInput,
    ligand: str,
    reference: Optional[str] = None,
    model: Optional[str] = None,
    use_vina: bool = False,
    radius: float = 8.0,
    verbose: int = 0,
) -> float:
    """
    Calculate score only (without docking sampling).

    Args:
        receptor: Receptor PDB file path, or a pre-defined Pocket object
        ligand: Ligand file path
        reference: Reference ligand file path
        model: Model weights path
        use_vina: Whether to use Vina scoring function
        radius: Pocket extraction radius
        verbose: Log level, 0=quiet, 1=warn, 2=info (default: 0)

    Returns:
        float: Docking score
    """
    from . import set_verbose, get_verbose

    prev_verbose = get_verbose()
    set_verbose(verbose)
    try:
        return _scoring_impl(
            receptor, ligand, reference, model, use_vina, radius,
        )
    finally:
        set_verbose(prev_verbose)

