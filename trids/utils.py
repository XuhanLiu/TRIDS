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

Utility functions for TRIDS.

Example:
    >>> import trids
    >>> trids.set_device("gpu", 0)
    >>> rmsd = trids.utils.calculate_rmsd(molecule1.coordinates, molecule2.coordinates)
    >>> print(f"RMSD: {rmsd}")
    >>> trids.utils.set_seed(42)
    >>> print(f"Random seed: {torch.random.get_rng_state()}")
"""

from typing import Optional, Tuple, List, Union
import os
import torch

_verbose = 0
_device = "cuda:0" if torch.cuda.is_available() else "cpu"
_num_threads: Optional[int] = 1 if torch.cuda.is_available() else os.cpu_count()


def _as_float_tensor(coords: Union[torch.Tensor, list, tuple]) -> torch.Tensor:
    if isinstance(coords, torch.Tensor):
        return coords.float()
    return torch.tensor(coords, dtype=torch.float32)


def log(msg: str, *, level: int = 2) -> None:
    """Print *msg* when global verbose level is at least *level*."""
    if _verbose >= level:
        print(msg)


def set_log_level(level: int) -> None:
    """
    Set Python-side log verbosity.

    Args:
        level: 0=quiet (errors only), 1=warnings, 2=info messages
    """
    if level not in (0, 1, 2):
        raise ValueError("verbose must be 0 (error), 1 (warn), or 2 (info)")
    global _verbose
    _verbose = level


def get_log_level() -> int:
    """Return current Python-side log verbosity level."""
    return _verbose


def set_device(arg1: str, arg2: int) -> None:
    """
    Set the global compute device for TRIDS operations.

    Args:
        arg1: Device type, ``"cpu"`` or ``"gpu"``
        arg2: For ``"cpu"``, number of CPU threads to use; for ``"gpu"``,
              zero-based GPU index
    """
    kind = arg1.strip().lower()
    if kind not in ("cpu", "gpu"):
        raise ValueError('arg1 must be "cpu" or "gpu"')

    global _device, _num_threads
    if kind == "cpu":
        if arg2 < 1:
            raise ValueError("arg2 must be >= 1 when arg1 is cpu")
        torch.set_num_threads(arg2)
        _num_threads = arg2
        _device = "cpu"
        return

    if arg2 < 0:
        raise ValueError("arg2 must be >= 0 when arg1 is gpu")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    if arg2 >= torch.cuda.device_count():
        raise ValueError(
            f"GPU index {arg2} is out of range; "
            f"available GPUs: 0..{torch.cuda.device_count() - 1}"
        )
    _device = f"cuda:{arg2}"
    _num_threads = None


def get_device() -> str:
    """Return the global compute device string, e.g. ``cuda:0`` or ``cpu``."""
    return _device


def get_num_threads() -> Optional[int]:
    """Return configured CPU thread count, or ``None`` when using GPU."""
    return _num_threads


def calculate_rmsd(
    coords1: torch.Tensor, 
    coords2: torch.Tensor,
    align: bool = True
) -> float:
    """
    Calculate RMSD between two conformations
    
    Args:
        coords1: First conformation coordinates [N, 3]
        coords2: Second conformation coordinates [N, 3]
        align: Whether to perform optimal superposition first
        
    Returns:
        float: RMSD value (Angstrom)
    """
    coords1 = _as_float_tensor(coords1)
    coords2 = _as_float_tensor(coords2)

    if coords1.shape != coords2.shape:
        raise ValueError("Coordinate shapes must match")
    
    if align:
        coords2 = kabsch_align(coords1, coords2)
    
    diff = coords1 - coords2
    rmsd = torch.sqrt(torch.mean(torch.sum(diff ** 2, dim=1)))
    return rmsd.item()


def kabsch_align(
    target: torch.Tensor,
    mobile: torch.Tensor,
) -> torch.Tensor:
    """
    Use Kabsch algorithm to superimpose mobile onto target
    
    Args:
        target: Target coordinates [N, 3]
        mobile: Coordinates to be superimposed [N, 3]
        
    Returns:
        torch.Tensor: Superimposed coordinates [N, 3]
    """
    target = _as_float_tensor(target)
    mobile = _as_float_tensor(mobile)

    target_center = target.mean(dim=0)
    mobile_center = mobile.mean(dim=0)

    target_centered = target - target_center
    mobile_centered = mobile - mobile_center

    H = mobile_centered.T @ target_centered
    U, _, Vt = torch.linalg.svd(H)

    R = Vt.T @ U.T

    if torch.linalg.det(R) < 0:
        Vt = Vt.clone()
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    return (mobile - mobile_center) @ R.T + target_center


def set_seed(seed: int):
    """
    Set random seed for reproducibility

    Args:
        seed: Random seed
    """
    import random
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def device_from_index(device_id: int = 0) -> torch.device:
    """
    Build a ``torch.device`` from a CUDA device index.

    Args:
        device_id: CUDA device ID (-1 for CPU)

    Returns:
        torch.device: Compute device
    """
    if device_id < 0 or not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(f"cuda:{device_id}")


def check_cuda() -> Tuple[bool, bool, Optional[str]]:
    """
    Check CUDA availability
    
    Returns:
        Tuple[bool, bool, str]: (cuda_available, cudnn_available, cuda_version)
    """
    cuda_available = torch.cuda.is_available()
    cudnn_available = torch.backends.cudnn.is_available() if cuda_available else False
    cuda_version = torch.version.cuda if cuda_available else None
    
    return cuda_available, cudnn_available, cuda_version


def print_cuda_info():
    """Print CUDA information"""
    cuda_available, cudnn_available, cuda_version = check_cuda()
    
    print(f"CUDA available: {cuda_available}")
    if cuda_available:
        print(f"CUDA version: {cuda_version}")
        print(f"cuDNN available: {cudnn_available}")
        print(f"GPU count: {torch.cuda.device_count()}")
        for i in range(torch.cuda.device_count()):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")


def coords_to_cpu(coords: torch.Tensor) -> torch.Tensor:
    """Return a detached CPU copy of coordinate tensor."""
    return coords.detach().cpu()


def coords_to_device(
    coords: Union[torch.Tensor, list, tuple],
    device: Union[str, torch.device] = "cpu",
) -> torch.Tensor:
    """Convert coordinates to a float tensor on the given device."""
    return _as_float_tensor(coords).to(device)


def center_of_mass(coords: torch.Tensor, masses: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Calculate center of mass
    
    Args:
        coords: Atom coordinates [N, 3]
        masses: Atom masses [N] (optional, default equal weights)
        
    Returns:
        torch.Tensor: Center of mass coordinates [3]
    """
    if masses is None:
        return coords.mean(dim=0)
    
    total_mass = masses.sum()
    return (coords * masses.unsqueeze(-1)).sum(dim=0) / total_mass


def translate_coords(coords: torch.Tensor, translation: torch.Tensor) -> torch.Tensor:
    """
    Translate coordinates
    
    Args:
        coords: Atom coordinates [N, 3] or [batch, N, 3]
        translation: Translation vector [3] or [batch, 3]
        
    Returns:
        torch.Tensor: Translated coordinates
    """
    if coords.dim() == 2 and translation.dim() == 1:
        return coords + translation.unsqueeze(0)
    elif coords.dim() == 3 and translation.dim() == 2:
        return coords + translation.unsqueeze(1)
    else:
        return coords + translation


def rotate_coords(coords: torch.Tensor, rotation_matrix: torch.Tensor) -> torch.Tensor:
    """
    Rotate coordinates
    
    Args:
        coords: Atom coordinates [N, 3] or [batch, N, 3]
        rotation_matrix: Rotation matrix [3, 3] or [batch, 3, 3]
        
    Returns:
        torch.Tensor: Rotated coordinates
    """
    if coords.dim() == 2 and rotation_matrix.dim() == 2:
        return coords @ rotation_matrix.T
    elif coords.dim() == 3 and rotation_matrix.dim() == 3:
        return torch.bmm(coords, rotation_matrix.transpose(-1, -2))
    else:
        raise ValueError("Incompatible dimensions for rotation")

