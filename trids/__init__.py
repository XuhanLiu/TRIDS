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

Example:
    >>> import trids
    >>> trids.set_device("gpu", 0)
    >>> results = trids.docking(
    ...     receptor="protein.pdb",
    ...     ligand="ligand.sdf",
    ...     reference="ref_ligand.sdf"
    ... )
    >>> print(f"Best score: {results[0].score}")
"""

import os
import sys
import importlib.util
from pathlib import Path
from typing import Optional

__version__ = "1.0.0"
__author__ = "Dr. Xuhan Liu"
__email__ = "xuhanliu@qq.com"


# ==================== Path Lookup ====================

def get_lib_dir() -> Path:
    """
    Get the lib directory path
    
    Search order:
    1. conda environment's lib/trids{version} directory
    2. sys.prefix's lib directory (virtual environment)
    3. Project root's lib directory (development mode)
    
    Returns:
        Path: lib directory path
    """
    # 1. Try conda environment's lib directory
    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        conda_lib = Path(conda_prefix) / "lib" / f"trids{__version__}"
        if conda_lib.exists():
            return conda_lib
        # On Windows, conda may use Library/lib instead of lib
        if sys.platform == "win32":
            conda_lib_win = Path(conda_prefix) / "Library" / "lib" / f"trids{__version__}"
            if conda_lib_win.exists():
                return conda_lib_win
    
    # 2. Try sys.prefix (virtual environment)
    sys_lib = Path(sys.prefix) / "lib" / f"trids{__version__}"
    if sys_lib.exists():
        return sys_lib
    if sys.platform == "win32":
        sys_lib_win = Path(sys.prefix) / "Library" / "lib" / f"trids{__version__}"
        if sys_lib_win.exists():
            return sys_lib_win
    
    # 3. Fall back to project root's lib directory (development mode)
    # trids/__init__.py -> trids/ -> project_root/ -> lib/
    project_lib = Path(__file__).parent.parent / "lib"
    if project_lib.exists():
        return project_lib
    
    # 4. Last resort: return conda lib directory (even if it doesn't exist)
    if conda_prefix:
        if sys.platform == "win32":
            return Path(conda_prefix) / "Library" / "lib" / f"trids{__version__}"
        return Path(conda_prefix) / "lib" / f"trids{__version__}"
    if sys.platform == "win32":
        return Path(sys.prefix) / "Library" / "lib" / f"trids{__version__}"
    return Path(sys.prefix) / "lib" / f"trids{__version__}"


def get_model_path(model_name: str) -> Path:
    """
    Get the model file path
    
    Args:
        model_name: Model file name (e.g., "libtrids_site_binding.pt")
        
    Returns:
        Path: Full path to the model file
        
    Raises:
        FileNotFoundError: If the model file does not exist
    """
    lib_dir = get_lib_dir()
    model_path = lib_dir / model_name
    
    if model_path.exists():
        return model_path
    
    raise FileNotFoundError(
        f"Model file not found: {model_name}\n"
        f"Searched in: {lib_dir}"
    )


# ==================== Load C++ Core Module ====================

# Must import torch first to ensure its dynamic libraries are loaded
# This allows the _core module to find libtorch*.so libraries
import torch

_HAS_CORE = False
_core = None

def _load_core_module():
    """Load the C++ core module"""
    global _HAS_CORE, _core
    
    lib_dir = get_lib_dir()
    
    # Find _core module file (.so on Linux, .pyd on Windows)
    _ext = ".pyd" if sys.platform == "win32" else ".so"
    core_file = None
    try:
        for f in lib_dir.iterdir():
            if f.name.startswith('_core') and f.suffix == _ext:
                core_file = f
                break
    except Exception as e:
        print(f"Warning: Error scanning lib directory {lib_dir}: {e}")
        return
    
    if core_file is None:
        print(f"Warning: TRIDS C++ core not found in {lib_dir}. Some features may be unavailable.")
        return
    
    # Add lib directory to dynamic library search path
    if sys.platform == "win32":
        # Windows: use os.add_dll_directory (Python 3.8+) and prepend to PATH
        try:
            os.add_dll_directory(str(lib_dir))
        except (OSError, AttributeError):
            pass
        _path = os.environ.get("PATH", "")
        if str(lib_dir) not in _path:
            os.environ["PATH"] = str(lib_dir) + ";" + _path
    else:
        ld_path = os.environ.get('LD_LIBRARY_PATH', '')
        if str(lib_dir) not in ld_path:
            os.environ['LD_LIBRARY_PATH'] = str(lib_dir) + ':' + ld_path
    
    # Use importlib to load _core module
    try:
        spec = importlib.util.spec_from_file_location("_core", core_file)
        _core = importlib.util.module_from_spec(spec)
        
        # Register in sys.modules
        sys.modules['trids._core'] = _core
        
        spec.loader.exec_module(_core)
        if not hasattr(_core, "_Engine"):
            raise ImportError("_core is missing required symbols (rebuild with BUILD_PYTHON=ON)")
        _HAS_CORE = True
    except Exception as e:
        print(f"Warning: Failed to load _core module from {core_file}: {e}")
        _core = None
        _HAS_CORE = False
        if "trids._core" in sys.modules:
            del sys.modules["trids._core"]

# Load C++ core module
_load_core_module()


def require_core():
    """Raise if the C++ core module is unavailable."""
    if not _HAS_CORE:
        raise RuntimeError("C++ core module not available")


# ==================== Import Submodules ====================

from .molecule import Molecule, Receptor, Pocket
from .scorer import TriScorer, VinaScorer
from .dock import Engine, DockingResult
from .api import docking, scoring
from . import utils


def set_verbose(level: int) -> None:
    """
    Set log verbosity for Python and C++ layers.

    Args:
        level: 0=quiet (default), 1=warnings, 2=info
    """
    utils.set_log_level(level)
    if _HAS_CORE and hasattr(_core, "set_verbose"):
        _core.set_verbose(level)


def get_verbose() -> int:
    """Return current log verbosity level."""
    return utils.get_log_level()


def set_device(arg1: str, arg2: int) -> None:
    """
    Set the global compute device for TRIDS operations.

    Args:
        arg1: Device type, ``"cpu"`` or ``"gpu"``
        arg2: For ``"cpu"``, number of CPU threads; for ``"gpu"``, GPU index
    """
    utils.set_device(arg1, arg2)


def get_device() -> str:
    """Return the global compute device string."""
    return utils.get_device()


def get_num_threads() -> Optional[int]:
    """Return configured CPU thread count, or ``None`` when using GPU."""
    return utils.get_num_threads()


set_verbose(0)


# ==================== Public API ====================

__all__ = [
    # Molecule classes
    "Molecule",
    "Receptor", 
    "Pocket",
    # Scoring functions
    "TriScorer",
    "VinaScorer",
    # Docking
    "Engine",
    "DockingResult",
    # High-level API
    "docking",
    "scoring",
    # Logging
    "set_verbose",
    "get_verbose",
    "set_device",
    "get_device",
    "get_num_threads",
    # Utilities
    "utils",
    # Path lookup
    "get_lib_dir",
    "get_model_path",
]


def get_version() -> str:
    """Get version number"""
    return __version__


def check_installation() -> bool:
    """
    Check if installation is complete
    
    Returns:
        bool: True if C++ core module is available
    """
    return _HAS_CORE


def print_info():
    """Print version and environment information"""
    print(f"TRIDS version: {__version__}")
    print(f"Author: {__author__}")
    
    if check_installation():
        print("C++ core: Available")
    else:
        print("C++ core: Not available (run `pip install -e .` to build)")
    
    utils.print_cuda_info()

