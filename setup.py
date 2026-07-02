"""
TriDS: Deep Learning-based Molecular Docking Framework

Automated setup script that:
1. Automatically compiles C++ code using CMake
2. Installs lib files to conda env's lib/trids{version} directory

Usage:
    python setup.py install
    pip install -e . --no-build-isolation

Note:
    --no-build-isolation is required so the build uses the environment's PyTorch.
"""

import base64
import hashlib
import os
import re
import shutil
import subprocess
import sys
import sysconfig
import tempfile
import zipfile
from pathlib import Path

from setuptools import setup
from setuptools.command.bdist_wheel import bdist_wheel
from setuptools.command.build_py import build_py
from setuptools.command.develop import develop
from setuptools.command.editable_wheel import editable_wheel
from setuptools.command.install import install
from setuptools.command.install_lib import install_lib

VERSION = "1.0.0"
SOURCE_DIR = Path(__file__).parent.resolve()
PACKAGE_SOURCE = SOURCE_DIR / "trids"
IS_WINDOWS = sys.platform == "win32"
CORE_EXT = ".pyd" if IS_WINDOWS else ".so"
MSVC_CONFIGS = ("Release", "RelWithDebInfo", "MinSizeRel", "Debug")
_trids_asset_outputs: list[str] | None = None


def _env_base() -> Path:
    prefix = os.environ.get("CONDA_PREFIX")
    return Path(prefix) if prefix else Path(sys.prefix)


def _install_path(*parts: str) -> Path:
    base = _env_base()
    if IS_WINDOWS:
        return base.joinpath("Library", *parts)
    return base.joinpath(*parts)


def get_lib_install_dir() -> Path:
    return _install_path("lib", f"trids{VERSION}")


def get_bin_install_dir() -> Path:
    return _install_path("bin")


def get_include_install_dir() -> Path:
    return _install_path("include", "trids")


def get_cmake_install_dir() -> Path:
    return _install_path("lib", "cmake", f"trids{VERSION}")


def get_build_dir(source_dir: Path = SOURCE_DIR) -> Path:
    return source_dir / "build"


def _artifact_search_dirs(directory: Path) -> list[Path]:
    dirs = [directory]
    if IS_WINDOWS:
        dirs.extend(directory / cfg for cfg in MSVC_CONFIGS if (directory / cfg).exists())
    return dirs


def _resolve_artifact_root(directory: Path) -> Path | None:
    for search_dir in _artifact_search_dirs(directory):
        if IS_WINDOWS:
            has_dll = any((search_dir / name).is_file() for name in ("trids.dll", "libtrids.dll"))
            if (
                (search_dir / "trids.exe").is_file()
                and has_dll
                and any(search_dir.glob("_core*.pyd"))
            ):
                return search_dir
        elif (
            (search_dir / "trids").is_file()
            and (search_dir / "libtrids.so").is_file()
            and any(search_dir.glob(f"_core*{CORE_EXT}"))
        ):
            return search_dir
    return None


def _artifacts_complete(directory: Path) -> bool:
    return _resolve_artifact_root(directory) is not None


def _artifact_dir(directory: Path) -> Path:
    return _resolve_artifact_root(directory) or directory


def _find_file(directory: Path, names: str | tuple[str, ...]) -> Path | None:
    if isinstance(names, str):
        names = (names,)
    for search_dir in _artifact_search_dirs(directory):
        for name in names:
            candidate = search_dir / name
            if candidate.is_file():
                return candidate
    return None


def _copy_file(src: Path, dest: Path) -> str:
    dest.parent.mkdir(parents=True, exist_ok=True)
    label = f"{src.name} -> {dest.name}" if src.name != dest.name else src.name
    print(f"  Copying {label}")
    shutil.copy2(src, dest)
    return str(dest)


def _copy_glob(source: Path, dest_dir: Path, patterns: tuple[str, ...]) -> list[str]:
    installed: list[str] = []
    for pattern in patterns:
        for src_file in source.glob(pattern):
            installed.append(_copy_file(src_file, dest_dir / src_file.name))
    return installed


def find_prebuilt_bin_dir(source_dir: Path) -> Path | None:
    if IS_WINDOWS:
        win_root = source_dir / "bin" / "windows"
        if not win_root.is_dir():
            return None
        candidates = [win_root, *sorted(win_root.rglob("*"), key=lambda path: len(path.parts))]
        for candidate in candidates:
            if candidate.is_dir() and _artifacts_complete(candidate):
                return _artifact_dir(candidate)
        return None

    linux_root = source_dir / "bin" / "linux"
    if not linux_root.is_dir():
        return None

    candidates: list[Path] = []
    try:
        import torch

        match = re.match(r"(\d+)\.(\d+)", (torch.version.cuda or "").strip())
        if match:
            candidates.append(linux_root / f"cu{match.group(1)}{match.group(2)}")
    except Exception:
        pass
    candidates.extend(sorted(linux_root.glob("cu*"), key=lambda path: path.name, reverse=True))

    for candidate in dict.fromkeys(candidates):
        if _artifacts_complete(candidate):
            return candidate
    return None


def get_artifact_dir(source_dir: Path) -> Path:
    return find_prebuilt_bin_dir(source_dir) or _artifact_dir(get_build_dir(source_dir))


def _find_core_module(prebuilt: Path | None, build_dir: Path) -> Path | None:
    if prebuilt is not None:
        for search_dir in _artifact_search_dirs(prebuilt):
            found = next(search_dir.glob(f"_core*{CORE_EXT}"), None)
            if found is not None:
                return found
    if build_dir.exists():
        return next(build_dir.rglob(f"_core*{CORE_EXT}"), None)
    return None


def _patch_elf_rpath(binary: Path, rpath: str) -> bool:
    patchelf = shutil.which("patchelf")
    if not patchelf:
        return False
    return subprocess.run(
        [patchelf, "--set-rpath", rpath, str(binary)],
        capture_output=True,
        text=True,
    ).returncode == 0


def _create_windows_launcher(wrapper: Path, real_binary: Path) -> None:
    """Write a .bat launcher that avoids llvm-openmp DLL clashes in Library/bin."""
    base = _env_base()
    torch_lib = base / "Lib" / "site-packages" / "torch" / "lib"
    bin_dir = get_bin_install_dir()
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    wrapper.write_text(
        "@echo off\r\n"
        f'set "PATH={torch_lib};{bin_dir};%PATH%"\r\n'
        "set \"KMP_DUPLICATE_LIB_OK=TRUE\"\r\n"
        f'"{real_binary}" %*\r\n',
        encoding="utf-8",
    )


def install_lib_to_env(source_dir: Path, artifact_dir: Path) -> list[str]:
    lib_source = source_dir / "lib"
    lib_dest = get_lib_install_dir()
    artifact_dir = _artifact_dir(artifact_dir)
    print(f"Installing lib files to {lib_dest}")
    print(f"  Native artifacts from {artifact_dir}")

    if not lib_source.is_dir():
        sys.exit(f"Error: Source lib directory not found: {lib_source}")
    if not artifact_dir.exists():
        sys.exit(f"Error: Artifact directory not found: {artifact_dir}")

    installed = _copy_glob(lib_source, lib_dest, ("*.pt",))
    patterns = ("*.pyd", "*.dll") if IS_WINDOWS else ("*.so",)
    for src_dir in _artifact_search_dirs(artifact_dir):
        for pattern in patterns:
            for src_file in src_dir.glob(pattern):
                dest_name = src_file.name
                installed.append(_copy_file(src_file, lib_dest / dest_name))

    print(f"Lib files installed to: {lib_dest}")
    return installed


def install_bin_to_env(artifact_dir: Path) -> list[str]:
    artifact_dir = _artifact_dir(artifact_dir)
    exe_name = "trids.exe" if IS_WINDOWS else "trids"
    trids_exe = _find_file(artifact_dir, exe_name)
    if trids_exe is None:
        sys.exit(f"Error: {exe_name} not found in {artifact_dir}")

    if IS_WINDOWS:
        lib_dest = get_lib_install_dir()
        bin_dest = get_bin_install_dir()
        print(f"Installing trids executable to {lib_dest}")
        real_binary = lib_dest / exe_name
        installed = [_copy_file(trids_exe, real_binary)]

        wrapper = bin_dest / "trids.bat"
        print(f"Installing trids launcher to {wrapper}")
        _create_windows_launcher(wrapper, real_binary)
        installed.append(str(wrapper))
        print(f"Executable installed to: {wrapper}")
        return installed

    lib_dest = get_lib_install_dir()
    bin_dest = get_bin_install_dir()
    print(f"Installing trids executable to {lib_dest}")
    real_binary = lib_dest / "trids"
    _copy_file(trids_exe, real_binary)
    real_binary.chmod(0o755)

    try:
        import torch

        rpath = f"$ORIGIN:{Path(torch.__file__).resolve().parent / 'lib'}"
    except Exception:
        rpath = "$ORIGIN"
    if not _patch_elf_rpath(real_binary, rpath):
        print("  Note: patchelf not found; bin/trids wrapper will set LD_LIBRARY_PATH")

    wrapper = bin_dest / "trids"
    print(f"Installing trids launcher to {wrapper}")
    wrapper.parent.mkdir(parents=True, exist_ok=True)
    wrapper.write_text(
        "#!/bin/sh\n"
        f'LIBDIR="{lib_dest}"\n'
        'export LD_LIBRARY_PATH="$LIBDIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"\n'
        'exec "$LIBDIR/trids" "$@"\n'
    )
    wrapper.chmod(0o755)
    print(f"Executable installed to: {wrapper}")
    return [str(real_binary), str(wrapper)]


def install_include_to_env(source_dir: Path) -> list[str]:
    include_source = source_dir / "include"
    include_dest = get_include_install_dir()
    if not include_source.is_dir():
        sys.exit(f"Error: include directory not found: {include_source}")

    print(f"Installing header files to {include_dest}")
    installed = _copy_glob(include_source, include_dest, ("*.h", "*.hpp", "*.cuh"))
    print(f"  Copied {len(installed)} header files")
    return installed


def install_cmake_config(source_dir: Path) -> list[str]:
    cmake_template = source_dir / "cmake" / "trids-config.cmake.in"
    cmake_dest = get_cmake_install_dir()
    if not cmake_template.is_file():
        sys.exit(f"Error: CMake config template not found: {cmake_template}")

    print(f"Installing CMake config to {cmake_dest}")
    cmake_dest.mkdir(parents=True, exist_ok=True)

    package_init = (
        "# Package initialization\n"
        "macro(set_and_check _var _file)\n"
        "  set(${_var} \"${_file}\")\n"
        "  if(NOT EXISTS \"${_file}\")\n"
        "    message(FATAL_ERROR \"File or directory ${_file} referenced by variable ${_var} does not exist!\")\n"
        "  endif()\n"
        "endmacro()\n"
    )
    config_file = cmake_dest / "trids-config.cmake"
    config_file.write_text(
        cmake_template.read_text()
        .replace("@TRIDS_VERSION@", VERSION)
        .replace("@PACKAGE_INIT@", package_init)
    )
    version_file = cmake_dest / "trids-config-version.cmake"
    version_file.write_text(
        f'''# TriDS version file
set(PACKAGE_VERSION "{VERSION}")

if(PACKAGE_VERSION VERSION_LESS PACKAGE_FIND_VERSION)
    set(PACKAGE_VERSION_COMPATIBLE FALSE)
else()
    set(PACKAGE_VERSION_COMPATIBLE TRUE)
    if(PACKAGE_FIND_VERSION STREQUAL PACKAGE_VERSION)
        set(PACKAGE_VERSION_EXACT TRUE)
    endif()
endif()
'''
    )
    print(f"  Generated {config_file.name} and {version_file.name}")
    return [str(config_file), str(version_file)]


def install_all_assets(source_dir: Path, artifact_dir: Path) -> list[str]:
    if IS_WINDOWS:
        print("Windows install layout:")
        print(f"  Models / _core.pyd / trids.dll / trids.exe -> {get_lib_install_dir()}")
        print(f"  trids.bat (CLI launcher)                   -> {get_bin_install_dir()}")
        print(f"  Headers                         -> {get_include_install_dir()}")
        print(f"  CMake config                    -> {get_cmake_install_dir()}")
        print(f"  Conda prefix                    -> {_env_base()}")

    installed: list[str] = []
    installed.extend(install_lib_to_env(source_dir, artifact_dir))
    installed.extend(install_bin_to_env(artifact_dir))
    installed.extend(install_include_to_env(source_dir))
    installed.extend(install_cmake_config(source_dir))
    return installed


def _record_relative_path(path: Path, root: Path) -> str | None:
    try:
        return Path(os.path.relpath(path.resolve(), root.resolve())).as_posix()
    except ValueError:
        return None


def write_installed_files_txt(install_cmd) -> None:
    ei_cmd = install_cmd.get_finalized_command("egg_info")
    install_lib_cmd = install_cmd.get_finalized_command("install_lib")
    egg_info_dir = Path(install_lib_cmd.install_dir) / f"{ei_cmd._get_egg_basename()}.egg-info"
    if not egg_info_dir.is_dir():
        return

    entries: list[str] = []

    def add_path(path: Path):
        if path.is_file() or path.is_symlink():
            rel = _record_relative_path(path, egg_info_dir)
            if rel and rel not in entries:
                entries.append(rel)

    for output in getattr(install_cmd, "_extra_install_outputs", []):
        add_path(Path(output))
    for output in install_lib_cmd.get_outputs():
        add_path(Path(output))
    for item in egg_info_dir.rglob("*"):
        if item.is_file() and item.name != "installed-files.txt":
            add_path(item)

    installed_files = egg_info_dir / "installed-files.txt"
    installed_files.write_text("\n".join(entries) + ("\n" if entries else ""))
    print(f"  Wrote {installed_files} ({len(entries)} files)")


def append_assets_to_wheel_record(wheel_path: Path, asset_paths: list[str]) -> int:
    entries: list[str] = []
    seen: set[str] = set()
    site_packages = Path(sysconfig.get_path("purelib")).resolve()

    for raw in asset_paths:
        path = Path(raw)
        if not path.is_file():
            continue
        rel = _record_relative_path(path, site_packages)
        if rel is None or rel in seen:
            continue
        seen.add(rel)
        data = path.read_bytes()
        digest = base64.urlsafe_b64encode(hashlib.sha256(data).digest()).rstrip(b"=").decode("ascii")
        entries.append(f"{rel},sha256={digest},{len(data)}")

    if not entries:
        return 0

    wheel_path = Path(wheel_path)
    with zipfile.ZipFile(wheel_path, "r") as zf:
        record_name = next((n for n in zf.namelist() if n.endswith(".dist-info/RECORD")), None)
        if record_name is None:
            return 0
        record_lines = zf.read(record_name).decode("utf-8").splitlines()
        archive = {info.filename: zf.read(info.filename) for info in zf.infolist()}

    existing = {line.split(",", 1)[0] for line in record_lines if line}
    record_lines.extend(entry for entry in entries if entry.split(",", 1)[0] not in existing)
    archive[record_name] = ("\n".join(record_lines) + "\n").encode("utf-8")

    fd, tmp_path = tempfile.mkstemp(suffix=".whl")
    os.close(fd)
    try:
        with zipfile.ZipFile(tmp_path, "w", compression=zipfile.ZIP_DEFLATED) as zout:
            for name, data in archive.items():
                zout.writestr(name, data)
        shutil.move(tmp_path, wheel_path)
    except Exception:
        os.unlink(tmp_path)
        raise
    return len(entries)


def _find_vcvars64() -> Path | None:
    if not IS_WINDOWS:
        return None

    program_files_x86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
    vswhere = Path(program_files_x86) / "Microsoft Visual Studio" / "Installer" / "vswhere.exe"
    if vswhere.is_file():
        result = subprocess.run(
            [
                str(vswhere), "-latest", "-products", "*",
                "-requires", "Microsoft.VisualStudio.Component.VC.Tools.x86.x64",
                "-property", "installationPath",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode == 0 and result.stdout.strip():
            vcvars = Path(result.stdout.strip()) / "VC" / "Auxiliary" / "Build" / "vcvars64.bat"
            if vcvars.is_file():
                return vcvars

    conda_prefix = os.environ.get("CONDA_PREFIX")
    if conda_prefix:
        activate_vs = Path(conda_prefix) / "etc" / "conda" / "activate.d" / "vs2022_compiler_vars.bat"
        if activate_vs.is_file():
            return activate_vs
    return None


def _run(cmd: list[str], cwd: Path, env: dict | None = None) -> None:
    print(f"  Command: {' '.join(cmd)}")
    print("-" * 60)
    sys.stdout.flush()
    run_env = env or os.environ.copy()
    if IS_WINDOWS:
        vcvars = _find_vcvars64()
        if vcvars is not None:
            proc = subprocess.run(
                f'call "{vcvars}" && {subprocess.list2cmdline(cmd)}',
                cwd=cwd,
                env=run_env,
                shell=True,
            )
        else:
            print("  Warning: vcvars64.bat not found; MSVC tools may be missing from PATH")
            proc = subprocess.run(cmd, cwd=cwd, env=run_env)
    else:
        proc = subprocess.run(cmd, cwd=cwd, env=run_env)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed: {' '.join(cmd)}")
    print("-" * 60)


def _windows_build_env() -> dict:
    env = os.environ.copy()
    env["Python_EXECUTABLE"] = sys.executable
    conda = _env_base()
    env["PATH"] = os.pathsep.join([
        str(conda),
        str(conda / "Library" / "bin"),
        str(conda / "Scripts"),
        env.get("PATH", ""),
    ])
    return env


def _should_wipe_cmake_cache(build_dir: Path) -> bool:
    cache = build_dir / "CMakeCache.txt"
    if not cache.is_file():
        return False
    text = cache.read_text(encoding="utf-8", errors="ignore")
    return "pip-build-env" in text or (IS_WINDOWS and "CMAKE_GENERATOR:INTERNAL=Ninja" not in text)


def build_cpp_extension(source_dir: Path, build_dir: Path, force_rebuild: bool = False):
    print("=" * 60)
    print("Building TriDS C++ Extension")
    print("=" * 60)

    if force_rebuild and build_dir.exists():
        print("Force rebuild: clearing build directory...")
        shutil.rmtree(build_dir)

    build_dir.mkdir(parents=True, exist_ok=True)
    if _should_wipe_cmake_cache(build_dir):
        print("Detected stale or incompatible CMake cache, reconfiguring...")
        shutil.rmtree(build_dir)
        build_dir.mkdir(parents=True, exist_ok=True)

    cmake_cache = build_dir / "CMakeCache.txt"
    build_env = None
    if not cmake_cache.exists():
        print(f"\n[1/2] Configuring CMake in {build_dir}...")
        if IS_WINDOWS:
            if not os.environ.get("CONDA_PREFIX"):
                raise RuntimeError("CONDA_PREFIX is not set. Activate your conda env before installing.")
            build_env = _windows_build_env()
            cmake_args = [
                "cmake", str(source_dir / "cmake" / "windows"),
                "-G", "Ninja", "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_PYTHON=ON",
            ]
            ninja = _env_base() / "Library" / "bin" / "ninja.exe"
            if ninja.is_file():
                cmake_args.append(f"-DCMAKE_MAKE_PROGRAM={ninja}")
        else:
            build_env = os.environ.copy()
            build_env["Python_EXECUTABLE"] = sys.executable
            cmake_args = ["cmake", str(source_dir / "cmake" / "linux"), "-DBUILD_PYTHON=ON"]
        _run(cmake_args, build_dir, build_env)
    else:
        print(f"\n[1/2] Using existing CMake configuration in {build_dir}")
        if IS_WINDOWS:
            build_env = _windows_build_env()

    print("\n[2/2] Building with 8 parallel jobs...")
    build_cmd = [
        "cmake", "--build", str(build_dir),
        "--target", "_core", "--target", "trids_exe", "--parallel", "8",
    ]
    if IS_WINDOWS:
        build_cmd.extend(["--config", "Release"])
    _run(build_cmd, build_dir, build_env)

    core_module = next(build_dir.rglob(f"_core*{CORE_EXT}"), None)
    if core_module is None:
        raise RuntimeError(f"Compiled _core module not found in {build_dir}")

    print(f"\nBuild successful: {core_module.name}")
    print("=" * 60)


def ensure_cpp_extension_built(force_rebuild: bool = False) -> bool:
    prebuilt = find_prebuilt_bin_dir(SOURCE_DIR)
    build_dir = get_build_dir(SOURCE_DIR)
    existing_module = _find_core_module(prebuilt, build_dir)

    need_build = force_rebuild or existing_module is None
    if not need_build:
        api_cpp = SOURCE_DIR / "csrc" / "python" / "api.cpp"
        if api_cpp.is_file() and api_cpp.stat().st_mtime > existing_module.stat().st_mtime:
            print("Source code has been updated, rebuilding...")
            need_build = True

    if not need_build:
        print(f"Using existing C++ extension: {existing_module.name}")
        return True

    try:
        build_cpp_extension(SOURCE_DIR, build_dir, force_rebuild=force_rebuild)
        return True
    except Exception as exc:
        print(f"Failed to build C++ extension: {exc}")
        if prebuilt is not None:
            print(f"Prebuilt artifacts remain available in {prebuilt}")
            return True
        print("\nYou can try building manually:")
        print("  mkdir build && cd build")
        print(f"  cmake ../cmake/{'windows' if IS_WINDOWS else 'linux'} -DBUILD_PYTHON=ON")
        print("  cmake --build . --target _core --target trids_exe --config Release -j 8")
        return False


def ensure_trids_assets_installed() -> list[str]:
    global _trids_asset_outputs
    if _trids_asset_outputs is not None:
        return _trids_asset_outputs

    if "pip-build-env" in sys.prefix:
        print("Warning: Running in pip isolated build environment.")
        print("C++ extension will NOT be built automatically.")
        print("Please use: pip install -e . --no-build-isolation")
        _trids_asset_outputs = []
        return _trids_asset_outputs

    artifact_dir = find_prebuilt_bin_dir(SOURCE_DIR)
    if artifact_dir is not None:
        print(f"Using prebuilt artifacts from {artifact_dir}")
    elif not ensure_cpp_extension_built():
        _trids_asset_outputs = []
        return _trids_asset_outputs
    else:
        artifact_dir = get_artifact_dir(SOURCE_DIR)

    if not _artifacts_complete(artifact_dir):
        print(f"Error: Incomplete TRIDS artifacts in {artifact_dir}")
        if IS_WINDOWS:
            print("Expected under build/ or bin/windows/: trids.exe, trids.dll, _core*.pyd")
        _trids_asset_outputs = []
        return _trids_asset_outputs

    _trids_asset_outputs = install_all_assets(SOURCE_DIR, artifact_dir)
    return _trids_asset_outputs


def _python_package_output_mapping(build_lib: str) -> dict[str, str]:
    build_root = Path(build_lib)
    return {
        str(build_root / src.relative_to(SOURCE_DIR)): str(src)
        for src in sorted(PACKAGE_SOURCE.rglob("*.py"))
    }


def _build_py_outputs(cmd) -> list[str]:
    return list(_python_package_output_mapping(cmd.build_lib).keys()) + getattr(
        cmd, "_extra_install_outputs", []
    )


class TridsAssetMixin:
    _extra_install_outputs: list[str] = []

    def _install_trids_assets(self):
        self._extra_install_outputs = ensure_trids_assets_installed()

    def get_outputs(self):
        return super().get_outputs() + getattr(self, "_extra_install_outputs", [])


class BuildPyCommand(TridsAssetMixin, build_py):
    """Deploy native assets; Python sources install from tree, not via build/lib."""

    def run(self):
        self._install_trids_assets()

    def get_output_mapping(self):
        return _python_package_output_mapping(self.build_lib)

    def get_outputs(self, include_bytecode=True):
        return _build_py_outputs(self)


class InstallLibCommand(install_lib):
    """Install the trids package directly from source into site-packages."""

    def run(self):
        ensure_trids_assets_installed()
        super().run()

    def install(self):
        dest_pkg = Path(self.install_dir) / "trids"
        print(f"Installing Python package from {PACKAGE_SOURCE} -> {dest_pkg}")
        return self.copy_tree(str(PACKAGE_SOURCE), str(dest_pkg))


class WheelAssetRecordMixin:
    def _register_wheel_assets(self, wheel_path):
        if not _trids_asset_outputs:
            return
        count = append_assets_to_wheel_record(Path(wheel_path), _trids_asset_outputs)
        if count:
            print(f"  Registered {count} asset paths in wheel RECORD for pip uninstall")


class EditableWheelCommand(WheelAssetRecordMixin, editable_wheel):
    def _create_wheel_file(self, bdist_wheel_cmd):
        wheel_path = super()._create_wheel_file(bdist_wheel_cmd)
        self._register_wheel_assets(wheel_path)
        return wheel_path


class BdistWheelCommand(WheelAssetRecordMixin, bdist_wheel):
    def run(self):
        build_lib = self.get_finalized_command("build_py").build_lib
        for dest, src in _python_package_output_mapping(build_lib).items():
            dest_path = Path(dest)
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest_path)
        super().run()
        for wheel in sorted(Path(self.dist_dir or "dist").glob("*.whl")):
            self._register_wheel_assets(wheel)


class AssetInstallFinalizeMixin(TridsAssetMixin):
    def run(self):
        self._install_trids_assets()
        super().run()
        write_installed_files_txt(self)


class DevelopCommand(AssetInstallFinalizeMixin, develop):
    pass


class InstallCommand(AssetInstallFinalizeMixin, install):
    pass


AUTO_BUILD = os.environ.get("TRIDS_SKIP_BUILD", "0") != "1"
cmdclass = {
    "build_py": BuildPyCommand,
    "install_lib": InstallLibCommand,
    "bdist_wheel": BdistWheelCommand,
    "editable_wheel": EditableWheelCommand,
    "develop": DevelopCommand,
    "install": InstallCommand,
} if AUTO_BUILD else {}

if not AUTO_BUILD:
    print("Skipping automatic C++ build (TRIDS_SKIP_BUILD=1)")

setup(packages=["trids"], include_package_data=True, cmdclass=cmdclass)
