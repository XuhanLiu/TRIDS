#!/usr/bin/env python3
"""CUDA compiler launcher for MSVC+Ninja on Windows."""
import os
import subprocess
import sys
import tempfile


def _long_path(path: str) -> str:
    path = os.path.normpath(path.strip('"'))
    if os.name != "nt":
        return path
    try:
        import ctypes
        buf = ctypes.create_unicode_buffer(32768)
        if ctypes.windll.kernel32.GetLongPathNameW(path, buf, 32768):
            return buf.value or path
    except (AttributeError, OSError):
        pass
    return path


def _nvcc() -> str:
    if len(sys.argv) > 1 and sys.argv[1].lower().endswith("nvcc.exe"):
        return os.path.normpath(sys.argv[1].strip('"'))
    prefix = os.environ.get("CONDA_PREFIX", "")
    path = os.path.join(prefix, "Library", "bin", "nvcc.exe")
    if prefix and os.path.isfile(path):
        return path
    raise SystemExit("trids_nvcc_driver: nvcc.exe not found")


def _prepare(args: list[str]) -> list[str]:
    skip = {"-MD", "-MT", "-MF", "-FS", "/FS", "/RTC1", "/Zi", "/Od", "/MDd", "/Ob0"}
    out: list[str] = []
    i = 0
    while i < len(args):
        a = args[i]
        if a in skip or ",-FS" in a or a.startswith("-Xcompiler=-Fd"):
            i += 2 if a in ("-MT", "-MF") and i + 1 < len(args) else 1
            continue
        if a.startswith("-ccbin="):
            out.append(f"-ccbin={_long_path(a.split('=', 1)[1])}")
            i += 1
            continue
        if a == "-Xcompiler" and i + 1 < len(args):
            if args[i + 1] not in skip:
                out.append(f"-Xcompiler={args[i + 1]}")
            i += 2
            continue
        if a.startswith("-Xcompiler="):
            for piece in a[len("-Xcompiler=") :].split():
                if piece not in skip:
                    out.append(f"-Xcompiler={piece}")
            i += 1
            continue
        if a.startswith("/wd") or a.startswith("/Zc:"):
            out.append(f"-Xcompiler={a}")
            i += 1
            continue
        out.append(a)
        i += 1
    return out


def main() -> int:
    os.environ.pop("NVCC_PREPEND_FLAGS", None)
    argv = sys.argv[1:]
    nvcc = _nvcc()
    if argv and os.path.normcase(argv[0]) == os.path.normcase(nvcc):
        argv = argv[1:]
    argv = _prepare(argv)
    fd, rsp = tempfile.mkstemp(suffix=".nvcc.rsp", prefix="trids_", text=True)
    os.close(fd)
    with open(rsp, "w", encoding="utf-8", newline="\n") as f:
        f.write(subprocess.list2cmdline(argv))
    try:
        return subprocess.call([nvcc, f"--options-file={rsp}"])
    finally:
        try:
            os.remove(rsp)
        except OSError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
