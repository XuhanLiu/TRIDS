"""Smoke test for the TRIDS PyMOL plugin in nogui mode."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PLUGIN_PATH = PROJECT_ROOT / "plugins" / "pymol" / "trids_gui.py"
SHARE = PROJECT_ROOT / "tests"


def main() -> int:
    import pymol
    from pymol import cmd

    pymol.finish_launching(["pymol", "-cq"])

    print("=== Loading TRIDS PyMOL plugin ===")
    cmd.run(str(PLUGIN_PATH))

    # Command registration
    commands = {"tridev", "trids", "triscore", "trisite", "trinfo"}
    missing = [name for name in commands if name not in cmd.keyword]
    if missing:
        print(f"[FAIL] Missing commands: {missing}")
        return 1
    print(f"[PASS] Registered commands: {sorted(commands)}")

    # A-menu patch
    import pymol.menu as pymol_menu

    if not getattr(pymol_menu, "_trids_patched", False):
        print("[FAIL] A-menu was not patched (_trids_patched is False)")
        return 1
    print("[PASS] A-menu patch flag set")

    menu = pymol_menu.mol_action(cmd, "test_obj")
    trids_entry = next((item for item in menu if item[1] == "TRIDS"), None)
    if trids_entry is None:
        print("[FAIL] TRIDS submenu not found in mol_action menu")
        return 1

    submenu = trids_entry[2]
    labels = [item[1] for item in submenu if item[0] == 1]
    expected = {"Binding Site", "Docking", "Scoring"}
    if set(labels) != expected:
        print(f"[FAIL] Unexpected TRIDS submenu items: {labels}")
        return 1
    for item in submenu:
        if item[0] != 1:
            continue
        if item[1] == "Binding Site":
            binding_menu = item[2]
            if not isinstance(binding_menu, list):
                print(f"[FAIL] Binding Site submenu must be a list, got {type(binding_menu)}")
                return 1
            binding_labels = [entry[1] for entry in binding_menu if entry[0] == 1]
            if binding_labels[0] != "auto":
                print(f"[FAIL] Binding Site submenu must start with auto, got {binding_labels}")
                return 1
            continue
        if item[1] in {"Docking", "Scoring"}:
            if not isinstance(item[2], list):
                print(f"[FAIL] {item[1]} submenu must be a list, got {type(item[2])}")
                return 1
            continue
        if not isinstance(item[2], str):
            print(f"[FAIL] Menu item {item[1]!r} must use string command, got {type(item[2])}")
            return 1
    print(f"[PASS] A-menu TRIDS items: {labels}")

    # trinfo / tridev (PyMOL extended commands are invoked via cmd.do)
    print("\n=== Running trinfo ===")
    cmd.do("trinfo")
    print("[PASS] trinfo executed")

    print("\n=== Running tridev ===")
    cmd.do("tridev gpu, 0")
    cmd.do("tridev")
    print("[PASS] tridev executed")

    # Load test structures and run trisite (lightweight functional test)
    target = SHARE / "target.pdb"
    ligand = SHARE / "ligand.sdf"
    if not target.is_file():
        print(f"[SKIP] functional tests: missing {target}")
        return 0

    print("\n=== Running trisite on tests/target.pdb ===")
    cmd.load(str(target), "protein")
    cmd.do("trisite protein, , 8.0, 0")
    print("[PASS] trisite executed")

    if ligand.is_file():
        print("\n=== Running triscore with trisite pocket ===")
        cmd.load(str(ligand), "ligand")
        cmd.do("triscore pkt_pred_0, ligand")
        print("[PASS] triscore executed")

    print("\n=== All PyMOL plugin smoke tests passed ===")
    return 0


if __name__ == "__main__":
    sys.exit(main())
