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
 

PyMOL Plugin for TRIDS - Deep Learning-based Molecular Docking

This plugin integrates TRIDS into PyMOL, allowing users to run docking,
scoring, and pocket prediction directly from the PyMOL command line.

Installation:
    Method 1: (recommended)
    * Run PyMOL
    * Menu: Plugin -> Plugin Manager -> Install New Plugin -> Choose File ...
    * Choose: ~/plugins/trids_pymol.py
    * Restart PyMOL
    * If installation succeed, the menu in plugin has "TRIDS"

    Method 2: (Manual)
    Copy trids_pymol.py into the startup folder of PyMOL:
    
        $ cp plugins/trids_pymol.py ~/.pymol/startup/                               # on  Linux

    or

        $ copy plugins/trids_pymol.py C:\\Users\\<Username>\\.pymol\\startup\\      # on Windows

    Method 3: (Temporary) 
    Run the following code in the command line of PyMOL

        PyMOL> run ~/plugins/trids_pymol.py

Usage (PyMOL command line):
    device [dtype [, num]]       # set or show compute device
    trids pocket, ligand [, options...]
    triscore pocket, ligand [, options...]
    trisite receptor [, reference [, options...]]
    trinfo

Usage (object panel A menu):
    Click A next to an object/selection, then TRIDS > trisite / trids / triscore
"""

from __future__ import annotations

import os
import re
import sys
import tempfile
import traceback
from pathlib import Path
from typing import List, Optional, Tuple
from pymol.Qt import QtWidgets


def _setup_trids_path() -> None:
    """Make TRIDS importable when the plugin is loaded outside the project cwd."""
    plugin_dir = Path(__file__).resolve().parent
    for candidate in (plugin_dir.parent, plugin_dir.parent.parent):
        pkg = candidate / "trids"
        if pkg.is_dir() and (pkg / "__init__.py").is_file():
            root = str(candidate)
            if root not in sys.path:
                sys.path.insert(0, root)
            return


_setup_trids_path()

# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def _print(msg: str) -> None:
    """Print to PyMOL console."""
    try:
        from pymol import cmd
        cmd.feedback("enable", "all", "results")
    except Exception:
        pass
    print(msg)


def _ensure_trids():
    """Import trids and return the module, or raise with a helpful message."""
    try:
        import trids
        return trids
    except ImportError as exc:
        raise ImportError(
            "TRIDS is not installed.  Install with:\n"
            "  pip install trids   (or)   pip install -e /path/to/trids\n"
            f"Original error: {exc}"
        ) from exc


def _cmd_arg(value: str) -> str:
    """Strip whitespace and surrounding quotes from PyMOL command arguments."""
    value = str(value).strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        value = value[1:-1].strip()
    return value


def _pymol_cmd_arg(arg: str) -> str:
    """Format one PyMOL command-line argument; quote only when needed."""
    arg = arg.strip()
    if not arg:
        return '""'
    if re.fullmatch(r"[\w.+-]+", arg):
        return arg
    safe = arg.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{safe}"'


def menu_cmd_args(command: str, *args: str) -> str:
    """Build a PyMOL menu command string executed on click (not on hover)."""
    quoted = [_pymol_cmd_arg(arg) for arg in args]
    cmdline = command if not quoted else f"{command} " + ", ".join(quoted)
    return f"cmd.do({cmdline!r})"


def _selection_to_file(selection: str, suffix: str, fmt: str) -> str:
    """Export a PyMOL selection/object to a temp file, or return an existing path."""
    from pymol import cmd

    selection = _cmd_arg(selection)
    if os.path.isfile(selection):
        return selection

    tmp = tempfile.NamedTemporaryFile(suffix=suffix, delete=False)
    tmp.close()
    cmd.save(tmp.name, selection, format=fmt)
    return tmp.name


def _selection_to_pdb(selection: str) -> str:
    return _selection_to_file(selection, ".pdb", "pdb")


def _selection_to_sdf(selection: str) -> str:
    return _selection_to_file(selection, ".sdf", "sdf")


def _reference_ligand(reference: str):
    trids = _ensure_trids()
    ref_lig = trids.Molecule.from_file(_selection_to_sdf(reference))
    ref_lig.delete_hydrogens()
    return ref_lig


def _make_pockets(rec, reference: str, cutoff: float):
    """Return (pockets, prefix) for reference-based or predicted binding sites."""
    if reference:
        return [rec.extract_pocket(_reference_ligand(reference), radius=cutoff)], "site_ref"
    return rec.predict_pockets(cutoff=cutoff), "site_pred"


# ---------------------------------------------------------------------------
# Pocket cache (in-memory C++ objects + session metadata on cmd)
# ---------------------------------------------------------------------------

_POCKET_CACHE: dict = {}
_POCKET_META_ATTR = "_trids_pocket_meta"


def _pocket_meta_store(cmd) -> dict:
    """Session-scoped pocket metadata; survives plugin module reload."""
    if not hasattr(cmd, _POCKET_META_ATTR):
        setattr(cmd, _POCKET_META_ATTR, {})
    return getattr(cmd, _POCKET_META_ATTR)


def _register_pocket_cache(
    name: str,
    pocket,
    receptor: str,
    box_min: Optional[Tuple[float, float, float]] = None,
    box_max: Optional[Tuple[float, float, float]] = None,
    *,
    reference: str = "",
    cutoff: float = 8.0,
    index: int = 0,
) -> None:
    from pymol import cmd

    _POCKET_CACHE[name] = {
        "pocket": pocket,
        "receptor": receptor,
        "box_min": box_min,
        "box_max": box_max,
    }
    _pocket_meta_store(cmd)[name] = {
        "receptor": receptor,
        "reference": reference,
        "cutoff": cutoff,
        "index": index,
        "mode": "ref" if name.startswith("site_ref_") else "pred",
        "box_min": box_min,
        "box_max": box_max,
    }


def _rebuild_docking_pocket(pocket_name: str):
    """Rebuild a TRIDS pocket object from session metadata."""
    from pymol import cmd

    meta = _pocket_meta_store(cmd).get(pocket_name)
    if meta is None:
        return None

    trids = _ensure_trids()
    rec = trids.Receptor.from_pdb(_selection_to_pdb(meta["receptor"]))
    rec.delete_hydrogens()

    reference = meta.get("reference") or "" if meta["mode"] == "ref" else ""
    pockets, _ = _make_pockets(rec, reference, float(meta["cutoff"]))
    index = int(meta.get("index", 0))
    if index < 0 or index >= len(pockets):
        return None
    pocket = pockets[index]

    _POCKET_CACHE[pocket_name] = {
        "pocket": pocket,
        "receptor": meta["receptor"],
        "box_min": meta.get("box_min"),
        "box_max": meta.get("box_max"),
    }
    _print(f"TRIDS> Rebuilt pocket cache for {pocket_name}.")
    return pocket


def _resolve_docking_pocket(receptor: str):
    """Return cached pocket; receptor must be a trisite pocket object."""
    receptor = _cmd_arg(receptor)
    if not receptor:
        raise ValueError(
            "Pocket object is required. "
            "Use a trisite result (site_ref_* or site_pred_*)."
        )
    if not _is_pocket_object(receptor):
        raise ValueError(
            f"Receptor must be a pocket object from trisite "
            f"(site_ref_* or site_pred_*), got '{receptor}'. "
            "Run trisite first."
        )
    entry = _POCKET_CACHE.get(receptor)
    pocket = None if entry is None else entry["pocket"]
    if pocket is None:
        pocket = _rebuild_docking_pocket(receptor)
    if pocket is None:
        raise ValueError(
            f"Pocket '{receptor}' is not cached. "
            "Run trisite on the receptor again before docking or scoring."
        )
    return pocket


def __init_plugin__(app=None):
    """PyMOL Plugin Manager entry point."""
    from pymol.plugins import addmenuitemqt

    addmenuitemqt("TRIDS settings", _open_gui)
    _register_action_menu()


def _list_docking_objects() -> List[str]:
    """Return all loaded PyMOL objects and selections."""
    from pymol import cmd

    names = list(cmd.get_names("objects"))
    for name in cmd.get_names("selections"):
        if name not in names:
            names.append(name)
    return names


def _is_small_molecule(name: str) -> bool:
    """Return True if the object/selection looks like a small molecule."""
    from pymol import cmd

    try:
        n_total = cmd.count_atoms(name)
    except Exception:
        return False
    if n_total == 0 or n_total > 999:
        return False

    n_polymer = cmd.count_atoms(f"({name}) and polymer")
    if n_polymer > 0 and n_polymer / n_total >= 0.2:
        return False

    if cmd.count_atoms(f"({name}) and organic") > 0:
        return True
    if n_polymer == 0 and cmd.count_atoms(f"({name}) and het") > 0:
        return True
    return n_polymer == 0 and n_total <= 300


def _is_pocket_object(name: str) -> bool:
    """Return True if the object is a TRIDS pocket visualization (site_ref_* / site_pred_*)."""
    if name.endswith("_box"):
        return False
    return name.startswith("site_ref_") or name.startswith("site_pred_")


def _list_ligand_objects() -> List[str]:
    """Return all loaded small-molecule objects/selections."""
    return [name for name in _list_docking_objects() if _is_small_molecule(name)]


def _list_pocket_objects() -> List[str]:
    """Return all loaded TRIDS pocket objects."""
    return [name for name in _list_docking_objects() if _is_pocket_object(name)]


def _list_reference_ligands(receptor: str) -> List[str]:
    """Return loaded small-molecule objects/selections usable as reference ligands."""
    return [name for name in _list_ligand_objects() if name != receptor]


def _build_binding_site_menu(receptor: str) -> list:
    """Build the Binding Site submenu with auto and available ligands."""
    if _is_small_molecule(receptor):
        items = [[2, "(Not a receptor)", ""]]
    else:
        items = [
            [2, "Binding Site:", ""],
            [1, "prediction", menu_cmd_args("trisite", receptor)],
        ]
        ligands = _list_reference_ligands(receptor)
        if ligands:
            items.append([0, "", ""])
            items.append([2, "Reference:", ""])
            for lig in ligands:
                items.append([1, lig, menu_cmd_args("trisite", receptor, lig)])
    return items


def _build_docking_scoring_menu(current: str, command: str, title: str) -> list:
    """Build Docking/Scoring submenu pairing pockets with ligands."""
    items = [[2, f"{title}:", ""]]
    if _is_pocket_object(current):
        partners = _list_ligand_objects()
        label = "Ligand"
    elif _is_small_molecule(current):
        partners = _list_pocket_objects()
        label = "Pocket"
    else:
        partners = []
        items.append([2, "(Not binding site)", ""])
        return items

    if not partners:
        items.append([1, f"(no {label.lower()}s)", ""])
        return items

    items.append([2, f"{label}:", ""])
    for partner in partners:
        if _is_pocket_object(current):
            receptor, ligand = current, partner
        else:
            receptor, ligand = partner, current
        items.append([1, partner, menu_cmd_args(command, receptor, ligand)])
    return items


def _trids_action_submenu(self_cmd, sele: str):
    """Build the TRIDS submenu for the object-panel A button."""
    return [
        [2, "TRIDS:", ""],
        [1, "Binding Site", _build_binding_site_menu(sele)],
        [1, "Docking", _build_docking_scoring_menu(sele, "trids", "Docking")],
        [1, "Scoring", _build_docking_scoring_menu(sele, "triscore", "Scoring")],
    ]


def _wrap_action_menu(original):
    """Append a TRIDS submenu to an existing PyMOL A-menu builder."""

    def wrapped(self_cmd, sele):
        menu = original(self_cmd, sele)
        menu.extend([
            [0, "", ""],
            [1, "TRIDS", _trids_action_submenu(self_cmd, sele)],
        ])
        return menu

    return wrapped


def _register_action_menu():
    """Inject TRIDS items into the right-panel A (Action) menus."""
    try:
        import pymol.menu as pymol_menu
    except ImportError:
        return

    if getattr(pymol_menu, "_trids_patched", False):
        return

    for name in ("mol_action", "sele_action", "group_action"):
        if hasattr(pymol_menu, name):
            setattr(pymol_menu, name, _wrap_action_menu(getattr(pymol_menu, name)))

    pymol_menu._trids_patched = True


def _get_device_status() -> str:
    """Return a one-line description of the current TRIDS compute device."""
    try:
        trids = _ensure_trids()
        msg = f"Current device: {trids.get_device()}"
        threads = trids.get_num_threads()
        if threads is not None:
            msg += f"  (threads: {threads})"
        return msg
    except Exception as exc:
        return f"Error: {exc}"


def _get_trinfo_text() -> str:
    """Return TRIDS installation and environment details as plain text."""
    try:
        trids = _ensure_trids()
        lines = [
            f"Version         : {trids.__version__}",
            f"Device          : {trids.get_device()}",
        ]
        num_threads = trids.get_num_threads()
        if num_threads is not None:
            lines.append(f"Num threads     : {num_threads}")

        import torch
        lines.extend([
            f"PyTorch version : {torch.__version__}",
            f"CUDA available  : {torch.cuda.is_available()}",
        ])
        if torch.cuda.is_available():
            lines.append(f"CUDA version    : {torch.version.cuda}")
            for i in range(torch.cuda.device_count()):
                lines.append(f"  GPU {i}         : {torch.cuda.get_device_name(i)}")
        lines.append("\n\n")
        lines.append("Usage (PyMOL command line):\n")
        lines.append("    # set or show compute device\n")
        lines.append("    device [dtype [, num]]\n")
        lines.append("    # predict pockets in a receptor\n")
        lines.append("    trisite receptor [, reference [, options...]]\n")
        lines.append("    # dock a ligand into a trisite pocket\n")
        lines.append("    trids pocket, ligand [, options...]\n")
        lines.append("    # score a ligand pose in a trisite pocket\n")
        lines.append("    triscore pocket, ligand [, options...]\n")
        lines.append("    # show TRIDS info\n")
        lines.append("    trinfo\n")
        lines.append("\n")
        lines.append("Usage (object panel A menu):\n")
        lines.append("    Click A next to an object/selection, then TRIDS > Binding Site / Docking / Scoring\n")
        lines.append("\n")
        return "\n".join(lines)
    except Exception:
        return f"TRIDS Error:\n{traceback.format_exc()}"


def _make_object_combo(
    objects: List[str],
    allow_none: bool = False,
) -> "QtWidgets.QComboBox":
    from pymol.Qt import QtWidgets

    combo = QtWidgets.QComboBox()
    combo.setEditable(True)
    if allow_none:
        combo.addItem("(none)")
    if objects:
        combo.addItems(objects)
    else:
        combo.addItem("")
    return combo


def _refresh_object_combo(
    combo: "QtWidgets.QComboBox",
    objects: List[str],
    allow_none: bool = False,
) -> None:
    current = combo.currentText()
    combo.blockSignals(True)
    combo.clear()
    if allow_none:
        combo.addItem("(none)")
    if objects:
        combo.addItems(objects)
    else:
        combo.addItem("")
    idx = combo.findText(current)
    if idx >= 0:
        combo.setCurrentIndex(idx)
    else:
        combo.setEditText(current)
    combo.blockSignals(False)


def _combo_value(combo: "QtWidgets.QComboBox", optional: bool = False) -> str:
    value = combo.currentText().strip()
    if optional and value in ("", "(none)"):
        return ""
    return value


def _open_gui():
    """Open the TRIDS settings dialog with tabbed parameter panels."""
    from pymol.Qt import QtWidgets

    dialog = QtWidgets.QDialog()
    dialog.setWindowTitle("TRIDS Settings")
    dialog.setMinimumWidth(440)
    dialog.setMinimumHeight(420)

    root = QtWidgets.QVBoxLayout(dialog)
    tabs = QtWidgets.QTabWidget()
    root.addWidget(tabs)

    object_combo_meta: List[Tuple[QtWidgets.QComboBox, bool]] = []
    pocket_combo_meta: List[QtWidgets.QComboBox] = []
    initial_objects = _list_docking_objects()
    initial_pockets = _list_pocket_objects()

    def refresh_objects() -> List[str]:
        objects = _list_docking_objects()
        pockets = _list_pocket_objects()
        for combo, allow_none in object_combo_meta:
            _refresh_object_combo(combo, objects, allow_none=allow_none)
        for combo in pocket_combo_meta:
            _refresh_object_combo(combo, pockets, allow_none=False)
        return objects

    def register_combo(combo: QtWidgets.QComboBox, allow_none: bool = False) -> QtWidgets.QComboBox:
        object_combo_meta.append((combo, allow_none))
        return combo

    def register_pocket_combo(combo: QtWidgets.QComboBox) -> QtWidgets.QComboBox:
        pocket_combo_meta.append(combo)
        return combo

    # --- tridev ---
    tridev_tab = QtWidgets.QWidget()
    tridev_form = QtWidgets.QFormLayout(tridev_tab)
    tridev_status = QtWidgets.QLabel(_get_device_status())
    tridev_status.setWordWrap(True)
    tridev_form.addRow("Status:", tridev_status)

    dev_combo = QtWidgets.QComboBox()
    dev_combo.addItems(["cpu", "gpu"])
    tridev_form.addRow("Device:", dev_combo)

    num_spin = QtWidgets.QSpinBox()
    num_spin.setRange(0, 128)
    num_spin.setValue(0)
    tridev_form.addRow("Index / threads:", num_spin)

    tridev_btn_row = QtWidgets.QHBoxLayout()
    show_dev_btn = QtWidgets.QPushButton("Show current")
    apply_dev_btn = QtWidgets.QPushButton("Apply")
    tridev_btn_row.addWidget(show_dev_btn)
    tridev_btn_row.addWidget(apply_dev_btn)
    tridev_btn_row.addStretch()
    tridev_form.addRow(tridev_btn_row)

    def on_show_device() -> None:
        tridev()
        tridev_status.setText(_get_device_status())

    def on_apply_device() -> None:
        tridev(dev_combo.currentText(), str(num_spin.value()))
        tridev_status.setText(_get_device_status())

    show_dev_btn.clicked.connect(on_show_device)
    apply_dev_btn.clicked.connect(on_apply_device)
    tabs.addTab(tridev_tab, "Device")

    # --- trisite ---
    trisite_tab = QtWidgets.QWidget()
    trisite_form = QtWidgets.QFormLayout(trisite_tab)
    trisite_receptor = register_combo(_make_object_combo(initial_objects))
    trisite_reference = register_combo(_make_object_combo(initial_objects, allow_none=True))
    trisite_cutoff = QtWidgets.QDoubleSpinBox()
    trisite_cutoff.setRange(1.0, 50.0)
    trisite_cutoff.setDecimals(1)
    trisite_cutoff.setValue(8.0)
    trisite_show = QtWidgets.QCheckBox("Visualize pockets in PyMOL")
    trisite_show.setChecked(True)
    trisite_form.addRow("Receptor:", trisite_receptor)
    trisite_form.addRow("Reference:", trisite_reference)
    trisite_form.addRow("Cutoff (Å):", trisite_cutoff)
    trisite_form.addRow("", trisite_show)
    trisite_run = QtWidgets.QPushButton("Launch")
    trisite_form.addRow(trisite_run)

    def on_run_trisite() -> None:
        receptor = _combo_value(trisite_receptor)
        if not receptor:
            QtWidgets.QMessageBox.warning(dialog, "binding site", "Receptor is required.")
            return
        trisite(
            receptor,
            _combo_value(trisite_reference, optional=True),
            str(trisite_cutoff.value()),
            "1" if trisite_show.isChecked() else "0",
        )

    trisite_run.clicked.connect(on_run_trisite)
    tabs.addTab(trisite_tab, "Binding Site")

    # --- triscore ---
    triscore_tab = QtWidgets.QWidget()
    triscore_form = QtWidgets.QFormLayout(triscore_tab)
    triscore_pocket = register_pocket_combo(_make_object_combo(initial_pockets))
    triscore_ligand = register_combo(_make_object_combo(initial_objects))
    triscore_vina = QtWidgets.QCheckBox("Use Vina scoring")
    triscore_result = QtWidgets.QLabel("Score: —")
    triscore_form.addRow("Pocket:", triscore_pocket)
    triscore_form.addRow("Ligand:", triscore_ligand)
    triscore_form.addRow("", triscore_vina)
    triscore_form.addRow(triscore_result)
    triscore_run = QtWidgets.QPushButton("Launch")
    triscore_form.addRow(triscore_run)

    def on_run_triscore() -> None:
        pocket = _combo_value(triscore_pocket)
        ligand = _combo_value(triscore_ligand)
        if not pocket or not ligand:
            QtWidgets.QMessageBox.warning(
                dialog, "Scoring", "Pocket and ligand are required."
            )
            return
        if pocket == ligand:
            QtWidgets.QMessageBox.warning(
                dialog, "Scoring", "Pocket and ligand must be different."
            )
            return
        score = triscore(
            pocket,
            ligand,
            "1" if triscore_vina.isChecked() else "0",
        )
        if score is not None:
            triscore_result.setText(f"Score: {score:.4f}")

    triscore_run.clicked.connect(on_run_triscore)
    tabs.addTab(triscore_tab, "Scoring")

    # --- trids ---
    trids_tab = QtWidgets.QWidget()
    trids_form = QtWidgets.QFormLayout(trids_tab)
    trids_pocket = register_pocket_combo(_make_object_combo(initial_pockets))
    trids_ligand = register_combo(_make_object_combo(initial_objects))
    trids_top_n = QtWidgets.QSpinBox()
    trids_top_n.setRange(1, 64)
    trids_top_n.setValue(8)
    trids_streams = QtWidgets.QSpinBox()
    trids_streams.setRange(1, 8192)
    trids_streams.setValue(1024)
    trids_depth = QtWidgets.QSpinBox()
    trids_depth.setRange(1, 128)
    trids_depth.setValue(8)
    trids_vina = QtWidgets.QCheckBox("Use Vina scoring")
    trids_name = QtWidgets.QLineEdit()
    trids_name.setPlaceholderText("auto: dock_<ligand>")
    trids_form.addRow("Pocket:", trids_pocket)
    trids_form.addRow("Ligand:", trids_ligand)
    trids_form.addRow("Top N poses:", trids_top_n)
    trids_form.addRow("Streams:", trids_streams)
    trids_form.addRow("Depth:", trids_depth)
    trids_form.addRow("", trids_vina)
    trids_form.addRow("Output object:", trids_name)
    trids_run = QtWidgets.QPushButton("Launch")
    trids_form.addRow(trids_run)

    def on_run_trids() -> None:
        pocket = _combo_value(trids_pocket)
        ligand = _combo_value(trids_ligand)
        if not pocket or not ligand:
            QtWidgets.QMessageBox.warning(
                dialog, "Docking", "Pocket and ligand are required."
            )
            return
        if pocket == ligand:
            QtWidgets.QMessageBox.warning(
                dialog, "Docking", "Pocket and ligand must be different."
            )
            return
        name = trids_name.text().strip() or None
        trids(
            pocket,
            ligand,
            str(trids_top_n.value()),
            str(trids_streams.value()),
            str(trids_depth.value()),
            "1" if trids_vina.isChecked() else "0",
            name,
        )

    trids_run.clicked.connect(on_run_trids)
    tabs.addTab(trids_tab, "Docking")

    # --- about ---
    about_tab = QtWidgets.QWidget()
    about_layout = QtWidgets.QVBoxLayout(about_tab)
    about_text = QtWidgets.QPlainTextEdit()
    about_text.setReadOnly(True)
    about_text.setPlainText(_get_trinfo_text())
    about_layout.addWidget(about_text)
    tabs.addTab(about_tab, "About")

    # --- footer ---
    footer = QtWidgets.QHBoxLayout()
    refresh_btn = QtWidgets.QPushButton("Refresh objects")
    close_btn = QtWidgets.QPushButton("Close")
    footer.addWidget(refresh_btn)
    footer.addStretch()
    footer.addWidget(close_btn)
    root.addLayout(footer)

    refresh_btn.clicked.connect(refresh_objects)
    close_btn.clicked.connect(dialog.accept)

    dialog.exec_()


# ---------------------------------------------------------------------------
# PyMOL object helpers
# ---------------------------------------------------------------------------

def _load_docking_results_into_pymol(results, name: str) -> None:
    """Load docking poses as one PyMOL object with one state per pose."""
    from pymol import cmd

    if not results:
        return

    for i, r in enumerate(results):
        tmp = tempfile.NamedTemporaryFile(suffix=".sdf", delete=False)
        tmp.close()
        try:
            r.molecule.to_sdf(tmp.name)
            state = r.rank if r.rank > 0 else 1
            cmd.load(tmp.name, name, state=state)
            cmd.set_title(name, state, f"{i+1}/{len(results)}")
        finally:
            os.unlink(tmp.name)

    cmd.frame(1)


def _resolve_receptor_object(receptor: str, rec_path: str) -> tuple[str, Optional[str]]:
    """Return (selection_name, temp_object_to_delete).

    When *receptor* is a file path, load it into PyMOL under a temporary object
    name so downstream selections/visualization work.
    """
    from pymol import cmd

    if os.path.isfile(receptor):
        obj_name = "_trids_receptor"
        cmd.load(rec_path, obj_name)
        return obj_name, obj_name
    return receptor, None


def _prepare_docking_inputs(receptor: str, ligand: str):
    """Resolve pocket object and ligand export path for docking or scoring."""
    receptor, ligand = _cmd_arg(receptor), _cmd_arg(ligand)
    if not receptor or not ligand:
        raise ValueError("pocket and ligand are required")
    return _resolve_docking_pocket(receptor), receptor, _selection_to_sdf(ligand)


# ---------------------------------------------------------------------------
# tridev
# ---------------------------------------------------------------------------

def tridev(dev: str = "", num: str = ""):
    """
DESCRIPTION

    Set or display the global TRIDS compute device for subsequent commands.

Usage:

    tridev
    tridev dev [, num]
    tridev dev=cpu, num=8
    tridev gpu, 0

Options:

    dev = str: device type, ``cpu`` or ``gpu``
    num = int: for ``cpu``, number of CPU threads; for ``gpu``, GPU index

Examples:

    tridev
    tridev gpu, 0
    tridev cpu, 8
    tridev dev=gpu, num=1
    """
    try:
        trids = _ensure_trids()

        if not dev:
            msg = f"TRIDS> Current device: {trids.get_device()}"
            threads = trids.get_num_threads()
            if threads is not None:
                msg += f" (num_threads: {threads})"
            _print(msg)
            return

        if not num:
            _print("Error: num is required. Use 'help tridev' for usage.")
            return

        trids.set_device(dev, int(num))
        msg = f"TRIDS> Device set to: {trids.get_device()}"
        threads = trids.get_num_threads()
        if threads is not None:
            msg += f" (num_threads: {threads})"
        _print(msg)

    except Exception:
        _print(f"TRIDS Error:\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------
# trids
# ---------------------------------------------------------------------------

def trids(
    receptor: str = "",
    ligand: str = "",
    top_n: str = "8",
    streams: str = "1024",
    depth: str = "8",
    use_vina: str = "0",
    name: str = None,
):
    """
DESCRIPTION

    Run TRIDS molecular docking into a trisite-defined pocket.

Usage:

    trids pocket, ligand [, top_n [, streams [, depth [, use_vina [, name ]]]]]

Options:

    pocket      = str: trisite pocket object (site_ref_* or site_pred_*)
    ligand      = str: PyMOL selection as the ligand
    top_n       = int: number of top conformations to return (default: 8)
    streams     = int: number of parallel sampling tasks (default: 1024)
    depth       = int: Monte Carlo search depth (default: 8)
    use_vina    = 0/1: use Vina scoring function instead of TRI (default: 0)
    name        = str: base name for the loaded multi-state object (default: trids)

Examples:

    device gpu, 0
    trisite protein
    trids site_pred_0, ligand
    trids site_ref_0, ligand, top_n=5
    """
    try:
        trids = _ensure_trids()
        from pymol import cmd

        if not receptor or not ligand:
            _print("Error: pocket and ligand are required. "
                   "Use 'help trids' for usage.")
            return

        pocket, receptor, lig_path = _prepare_docking_inputs(receptor, ligand)

        top_n_i = int(top_n)
        stream_i = int(streams)
        depth_i = int(depth)
        use_vina_b = bool(int(use_vina))

        _print(f"TRIDS> Starting docking ...")
        _print(f"TRIDS>   Pocket   : {receptor}")
        _print(f"TRIDS>   Ligand   : {ligand}")
        _print(f"TRIDS>   Device   : {trids.get_device()}")

        results = trids.docking(
            receptor=pocket,
            ligand=lig_path,
            top_n=top_n_i,
            streams=stream_i,
            depth=depth_i,
            use_vina=use_vina_b,
        )

        if name is None:
            name = f"dock_{ligand}"
        _print(f"TRIDS> Docking complete. {len(results)} pose(s) returned.")
        _load_docking_results_into_pymol(results, name)
        for r in results:
            _print(f"TRIDS>   State {r.rank}: score = {r.score:.3f}")

        _print(f"TRIDS> Loaded as object '{name}' with {len(results)} state(s).")
        cmd.zoom(name)

    except Exception:
        _print(f"TRIDS Error:\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------
# triscore
# ---------------------------------------------------------------------------

def triscore(
    receptor: str = "",
    ligand: str = "",
    use_vina: str = "0",
) -> Optional[float]:
    """
DESCRIPTION

    Calculate a docking score for the current ligand pose in a trisite pocket.

Usage:

    triscore pocket, ligand [, use_vina ]

Options:

    pocket        = str: trisite pocket object (site_ref_* or site_pred_*)
    ligand        = str: PyMOL selection as the ligand
    use_vina      = 0/1: use Vina scoring (default: 0)

Examples:

    device cpu, 8
    trisite protein
    triscore site_pred_0, ligand
    """
    try:
        trids = _ensure_trids()

        if not receptor or not ligand:
            _print("Error: pocket and ligand are required. "
                   "Use 'help triscore' for usage.")
            return None

        pocket, receptor, lig_path = _prepare_docking_inputs(receptor, ligand)

        _print(f"TRIDS> Scoring ... (device: {trids.get_device()})")
        _print(f"TRIDS>   Pocket   : {receptor}")
        score = trids.scoring(
            receptor=pocket,
            ligand=lig_path,
            use_vina=bool(int(use_vina)),
        )
        _print(f"TRIDS> Score = {score:.4f}")
        return float(score)

    except Exception:
        _print(f"TRIDS Error:\n{traceback.format_exc()}")
        return None


def _looks_like_float(value: str) -> bool:
    try:
        float(value)
        return True
    except (TypeError, ValueError):
        return False


def _to_xyz(tensor_or_seq) -> Optional[Tuple[float, float, float]]:
    if tensor_or_seq is None:
        return None
    if hasattr(tensor_or_seq, "cpu"):
        tensor_or_seq = tensor_or_seq.cpu()
    if hasattr(tensor_or_seq, "tolist"):
        vals = tensor_or_seq.tolist()
    else:
        vals = list(tensor_or_seq)
    if len(vals) < 3:
        return None
    return float(vals[0]), float(vals[1]), float(vals[2])


def _safe_count_atoms(cmd, sele: str) -> int:
    try:
        return int(cmd.count_atoms(sele))
    except Exception:
        return 0


def _ensure_center_anchor(
    cmd,
    name: str,
    center: Tuple[float, float, float],
) -> str:
    if name in cmd.get_names("objects"):
        cmd.delete(name)
    cmd.pseudoatom(name, pos=[center[0], center[1], center[2]])
    return name


def _normalize_box(
    box_min: Tuple[float, float, float],
    box_max: Tuple[float, float, float],
) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    return (
        (min(box_min[0], box_max[0]), min(box_min[1], box_max[1]), min(box_min[2], box_max[2])),
        (max(box_min[0], box_max[0]), max(box_min[1], box_max[1]), max(box_min[2], box_max[2])),
    )


def _box_atom_selection(
    sele_base: str,
    box_min,
    box_max,
    pad: float = 0.0,
) -> str:
    (bx0, by0, bz0), (bx1, by1, bz1) = _normalize_box(box_min, box_max)
    if pad > 0.0:
        bx0 -= pad
        by0 -= pad
        bz0 -= pad
        bx1 += pad
        by1 += pad
        bz1 += pad
    return (
        f"({sele_base}) and (x >= {bx0} and x <= {bx1}) "
        f"and (y >= {by0} and y <= {by1}) "
        f"and (z >= {bz0} and z <= {bz1})"
    )


def _pocket_atom_selection(
    cmd,
    sele_base: str,
    box_min,
    box_max,
    center: Optional[Tuple[float, float, float]],
    reference: str,
    cutoff: float,
    center_anchor: str = "",
) -> str:
    """Pick receptor atoms for pocket visualization; prefer PyMOL distance selections."""
    candidates = []
    if reference and not os.path.isfile(reference):
        ref_sele = f"({reference})"
        if _safe_count_atoms(cmd, ref_sele) > 0:
            candidates.append(f"({sele_base}) within {cutoff} of {ref_sele}")
    if center is not None and center_anchor:
        _ensure_center_anchor(cmd, center_anchor, center)
        candidates.append(f"({sele_base}) within {cutoff} of ({center_anchor})")
    if box_min is not None and box_max is not None:
        # TRIDS box is ligand bbox + 2 A; pocket residues reach out to cutoff.
        candidates.append(_box_atom_selection(sele_base, box_min, box_max, pad=cutoff))

    for sele in candidates:
        if _safe_count_atoms(cmd, sele) > 0:
            return sele
    return candidates[-1] if candidates else sele_base


def _create_pocket_object(cmd, pocket_name: str, sele: str) -> bool:
    if _safe_count_atoms(cmd, sele) <= 0:
        _print(f"TRIDS>   Warning: no atoms matched for {pocket_name}; skipping surface.")
        return False
    if pocket_name in cmd.get_names("objects"):
        cmd.delete(pocket_name)
    cmd.create(pocket_name, sele)
    return True


def _draw_box_cgo(cmd, name: str, box_min, box_max, color: str) -> None:
    from pymol import cgo

    x0, y0, z0 = box_min
    x1, y1, z1 = box_max
    rgb = cmd.get_color_tuple(color)
    edges = (
        (x0, y0, z0, x1, y0, z0), (x1, y0, z0, x1, y1, z0), (x1, y1, z0, x0, y1, z0), (x0, y1, z0, x0, y0, z0),
        (x0, y0, z1, x1, y0, z1), (x1, y0, z1, x1, y1, z1), (x1, y1, z1, x0, y1, z1), (x0, y1, z1, x0, y0, z1),
        (x0, y0, z0, x0, y0, z1), (x1, y0, z0, x1, y0, z1), (x1, y1, z0, x1, y1, z1), (x0, y1, z0, x0, y1, z1),
    )
    obj = [cgo.BEGIN, cgo.LINES, cgo.LINEWIDTH, 2.0, cgo.COLOR, rgb[0], rgb[1], rgb[2]]
    for xa, ya, za, xb, yb, zb in edges:
        obj.extend([cgo.VERTEX, xa, ya, za, cgo.VERTEX, xb, yb, zb])
    obj.append(cgo.END)
    cmd.load_cgo(obj, name)


def _visualize_pockets(
    cmd,
    pockets: List,
    sele_base: str,
    show_b: bool,
    prefix: str = "trisite",
    reference: str = "",
    cutoff: float = 8.0,
) -> None:
    palette = [
        "red", "blue", "green", "yellow", "magenta",
        "cyan", "orange", "salmon", "lime", "slate",
    ]

    for i, pocket in enumerate(pockets):
        box_min = _to_xyz(getattr(pocket, "box_min", None))
        box_max = _to_xyz(getattr(pocket, "box_max", None))
        if box_min is not None and box_max is not None:
            box_min_str = f"({box_min[0]:.1f}, {box_min[1]:.1f}, {box_min[2]:.1f})"
            box_max_str = f"({box_max[0]:.1f}, {box_max[1]:.1f}, {box_max[2]:.1f})"
            box_info = f"box_min={box_min_str}, box_max={box_max_str}"
        else:
            box_info = "box=N/A"

        _print(
            f"TRIDS>   Pocket {i}: {pocket.num_residues} residues, "
            f"{pocket.num_atoms} atoms, {box_info}"
        )

        if not (show_b and box_min is not None and box_max is not None):
            continue

        pocket_name = f"{prefix}_{i}"
        color = palette[i % len(palette)]
        box_min, box_max = _normalize_box(box_min, box_max)
        center = _to_xyz(getattr(pocket, "center", None))
        center_anchor = f"{prefix}_{i}_anchor"
        try:
            sele = _pocket_atom_selection(
                cmd,
                sele_base,
                box_min,
                box_max,
                center,
                reference,
                cutoff,
                center_anchor=center_anchor,
            )
            if _create_pocket_object(cmd, pocket_name, sele):
                cmd.show("surface", pocket_name)
                cmd.set("transparency", 0.5, pocket_name)
                cmd.color(color, pocket_name)
        finally:
            if center_anchor in cmd.get_names("objects"):
                cmd.delete(center_anchor)

        box_name = f"{prefix}_{i}_box"
        if box_name in cmd.get_names("objects"):
            cmd.delete(box_name)
        _draw_box_cgo(cmd, box_name, box_min, box_max, color)

    if show_b and pockets:
        cmd.zoom(f"{prefix}_*")


# ---------------------------------------------------------------------------
# trisite
# ---------------------------------------------------------------------------

def trisite(
    receptor: str = "",
    reference: str = "",
    cutoff: str = "8.0",
    show: str = "1",
):
    """
DESCRIPTION

    Define or predict binding pockets on a receptor.

    Without a reference ligand, pockets are predicted by the TRIDS AI model.
    With a reference ligand, the binding site is extracted from the reference
    coordinates within the given radius.

Usage:

    trisite receptor [, reference [, cutoff [, show ]]]

Options:

    receptor  = str: PyMOL selection as the receptor
    reference = str: (optional) PyMOL selection as the reference ligand
    cutoff    = float: pocket radius in Angstrom (default: 8.0)
    show      = 0/1: auto-visualize pockets in PyMOL (default: 1)

Examples:

    device gpu, 0
    trisite protein
    trisite protein, ref_lig
    trisite protein, ref_lig, cutoff=10.0
    trisite protein, cutoff=10.0
    """
    try:
        trids = _ensure_trids()
        from pymol import cmd

        if not receptor:
            _print("Error: receptor is required. "
                   "Use 'help trisite' for usage.")
            return

        receptor = _cmd_arg(receptor)
        reference = _cmd_arg(reference)
        cutoff = _cmd_arg(cutoff)
        show = _cmd_arg(show)
        # Legacy: trisite receptor, cutoff [, show]
        if reference and _looks_like_float(reference):
            if cutoff in ("0", "1"):
                show = cutoff
            cutoff = reference
            reference = ""
        cutoff_f = float(cutoff)
        show_b = bool(int(show))

        rec_path = _selection_to_pdb(receptor)
        sele_base, tmp_obj = _resolve_receptor_object(receptor, rec_path)

        rec = trids.Receptor.from_pdb(rec_path)
        rec.delete_hydrogens()

        if reference:
            _print(
                f"TRIDS> Extracting binding site from reference "
                f"(device: {trids.get_device()}) ..."
            )
            _print(f"TRIDS>   Receptor : {receptor}")
            _print(f"TRIDS>   Reference: {reference}")
        else:
            _print(f"TRIDS> Predicting binding pockets ... (device: {trids.get_device()})")

        pockets, prefix = _make_pockets(rec, reference, cutoff_f)

        _print(f"TRIDS> Found {len(pockets)} pocket(s).")
        for i, pocket in enumerate(pockets):
            pocket_name = f"{prefix}_{i}"
            _register_pocket_cache(
                pocket_name,
                pocket,
                receptor,
                _to_xyz(getattr(pocket, "box_min", None)),
                _to_xyz(getattr(pocket, "box_max", None)),
                reference=reference,
                cutoff=cutoff_f,
                index=i,
            )
        try:
            _visualize_pockets(
                cmd, pockets, sele_base, show_b, prefix,
                reference=reference, cutoff=cutoff_f,
            )
        finally:
            if tmp_obj:
                cmd.delete(tmp_obj)

    except Exception:
        _print(f"TRIDS Error:\n{traceback.format_exc()}")


# ---------------------------------------------------------------------------
# trinfo
# ---------------------------------------------------------------------------

def trinfo():
    """
DESCRIPTION

    Display TRIDS installation and environment information.

Usage:

    trinfo
    """
    _print(_get_trinfo_text())


# ---------------------------------------------------------------------------
# Register commands with PyMOL
# ---------------------------------------------------------------------------

def _register_commands():
    """Register all TRIDS commands with PyMOL's cmd module."""
    try:
        from pymol import cmd

        cmd.extend("tridev", tridev)
        cmd.extend("trids", trids)
        cmd.extend("triscore", triscore)
        cmd.extend("trisite", trisite)
        cmd.extend("trinfo", trinfo)

        zoom = cmd.auto_arg[0]["zoom"]
        for command in ("tridev", "trinfo", "trids", "triscore", "trisite"):
            cmd.auto_arg[0][command] = zoom
        for command in ("trids", "triscore", "trisite"):
            cmd.auto_arg[1][command] = zoom

        _register_action_menu()

        print(
            "TRIDS Plugin loaded. Commands: tridev, trids, triscore, trisite, trinfo; "
            "A-menu: TRIDS"
        )
    except ImportError:
        pass


_register_commands()
