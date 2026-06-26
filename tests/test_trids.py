#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
TRIDS Python API Test Script

Tests basic functionality of TRIDS Python interface, including:
1. Module import
2. Molecule loading and manipulation
3. Scoring function tests
4. Docking functionality tests

Usage:
    python test_pytrids.py -g 0      # GPU 0
    python test_pytrids.py -g -1     # CPU
"""

import os
import sys
import time
import argparse
from pathlib import Path

# Parse command line arguments
parser = argparse.ArgumentParser(description="TRIDS Python API Test Script")
parser.add_argument(
    "-g", "--gpu",
    type=int,
    default=0,
    help="Device to run tests on (default: 0, CPU: -1)."
)
args = parser.parse_args()

# Set device from command line argument
if args.gpu < 0:
    DEFAULT_DEVICE_KIND = "cpu"
    DEFAULT_DEVICE_ARG2 = os.cpu_count() or 1
else:
    DEFAULT_DEVICE_KIND = "gpu"
    DEFAULT_DEVICE_ARG2 = args.gpu

# Ensure trids module can be found
SCRIPT_DIR = Path(__file__).parent.absolute()
PROJECT_ROOT = SCRIPT_DIR.parent

print("=" * 60)
print("TRIDS Python API Test")
print("=" * 60)
print(f"Using device: {DEFAULT_DEVICE_KIND}, arg2={DEFAULT_DEVICE_ARG2}")

# ==================== 1. Test Module Import ====================
print("\n[1] Testing module import...")

try:
    import trids
    trids.set_device(DEFAULT_DEVICE_KIND, DEFAULT_DEVICE_ARG2)
    print(f"    [DONE] trids module imported successfully")
    print(f"    [DONE] Version: {trids.__version__}")
except ImportError as e:
    print(f"    [FAIL] Import failed: {e}")
    sys.exit(1)

# Check C++ core module
if trids.check_installation():
    print("    [DONE] C++ core module available")
else:
    print("    [FAIL] C++ core module not available")
    print("    Please run: pip install -e .")
    sys.exit(1)

# Print environment info
print("\n[2] Environment info:")
trids.utils.print_cuda_info()

# ==================== 2. Test Molecule Loading ====================
print("\n[3] Testing molecule loading...")

# Example file paths
receptor_file = SCRIPT_DIR / "target.pdb"
ligand_init_file = SCRIPT_DIR / "ligand.sdf"
ligand_ref_file = SCRIPT_DIR / "reference.sdf"

# Check if files exist
for f in [receptor_file, ligand_init_file, ligand_ref_file]:
    if not f.exists():
        print(f"    [FAIL] File not found: {f}")
        sys.exit(1)

try:
    # Load receptor
    receptor = trids.Receptor.from_pdb(str(receptor_file))
    print(f"    [DONE] Receptor loaded successfully: {receptor}")
    
    # Delete hydrogen atoms
    receptor.delete_hydrogens()
    print(f"    [DONE] After removing hydrogens: {receptor.num_atoms} atoms, {receptor.num_residues} residues")
    
except Exception as e:
    print(f"    [FAIL] Receptor loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

try:
    # Load ligand
    ligand_init = trids.Molecule.from_file(str(ligand_init_file))
    print(f"    [DONE] Ligand (initial) loaded successfully: {ligand_init}")
    
    ligand_ref = trids.Molecule.from_file(str(ligand_ref_file))
    print(f"    [DONE] Ligand (reference) loaded successfully: {ligand_ref}")
    
    # Delete hydrogen atoms
    ligand_init.delete_hydrogens()
    ligand_ref.delete_hydrogens()
    print(f"    [DONE] After removing hydrogens: initial ligand {ligand_init.num_atoms} atoms, reference ligand {ligand_ref.num_atoms} atoms")
    
except Exception as e:
    print(f"    [FAIL] Ligand loading failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ==================== 3. Test Coordinate Operations ====================
print("\n[4] Testing coordinate operations...")

try:
    import torch
    
    coords = ligand_ref.coordinates
    print(f"    [DONE] Coordinates retrieved successfully: shape = {coords.shape}")
    print(f"    [DONE] Coordinate range: x=[{coords[:, 0].min():.2f}, {coords[:, 0].max():.2f}], "
          f"y=[{coords[:, 1].min():.2f}, {coords[:, 1].max():.2f}], "
          f"z=[{coords[:, 2].min():.2f}, {coords[:, 2].max():.2f}]")
    
except Exception as e:
    print(f"    [FAIL] Coordinate operation failed: {e}")
    import traceback
    traceback.print_exc()

# ==================== 4. Test Pocket Extraction ====================
print("\n[5] Testing pocket extraction...")

try:
    pocket = receptor.extract_pocket(ligand_ref)
    print(f"    [DONE] Pocket extracted successfully: {pocket}")
    print(f"    [DONE] Pocket center: {pocket.center}")
    
except Exception as e:
    print(f"    [FAIL] Pocket extraction failed: {e}")
    import traceback
    traceback.print_exc()

# ==================== 5. Test Graph Conversion ====================
print("\n[6] Testing molecular graph conversion...")

try:
    node_feats, edge_feats, edge_index = ligand_ref.to_graph()
    print(f"    [DONE] Ligand graph conversion successful:")
    print(f"      - Node features: {node_feats.shape}")
    print(f"      - Edge features: {edge_feats.shape}")
    print(f"      - Edge index: {edge_index.shape}")
    
except Exception as e:
    print(f"    [FAIL] Ligand graph conversion failed: {e}")
    import traceback
    traceback.print_exc()

try:
    node_feats, edge_feats, edge_index = pocket.to_graph()
    print(f"    [DONE] Pocket graph conversion successful:")
    print(f"      - Node features: {node_feats.shape}")
    print(f"      - Edge features: {edge_feats.shape}")
    print(f"      - Edge index: {edge_index.shape}")
    
except Exception as e:
    print(f"    [FAIL] Pocket graph conversion failed: {e}")
    import traceback
    traceback.print_exc()

# ==================== 6. Test SMILES to Molecule ====================
print("\n[7] Testing SMILES to molecule...")

try:
    smiles = "c1ccccc1"  # Benzene ring
    mol_from_smiles = trids.Molecule.from_smiles(smiles, gen_3d=True)
    print(f"    [DONE] SMILES '{smiles}' to molecule successful: {mol_from_smiles}")
    
except Exception as e:
    print(f"    [FAIL] SMILES to molecule failed: {e}")
    import traceback
    traceback.print_exc()

# ==================== 7. Test RMSD Calculation ====================
print("\n[8] Testing RMSD calculation...")

try:
    coords1 = ligand_ref.coordinates
    coords2 = ligand_init.coordinates[:coords1.shape[0]]  # Atom count may differ
    
    if coords1.shape == coords2.shape:
        rmsd = trids.utils.calculate_rmsd(coords1, coords2, align=True)
        print(f"    [DONE] RMSD calculation successful: {rmsd:.4f} Å")
    else:
        print(f"    [WARN] Skipping RMSD calculation (different atom counts: {coords1.shape[0]} vs {coords2.shape[0]})")
    
except Exception as e:
    print(f"    [FAIL] RMSD calculation failed: {e}")
    import traceback
    traceback.print_exc()

# ==================== 8. Test Scoring Function Creation (requires model) ====================
print("\n[9] Testing scoring function creation...")

# Check model file
model_path = PROJECT_ROOT / "lib" / "libtrids_sampling.pt"

if model_path.exists():
    try:
        scorer = trids.TriScorer(model_path=str(model_path))
        print(f"    [DONE] TRI scoring function created successfully: {scorer}")
        
        # Set pocket and ligand
        scorer.set_pocket(pocket)
        scorer.set_ligand(ligand_ref)
        scorer.profile()
        
        # Calculate score
        score = scorer.score(ligand_ref.coordinates)
        print(f"    [DONE] Reference ligand score: {score:.4f}")
        
    except Exception as e:
        print(f"    [FAIL] TRI scoring function test failed: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"    [WARN] Model file not found: {model_path}")
    print("    [WARN] Skipping TRI scoring function test")

# Test Vina scoring function
try:
    vina_scorer = trids.VinaScorer()
    print(f"    [DONE] Vina scoring function created successfully: {vina_scorer}")
    
    # Set pocket and ligand
    vina_scorer.set_pocket(pocket)
    vina_scorer.set_ligand(ligand_ref)
    vina_scorer.profile()
    
    # Calculate score
    vina_score = vina_scorer.score(ligand_ref.coordinates)
    print(f"    [DONE] Reference ligand Vina score: {vina_score:.4f}")
    
except Exception as e:
    print(f"    [FAIL] Vina scoring function test failed: {e}")
    import traceback
    traceback.print_exc()

# ==================== 10. Test Docking (using reference ligand pocket) ====================
print("\n[10] Testing Docking (using reference ligand to extract pocket)...")

if model_path.exists():
    try:
        # Create scoring function
        dock_scorer = trids.TriScorer(model_path=str(model_path))
        
        # Create docking engine
        engine = trids.Engine(
            scorer=dock_scorer,
            streams=64,    # Reduce streams for faster testing
            depth=16,      # Reduce depth for faster testing
            top_n=3,
        )
        print(f"    [DONE] Docking engine created successfully: {engine}")

        start_time = time.time()
        ref_pocket = receptor.extract_pocket(ligand_ref)
        trids.utils.set_seed(42)
        results = engine.dock(ligand_init, ref_pocket)
        elapsed = time.time() - start_time

        print(f"    [DONE] Docking completed ({elapsed:.2f}s), found {len(results)} conformations")
        for result in results:
            if result.molecule.num_atoms == ligand_ref.num_atoms:
                rmsd = trids.utils.calculate_rmsd(
                    result.molecule.coordinates,
                    ligand_ref.coordinates,
                    align=True,
                )
                print(f"      - Rank {result.rank}: score={result.score:.2f}, RMSD={rmsd:.2f} Å")
            else:
                print(f"      - Rank {result.rank}: score={result.score:.2f}")

    except Exception as e:
        print(f"    [FAIL] Docking (reference ligand pocket) test failed: {e}")
        import traceback
        traceback.print_exc()
else:
    print(f"    [WARN] Model file not found, skipping Docking test")

# ==================== 11. Test Docking (using AI predicted pocket) ====================
print("\n[11] Testing Docking (using AI model to predict pocket)...")

binding_model_path = PROJECT_ROOT / "lib" / "libtrids_site_binding.pt"

if model_path.exists() and binding_model_path.exists():
    try:
        # Use AI model to predict pockets
        start_time = time.time()
        predicted_pockets = receptor.predict_pockets()
        pred_time = time.time() - start_time
        print(f"    [DONE] AI model pocket prediction completed ({pred_time:.2f}s)")
        print(f"    [DONE] Predicted {len(predicted_pockets)} binding sites")
        
        if len(predicted_pockets) > 0:
            # Use first predicted pocket for docking
            pred_pocket = predicted_pockets[0]
            print(f"    [DONE] Using predicted pocket 0: {pred_pocket.num_residues} residues, center: {pred_pocket.center}")
            
            # Create scoring function and docking engine
            dock_scorer2 = trids.TriScorer(model_path=str(model_path))
            engine2 = trids.Engine(
                scorer=dock_scorer2,
                streams=64,
                depth=16,
                top_n=3,
            )
            
            # Execute docking
            trids.utils.set_seed(42)
            start_time = time.time()
            results2 = engine2.dock(ligand_init, pred_pocket)
            elapsed = time.time() - start_time
            
            print(f"    [DONE] Docking completed ({elapsed:.2f}s), found {len(results2)} conformations")
            for result in results2:
                # Calculate RMSD to reference ligand
                if result.molecule.num_atoms == ligand_ref.num_atoms:
                    rmsd = trids.utils.calculate_rmsd(
                        result.molecule.coordinates, 
                        ligand_ref.coordinates, 
                        align=True
                    )
                    print(f"      - Rank {result.rank}: score={result.score:.2f}, RMSD={rmsd:.2f} Å")
                else:
                    print(f"      - Rank {result.rank}: score={result.score:.2f}")
        else:
            print(f"    [WARN] No pockets predicted, skipping docking test")
        
    except Exception as e:
        print(f"    [FAIL] Docking (AI predicted pocket) test failed: {e}")
        import traceback
        traceback.print_exc()
else:
    missing_files = []
    if not model_path.exists():
        missing_files.append(str(model_path))
    if not binding_model_path.exists():
        missing_files.append(str(binding_model_path))
    print(f"    [WARN] Model file(s) not found: {', '.join(missing_files)}")
    print(f"    [WARN] Skipping AI predicted pocket Docking test")

# ==================== Summary ====================
print("\n" + "=" * 60)
print("Test completed!")
print("=" * 60)


