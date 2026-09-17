"""
Preprocess PDBbind v2016 raw files into .npy feature arrays for HQDeepDTAF.

Requirements:
    pip install biopython openbabel-wheel numpy

Usage:
    python preprocess.py --pdbbind_dir /path/to/pdbbind2016 --output_dir data/processed

PDBbind directory layout expected:
    pdbbind2016/
    ├── v2016-core/
    │   ├── 1a1e/
    │   │   ├── 1a1e_protein.pdb
    │   │   ├── 1a1e_pocket.pdb
    │   │   └── 1a1e_ligand.sdf
    │   └── ...
    ├── v2016-other-PL/         (general set for training)
    │   └── ...
    └── index/
        ├── INDEX_core_data.2016
        └── INDEX_general_PL_data.2016

Outputs per complex (in output_dir):
    <pdb_id>_seq.npy   float32 (1000, 40)
    <pdb_id>_pkt.npy   float32 (63, 40)
    <pdb_id>_smi.npy   int64   (150,)
    <pdb_id>_aff.npy   float32 scalar

Also writes:
    data/splits/train.txt
    data/splits/test.txt
"""

import argparse
import os
import re
import warnings
import numpy as np

from dataset import (
    encode_sequence, label_smiles,
    MAX_SEQ_LEN, MAX_PKT_LEN, MAX_SMI_LEN,
    AA_TYPES,
)

# Try to import BioPython for structure parsing and DSSP
try:
    from Bio import PDB
    from Bio.PDB.DSSP import DSSP
    BIOPYTHON_OK = True
except ImportError:
    BIOPYTHON_OK = False
    warnings.warn("BioPython not found. Install with: pip install biopython\n"
                  "Secondary structure will default to coil ('C') for all residues.")

# Try to import openbabel for SDF -> SMILES
try:
    from openbabel import openbabel as ob
    OPENBABEL_OK = True
except ImportError:
    OPENBABEL_OK = False
    warnings.warn("OpenBabel not found. Install with: pip install openbabel-wheel\n"
                  "Ligand SMILES will be set to empty strings.")

PARSER = PDB.PDBParser(QUIET=True) if BIOPYTHON_OK else None

# ---------------------------------------------------------------------------
# DSSP secondary structure mapping to 8-class scheme
# DSSP uses single letters; we map to our SSE_TYPES = (B,C,E,G,H,I,S,T)
# ---------------------------------------------------------------------------
DSSP_MAP = {
    'H': 'H',  # Alpha helix
    'B': 'B',  # Beta bridge
    'E': 'E',  # Extended strand
    'G': 'G',  # 3-10 helix
    'I': 'I',  # Pi helix
    'T': 'T',  # Turn
    'S': 'S',  # Bend
    '-': 'C',  # Coil / loop (default)
    ' ': 'C',
}


def parse_pdb_sequence(pdb_path: str) -> tuple[str, str]:
    """Parse a PDB file and return (aa_sequence, sse_sequence) strings.

    Returns:
        aa_seq:  one-letter amino acid codes (unknown → 'X')
        sse_seq: one-letter SSE codes from DSSP (default 'C' if DSSP fails)
    """
    if not BIOPYTHON_OK:
        # Fallback: extract sequence from SEQRES or ATOM records, no SSE
        aa_seq = _parse_sequence_from_atoms(pdb_path)
        sse_seq = 'C' * len(aa_seq)
        return aa_seq, sse_seq

    try:
        structure = PARSER.get_structure('prot', pdb_path)
        model = structure[0]

        # Get residue sequence from ATOM records
        three_to_one = _three_to_one_map()
        aa_seq = ''
        residue_list = []
        for chain in model:
            for res in chain:
                if res.id[0] == ' ':  # ATOM records only (skip HETATM)
                    aa = three_to_one.get(res.resname.strip(), 'X')
                    aa_seq += aa
                    residue_list.append((chain.id, res.id))

        # Run DSSP for secondary structure
        try:
            dssp = DSSP(model, pdb_path, dssp='mkdssp')
            sse_seq = ''
            for chain_id, res_id in residue_list:
                key = (chain_id, res_id)
                if key in dssp:
                    ss = dssp[key][2]  # secondary structure character
                    sse_seq += DSSP_MAP.get(ss, 'C')
                else:
                    sse_seq += 'C'
        except Exception:
            sse_seq = 'C' * len(aa_seq)

        return aa_seq, sse_seq

    except Exception as e:
        warnings.warn(f"PDB parsing failed for {pdb_path}: {e}")
        return '', ''


def _parse_sequence_from_atoms(pdb_path: str) -> str:
    """Fallback: extract residue sequence from ATOM records without BioPython."""
    three_to_one = _three_to_one_map()
    seen = set()
    seq = ''
    with open(pdb_path) as f:
        for line in f:
            if line.startswith('ATOM'):
                chain = line[21]
                res_name = line[17:20].strip()
                res_seq = line[22:26].strip()
                ins_code = line[26].strip()
                key = (chain, res_seq, ins_code)
                if key not in seen:
                    seen.add(key)
                    seq += three_to_one.get(res_name, 'X')
    return seq


def sdf_to_smiles(sdf_path: str) -> str:
    """Convert SDF ligand file to SMILES string using OpenBabel."""
    if not OPENBABEL_OK:
        return ''
    try:
        conv = ob.OBConversion()
        conv.SetInAndOutFormats('sdf', 'smi')
        mol = ob.OBMol()
        conv.ReadFile(mol, sdf_path)
        smiles = conv.WriteString(mol).strip().split()[0]  # first token is SMILES
        return smiles
    except Exception as e:
        warnings.warn(f"SDF→SMILES failed for {sdf_path}: {e}")
        return ''


def parse_index_file(index_path: str) -> dict[str, float]:
    """Parse PDBbind INDEX file and return {pdb_id: -logKd} dict."""
    affinities = {}
    with open(index_path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            if len(parts) >= 4:
                pdb_id = parts[0].lower()
                try:
                    affinity = float(parts[3])
                    affinities[pdb_id] = affinity
                except ValueError:
                    continue
    return affinities


def process_complex(pdb_id: str, pdb_dir: str, output_dir: str, affinity: float) -> bool:
    """Process one protein-ligand complex and save .npy feature files."""
    protein_pdb = os.path.join(pdb_dir, pdb_id, f'{pdb_id}_protein.pdb')
    pocket_pdb  = os.path.join(pdb_dir, pdb_id, f'{pdb_id}_pocket.pdb')
    ligand_sdf  = os.path.join(pdb_dir, pdb_id, f'{pdb_id}_ligand.sdf')

    if not os.path.exists(protein_pdb) or not os.path.exists(pocket_pdb):
        warnings.warn(f"Missing PDB files for {pdb_id}, skipping.")
        return False

    # Protein sequence + SSE
    seq_aa, seq_sse = parse_pdb_sequence(protein_pdb)
    if not seq_aa:
        return False
    seq_feat = encode_sequence(seq_aa, seq_sse, max_len=MAX_SEQ_LEN)   # (1000, 40)

    # Pocket sequence + SSE
    pkt_aa, pkt_sse = parse_pdb_sequence(pocket_pdb)
    if not pkt_aa:
        return False
    pkt_feat = encode_sequence(pkt_aa, pkt_sse, max_len=MAX_PKT_LEN)   # (63, 40)

    # Ligand SMILES
    smiles = sdf_to_smiles(ligand_sdf) if os.path.exists(ligand_sdf) else ''
    smi_feat = label_smiles(smiles, max_len=MAX_SMI_LEN)                # (150,)

    # Save
    np.save(os.path.join(output_dir, f'{pdb_id}_seq.npy'), seq_feat)
    np.save(os.path.join(output_dir, f'{pdb_id}_pkt.npy'), pkt_feat)
    np.save(os.path.join(output_dir, f'{pdb_id}_smi.npy'), smi_feat)
    np.save(os.path.join(output_dir, f'{pdb_id}_aff.npy'), np.float32(affinity))
    return True


def _three_to_one_map() -> dict[str, str]:
    return {
        'ALA': 'A', 'ARG': 'R', 'ASN': 'N', 'ASP': 'D', 'CYS': 'C',
        'GLN': 'Q', 'GLU': 'E', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
        'LEU': 'L', 'LYS': 'K', 'MET': 'M', 'PHE': 'F', 'PRO': 'P',
        'SER': 'S', 'THR': 'T', 'TRP': 'W', 'TYR': 'Y', 'VAL': 'V',
        'MSE': 'M',  # selenomethionine
        'HSD': 'H', 'HSE': 'H', 'HSP': 'H',  # histidine variants
    }


def main():
    p = argparse.ArgumentParser(description='Preprocess PDBbind v2016 for HQDeepDTAF')
    p.add_argument('--pdbbind_dir', required=True, help='Root PDBbind directory')
    p.add_argument('--output_dir', default='data/processed')
    p.add_argument('--core_subdir', default='v2016-core', help='Core set subdirectory name')
    p.add_argument('--general_subdir', default='v2016-other-PL', help='General set subdirectory name')
    p.add_argument('--core_index', default='index/INDEX_core_data.2016')
    p.add_argument('--general_index', default='index/INDEX_general_PL_data.2016')
    args = p.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs('data/splits', exist_ok=True)

    core_index_path    = os.path.join(args.pdbbind_dir, args.core_index)
    general_index_path = os.path.join(args.pdbbind_dir, args.general_index)
    core_dir           = os.path.join(args.pdbbind_dir, args.core_subdir)
    general_dir        = os.path.join(args.pdbbind_dir, args.general_subdir)

    # Load affinity labels
    core_affinities    = parse_index_file(core_index_path)
    general_affinities = parse_index_file(general_index_path)

    # Remove core set complexes from general set (they overlap)
    general_only = {k: v for k, v in general_affinities.items() if k not in core_affinities}

    print(f"Core (test):    {len(core_affinities)} complexes")
    print(f"General (train): {len(general_only)} complexes")

    # Process test (core) set
    test_ids = []
    for pdb_id, aff in core_affinities.items():
        ok = process_complex(pdb_id, core_dir, args.output_dir, aff)
        if ok:
            test_ids.append(pdb_id)
    print(f"Processed {len(test_ids)} core complexes")

    # Process train (general) set
    train_ids = []
    for pdb_id, aff in general_only.items():
        ok = process_complex(pdb_id, general_dir, args.output_dir, aff)
        if ok:
            train_ids.append(pdb_id)
    print(f"Processed {len(train_ids)} general complexes")

    # Write split files
    with open('data/splits/train.txt', 'w') as f:
        f.write('\n'.join(train_ids))
    with open('data/splits/test.txt', 'w') as f:
        f.write('\n'.join(test_ids))

    print(f"\nDone. Features saved to {args.output_dir}/")
    print(f"Split files: data/splits/train.txt ({len(train_ids)}), data/splits/test.txt ({len(test_ids)})")


if __name__ == '__main__':
    main()
