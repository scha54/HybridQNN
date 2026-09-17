"""
PDBbind v2016 dataset for HQDeepDTAF.

Directory layout expected after preprocessing:
    data/
    ├── processed/
    │   ├── <pdb_id>_seq.npy      # float32 (1000, 40)
    │   ├── <pdb_id>_pkt.npy      # float32 (63, 40)
    │   ├── <pdb_id>_smi.npy      # int64   (150,)
    │   └── <pdb_id>_aff.npy      # float32 scalar
    └── splits/
        ├── train.txt             # one PDB ID per line
        └── test.txt

Run preprocess.py first to generate processed/ from raw PDBbind files.
Use DummyDataset to verify the model runs without real data.
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# SMILES vocabulary (exact match to KailiWang1/DeepDTAF src/dataset.py)
# ---------------------------------------------------------------------------
CHAR_SMI_SET = {
    "(": 1,  ".": 2,  "0": 3,  "2": 4,  "4": 5,  "6": 6,  "8": 7,  "@": 8,
    "B": 9,  "D": 10, "F": 11, "H": 12, "L": 13, "N": 14, "P": 15, "R": 16,
    "T": 17, "V": 18, "Z": 19, "\\": 20,"b": 21, "d": 22, "f": 23, "h": 24,
    "l": 25, "n": 26, "r": 27, "t": 28, "#": 29, "%": 30, ")": 31, "+": 32,
    "-": 33, "/": 34, "1": 35, "3": 36, "5": 37, "7": 38, "9": 39, "=": 40,
    "A": 41, "C": 42, "E": 43, "G": 44, "I": 45, "K": 46, "M": 47, "O": 48,
    "S": 49, "U": 50, "W": 51, "Y": 52, "[": 53, "]": 54, "a": 55, "c": 56,
    "e": 57, "g": 58, "i": 59, "m": 60, "o": 61, "s": 62, "u": 63, "y": 64,
}
CHAR_SMI_SET_LEN = len(CHAR_SMI_SET)  # 64

# ---------------------------------------------------------------------------
# Protein feature encoding constants  (PT_FEATURE_SIZE = 40)
# Feature order: c1(4) + c2(7) + sse(8) + aa(21) = 40
# ---------------------------------------------------------------------------
PT_FEATURE_SIZE = 40

# 21 amino acid types (20 standard + X for unknown/non-standard)
AA_TYPES = ('G', 'A', 'V', 'L', 'I', 'M', 'F', 'P', 'W',
            'S', 'T', 'Y', 'C', 'Q', 'N', 'D', 'E', 'K', 'R', 'H', 'X')
AA_INDEX = {aa: i for i, aa in enumerate(AA_TYPES)}

# 8 secondary structure types (SSPro 8-class / DSSP)
SSE_TYPES = ('B', 'C', 'E', 'G', 'H', 'I', 'S', 'T')
SSE_INDEX = {s: i for i, s in enumerate(SSE_TYPES)}

# c1: 4 chemical-nature groups
C1_GROUPS = {
    'non_polar': frozenset(['G', 'A', 'V', 'L', 'I', 'M', 'F', 'P', 'W']),
    'polar':     frozenset(['S', 'T', 'Y', 'C', 'Q', 'N']),
    'acidic':    frozenset(['D', 'E']),
    'basic':     frozenset(['K', 'R', 'H']),
}
C1_KEYS = ('non_polar', 'polar', 'acidic', 'basic')

# c2: 7 hydrophobicity/charge cluster groups
C2_GROUPS = {
    1: frozenset(['A', 'G', 'V']),
    2: frozenset(['I', 'L', 'F', 'P']),
    3: frozenset(['Y', 'M', 'T', 'S']),
    4: frozenset(['H', 'N', 'Q', 'W']),
    5: frozenset(['R', 'K']),
    6: frozenset(['D', 'E']),
    7: frozenset(['C']),
}
C2_KEYS = (1, 2, 3, 4, 5, 6, 7)

# Sequence length constants
MAX_SEQ_LEN = 1000   # protein sequence
MAX_PKT_LEN = 63     # binding pocket
MAX_SMI_LEN = 150    # SMILES string


# ---------------------------------------------------------------------------
# Encoding functions
# ---------------------------------------------------------------------------

def encode_residue(aa: str, sse: str = 'C') -> np.ndarray:
    """Encode one residue to a 40-dim float32 feature vector.

    Order: c1(4) + c2(7) + sse(8) + aa(21) = 40.
    Unknown residue 'X' gets uniform fractional values for c1/c2.
    """
    feat = np.zeros(40, dtype=np.float32)
    aa = aa if aa in AA_INDEX else 'X'

    # c1: dims 0-3
    for i, key in enumerate(C1_KEYS):
        feat[i] = (1.0 / len(C1_KEYS)) if aa == 'X' else (1.0 if aa in C1_GROUPS[key] else 0.0)

    # c2: dims 4-10
    for i, key in enumerate(C2_KEYS):
        feat[4 + i] = (1.0 / len(C2_KEYS)) if aa == 'X' else (1.0 if aa in C2_GROUPS[key] else 0.0)

    # sse: dims 11-18
    sse_char = sse if sse in SSE_INDEX else 'C'
    feat[11 + SSE_INDEX[sse_char]] = 1.0

    # aa: dims 19-39
    feat[19 + AA_INDEX[aa]] = 1.0

    return feat


def encode_sequence(aa_seq: str, sse_seq: str = '', max_len: int = MAX_SEQ_LEN) -> np.ndarray:
    """Encode a protein/pocket sequence to (max_len, 40) float32 array."""
    result = np.zeros((max_len, PT_FEATURE_SIZE), dtype=np.float32)
    for i, aa in enumerate(aa_seq[:max_len]):
        sse = sse_seq[i] if i < len(sse_seq) else 'C'
        result[i] = encode_residue(aa, sse)
    return result


def label_smiles(smiles: str, max_len: int = MAX_SMI_LEN) -> np.ndarray:
    """Encode SMILES to int64 array (0-based indices for nn.Embedding)."""
    result = np.zeros(max_len, dtype=np.int64)
    for i, ch in enumerate(smiles[:max_len]):
        if ch in CHAR_SMI_SET:
            result[i] = CHAR_SMI_SET[ch] - 1  # 1-based dict → 0-based embedding
    return result


# ---------------------------------------------------------------------------
# Datasets
# ---------------------------------------------------------------------------

class PDBbindDataset(Dataset):
    """Loads pre-processed PDBbind v2016 features from disk.

    Pre-processing (run preprocess.py first):
        <processed_dir>/<pdb_id>_seq.npy  — float32 (1000, 40)
        <processed_dir>/<pdb_id>_pkt.npy  — float32 (63, 40)
        <processed_dir>/<pdb_id>_smi.npy  — int64   (150,)
        <processed_dir>/<pdb_id>_aff.npy  — float32 scalar
    """

    def __init__(self, processed_dir: str, split_file: str):
        """
        Args:
            processed_dir: path to directory with .npy feature files
            split_file: text file with one PDB ID per line
        """
        self.processed_dir = processed_dir
        with open(split_file) as f:
            self.ids = [line.strip() for line in f if line.strip()]

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        pid = self.ids[idx]
        d = self.processed_dir
        seq = torch.from_numpy(np.load(os.path.join(d, f'{pid}_seq.npy')))   # (1000, 40)
        pkt = torch.from_numpy(np.load(os.path.join(d, f'{pid}_pkt.npy')))   # (63, 40)
        smi = torch.from_numpy(np.load(os.path.join(d, f'{pid}_smi.npy')))   # (150,)
        aff = torch.tensor(float(np.load(os.path.join(d, f'{pid}_aff.npy'))), dtype=torch.float32)
        return seq, pkt, smi, aff


class DummyDataset(Dataset):
    """Generates random tensors with correct shapes for smoke-testing the model.

    No real data or preprocessing required.

    Usage:
        ds = DummyDataset(n=32)
        loader = DataLoader(ds, batch_size=4)
        model = DeepDTAF()
        for seq, pkt, smi, aff in loader:
            out = model(seq, pkt, smi)
    """

    def __init__(self, n: int = 64):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        seq = torch.zeros(MAX_SEQ_LEN, PT_FEATURE_SIZE, dtype=torch.float32)
        pkt = torch.zeros(MAX_PKT_LEN, PT_FEATURE_SIZE, dtype=torch.float32)
        smi = torch.zeros(MAX_SMI_LEN, dtype=torch.int64)
        aff = torch.tensor(5.0 + torch.rand(1).item(), dtype=torch.float32)
        return seq, pkt, smi, aff
