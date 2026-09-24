# HQDeepDTAF — Implementation Guide

Hybrid Quantum Neural Network for protein-ligand binding affinity prediction.

**Paper:** Jeong et al., *Hybrid quantum neural networks for efficient protein-ligand binding affinity prediction*, EPJ Quantum Technology (2025) 12:120  
**Original repo:** https://github.com/drmoon-1st/HQDeepDTAF

---

## Quick Reference

| Goal | File | Command / Action |
|------|------|-----------------|
| Smoke test (no data) | `HQDeepDTAF_smoketest.ipynb` | Upload to Colab → Run all |
| Full training | `HQDeepDTAF_fullrun.ipynb` | Upload to Colab → configure paths → Run all |
| Local smoke test | `train.py` | `python train.py --dummy --epochs 2 --runs 1` |
| Preprocess PDBbind | `preprocess.py` | `python preprocess.py --pdbbind_dir /path/to/pdbbind2016` |

---

## Repository Contents

```
HybridQNN/
├── model.py                     # HQDeepDTAF model, VQC circuit, test() loop
├── dataset.py                   # SMILES vocabulary, feature encoding, PDBbindDataset, DummyDataset
├── metrics.py                   # c_index, RMSE, MAE, SD, CORR
├── train.py                     # Training script with --dummy flag
├── preprocess.py                # PDBbind v2016 → .npy feature pipeline
├── HQDeepDTAF_smoketest.ipynb   # Self-contained Colab smoke test (no data needed)
├── HQDeepDTAF_fullrun.ipynb     # Full training on PDBbind v2016 via Google Colab
└── context.md                   # Architecture notes and paper summary
```

---

## Option 1 — Smoke Test (~15 min, no data needed)

Verifies the model architecture runs end-to-end using synthetic data.

1. Open [colab.research.google.com](https://colab.research.google.com)
2. **File → Upload notebook** → select `HQDeepDTAF_smoketest.ipynb`
3. **Runtime → Run all**

Expected final output:
```
Smoke test PASSED — model runs end-to-end.
```

---

## Option 2 — Full Training on PDBbind v2016

### Prerequisites

| Requirement | Notes |
|-------------|-------|
| Google account | For Colab + Google Drive |
| ~25 GB Google Drive space | PDBbind raw (~20 GB) + processed (~5 GB) |
| PDBbind v2016 registration | Free academic registration at pdbbind.org.cn |
| Colab runtime | Free CPU works; Pro recommended for longer sessions |

---

### Step 1 — Download PDBbind v2016

Register and download from [pdbbind.org.cn](http://www.pdbbind.org.cn). You need three files:

| File | Size | Contents |
|------|------|----------|
| `PDBbind_v2016_plain_text_index.tar.gz` | ~1 MB | Affinity labels |
| `PDBbind_v2016_core_set.tar.gz` | ~400 MB | 290 test complexes |
| `PDBbind_v2016_other_PL_set.tar.gz` | ~18 GB | ~3,700 train complexes |

Extract so the layout is:
```
pdbbind2016/
├── v2016-core/
│   ├── 1a1e/
│   │   ├── 1a1e_protein.pdb
│   │   ├── 1a1e_pocket.pdb
│   │   └── 1a1e_ligand.sdf
│   └── ...
├── v2016-other-PL/
│   └── ...
└── index/
    ├── INDEX_core_data.2016
    └── INDEX_general_PL_data.2016
```

---

### Step 2 — Set Up Google Drive

Create a folder `HybridQNN/` in your Google Drive and upload:

```
MyDrive/HybridQNN/
├── metrics.py
├── dataset.py
├── model.py
├── preprocess.py
├── train.py
└── pdbbind2016/          ← extracted PDBbind data
    ├── v2016-core/
    ├── v2016-other-PL/
    └── index/
```

The preprocessing outputs (`data/processed/`, `data/splits/`) and checkpoints will be saved here automatically.

---

### Step 3 — Run the Full Training Notebook

1. Open [colab.research.google.com](https://colab.research.google.com)
2. **File → Upload notebook** → select `HQDeepDTAF_fullrun.ipynb`
3. Edit **Cell 3 (Configure Paths)** if your Drive folder is named differently
4. **Runtime → Run all**

The notebook will:
- Install all dependencies (including `dssp` binary for secondary structure)
- Mount your Google Drive
- Verify source files and PDBbind structure
- Run preprocessing (~2–4 hours, saved directly to Drive)
- Train for 5 runs × 20 epochs with epoch-level checkpointing

**If the Colab session times out**, simply re-run the notebook — it will resume from the last completed epoch automatically.

---

### Step 4 — Retrieve Results

After training, download from Drive:
- `HybridQNN/checkpoints/best_model.pt` — trained model weights
- `HybridQNN/checkpoints/progress.json` — run history and metrics

---

## Training Configuration

```python
optimizer    = AdamW
lr           = 0.005
weight_decay = 0.01
loss         = MSELoss
epochs       = 20
batch_size   = 16
runs         = 5          # paper selects run with lowest train MSE
```

---

## Estimated Training Time

| Environment | Preprocessing | Training (5 runs × 20 epochs) |
|-------------|--------------|-------------------------------|
| Colab free CPU | 2–4 hours | 5–10 days (use checkpointing) |
| Colab Pro (A100) | 2–4 hours | 2–4 days (GPU helps classical layers only) |
| 32-core workstation | 1–2 hours | 2–3 days |
| Paper hardware (Ryzen 9 7950X) | — | ~2 days |

> PennyLane's `default.qubit` does not use the GPU — the VQC simulation always runs on CPU.

---

## Model Architecture

```
Protein seq (1000 × 40)  → Linear(40, 128) → DilatedResBlockA (rates 1,2,4,8,16) → MaxPool → 128-dim
Pocket  seq (63 × 40)    → Linear(40, 128) → Conv1d(32→64→128)                    → MaxPool → 128-dim
SMILES      (150,)       → Embedding(64, 128) → DilatedResBlockB (rates 1,2,4,8)  → MaxPool → 128-dim

Concat(384) → Dropout(0.2) → Linear(384, n_qubits=10)
→ ReUploadingVQC (10 qubits, 20 layers, 830 quantum params)
→ Linear(10, 1) → PReLU → binding affinity
```

**Quantum circuit:** PennyLane `default.qubit`, parameter-shift gradients, wrapped as `qml.qnn.TorchLayer`

---

## Evaluation Metrics

| Metric | Direction | Description |
|--------|-----------|-------------|
| MAE | ↓ lower is better | Mean absolute error |
| RMSE | ↓ lower is better | Root mean squared error |
| R (CORR) | ↑ higher is better | Pearson correlation |
| SD | ↓ lower is better | Std dev of residuals after linear fit |
| CI | ↑ higher is better | Concordance index |

**Paper best results (HQDeepDTAF-NN-Angle, 9 qubits, PDBbind v2016 Core):**

| MAE | RMSE | R | SD | CI |
|-----|------|---|----|----|
| 1.082 | 1.368 | 0.783 | 1.355 | 0.792 |

---

## Troubleshooting

### `pip install` returns HTTP 403 Forbidden
Corporate network blocks PyPI. Use the Colab notebooks (install runs on Google's servers) or ask IT for the internal pip mirror URL.

### `git push` SSL certificate error
```powershell
git config --global http.sslBackend schannel
```
Uses Windows' native SSL library which trusts the corporate CA certificate.

### `mkdssp` not found during preprocessing
On Colab, the `dssp` apt package installs the binary. Locally on Windows, download from [swift.cmbi.ru.nl/gv/dssp](https://swift.cmbi.ru.nl/gv/dssp/) or install via conda: `conda install -c salilab dssp`.

### Colab session timed out mid-training
Re-run the notebook. The training cell reads `progress.json` and resumes from the last completed epoch.

### `Squeeze()` error with batch size 1
The DataLoader uses `drop_last=True` to prevent batches of size 1, which would cause `Squeeze()` to drop the batch dimension.
