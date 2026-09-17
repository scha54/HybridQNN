# HQDeepDTAF — Local Implementation

Hybrid Quantum Neural Network for protein-ligand binding affinity prediction.

**Paper:** Jeong et al., *Hybrid quantum neural networks for efficient protein-ligand binding affinity prediction*, EPJ Quantum Technology (2025) 12:120  
**Original repo:** https://github.com/drmoon-1st/HQDeepDTAF

---

## Repository Contents

```
HybridQNN/
├── model.py                     # HQDeepDTAF model + VQC + test() loop
├── dataset.py                   # SMILES vocab, feature encoding, PDBbindDataset, DummyDataset
├── metrics.py                   # c_index, RMSE, MAE, SD, CORR
├── train.py                     # Training script (--dummy flag for smoke test)
├── preprocess.py                # PDBbind v2016 → .npy preprocessing pipeline
├── HQDeepDTAF_smoketest.ipynb   # Self-contained Colab smoke test notebook
└── context.md                   # Architecture notes and paper summary
```

---

## Smoke Test (Google Colab — no local install needed)

Use this if your local machine cannot reach PyPI (e.g. corporate network restrictions).

### Steps

1. Open [colab.research.google.com](https://colab.research.google.com) in a browser (sign in with a personal Google account)
2. **File → Upload notebook** → select `HQDeepDTAF_smoketest.ipynb`
3. **Runtime → Run all** (or run cells top-to-bottom with Shift+Enter)

### What the notebook does

| Cell | Action |
|------|--------|
| 1 | Installs `torch`, `pennylane`, `numpy`, `tqdm` |
| 2 | Writes `metrics.py` |
| 3 | Writes `dataset.py` |
| 4 | Writes `model.py` |
| 5 | Runs 2 epochs on 64 synthetic samples, evaluates on 16, prints all 5 metrics |

### Expected output

```
Building DummyDataset (64 train / 16 test)...
Instantiating DeepDTAF (this compiles the quantum circuit — may take ~30s)...

Starting smoke test: 2 epochs, batch_size=16
NOTE: Each batch runs the quantum circuit — expect ~1-3 min per epoch on Colab CPU.

Epoch 1/2  train_MSE=X.XXXX
Epoch 2/2  train_MSE=X.XXXX

Running test evaluation...

========== Smoke Test Results ==========
  loss      : X.XXXX
  c_index   : X.XXXX
  RMSE      : X.XXXX
  MAE       : X.XXXX
  SD        : X.XXXX
  CORR      : X.XXXX

Smoke test PASSED — model runs end-to-end.
```

Metric values will not be meaningful (synthetic data), but a clean run confirms the full architecture works.

### Timing (Colab CPU)

| Phase | Time |
|-------|------|
| Package install | ~1 min |
| Circuit compile (model instantiation) | ~30 sec |
| Per epoch (4 batches, VQC per sample) | ~2–5 min |
| **Total** | **~10–15 min** |

---

## Local Smoke Test (if packages are available)

```bash
python train.py --dummy --epochs 2 --runs 1
```

### Requirements

```bash
pip install torch pennylane numpy tqdm
```

> **Corporate network blocked?** If `pip install` returns HTTP 403, use the Colab notebook above or ask IT for the internal pip mirror URL.

---

## Full Training on PDBbind v2016

### 1. Download the dataset

Register and download from [pdbbind.org.cn](http://www.pdbbind.org.cn). You need:
- `v2016-core/` — test set (290 complexes)
- `v2016-other-PL/` — training set (~3,700 complexes)
- `index/INDEX_core_data.2016` and `index/INDEX_general_PL_data.2016`

### 2. Preprocess

```bash
pip install biopython openbabel-wheel
python preprocess.py --pdbbind_dir /path/to/pdbbind2016 --output_dir data/processed
```

Outputs one `.npy` file per complex per modality: `<pdb_id>_{seq,pkt,smi,aff}.npy`  
Also writes `data/splits/train.txt` and `data/splits/test.txt`.

### 3. Train

```bash
python train.py \
  --data_dir data/processed \
  --train_split data/splits/train.txt \
  --test_split data/splits/test.txt \
  --epochs 20 \
  --batch_size 16 \
  --runs 5
```

Training repeats 5 times; the run with lowest training MSE is saved to `checkpoints/best_model.pt`.

### Paper best results (HQDeepDTAF-NN-Angle, 9 qubits)

| MAE | RMSE | R | SD | CI |
|-----|------|---|----|----|
| 1.082 | 1.368 | 0.783 | 1.355 | 0.792 |

---

## Model Architecture

```
Protein seq (1000 × 40)  → Linear(40,128) → DilatedResBlockA → AdaptiveMaxPool → 128-dim
Pocket seq  (63 × 40)    → Linear(40,128) → Conv1d stack     → AdaptiveMaxPool → 128-dim
SMILES      (150,)       → Embedding(64,128) → DilatedResBlockB → AdaptiveMaxPool → 128-dim

Concat → 384-dim → Dropout(0.2) → Linear(384, n_qubits)
→ ReUploadingVQC (n_qubits=10, qnn_layers=20) → Linear(n_qubits, 1) → PReLU → affinity
```

**Quantum circuit:** `default.qubit` (PennyLane CPU simulator), `diff_method='best'` (parameter-shift rule), wrapped as `qml.qnn.TorchLayer`.

**Quantum parameters:** 830 (entangling: 21×1×10×3=630, embedding: 20×10=200)

---

## Evaluation Metrics

| Metric | Direction | Description |
|--------|-----------|-------------|
| MAE | ↓ lower is better | Mean absolute error |
| RMSE | ↓ lower is better | Root mean squared error |
| R (CORR) | ↑ higher is better | Pearson correlation |
| SD | ↓ lower is better | Std dev of residuals after linear fit |
| CI | ↑ higher is better | Concordance index |
