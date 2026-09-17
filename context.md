# HQDeepDTAF — Local Run Context

**Paper:** Jeong et al., *Hybrid quantum neural networks for efficient protein-ligand binding affinity prediction*, EPJ Quantum Technology (2025) 12:120  
**Code:** https://github.com/drmoon-1st/HQDeepDTAF

---

## 1. Overview

HQDeepDTAF (Hybrid Quantum DeepDTAF) predicts protein-ligand binding affinity by replacing the classical dense layers of DeepDTAF with a Hybrid Quantum Neural Network (HQNN). The model takes three input modalities — protein sequence, binding pocket, and ligand SMILES — encodes each through dilated convolutional modules, then fuses them and passes the result through a variational quantum circuit (VQC) for final affinity prediction.

---

## 2. Repository Structure

```
HQDeepDTAF/
├── model.py            # Main HQDeepDTAF model + GetVQC factory + test() loop
├── classical_models    # Classical baselines: DeepDTA, TopologyNet, AtomicNN
├── metrics.py          # IMPLEMENTED — c_index, RMSE, MAE, SD, CORR
├── imgs/
│   ├── mainfig.png     # Architecture diagram
│   ├── result1.png
│   └── result2.png
│
# Missing from public release (must be implemented):
├── dataset.py          # Provides PT_FEATURE_SIZE, data loaders
└── train.py            # Training loop (not released; test() IS in model.py)
```

> **Status:** `metrics.py` is now implemented locally. `dataset.py` still needs to be implemented before running. `GetVQC` is fully defined inside `model.py` — no missing file.

---

## 3. Dependencies

No `requirements.txt` is provided. Install manually:

```bash
pip install torch numpy tqdm pennylane
```

| Package    | Role                                      |
|------------|-------------------------------------------|
| `torch`    | Classical neural network layers, training |
| `pennylane`| Quantum circuit simulation (QNode, VQC)   |
| `numpy`    | Numerical operations                      |
| `tqdm`     | Progress bars                             |

**Quantum backend used in the paper:** `default.qubit` (PennyLane CPU simulator). No real quantum hardware was used — all results come from classical simulation.

---

## 4. Dataset

**Source:** PDBbind database v2016 — Core Set  
**Download:** http://www.pdbbind.org.cn (registration required)

### Provided files per complex:
- `*.pdb` — protein structure file (extract protein + pocket sequences)
- `*_pocket.pdb` — binding pocket PDB file
- `*_ligand.sdf` — ligand structure file (convert to SMILES via Open Babel)

### Convert SDF → SMILES:
```bash
pip install openbabel-wheel   # or install Open Babel system-wide
obabel input.sdf -O output.smi --gen2D
```

### Fixed sequence lengths (pad/truncate to these):
| Input         | Fixed length |
|---------------|-------------|
| Protein seq   | 1000        |
| SMILES string | 150         |
| Pocket seq    | 63          |

Sequences shorter than the limit are zero-padded; longer ones are truncated.

### SMILES character vocabulary (64 chars):
Characters are mapped to integers, e.g. `'C'→42`, `'O'→48`, `'='→40`, `')'→31`, `'('→1`.

### Protein/Pocket feature vector (40-dim per residue):
- 21-dim one-hot: amino acid identity (21 residue types including non-standard)
- 8-dim one-hot: secondary structure elements (SSEs) from SSPro
- 11-dim: physicochemical attributes

### Train/Test split:
- **Training:** General set from PDBbind v2016
- **Test:** Core 2016 set (290 complexes); Smith-Waterman similarity ≤ 60% to any training sequence for 99% of protein pairs

---

## 5. Model Architecture

### 5.1 Top-level config (`model.py`)

```python
qnn_type      = 'ReUploadingVQC'   # or 'NormalVQC'
n_qubits      = 10                 # paper best: 9
qnn_layers    = 20
CHAR_SMI_SET_LEN = 64
```

### 5.2 Module pipeline

```
Protein (1000 × 40)   → Linear(PT_FEATURE_SIZE, 128) → Embedding → DilatedParllelResidualBlockA (rates 1,2,4,8,16) → AdaptiveMaxPool1d(1) → 128-dim
Pocket  (63 × 40)     → Linear(PT_FEATURE_SIZE, 128) → Embedding → Conv1d stack (32→64→128)                        → AdaptiveMaxPool1d(1) → 128-dim
SMILES  (150,)        → nn.Embedding(64, 128)          →            DilatedParllelResidualBlockB (rates 1,2,4,8)    → AdaptiveMaxPool1d(1) → 128-dim

Concat(protein, pocket, SMILES) → 384-dim → Dropout(0.2)
→ Linear(384, n_qubits)         → n_qubits-dim classical embedding h
→ VQC(n_qubits, qnn_layers)     → n_qubits-dim quantum output ψ
→ Linear(n_qubits, 1) → PReLU  → scalar binding affinity
```

### 5.3 Dilated convolution blocks

- **DilatedParllelResidualBlockA** (protein): 5 parallel branches with dilation rates `[1, 2, 4, 8, 16]`, hierarchical fusion with residual connections.
- **DilatedParllelResidualBlockB** (ligand): 4 parallel branches with dilation rates `[1, 2, 4, 8]`, hierarchical fusion with residual connections.
- Pocket: plain `Conv1d` stack with filter counts `32 → 64 → 128`.

### 5.4 Quantum circuit (`ReUploadingVQC`) — actual code

```
BasisStatePreparation(|0...0⟩)
AngleEmbedding(h, wires=range(n_qubits))          # initial embedding
For l = 0..qnn_layers-1:
    StronglyEntanglingLayers(entangling_weights[l])
    AngleEmbedding(h * embedding_weights[l], wires=range(n_qubits))  # re-upload with learned scale
StronglyEntanglingLayers(entangling_weights[-1])   # final entangling layer (qnn_layers+1 total)
Return: [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]
```

**Two trainable weight tensors:**
- `entangling_weights`: shape `(qnn_layers+1, 1, n_qubits, 3)` = (21, 1, 10, 3) with defaults
- `embedding_weights`: shape `(qnn_layers, n_qubits)` = (20, 10) — per-layer input scaling

- **PennyLane device:** `qml.device("default.qubit", wires=n_qubits)`
- **Interface:** `torch`, `diff_method='best'` (uses parameter-shift rule)
- **Wrapping:** `qml.qnn.TorchLayer(qnode, weight_shapes)` — integrates as a `nn.Module`

**Parameter count (QNN, n_qubits=10, qnn_layers=20):**  
`entangling: 21×1×10×3 = 630` + `embedding: 20×10 = 200` = **830 quantum parameters**

### 5.5 Model variants from the paper

| Variant | Embedding | Qubits | Layers | # C params | # Q params | Best MAE |
|---------|-----------|--------|--------|------------|------------|----------|
| HQDeepDTAF-Amplitude | Pure amplitude | 9 | 20 | 96,551 | 540 | 1.298 |
| HQDeepDTAF-NN-Amplitude | NN + amplitude | 9 | 20 | 145,831 | 540 | 1.086 |
| HQDeepDTAF-NN-Angle | NN + angle | 9 | 20 | 100,016 | 540 | **1.082** |

**Best model:** `HQDeepDTAF-NN-Angle` with 9 qubits, 20 layers.

---

## 6. Training Configuration

```python
optimizer   = AdamW
lr          = 0.005          # max learning rate
weight_decay = 0.01
loss        = nn.MSELoss()
epochs      = 20
batch_size  = 16
```

Training is repeated **5 times**; the run with the lowest training MSE is selected for reporting. Metrics are averaged over these 5 runs.

### AdamW update rule:
```
θ_{t+1} = θ_t - η * (m̂_t / (√v̂_t + ε) + λ·θ_t)
```

### Gradient computation (quantum layers):
Parameter-shift rule:  
```
∇_θ ⟨B̂⟩(θ) = ½[⟨B̂⟩(θ + π/2) - ⟨B̂⟩(θ - π/2)]
```

---

## 7. Evaluation Metrics

Implemented in the missing `metrics.py`:

| Metric | Direction | Formula |
|--------|-----------|---------|
| MAE | ↓ lower is better | mean\|y - ŷ\| |
| RMSE | ↓ lower is better | √(mean(y - ŷ)²) |
| R (Pearson) | ↑ higher is better | Pearson correlation of y and ŷ |
| SD | ↓ lower is better | √(1/(N-1) Σ[yᵢ - (apᵢ + b)]²) |
| CI | ↑ higher is better | Concordance index |

**Best reported results (HQDeepDTAF-NN-Angle, 9 qubits):**  
MAE=1.082, RMSE=1.368, R=0.783, SD=1.355, CI=0.792

---

## 8. Noise Simulation

To evaluate NISQ robustness, add noise to the PennyLane device:

```python
# Depolarizing noise
dev_noisy = qml.device("default.mixed", wires=n_qubits)
# Then insert noise channels after each gate using qml.DepolarizingChannel

# Amplitude damping noise
# Use qml.AmplitudeDamping
```

Tested noise rates: `[0.001, 0.01, 0.1, 0.2]`

At noise rate ≤ 0.01, hybrid models still outperform classical DeepDTAF.  
At noise rate ≥ 0.1, all HQDeepDTAF variants degrade below DeepDTAF.

---

## 9. Hardware & Feasibility

| Model | Circuit depth (9 qubits, 20 layers) | NISQ feasible? |
|-------|--------------------------------------|----------------|
| HQDeepDTAF-NN-Angle | O(260) | **Yes** (< IBM 300-circuit limit) |
| HQDeepDTAF-NN-Amplitude | O(10,480) | No (exceeds coherence limits) |

**IBM Quantum compatibility:** Only NN-Angle is deployable on current IBM quantum hardware.

**Paper hardware:** AMD Ryzen 9 7950X, 128 GB RAM — classical simulation only (no QPU used).

---

## 10. Classical Baselines (in `classical_models`)

| Model | # C params | MAE |
|-------|------------|-----|
| DeepDTAF | 154,114 | 1.109 |
| DeepDTA | 1,523,396 | 1.208 |
| Pafnucy | 15,859,081 | 1.337 |
| TopologyNet | 33,425,345 | 1.195 |
| AEScore | 155,131 | 1.359 |
| Kdeep | 132,954,465 | 1.182 |

---

## 11. Function Approximation Benchmarks

Used to validate HQNN expressivity before the drug-target task:

| Function | Type | Interval | Train | Test |
|----------|------|----------|-------|------|
| sin(5x)/(5x) | Univariate | x ∈ (0, 3] | 200 | 100 |
| sin(5x₁)/(5x₁) + sin(5x₂)/(5x₂) | Multivariate | x₁,x₂ ∈ (0, 3] | 200 | 100 |

**Key result:** HQNN (1 qubit, 5 layers, 4+15 params) achieves MSE=0.0002, outperforming NN with 3265 parameters.

---

## 12. Missing Components — What to Implement

### `metrics.py` — DONE
Implemented at `metrics.py`. All five functions: `c_index`, `RMSE`, `MAE`, `SD`, `CORR`. No external dependencies beyond numpy.

### `dataset.py` (still needed — minimum interface):
```python
PT_FEATURE_SIZE = 40  # per-residue feature dimension (21 AA + 8 SSE + 11 physicochem)

class PDBbindDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir, split='train'):
        # Load protein sequences (max 1000 residues), encode as float32 (N, 1000, 40)
        # Load pocket sequences (max 63 residues), encode as float32 (N, 63, 40)
        # Load SMILES strings (max 150 chars), encode as int64 (N, 150) with 64-char vocab
        # Load binding affinity labels: -log10(Kd/Ki) as float32
        pass

    def __getitem__(self, idx):
        # model.forward signature: forward(seq, pkt, smi)
        # seq: float32 tensor (L=1000, 40) — model applies seq_embed (Linear) then transposes
        # pkt: float32 tensor (L=63, 40)  — reuses same seq_embed linear layer
        # smi: int64 tensor (L=150,)      — model applies nn.Embedding(64, 128)
        return protein_tensor, pocket_tensor, smiles_tensor, affinity_label
```

> **Note:** Both protein (`seq`) and pocket (`pkt`) pass through the same `self.seq_embed = nn.Linear(PT_FEATURE_SIZE, 128)` linear layer — they must both be float tensors of shape `(L, 40)`, not integer indices.

---

## 13. Quick Start (once dataset.py and metrics.py are in place)

```python
import torch
from model import DeepDTAF

model = DeepDTAF()
model.train()

optimizer = torch.optim.AdamW(model.parameters(), lr=0.005, weight_decay=0.01)
loss_fn = torch.nn.MSELoss()

# Training loop
for epoch in range(20):
    for protein, pocket, smiles, label in train_loader:
        pred = model(protein, pocket, smiles)
        loss = loss_fn(pred.squeeze(), label)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
```

---

## 14. Key Paper Findings

- HQNN with classical embedding (NN-Angle) outperforms pure QNN and classical NN at comparable or fewer parameters.
- The classical embedding network is essential — it adapts input representation to task-specific latent space before quantum encoding, enabling the UAT to hold empirically.
- Trainable quantum parameters beat frozen ones: NN-Angle (Training) MAE=1.082 vs. NN-Angle (Freezing) MAE=1.191.
- Convergence speed of hybrid models is comparable to classical models (not faster in epochs), but per-update cost is lower due to fewer parameters.
- Circuit depth scales as O((4+n)L) for angle embedding vs. O(poly(2ⁿ)L) for amplitude — angle embedding is the NISQ-feasible choice.
