"""
Training script for HQDeepDTAF.

Quick smoke test (no real data needed):
    python train.py --dummy

Full training on PDBbind v2016:
    python train.py --data_dir data/processed --train_split data/splits/train.txt --test_split data/splits/test.txt

Best results from the paper (HQDeepDTAF-NN-Angle, 9 qubits):
    MAE=1.082, RMSE=1.368, R=0.783, SD=1.355, CI=0.792
"""

import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model import DeepDTAF, test
from dataset import PDBbindDataset, DummyDataset


def parse_args():
    p = argparse.ArgumentParser(description='Train HQDeepDTAF')
    p.add_argument('--dummy', action='store_true', help='Use random dummy data for smoke testing')
    p.add_argument('--data_dir', default='data/processed', help='Directory with preprocessed .npy files')
    p.add_argument('--train_split', default='data/splits/train.txt', help='Train PDB ID list')
    p.add_argument('--test_split', default='data/splits/test.txt', help='Test PDB ID list')
    p.add_argument('--epochs', type=int, default=20)
    p.add_argument('--batch_size', type=int, default=16)
    p.add_argument('--lr', type=float, default=0.005)
    p.add_argument('--weight_decay', type=float, default=0.01)
    p.add_argument('--runs', type=int, default=5, help='Number of independent runs (paper uses 5, picks lowest train MSE)')
    p.add_argument('--save_dir', default='checkpoints', help='Directory to save best model')
    p.add_argument('--device', default='cpu', help='Device: cpu or cuda')
    return p.parse_args()


def run_once(args, train_loader, test_loader, device, run_id=0):
    model = DeepDTAF().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = nn.MSELoss()

    best_train_loss = float('inf')
    best_eval = None

    print(f"\n--- Run {run_id + 1} ---")
    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        n_samples = 0

        for seq, pkt, smi, label in train_loader:
            seq = seq.to(device)
            pkt = pkt.to(device)
            smi = smi.to(device)
            label = label.to(device)

            pred = model(seq, pkt, smi)
            loss = loss_fn(pred.view(-1), label.view(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * len(label)
            n_samples += len(label)

        train_loss = epoch_loss / n_samples
        print(f"  Epoch {epoch:2d}/{args.epochs}  train_MSE={train_loss:.4f}", end='')

        if train_loss < best_train_loss:
            best_train_loss = train_loss
            # Evaluate on test set when we have a new best training loss
            eval_result = test(model, test_loader, loss_fn, device, show=False)
            best_eval = eval_result
            print(f"  ← new best | test MAE={eval_result['MAE']:.3f}  RMSE={eval_result['RMSE']:.3f}  R={eval_result['CORR']:.3f}", end='')

        print()

    return best_train_loss, best_eval, model


def main():
    args = parse_args()
    device = torch.device(args.device)

    # -----------------------------------------------------------------------
    # Data
    # -----------------------------------------------------------------------
    if args.dummy:
        print("Using DummyDataset (smoke test — no real data)")
        train_ds = DummyDataset(n=64)
        test_ds = DummyDataset(n=16)
    else:
        train_ds = PDBbindDataset(args.data_dir, args.train_split)
        test_ds = PDBbindDataset(args.data_dir, args.test_split)
        print(f"Train: {len(train_ds)} complexes | Test: {len(test_ds)} complexes")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0)

    os.makedirs(args.save_dir, exist_ok=True)

    # -----------------------------------------------------------------------
    # Multiple runs — paper selects run with lowest train MSE
    # -----------------------------------------------------------------------
    all_results = []
    best_run_loss = float('inf')
    best_run_model = None

    for run_id in range(args.runs):
        train_loss, eval_result, model = run_once(args, train_loader, test_loader, device, run_id)
        all_results.append(eval_result)

        if train_loss < best_run_loss:
            best_run_loss = train_loss
            best_run_model = model
            torch.save(model.state_dict(), os.path.join(args.save_dir, 'best_model.pt'))
            print(f"  → Saved best model (run {run_id + 1})")

    # -----------------------------------------------------------------------
    # Report averaged metrics across all runs
    # -----------------------------------------------------------------------
    print("\n========== Results (averaged over all runs) ==========")
    for key in ('MAE', 'RMSE', 'CORR', 'SD', 'c_index'):
        vals = [r[key] for r in all_results if r is not None]
        if vals:
            print(f"  {key:8s}: {sum(vals)/len(vals):.4f}")

    print("\n========== Best run test metrics ==========")
    best_eval = all_results[0]
    best_loss = float('inf')
    for r in all_results:
        if r is not None and r['loss'] < best_loss:
            best_loss = r['loss']
            best_eval = r
    for key in ('MAE', 'RMSE', 'CORR', 'SD', 'c_index'):
        if best_eval:
            print(f"  {key:8s}: {best_eval[key]:.4f}")

    print(f"\nBest model saved to {args.save_dir}/best_model.pt")


if __name__ == '__main__':
    main()
