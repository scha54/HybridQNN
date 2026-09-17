import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import metrics

from dataset import PT_FEATURE_SIZE

import torch.nn.functional as F
import pennylane as qml


def GetVQC(n_qubits, qnn_layers, qnn_type):
    if qnn_type == 'ReUploadingVQC':
        def ReUploadingVQC(inputs, entangling_weights, embedding_weights):
            """VQC with data re-uploading: learned per-layer input scaling."""
            # default.qubit starts in |0...0> so no explicit BasisState needed
            # (qml.BasisStatePreparation was removed in PennyLane 0.40)
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            for i in range(qnn_layers):
                qml.StronglyEntanglingLayers(entangling_weights[i], wires=range(n_qubits))
                features = inputs * embedding_weights[i]
                qml.AngleEmbedding(features=features, wires=range(n_qubits))
            qml.StronglyEntanglingLayers(entangling_weights[-1], wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

        entangling_weights_shape = (qnn_layers + 1,) + qml.StronglyEntanglingLayers.shape(n_layers=1, n_wires=n_qubits)
        embedding_weights_shape = (qnn_layers, n_qubits)
        weight_shapes = {
            'entangling_weights': entangling_weights_shape,
            'embedding_weights': embedding_weights_shape,
        }
        return ReUploadingVQC, weight_shapes

    elif qnn_type == 'NormalVQC':
        def NormalVQC(inputs, entangling_weights):
            """Standard VQC without data re-uploading."""
            qml.AngleEmbedding(features=inputs, wires=range(n_qubits))
            qml.StronglyEntanglingLayers(entangling_weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(wires=i)) for i in range(n_qubits)]

        entangling_weights_shape = qml.StronglyEntanglingLayers.shape(n_layers=qnn_layers, n_wires=n_qubits)
        weight_shapes = {'entangling_weights': entangling_weights_shape}
        return NormalVQC, weight_shapes


qnn_type = 'ReUploadingVQC'  # or 'NormalVQC'
n_qubits = 10                 # paper best: 9; code default: 10
qnn_layers = 20
CHAR_SMI_SET_LEN = 64

counter = 0


class Squeeze(nn.Module):
    def forward(self, input: torch.Tensor):
        return input.squeeze()


class CDilated(nn.Module):
    def __init__(self, nIn, nOut, kSize, stride=1, d=1):
        super().__init__()
        padding = int((kSize - 1) / 2) * d
        self.conv = nn.Conv1d(nIn, nOut, kSize, stride=stride, padding=padding, bias=False, dilation=d)

    def forward(self, input):
        global counter
        output = self.conv(input)
        counter += 1
        return output


class DilatedParllelResidualBlockA(nn.Module):
    """Dilated parallel residual block for protein sequences (5 branches, rates 1-16)."""
    def __init__(self, nIn, nOut, add=True):
        super().__init__()
        n = int(nOut / 5)
        n1 = nOut - 4 * n
        self.c1 = nn.Conv1d(nIn, n, 1, padding=0)
        self.br1 = nn.Sequential(nn.BatchNorm1d(n), nn.PReLU())
        self.d1 = CDilated(n, n1, 3, 1, 1)
        self.d2 = CDilated(n, n, 3, 1, 2)
        self.d4 = CDilated(n, n, 3, 1, 4)
        self.d8 = CDilated(n, n, 3, 1, 8)
        self.d16 = CDilated(n, n, 3, 1, 16)
        self.br2 = nn.Sequential(nn.BatchNorm1d(nOut), nn.PReLU())
        if nIn != nOut:
            add = False
        self.add = add

    def forward(self, input):
        output1 = self.br1(self.c1(input))
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        d16 = self.d16(output1)
        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        add4 = add3 + d16
        combine = torch.cat([d1, add1, add2, add3, add4], 1)
        if self.add:
            combine = input + combine
        return self.br2(combine)


class DilatedParllelResidualBlockB(nn.Module):
    """Dilated parallel residual block for ligand SMILES (4 branches, rates 1-8)."""
    def __init__(self, nIn, nOut, add=True):
        super().__init__()
        n = int(nOut / 4)
        n1 = nOut - 3 * n
        self.c1 = nn.Conv1d(nIn, n, 1, padding=0)
        self.br1 = nn.Sequential(nn.BatchNorm1d(n), nn.PReLU())
        self.d1 = CDilated(n, n1, 3, 1, 1)
        self.d2 = CDilated(n, n, 3, 1, 2)
        self.d4 = CDilated(n, n, 3, 1, 4)
        self.d8 = CDilated(n, n, 3, 1, 8)
        self.br2 = nn.Sequential(nn.BatchNorm1d(nOut), nn.PReLU())
        if nIn != nOut:
            add = False
        self.add = add

    def forward(self, input):
        output1 = self.br1(self.c1(input))
        d1 = self.d1(output1)
        d2 = self.d2(output1)
        d4 = self.d4(output1)
        d8 = self.d8(output1)
        add1 = d2
        add2 = add1 + d4
        add3 = add2 + d8
        combine = torch.cat([d1, add1, add2, add3], 1)
        if self.add:
            combine = input + combine
        return self.br2(combine)


class DeepDTAF(nn.Module):
    """HQDeepDTAF: classical dilated-CNN encoders + variational quantum circuit."""

    def __init__(self):
        super().__init__()

        smi_embed_size = 128
        seq_embed_size = 128
        seq_oc = 128
        pkt_oc = 128
        smi_oc = 128

        self.smi_embed = nn.Embedding(CHAR_SMI_SET_LEN, smi_embed_size)
        self.seq_embed = nn.Linear(PT_FEATURE_SIZE, seq_embed_size)

        dev = qml.device("default.qubit", wires=n_qubits)
        VQC, weight_shapes = GetVQC(n_qubits, qnn_layers, qnn_type)
        qnode = qml.QNode(VQC, dev, interface='torch', diff_method='best')
        self.qlayerC = qml.qnn.TorchLayer(qnode, weight_shapes)

        conv_seq = []
        ic = seq_embed_size
        for oc in [32, 64, 64, seq_oc]:
            conv_seq.append(DilatedParllelResidualBlockA(ic, oc))
            ic = oc
        conv_seq.append(nn.AdaptiveMaxPool1d(1))
        conv_seq.append(Squeeze())
        self.conv_seq = nn.Sequential(*conv_seq)

        conv_pkt = []
        ic = seq_embed_size
        for oc in [32, 64, pkt_oc]:
            conv_pkt.append(nn.Conv1d(ic, oc, 3))
            conv_pkt.append(nn.BatchNorm1d(oc))
            conv_pkt.append(nn.PReLU())
            ic = oc
        conv_pkt.append(nn.AdaptiveMaxPool1d(1))
        conv_pkt.append(Squeeze())
        self.conv_pkt = nn.Sequential(*conv_pkt)

        conv_smi = []
        ic = smi_embed_size
        for oc in [32, 64, smi_oc]:
            conv_smi.append(DilatedParllelResidualBlockB(ic, oc))
            ic = oc
        conv_smi.append(nn.AdaptiveMaxPool1d(1))
        conv_smi.append(Squeeze())
        self.conv_smi = nn.Sequential(*conv_smi)

        self.cat_dropout = nn.Dropout(0.2)
        self.clf = nn.Sequential(nn.Linear(n_qubits, 1), nn.PReLU())
        self.clangle = nn.Linear(384, n_qubits)

    def forward(self, seq, pkt, smi):
        seq_embed = torch.transpose(self.seq_embed(seq), 1, 2)   # (N, 128, 1000)
        seq_conv = self.conv_seq(seq_embed)                        # (N, 128)

        pkt_embed = torch.transpose(self.seq_embed(pkt), 1, 2)   # (N, 128, 63)
        pkt_conv = self.conv_pkt(pkt_embed)                        # (N, 128)

        smi_embed = torch.transpose(self.smi_embed(smi), 1, 2)   # (N, 128, 150)
        smi_conv = self.conv_smi(smi_embed)                        # (N, 128)

        cat = self.cat_dropout(torch.cat([seq_conv, pkt_conv, smi_conv], dim=1))  # (N, 384)
        output_ = self.clangle(cat)                                # (N, n_qubits)
        qout = self.qlayerC(output_)                               # (N, n_qubits)
        return self.clf(qout)                                      # (N, 1)


def test(model: nn.Module, test_loader, loss_function, device, show=True):
    model.eval()
    test_loss = 0
    outputs = []
    targets = []
    with torch.no_grad():
        for idx, (*x, y) in tqdm(enumerate(test_loader), disable=not show, total=len(test_loader)):
            for i in range(len(x)):
                x[i] = x[i].to(device)
            y = y.to(device)
            y_hat = model(*x)
            test_loss += loss_function(y_hat.view(-1), y.view(-1)).item()
            outputs.append(y_hat.cpu().numpy().reshape(-1))
            targets.append(y.cpu().numpy().reshape(-1))

    targets = np.concatenate(targets).reshape(-1)
    outputs = np.concatenate(outputs).reshape(-1)
    test_loss /= len(test_loader.dataset)

    return {
        'loss': test_loss,
        'c_index': metrics.c_index(targets, outputs),
        'RMSE': metrics.RMSE(targets, outputs),
        'MAE': metrics.MAE(targets, outputs),
        'SD': metrics.SD(targets, outputs),
        'CORR': metrics.CORR(targets, outputs),
    }
