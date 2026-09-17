import numpy as np


def c_index(y_true, y_pred):
    """Concordance index: fraction of concordant pairs among all ordered pairs."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    concordant = 0.0
    total = 0
    for i in range(len(y_true)):
        for j in range(i + 1, len(y_true)):
            if y_true[i] == y_true[j]:
                continue
            total += 1
            if y_true[i] < y_true[j]:
                if y_pred[i] < y_pred[j]:
                    concordant += 1
                elif y_pred[i] == y_pred[j]:
                    concordant += 0.5
            else:
                if y_pred[i] > y_pred[j]:
                    concordant += 1
                elif y_pred[i] == y_pred[j]:
                    concordant += 0.5
    return concordant / total if total > 0 else 0.0


def RMSE(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def MAE(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def SD(y_true, y_pred):
    """Regression SD: std dev of residuals after fitting y_true = a*y_pred + b."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    A = np.vstack([y_pred, np.ones(len(y_pred))]).T
    a, b = np.linalg.lstsq(A, y_true, rcond=None)[0]
    residuals = y_true - (a * y_pred + b)
    return float(np.sqrt(np.sum(residuals ** 2) / (len(y_true) - 1)))


def CORR(y_true, y_pred):
    """Pearson correlation coefficient."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.corrcoef(y_true, y_pred)[0, 1])
