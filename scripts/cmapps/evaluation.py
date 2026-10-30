"""Shared metrics and the sequence experiments' positive-quantile alert rule."""
import math
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

def score(actual, pred, threshold=30):
    actual, pred = np.asarray(actual), np.asarray(pred)
    error = pred - actual
    terms = np.expm1(np.where(error < 0, -error / 13, error / 10))
    assert np.isfinite(terms).all()
    truth, flag = actual <= 30, pred <= threshold
    tp, fp = int((truth & flag).sum()), int((~truth & flag).sum())
    fn, tn = int((truth & ~flag).sum()), int((~truth & ~flag).sum())
    return {'n': len(actual), 'RMSE': float(np.sqrt(mean_squared_error(actual, pred))),
            'MAE': float(mean_absolute_error(actual, pred)), 'bias': float(error.mean()),
            'NASA_score_sum': float(terms.sum()), 'NASA_score_mean': float(terms.mean()),
            'threshold': float(threshold), 'precision': tp / max(tp + fp, 1),
            'recall': tp / max(tp + fn, 1), 'TP': tp, 'FP': fp, 'FN': fn, 'TN': tn}

def threshold(actual,pred):
    """Smallest predicted-RUL cutoff catching at least 90% of RUL <=30 events."""
    positives=np.asarray(pred)[np.asarray(actual)<=30]
    if not len(positives):raise ValueError('No calibration events')
    return float(np.sort(positives)[math.ceil(.9*len(positives))-1])
