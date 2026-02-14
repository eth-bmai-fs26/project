"""Evaluation metrics for the Tool Router classifier."""

import numpy as np


def confusion_matrix(y_true, y_pred, num_classes=3) -> np.ndarray:
    """Confusion matrix — rows = true class, columns = predicted class."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm
