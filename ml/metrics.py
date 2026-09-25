"""Classification metrics implemented from the confusion matrix."""
from __future__ import annotations

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> np.ndarray:
    """cm[i, j] = number of samples whose true class is i and predicted class is j."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def classification_report(y_true: np.ndarray, y_pred: np.ndarray, class_names: list[str]) -> dict:
    """Accuracy plus per-class / macro / weighted precision, recall and F1.

        precision_k = TP_k / (TP_k + FP_k)    of everything predicted k, how much was k
        recall_k    = TP_k / (TP_k + FN_k)    of everything truly k, how much was found
        F1_k        = 2 * P_k * R_k / (P_k + R_k)
    """
    k = len(class_names)
    cm = confusion_matrix(y_true, y_pred, k)
    tp = np.diag(cm).astype(float)
    predicted = cm.sum(axis=0).astype(float)
    actual = cm.sum(axis=1).astype(float)
    precision = np.divide(tp, predicted, out=np.zeros(k), where=predicted > 0)
    recall = np.divide(tp, actual, out=np.zeros(k), where=actual > 0)
    denom = precision + recall
    f1 = np.divide(2 * precision * recall, denom, out=np.zeros(k), where=denom > 0)
    support = actual
    weights = support / support.sum()

    per_class = [
        {
            "class": name,
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, name in enumerate(class_names)
    ]
    return {
        "accuracy": float(tp.sum() / cm.sum()),
        "precision_macro": float(precision.mean()),
        "recall_macro": float(recall.mean()),
        "f1_macro": float(f1.mean()),
        "precision_weighted": float((precision * weights).sum()),
        "recall_weighted": float((recall * weights).sum()),
        "f1_weighted": float((f1 * weights).sum()),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }
