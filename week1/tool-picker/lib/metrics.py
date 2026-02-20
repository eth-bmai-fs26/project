"""Evaluation metrics for the Tool Router classifier."""

import numpy as np
import matplotlib.pyplot as plt 
import pandas as pd

def confusion_matrix(y_true, y_pred, num_classes=3) -> np.ndarray:
    """Confusion matrix — rows = true class, columns = predicted class."""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def plot_confusion_matrix(cm: np.ndarray, label_names, show_dataframe=True):
    """
    Plot a confusion matrix with labels.

    Parameters
    ----------
    cm : np.ndarray
        Confusion matrix (square matrix).
    label_names : list
        List of class label names.
    show_dataframe : bool
        If True, display the confusion matrix as a DataFrame.
    """

    num_classes = cm.shape[0]

    # Optional DataFrame display
    if show_dataframe:
        cm_df = pd.DataFrame(
            cm,
            index=label_names,
            columns=label_names,
        )
        print(cm_df)

    # Plot
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm, cmap="Blues")

    ax.set_xticks(range(num_classes))
    ax.set_yticks(range(num_classes))
    ax.set_xticklabels(label_names, rotation=35, ha="right")
    ax.set_yticklabels(label_names)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(
                j, i, str(cm[i, j]),
                ha="center", va="center",
                color="white" if cm[i, j] > cm.max() / 2 else "black",
                fontsize=12, fontweight="bold"
            )

    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    plt.show()