from pathlib import Path
from typing import Dict, Iterable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)


# ============================================================
# CONFIGURATION
# ============================================================

CLASS_NAMES = {
    0: "No Failure",
    1: "TWF",
    2: "HDF",
    3: "PWF",
    4: "OSF",
}

CLASS_LABELS = list(CLASS_NAMES.keys())


# ============================================================
# CORE METRICS
# ============================================================

def calculate_metrics(
    y_true,
    y_pred,
    labels: Optional[Iterable[int]] = None,
) -> Dict[str, float]:
    """
    Calculate the main classification metrics.

    Macro-F1 is the primary metric for model comparison.
    """

    if labels is None:
        labels = sorted(np.unique(np.concatenate([
            np.asarray(y_true),
            np.asarray(y_pred),
        ])))

    metrics = {
        "macro_f1": f1_score(
            y_true,
            y_pred,
            labels=labels,
            average="macro",
            zero_division=0,
        ),
        "weighted_f1": f1_score(
            y_true,
            y_pred,
            labels=labels,
            average="weighted",
            zero_division=0,
        ),
        "accuracy": accuracy_score(
            y_true,
            y_pred,
        ),
    }

    return metrics


# ============================================================
# PER-CLASS F1
# ============================================================

def calculate_per_class_f1(
    y_true,
    y_pred,
    labels: Iterable[int] = CLASS_LABELS,
) -> Dict[int, float]:
    """
    Calculate F1-score separately for every class.
    """

    labels = list(labels)

    scores = f1_score(
        y_true,
        y_pred,
        labels=labels,
        average=None,
        zero_division=0,
    )

    return {
        int(label): float(score)
        for label, score in zip(labels, scores)
    }


# ============================================================
# COMPLETE MODEL EVALUATION
# ============================================================

def evaluate_model(
    model,
    X,
    y,
    model_name: str = "Model",
) -> Dict:

    predictions = model.predict(X)

    aggregate_metrics = calculate_metrics(
        y,
        predictions,
        labels=CLASS_LABELS,
    )

    per_class_f1 = calculate_per_class_f1(
        y,
        predictions,
        labels=CLASS_LABELS,
    )

    report = classification_report(
        y,
        predictions,
        labels=CLASS_LABELS,
        target_names=[
            CLASS_NAMES[i] for i in CLASS_LABELS
        ],
        output_dict=True,
        zero_division=0,
    )

    return {
        "model_name": model_name,
        "predictions": predictions,
        "macro_f1": aggregate_metrics["macro_f1"],
        "weighted_f1": aggregate_metrics["weighted_f1"],
        "accuracy": aggregate_metrics["accuracy"],
        "per_class_f1": per_class_f1,
        "classification_report": report,
    }


# ============================================================
# PRINT EVALUATION
# ============================================================

def print_evaluation(results: Dict) -> None:
    """
    Print a readable evaluation summary.
    """

    print("=" * 60)
    print(f"MODEL: {results['model_name']}")
    print("=" * 60)

    print(
        f"Macro F1    : {results['macro_f1']:.4f}"
    )
    print(
        f"Weighted F1 : {results['weighted_f1']:.4f}"
    )
    print(
        f"Accuracy    : {results['accuracy']:.4f}"
    )

    print("\nPer-class F1")
    print("-" * 40)

    for class_id, score in results["per_class_f1"].items():
        print(
            f"{class_id} - "
            f"{CLASS_NAMES[class_id]:<12} : "
            f"{score:.4f}"
        )


# ============================================================
# MODEL COMPARISON
# ============================================================

def compare_models(
    results_list: list[Dict],
) -> pd.DataFrame:
    """
    Create a model comparison table.

    Models are sorted by macro-F1 because this is the
    primary selection metric for the assignment.
    """

    rows = []

    for result in results_list:
        row = {
            "Model": result["model_name"],
            "Macro F1": result["macro_f1"],
            "Weighted F1": result["weighted_f1"],
            "Accuracy": result["accuracy"],
        }

        for class_id, score in result["per_class_f1"].items():
            row[f"F1 - {CLASS_NAMES[class_id]}"] = score

        rows.append(row)

    comparison = (
        pd.DataFrame(rows)
        .sort_values(
            by="Macro F1",
            ascending=False,
        )
        .reset_index(drop=True)
    )

    return comparison


# ============================================================
# CONFUSION MATRIX
# ============================================================

def plot_confusion_matrix(
    y_true,
    y_pred,
    model_name: str = "Model",
    output_path: Optional[str | Path] = None,
) -> None:
    """
    Plot and optionally save the confusion matrix.
    """

    cm = confusion_matrix(
        y_true,
        y_pred,
        labels=CLASS_LABELS,
    )

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    image = ax.imshow(cm)

    ax.set_title(
        f"Confusion Matrix - {model_name}"
    )

    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")

    tick_labels = [
        CLASS_NAMES[i]
        for i in CLASS_LABELS
    ]

    ax.set_xticks(
        range(len(CLASS_LABELS)),
        tick_labels,
        rotation=45,
        ha="right",
    )

    ax.set_yticks(
        range(len(CLASS_LABELS)),
        tick_labels,
    )

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center",
            )

    fig.colorbar(image, ax=ax)
    fig.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )
        fig.savefig(
            output_path,
            dpi=180,
            bbox_inches="tight",
        )
        print(f"Saved: {output_path}")

    plt.show()
    plt.close(fig)


# ============================================================
# SAVE METRICS
# ============================================================

def save_comparison(
    comparison: pd.DataFrame,
    output_path: str | Path = "reports/model_comparison.csv",
) -> None:
    """
    Save model comparison results.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    comparison.to_csv(
        output_path,
        index=False,
    )

    print(f"Saved: {output_path}")


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":

    print("evaluate.py loaded successfully.")
    print("\nClasses:")

    for class_id, class_name in CLASS_NAMES.items():
        print(f"{class_id}: {class_name}")

    print(
        "\nPrimary selection metric: Macro F1"
    )