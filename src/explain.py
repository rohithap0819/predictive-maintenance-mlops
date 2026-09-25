from pathlib import Path
from typing import Dict, List, Tuple

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

from src.data_validation import load_and_validate_data
from src.preprocessing import (
    FEATURES,
    prepare_training_data,
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

MODEL_PATH = Path("models/best_model.pkl")

OUTPUT_DIR = Path("reports")
OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

SHAP_OUTPUT_PATH = (
    OUTPUT_DIR / "shap_per_class.png"
)

CLASS_NAMES = {
    0: "No Failure",
    1: "TWF",
    2: "HDF",
    3: "PWF",
    4: "OSF",
}

FAILURE_CLASSES = [1, 2, 3, 4]

TOP_N_FEATURES = 8

# Number of validation observations to explain.
# Keeping this moderate makes SHAP faster.
MAX_EXPLAIN_SAMPLES = 500


# ============================================================
# LOAD MODEL
# ============================================================

def load_best_model(
    model_path: str | Path = MODEL_PATH,
):
    """
    Load the final tuned model saved by retrain.py.
    """

    model_path = Path(model_path)

    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found: {model_path}"
        )

    model = joblib.load(
        model_path
    )

    print(
        f"Loaded model: {model_path}"
    )

    return model


# ============================================================
# PREPARE EXPLANATION DATA
# ============================================================

def prepare_explanation_data(
    validation_features: pd.DataFrame,
) -> pd.DataFrame:
    """
    Prepare validation features in exactly the same column
    order used by the trained model.
    """

    missing_features = [
        feature
        for feature in FEATURES
        if feature not in validation_features.columns
    ]

    if missing_features:
        raise ValueError(
            "Missing model features: "
            f"{missing_features}"
        )

    X_explain = validation_features[
        FEATURES
    ].copy()

    if len(X_explain) > MAX_EXPLAIN_SAMPLES:
        X_explain = X_explain.sample(
            n=MAX_EXPLAIN_SAMPLES,
            random_state=RANDOM_STATE,
        )

    return X_explain


# ============================================================
# SHAP VALUE EXTRACTION
# ============================================================

def calculate_shap_values(
    model,
    X_explain: pd.DataFrame,
):
    """
    Calculate multiclass SHAP values.

    Handles the common TreeExplainer output formats:
    - list of arrays
    - 3D numpy array
    """

    explainer = shap.TreeExplainer(
        model
    )

    shap_values = explainer.shap_values(
        X_explain
    )

    if isinstance(
        shap_values,
        list,
    ):
        return shap_values

    array = np.asarray(
        shap_values
    )

    if array.ndim != 3:
        raise ValueError(
            "Unexpected SHAP output shape: "
            f"{array.shape}"
        )

    n_samples = len(
        X_explain
    )

    n_features = len(
        X_explain.columns
    )

    # Format:
    # samples × features × classes
    if (
        array.shape[0] == n_samples
        and array.shape[1] == n_features
    ):
        return [
            array[:, :, class_index]
            for class_index in range(
                array.shape[2]
            )
        ]

    # Format:
    # classes × samples × features
    if (
        array.shape[1] == n_samples
        and array.shape[2] == n_features
    ):
        return [
            array[class_index, :, :]
            for class_index in range(
                array.shape[0]
            )
        ]

    raise ValueError(
        "Could not interpret SHAP array shape: "
        f"{array.shape}"
    )


# ============================================================
# TOP FEATURES BY CLASS
# ============================================================

def get_top_features(
    shap_values,
    feature_names: List[str],
    class_id: int,
    top_n: int = TOP_N_FEATURES,
) -> List[Tuple[str, float]]:
    """
    Return the top features ranked by mean absolute SHAP value
    for one class.
    """

    values = np.asarray(
        shap_values[class_id]
    )

    mean_absolute_shap = (
        np.abs(values)
        .mean(axis=0)
    )

    order = np.argsort(
        mean_absolute_shap
    )[::-1][:top_n]

    return [
        (
            feature_names[index],
            float(
                mean_absolute_shap[index]
            ),
        )
        for index in order
    ]


# ============================================================
# CREATE MULTICLASS SHAP PLOT
# ============================================================

def create_shap_plot(
    shap_values,
    X_explain: pd.DataFrame,
    output_path: str | Path = SHAP_OUTPUT_PATH,
) -> Dict[int, List[Tuple[str, float]]]:
    """
    Create a 4-panel SHAP feature-importance plot for
    TWF, HDF, PWF and OSF.
    """

    output_path = Path(
        output_path
    )

    top_features_by_class = {}

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(16, 12),
    )

    axes = axes.ravel()

    for panel_index, class_id in enumerate(
        FAILURE_CLASSES
    ):

        values = np.asarray(
            shap_values[class_id]
        )

        mean_absolute_shap = (
            np.abs(values)
            .mean(axis=0)
        )

        order = np.argsort(
            mean_absolute_shap
        )[-TOP_N_FEATURES:][::-1]

        names = np.asarray(
            X_explain.columns
        )[order]

        scores = mean_absolute_shap[
            order
        ]

        top_features_by_class[
            class_id
        ] = [
            (
                str(name),
                float(score),
            )
            for name, score in zip(
                names,
                scores,
            )
        ]

        axes[panel_index].barh(
            names[::-1],
            scores[::-1],
        )

        axes[panel_index].set_title(
            f"{CLASS_NAMES[class_id]} "
            f"(class {class_id})"
        )

        axes[panel_index].set_xlabel(
            "Mean |SHAP value|"
        )

    fig.suptitle(
        "Multiclass SHAP - "
        "Top Feature Contributions by Failure Class",
        fontsize=16,
    )

    fig.tight_layout(
        rect=[0, 0, 1, 0.96]
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    fig.savefig(
        output_path,
        dpi=180,
        bbox_inches="tight",
    )

    plt.show()
    plt.close(fig)

    print(
        f"Saved SHAP plot: {output_path}"
    )

    return top_features_by_class


# ============================================================
# PRINT ENGINEERING INSIGHTS
# ============================================================

def print_engineering_insights(
    top_features_by_class: Dict[
        int,
        List[Tuple[str, float]]
    ],
) -> None:
    """
    Print top contributing features separately for each
    failure class.
    """

    print("\n" + "=" * 70)
    print("CLASS-SPECIFIC SHAP INSIGHTS")
    print("=" * 70)

    for class_id in FAILURE_CLASSES:

        print(
            f"\n{CLASS_NAMES[class_id]} "
            f"(class {class_id})"
        )

        print("-" * 50)

        for rank, (
            feature,
            score,
        ) in enumerate(
            top_features_by_class[
                class_id
            ][:5],
            start=1,
        ):

            print(
                f"{rank}. "
                f"{feature:<25} "
                f"{score:.6f}"
            )


# ============================================================
# SAVE SHAP SUMMARY
# ============================================================

def save_shap_summary(
    top_features_by_class: Dict[
        int,
        List[Tuple[str, float]]
    ],
    output_path: str | Path = (
        "reports/shap_feature_summary.csv"
    ),
) -> None:
    """
    Save the class-specific SHAP feature rankings.
    """

    rows = []

    for class_id in FAILURE_CLASSES:

        for rank, (
            feature,
            score,
        ) in enumerate(
            top_features_by_class[
                class_id
            ],
            start=1,
        ):

            rows.append(
                {
                    "Class_ID": class_id,
                    "Class_Name": CLASS_NAMES[
                        class_id
                    ],
                    "Rank": rank,
                    "Feature": feature,
                    "Mean_Absolute_SHAP": score,
                }
            )

    summary = pd.DataFrame(
        rows
    )

    output_path = Path(
        output_path
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    summary.to_csv(
        output_path,
        index=False,
    )

    print(
        f"Saved SHAP summary: {output_path}"
    )


# ============================================================
# COMPLETE EXPLAINABILITY PIPELINE
# ============================================================

def explain_model():
    """
    Complete SHAP explainability workflow.
    """

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

    model = load_best_model()

    # --------------------------------------------------------
    # Load validated training data
    # --------------------------------------------------------

    train, current, stress = (
        load_and_validate_data("data")
    )

    # --------------------------------------------------------
    # Recreate validation data using the same preprocessing
    # --------------------------------------------------------

    (
        X_train,
        X_val,
        y_train,
        y_val,
        X_res,
        y_res,
        target_encoder,
        type_encoder,
    ) = prepare_training_data(
        train
    )

    # --------------------------------------------------------
    # Prepare data to explain
    # --------------------------------------------------------

    X_explain = prepare_explanation_data(
        X_val
    )

    print(
        f"Explaining {len(X_explain)} "
        "validation observations."
    )

    # --------------------------------------------------------
    # SHAP
    # --------------------------------------------------------

    shap_values = calculate_shap_values(
        model,
        X_explain,
    )

    # --------------------------------------------------------
    # Plot
    # --------------------------------------------------------

    top_features = create_shap_plot(
        shap_values,
        X_explain,
        SHAP_OUTPUT_PATH,
    )

    # --------------------------------------------------------
    # Print insights
    # --------------------------------------------------------

    print_engineering_insights(
        top_features
    )

    # --------------------------------------------------------
    # Save machine-readable summary
    # --------------------------------------------------------

    save_shap_summary(
        top_features
    )

    print("\nExplainability pipeline completed.")

    return {
        "model": model,
        "X_explain": X_explain,
        "shap_values": shap_values,
        "top_features": top_features,
    }


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":

    explain_model()