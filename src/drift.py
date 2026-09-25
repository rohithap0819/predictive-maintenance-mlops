from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import ks_2samp
from evidently import Report
from evidently.presets import DataDriftPreset

from src.preprocessing import engineer_features


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

DRIFT_FEATURES = [
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
    "Power_W",
    "Temp_diff",
]

# KS-test threshold used for supplementary per-feature analysis
P_VALUE_THRESHOLD = 0.05

# A feature is considered meaningfully drifted when:
# p-value < 0.05
SIGNIFICANT_DRIFT_THRESHOLD = 0.05

# Dataset-level rule used for the engineering recommendation.
# If at least this many features are significantly drifted,
# the batch is considered materially shifted.
DATASET_DRIFT_FEATURE_COUNT = 2


# ============================================================
# PREPARE DATA FOR DRIFT MONITORING
# ============================================================

def prepare_drift_data(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add engineered features and retain only the numerical
    operational features used for drift monitoring.
    """

    df = engineer_features(df)

    missing = [
        column
        for column in DRIFT_FEATURES
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing drift features: {missing}"
        )

    return df[DRIFT_FEATURES].copy()


# ============================================================
# BASIC STATISTIC
# ============================================================

def compare_rotational_speed(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
) -> Dict[str, float]:
    """
    Compare mean rotational speed between two datasets.
    """

    reference_mean = float(
        reference_df["Rotational speed"].mean()
    )

    current_mean = float(
        current_df["Rotational speed"].mean()
    )

    absolute_difference = (
        current_mean - reference_mean
    )

    percentage_change = (
        absolute_difference / reference_mean * 100
        if reference_mean != 0
        else np.nan
    )

    return {
        "reference_mean": reference_mean,
        "current_mean": current_mean,
        "difference": absolute_difference,
        "percentage_change": percentage_change,
    }


# ============================================================
# EVIDENTLY REPORT
# ============================================================

def generate_evidently_report(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Generate and save an Evidently data-drift report.
    """

    report = Report(
        metrics=[
            DataDriftPreset()
        ]
    )

    evaluation = report.run(
        reference_data=reference_df,
        current_data=current_df,
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Evidently 0.7.x saves the evaluated result.
    evaluation.save_html(
        str(output_path)
    )

    print(f"Saved Evidently report: {output_path}")


# ============================================================
# PER-FEATURE DRIFT ANALYSIS
# ============================================================

def calculate_feature_drift(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Calculate a supplementary per-feature KS-test table.

    Evidently remains the main monitoring tool.
    The KS table makes the individual feature results easy
    to inspect programmatically.
    """

    rows = []

    for feature in DRIFT_FEATURES:

        reference_values = (
            reference_df[feature]
            .dropna()
            .astype(float)
        )

        current_values = (
            current_df[feature]
            .dropna()
            .astype(float)
        )

        statistic, p_value = ks_2samp(
            reference_values,
            current_values,
        )

        reference_mean = reference_values.mean()
        current_mean = current_values.mean()

        if reference_mean != 0:
            mean_change_pct = (
                (current_mean - reference_mean)
                / reference_mean
                * 100
            )
        else:
            mean_change_pct = np.nan

        rows.append(
            {
                "Feature": feature,
                "Reference_Mean": reference_mean,
                "Current_Mean": current_mean,
                "Mean_Change_%": mean_change_pct,
                "KS_Statistic": statistic,
                "P_Value": p_value,
                "Significant_Drift": (
                    p_value < SIGNIFICANT_DRIFT_THRESHOLD
                ),
            }
        )

    result = pd.DataFrame(rows)

    return result.sort_values(
        by="P_Value",
        ascending=True,
    ).reset_index(drop=True)


# ============================================================
# DRIFT SUMMARY
# ============================================================

def summarize_drift(
    drift_table: pd.DataFrame,
) -> Dict:
    """
    Summarize the per-feature drift results.
    """

    drifted_features = drift_table.loc[
        drift_table["Significant_Drift"],
        "Feature",
    ].tolist()

    drift_count = len(drifted_features)

    dataset_drift = (
        drift_count >= DATASET_DRIFT_FEATURE_COUNT
    )

    return {
        "dataset_drift": dataset_drift,
        "drifted_feature_count": drift_count,
        "drifted_features": drifted_features,
    }


# ============================================================
# SAVE DRIFT TABLE
# ============================================================

def save_drift_table(
    drift_table: pd.DataFrame,
    output_path: str | Path,
) -> None:
    """
    Save per-feature drift results.
    """

    output_path = Path(output_path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    drift_table.to_csv(
        output_path,
        index=False,
    )

    print(f"Saved drift table: {output_path}")


# ============================================================
# PLOT MEAN COMPARISON
# ============================================================

def plot_feature_means(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    dataset_name: str,
    output_path: str | Path | None = None,
) -> None:
    """
    Plot reference vs incoming feature means.
    """

    reference_means = (
        reference_df[DRIFT_FEATURES]
        .mean()
    )

    current_means = (
        current_df[DRIFT_FEATURES]
        .mean()
    )

    comparison = pd.DataFrame(
        {
            "Reference": reference_means,
            dataset_name: current_means,
        }
    )

    ax = comparison.plot(
        kind="bar",
        figsize=(12, 6),
    )

    ax.set_title(
        f"Reference vs {dataset_name} Feature Means"
    )
    ax.set_ylabel("Mean Value")
    ax.set_xlabel("Feature")
    plt.xticks(
        rotation=45,
        ha="right",
    )
    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        plt.savefig(
            output_path,
            dpi=180,
            bbox_inches="tight",
        )

        print(
            f"Saved feature mean plot: {output_path}"
        )

    plt.show()
    plt.close()


# ============================================================
# COMPLETE DATASET MONITORING
# ============================================================

def monitor_dataset(
    reference_df: pd.DataFrame,
    current_df: pd.DataFrame,
    dataset_name: str,
    output_dir: str | Path = "reports",
) -> Tuple[pd.DataFrame, Dict]:

    output_dir = Path(output_dir)
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Prepare data
    reference_features = prepare_drift_data(
        reference_df
    )

    current_features = prepare_drift_data(
        current_df
    )

    # --------------------------------------------------------
    # Basic rotational-speed comparison
    # --------------------------------------------------------

    speed_stats = compare_rotational_speed(
        reference_features,
        current_features,
    )

    print()
    print("=" * 60)
    print(f"{dataset_name.upper()} MONITORING")
    print("=" * 60)

    print(
        f"Reference mean rotational speed: "
        f"{speed_stats['reference_mean']:.2f}"
    )

    print(
        f"{dataset_name} mean rotational speed: "
        f"{speed_stats['current_mean']:.2f}"
    )

    print(
        f"Percentage change: "
        f"{speed_stats['percentage_change']:.2f}%"
    )

    # --------------------------------------------------------
    # Evidently report
    # --------------------------------------------------------

    html_path = (
        output_dir
        / f"drift_{dataset_name.lower()}.html"
    )

    generate_evidently_report(
        reference_features,
        current_features,
        html_path,
    )

    # --------------------------------------------------------
    # Feature-level analysis
    # --------------------------------------------------------

    drift_table = calculate_feature_drift(
        reference_features,
        current_features,
    )

    csv_path = (
        output_dir
        / f"drift_{dataset_name.lower()}_features.csv"
    )

    save_drift_table(
        drift_table,
        csv_path,
    )

    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    summary = summarize_drift(
        drift_table
    )

    print("\nPer-feature drift:")
    print(drift_table.to_string(index=False))

    print(
        "\nSignificantly drifted features:"
    )

    if summary["drifted_features"]:
        for feature in summary[
            "drifted_features"
        ]:
            print(f"  - {feature}")
    else:
        print("  None")

    print(
        f"\nDataset drift detected: "
        f"{summary['dataset_drift']}"
    )

    return drift_table, summary


# ============================================================
# RETRAINING DECISION
# ============================================================

def make_retraining_decision(
    current_summary: Dict,
    stress_summary: Dict,
) -> Dict:

    stress_drift = stress_summary[
        "dataset_drift"
    ]

    current_drift = current_summary[
        "dataset_drift"
    ]

    if stress_drift:
        decision = "RETRAIN"
        reason = (
            "Stress batch shows material distribution shift "
            "across multiple operational features."
        )

    elif current_drift:
        decision = "MONITOR"
        reason = (
            "Current batch shows some drift, but the "
            "evidence is not strong enough for immediate "
            "retraining."
        )

    else:
        decision = "NO_RETRAINING"
        reason = (
            "No material dataset-level drift detected."
        )

    return {
        "decision": decision,
        "reason": reason,
        "current_drift": current_drift,
        "stress_drift": stress_drift,
    }


# ============================================================
# LOCAL TEST / FULL MONITORING RUN
# ============================================================

if __name__ == "__main__":

    from src.data_validation import (
        load_and_validate_data,
    )

    # --------------------------------------------------------
    # Load validated datasets
    # --------------------------------------------------------

    train, current, stress = (
        load_and_validate_data("data")
    )

    # --------------------------------------------------------
    # Monitor current batch
    # --------------------------------------------------------

    current_table, current_summary = (
        monitor_dataset(
            reference_df=train,
            current_df=current,
            dataset_name="current",
            output_dir="reports",
        )
    )

    # --------------------------------------------------------
    # Monitor stress batch
    # --------------------------------------------------------

    stress_table, stress_summary = (
        monitor_dataset(
            reference_df=train,
            current_df=stress,
            dataset_name="stress",
            output_dir="reports",
        )
    )

    # --------------------------------------------------------
    # Retraining decision
    # --------------------------------------------------------

    decision = make_retraining_decision(
        current_summary,
        stress_summary,
    )

    print("\n")
    print("=" * 60)
    print("RETRAINING DECISION")
    print("=" * 60)

    print(
        f"Decision: {decision['decision']}"
    )

    print(
        f"Reason: {decision['reason']}"
    )