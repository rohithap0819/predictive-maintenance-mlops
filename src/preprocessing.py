from pathlib import Path
from typing import Tuple

import joblib
import numpy as np
import pandas as pd

from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42
TEST_SIZE = 0.20
SMOTE_K_NEIGHBORS = 3

TARGET_COLUMN = "Failure_Type"

RAW_FEATURES = [
    "Type",
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
]

ENGINEERED_FEATURES = [
    "Power_W",
    "Temp_diff",
]

FEATURES = [
    "Type_Code",
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
    "Power_W",
    "Temp_diff",
]


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add physically meaningful engineered features.

    Power_W:
        P = Torque * angular_velocity

    Temp_diff:
        Process temperature - Air temperature
    """

    df = df.copy()

    # Mechanical power in watts
    df["Power_W"] = (
        df["Torque"]
        * (2.0 * np.pi * df["Rotational speed"] / 60.0)
    )

    # Temperature difference
    df["Temp_diff"] = (
        df["Process temperature"]
        - df["Air temperature"]
    )

    return df


# ============================================================
# PREPARE MACHINE TYPE ENCODER
# ============================================================

def fit_type_encoder(train_df: pd.DataFrame) -> OrdinalEncoder:
    """
    Fit the machine Type encoder only on training data.
    """

    encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )

    encoder.fit(train_df[["Type"]])

    return encoder


def apply_type_encoder(
    df: pd.DataFrame,
    encoder: OrdinalEncoder,
) -> pd.DataFrame:
    """
    Apply the already-fitted Type encoder.
    """

    df = df.copy()

    df["Type_Code"] = encoder.transform(
        df[["Type"]]
    ).astype(int).ravel()

    return df


# ============================================================
# PREPARE TARGET ENCODER
# ============================================================

def fit_target_encoder(
    train_df: pd.DataFrame,
) -> LabelEncoder:
    """
    Fit the target encoder only on the training dataset.
    """

    encoder = LabelEncoder()
    encoder.fit(train_df[TARGET_COLUMN])

    return encoder


# ============================================================
# BUILD FEATURE MATRIX
# ============================================================

def build_feature_matrix(
    df: pd.DataFrame,
    type_encoder: OrdinalEncoder,
) -> pd.DataFrame:
    """
    Engineer features, encode machine Type, and return
    the final numeric feature matrix.
    """

    df = engineer_features(df)
    df = apply_type_encoder(df, type_encoder)

    X = df[FEATURES].copy()

    return X


# ============================================================
# COMPLETE TRAINING PREPARATION
# ============================================================

def prepare_training_data(
    train_df: pd.DataFrame,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    LabelEncoder,
    OrdinalEncoder,
]:
    """
    Prepare train/validation data correctly:

    1. Feature engineering
    2. Fit encoders only on training data
    3. Stratified 80/20 split
    4. Apply SMOTE only to training split
    """

    df = engineer_features(train_df)

    # --------------------------------------------------------
    # Fit encoders using training dataset
    # --------------------------------------------------------

    type_encoder = fit_type_encoder(df)
    target_encoder = fit_target_encoder(df)

    # Add machine type code
    df = apply_type_encoder(df, type_encoder)

    # --------------------------------------------------------
    # Create X and y
    # --------------------------------------------------------

    X = df[FEATURES].copy()

    y = target_encoder.transform(
        df[TARGET_COLUMN]
    )

    y = pd.Series(
        y,
        index=df.index,
        name=TARGET_COLUMN,
    )

    # --------------------------------------------------------
    # Stratified 80/20 train-validation split
    # --------------------------------------------------------

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    print("Train/Validation split")
    print("-" * 40)
    print(f"X_train: {X_train.shape}")
    print(f"X_val  : {X_val.shape}")
    print()

    print("Training class distribution BEFORE SMOTE")
    print(y_train.value_counts().sort_index())
    print()

    # --------------------------------------------------------
    # SMOTE ONLY ON TRAINING DATA
    # --------------------------------------------------------

    smote = SMOTE(
        k_neighbors=SMOTE_K_NEIGHBORS,
        random_state=RANDOM_STATE,
    )

    X_res, y_res = smote.fit_resample(
        X_train,
        y_train,
    )

    X_res = pd.DataFrame(
        X_res,
        columns=X_train.columns,
    )

    y_res = pd.Series(
        y_res,
        name=TARGET_COLUMN,
    )

    print("Training class distribution AFTER SMOTE")
    print(y_res.value_counts().sort_index())
    print()

    print("Validation class distribution")
    print(y_val.value_counts().sort_index())

    return (
        X_train,
        X_val,
        y_train,
        y_val,
        X_res,
        y_res,
        target_encoder,
        type_encoder,
    )


# ============================================================
# PREPARE CURRENT / STRESS DATA
# ============================================================

def prepare_inference_data(
    df: pd.DataFrame,
    type_encoder: OrdinalEncoder,
) -> pd.DataFrame:
    """
    Apply exactly the same feature engineering and Type
    encoder to current/stress datasets.
    """

    return build_feature_matrix(
        df,
        type_encoder,
    )


# ============================================================
# SAVE ENCODERS
# ============================================================

def save_encoders(
    target_encoder: LabelEncoder,
    type_encoder: OrdinalEncoder,
    output_dir: str | Path = "models",
) -> None:

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        target_encoder,
        output_dir / "label_encoder.pkl",
    )

    joblib.dump(
        type_encoder,
        output_dir / "type_encoder.pkl",
    )

    print(f"Saved encoders to: {output_dir}")


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":

    from src.data_validation import load_and_validate_data

    train, current, stress = load_and_validate_data("data")

    (
        X_train,
        X_val,
        y_train,
        y_val,
        X_res,
        y_res,
        target_encoder,
        type_encoder,
    ) = prepare_training_data(train)

    current_X = prepare_inference_data(
        current,
        type_encoder,
    )

    stress_X = prepare_inference_data(
        stress,
        type_encoder,
    )

    print("\nFinal feature columns:")
    print(FEATURES)

    print("\nCurrent shape:")
    print(current_X.shape)

    print("\nStress shape:")
    print(stress_X.shape)

    save_encoders(
        target_encoder,
        type_encoder,
        "models",
    )