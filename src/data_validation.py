from pathlib import Path
from typing import Tuple

import pandas as pd
import pandera as pa


# ============================================================
# DATA CONFIGURATION
# ============================================================

REQUIRED_COLUMNS = [
    "Type",
    "Air temperature",
    "Process temperature",
    "Rotational speed",
    "Torque",
    "Tool wear",
    "Failure_Type",
]

FAILURE_TYPES = {
    0: "No Failure",
    1: "TWF",
    2: "HDF",
    3: "PWF",
    4: "OSF",
}

MACHINE_TYPES = ["L", "M", "H"]


# ============================================================
# PANDERA SCHEMA
# ============================================================

RAW_SCHEMA = pa.DataFrameSchema(
    {
        "Type": pa.Column(
            pa.String,
            checks=pa.Check.isin(MACHINE_TYPES),
            nullable=False,
        ),
        "Air temperature": pa.Column(
            pa.Float64,
            nullable=False,
        ),
        "Process temperature": pa.Column(
            pa.Float64,
            nullable=False,
        ),
        "Rotational speed": pa.Column(
            pa.Int64,
            nullable=False,
        ),
        "Torque": pa.Column(
            pa.Float64,
            nullable=False,
        ),
        "Tool wear": pa.Column(
            pa.Int64,
            nullable=False,
        ),
        "Failure_Type": pa.Column(
            pa.Int64,
            checks=pa.Check.isin([0, 1, 2, 3, 4]),
            nullable=False,
        ),
        
    },
    strict=True,
    coerce=False,
)


# ============================================================
# DATA TYPE CLEANING
# ============================================================

def fix_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert raw CSV columns into the expected dtypes
    before Pandera validation.
    """

    df = df.copy()

    df["Type"] = df["Type"].astype("string")

    df["Air temperature"] = pd.to_numeric(
        df["Air temperature"], errors="raise"
    ).astype("float64")

    df["Process temperature"] = pd.to_numeric(
        df["Process temperature"], errors="raise"
    ).astype("float64")

    df["Rotational speed"] = pd.to_numeric(
        df["Rotational speed"], errors="raise"
    ).astype("int64")

    df["Torque"] = pd.to_numeric(
        df["Torque"], errors="raise"
    ).astype("float64")

    df["Tool wear"] = pd.to_numeric(
        df["Tool wear"], errors="raise"
    ).astype("int64")

    df["Failure_Type"] = pd.to_numeric(
        df["Failure_Type"], errors="raise"
    ).astype("int64")
    return df


# ============================================================
# VALIDATION
# ============================================================

def validate_dataset(
    df: pd.DataFrame,
    dataset_name: str = "dataset",
) -> pd.DataFrame:
    """
    Fix dtypes and validate one dataset using Pandera.
    """

    df = fix_dtypes(df)

    missing_columns = [
        column for column in REQUIRED_COLUMNS
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"{dataset_name} is missing columns: {missing_columns}"
        )

    unexpected_columns = [
        column for column in df.columns
        if column not in REQUIRED_COLUMNS
    ]

    if unexpected_columns:
        raise ValueError(
            f"{dataset_name} contains unexpected columns: "
            f"{unexpected_columns}"
        )

    validated_df = RAW_SCHEMA.validate(
        df,
        lazy=True,
    )

    print(
        f"[OK] {dataset_name} validated successfully "
        f"({validated_df.shape[0]} rows, {validated_df.shape[1]} columns)"
    )

    return validated_df


# ============================================================
# LOAD + VALIDATE ALL DATASETS
# ============================================================

def load_and_validate_data(
    data_dir: str | Path = "data",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    data_dir = Path(data_dir)

    train_path = data_dir / "train.csv"
    current_path = data_dir / "current.csv"
    stress_path = data_dir / "stress.csv"

    for path in [train_path, current_path, stress_path]:
        if not path.exists():
            raise FileNotFoundError(
                f"Required dataset not found: {path}"
            )

    train = pd.read_csv(train_path)
    current = pd.read_csv(current_path)
    stress = pd.read_csv(stress_path)

    print("Raw dataset shapes")
    print("-" * 40)
    print(f"Train   : {train.shape}")
    print(f"Current : {current.shape}")
    print(f"Stress  : {stress.shape}")
    print()

    train = validate_dataset(train, "train.csv")
    current = validate_dataset(current, "current.csv")
    stress = validate_dataset(stress, "stress.csv")

    return train, current, stress


# ============================================================
# LOCAL TEST
# ============================================================

if __name__ == "__main__":
    train, current, stress = load_and_validate_data("data")

    print("\nTraining sample:")
    print(train.head())