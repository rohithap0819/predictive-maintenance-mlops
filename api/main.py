from pathlib import Path

import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.preprocessing import (
    prepare_inference_data,
)


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

MODEL_PATH = PROJECT_ROOT / "models" / "best_model.pkl"
TYPE_ENCODER_PATH = PROJECT_ROOT / "models" / "type_encoder.pkl"
METADATA_PATH = PROJECT_ROOT / "models" / "best_model_metadata.pkl"


# ============================================================
# LOAD MODEL
# ============================================================

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}"
    )

if not TYPE_ENCODER_PATH.exists():
    raise FileNotFoundError(
        f"Type encoder not found: {TYPE_ENCODER_PATH}"
    )

model = joblib.load(MODEL_PATH)
type_encoder = joblib.load(TYPE_ENCODER_PATH)

metadata = {}

if METADATA_PATH.exists():
    metadata = joblib.load(METADATA_PATH)


# ============================================================
# FASTAPI APP
# ============================================================

app = FastAPI(
    title="Predictive Maintenance API",
    description=(
        "Machine failure prediction API powered by "
        "the Predictive Maintenance MLOps pipeline."
    ),
    version="1.0.0",
)


# ============================================================
# REQUEST SCHEMA
# ============================================================

class MachineInput(BaseModel):
    type: str = Field(
        ...,
        description="Machine type: L, M, or H",
        pattern="^[LMH]$",
    )

    air_temperature: float = Field(
        ...,
        description="Air temperature",
    )

    process_temperature: float = Field(
        ...,
        description="Process temperature",
    )

    rotational_speed: float = Field(
        ...,
        gt=0,
        description="Rotational speed in RPM",
    )

    torque: float = Field(
        ...,
        ge=0,
        description="Torque",
    )

    tool_wear: float = Field(
        ...,
        ge=0,
        description="Tool wear",
    )


# ============================================================
# RESPONSE MAPPING
# ============================================================

CLASS_NAMES = {
    0: "No Failure",
    1: "TWF",
    2: "HDF",
    3: "PWF",
    4: "OSF",
}


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model": "PredMaint_XGBoost",
        "model_version": metadata.get(
            "model_version",
            "local",
        ),
    }


# ============================================================
# MODEL INFORMATION
# ============================================================

@app.get("/model-info")
def model_info():
    return {
        "model_name": metadata.get(
            "model_name",
            "PredMaint_XGBoost",
        ),
        "model_version": metadata.get(
            "model_version",
            "local",
        ),
        "macro_f1": metadata.get(
            "macro_f1"
        ),
        "accuracy": metadata.get(
            "accuracy"
        ),
        "optuna_trials": metadata.get(
            "optuna_trials"
        ),
    }


# ============================================================
# SINGLE PREDICTION
# ============================================================

@app.post("/predict")
def predict(machine: MachineInput):

    try:
        input_df = pd.DataFrame(
            [
                {
                    "Type": machine.type,
                    "Air temperature": machine.air_temperature,
                    "Process temperature": machine.process_temperature,
                    "Rotational speed": machine.rotational_speed,
                    "Torque": machine.torque,
                    "Tool wear": machine.tool_wear,
                }
            ]
        )

        # Apply the exact same preprocessing used during training
        X = prepare_inference_data(
            input_df,
            type_encoder,
        )

        prediction = int(
            model.predict(X)[0]
        )

        probabilities = model.predict_proba(X)[0]

        probability_map = {
            CLASS_NAMES[index]: round(
                float(probability),
                6,
            )
            for index, probability in enumerate(
                probabilities
            )
        }

        return {
            "prediction": CLASS_NAMES[
                prediction
            ],
            "class_id": prediction,
            "probabilities": probability_map,
            "engineered_features": {
                "Power_W": round(
                    float(X["Power_W"].iloc[0]),
                    3,
                ),
                "Temp_diff": round(
                    float(X["Temp_diff"].iloc[0]),
                    3,
                ),
            },
        }

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "message": "Predictive Maintenance API",
        "docs": "/docs",
        "health": "/health",
        "prediction_endpoint": "/predict",
    }