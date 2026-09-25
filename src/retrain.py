from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn
import optuna

from sklearn.metrics import accuracy_score, f1_score
from xgboost import XGBClassifier

from src.data_validation import load_and_validate_data
from src.preprocessing import prepare_training_data


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42
N_TRIALS = 30

MLFLOW_DB = "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT = "PredMaint_Optuna"

REGISTERED_MODEL_NAME = "PredMaint_XGBoost"

MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)

CLASS_LIST = [0, 1, 2, 3, 4]

CLASS_NAMES = {
    0: "No Failure",
    1: "TWF",
    2: "HDF",
    3: "PWF",
    4: "OSF",
}


# ============================================================
# MLFLOW SETUP
# ============================================================

mlflow.set_tracking_uri(MLFLOW_DB)
mlflow.set_experiment(MLFLOW_EXPERIMENT)

optuna.logging.set_verbosity(
    optuna.logging.WARNING
)


# ============================================================
# GLOBAL DATA
# ============================================================

X_RES = None
Y_RES = None
X_VAL = None
Y_VAL = None


# ============================================================
# OBJECTIVE FUNCTION
# ============================================================

def objective(trial):
    """
    Optuna objective.

    Each trial trains XGBoost using a different set of
    hyperparameters and returns validation macro-F1.
    """

    params = {
        "n_estimators": trial.suggest_int(
            "n_estimators",
            100,
            500,
        ),

        "max_depth": trial.suggest_int(
            "max_depth",
            3,
            10,
        ),

        "learning_rate": trial.suggest_float(
            "learning_rate",
            0.01,
            0.30,
            log=True,
        ),

        "min_child_weight": trial.suggest_float(
            "min_child_weight",
            1.0,
            10.0,
        ),

        "subsample": trial.suggest_float(
            "subsample",
            0.60,
            1.00,
        ),

        "colsample_bytree": trial.suggest_float(
            "colsample_bytree",
            0.60,
            1.00,
        ),

        "gamma": trial.suggest_float(
            "gamma",
            0.0,
            2.0,
        ),

        "reg_alpha": trial.suggest_float(
            "reg_alpha",
            1e-8,
            1.0,
            log=True,
        ),

        "reg_lambda": trial.suggest_float(
            "reg_lambda",
            1e-8,
            5.0,
            log=True,
        ),

        "random_state": RANDOM_STATE,
        "eval_metric": "mlogloss",
        "verbosity": 0,
        "n_jobs": -1,
    }

    model = XGBClassifier(
        **params
    )

    model.fit(
        X_RES,
        Y_RES,
    )

    predictions = model.predict(
        X_VAL
    )

    macro_f1 = f1_score(
        Y_VAL,
        predictions,
        labels=CLASS_LIST,
        average="macro",
        zero_division=0,
    )

    return macro_f1


# ============================================================
# PER-CLASS F1
# ============================================================

def calculate_per_class_f1(
    model,
):
    """
    Calculate F1 for each failure class.
    """

    predictions = model.predict(
        X_VAL
    )

    scores = f1_score(
        Y_VAL,
        predictions,
        labels=CLASS_LIST,
        average=None,
        zero_division=0,
    )

    return {
        class_id: float(score)
        for class_id, score in zip(
            CLASS_LIST,
            scores,
        )
    }


# ============================================================
# OPTUNA TUNING
# ============================================================

def run_optuna():
    """
    Run the Optuna study.
    """

    print("=" * 70)
    print("OPTUNA XGBOOST HYPERPARAMETER TUNING")
    print("=" * 70)

    print(
        f"Number of trials: {N_TRIALS}"
    )

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(
            seed=RANDOM_STATE
        ),
    )

    study.optimize(
        objective,
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    print("\nBest validation macro-F1:")
    print(
        f"{study.best_value:.6f}"
    )

    print("\nBest parameters:")

    for parameter, value in (
        study.best_params.items()
    ):
        print(
            f"  {parameter}: {value}"
        )

    return study


# ============================================================
# TRAIN FINAL TUNED MODEL
# ============================================================

def train_best_model(
    study,
):
    """
    Train XGBoost using Optuna's best parameters.
    """

    best_params = {
        **study.best_params,

        "random_state": RANDOM_STATE,
        "eval_metric": "mlogloss",
        "verbosity": 0,
        "n_jobs": -1,
    }

    model = XGBClassifier(
        **best_params
    )

    model.fit(
        X_RES,
        Y_RES,
    )

    predictions = model.predict(
        X_VAL
    )

    macro_f1 = f1_score(
        Y_VAL,
        predictions,
        labels=CLASS_LIST,
        average="macro",
        zero_division=0,
    )

    weighted_f1 = f1_score(
        Y_VAL,
        predictions,
        labels=CLASS_LIST,
        average="weighted",
        zero_division=0,
    )

    accuracy = accuracy_score(
        Y_VAL,
        predictions,
    )

    per_class_f1 = calculate_per_class_f1(
        model
    )

    print("\n" + "=" * 70)
    print("FINAL TUNED XGBOOST")
    print("=" * 70)

    print(
        f"Macro F1    : {macro_f1:.4f}"
    )

    print(
        f"Weighted F1 : {weighted_f1:.4f}"
    )

    print(
        f"Accuracy    : {accuracy:.4f}"
    )

    print("\nPer-class F1:")

    for class_id, score in (
        per_class_f1.items()
    ):
        print(
            f"  {class_id} - "
            f"{CLASS_NAMES[class_id]:<12} : "
            f"{score:.4f}"
        )

    return (
        model,
        best_params,
        macro_f1,
        weighted_f1,
        accuracy,
        per_class_f1,
    )


# ============================================================
# LOG TO MLFLOW
# ============================================================

def log_tuned_model(
    model,
    best_params,
    macro_f1,
    weighted_f1,
    accuracy,
    per_class_f1,
):
    """
    Log the tuned model and metrics to MLflow.
    """

    with mlflow.start_run(
        run_name="Optuna_Tuned_XGBoost"
    ) as run:

        # ----------------------------------------------------
        # Parameters
        # ----------------------------------------------------

        mlflow.log_param(
            "model_name",
            "XGBoost",
        )

        mlflow.log_param(
            "optuna_trials",
            N_TRIALS,
        )

        mlflow.log_param(
            "selection_metric",
            "macro_f1",
        )

        mlflow.log_param(
            "smote_k_neighbors",
            3,
        )

        for parameter, value in (
            best_params.items()
        ):
            mlflow.log_param(
                parameter,
                str(value),
            )

        # ----------------------------------------------------
        # Aggregate metrics
        # ----------------------------------------------------

        mlflow.log_metric(
            "macro_f1",
            macro_f1,
        )

        mlflow.log_metric(
            "weighted_f1",
            weighted_f1,
        )

        mlflow.log_metric(
            "accuracy",
            accuracy,
        )

        # ----------------------------------------------------
        # Per-class metrics
        # ----------------------------------------------------

        for class_id, score in (
            per_class_f1.items()
        ):
            mlflow.log_metric(
                f"f1_class_{class_id}",
                score,
            )

        # ----------------------------------------------------
        # Model artifact
        # ----------------------------------------------------

        model_info = mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            input_example=X_VAL.head(5),
        )

        run_id = run.info.run_id

        model_uri = model_info.model_uri

    print(
        f"\nMLflow run ID: {run_id}"
    )

    return run_id, model_uri


# ============================================================
# REGISTER MODEL
# ============================================================

def register_model(
    model_uri,
):
    """
    Register the tuned model in MLflow Model Registry
    and assign the production alias.
    """

    print("\n" + "=" * 70)
    print("MLFLOW MODEL REGISTRY")
    print("=" * 70)

    registered = mlflow.register_model(
        model_uri=model_uri,
        name=REGISTERED_MODEL_NAME,
    )

    version = registered.version

    print(
        f"Registered model: "
        f"{REGISTERED_MODEL_NAME}"
    )

    print(
        f"Version: {version}"
    )

    # --------------------------------------------------------
    # Set production alias
    # --------------------------------------------------------

    client = mlflow.tracking.MlflowClient()

    try:

        client.set_registered_model_alias(
            REGISTERED_MODEL_NAME,
            "production",
            version,
        )

        print(
            "Production alias: production"
        )

    except Exception as exc:

        print(
            "Warning: could not set "
            f"production alias: {exc}"
        )

    return version


# ============================================================
# SAVE LOCAL ARTIFACTS
# ============================================================

def save_local_artifacts(
    model,
    target_encoder,
    type_encoder,
    metadata,
):
    """
    Save the final production model and supporting encoders.
    """

    joblib.dump(
        model,
        MODEL_DIR / "best_model.pkl",
    )

    joblib.dump(
        target_encoder,
        MODEL_DIR / "label_encoder.pkl",
    )

    joblib.dump(
        type_encoder,
        MODEL_DIR / "type_encoder.pkl",
    )

    joblib.dump(
        metadata,
        MODEL_DIR / "best_model_metadata.pkl",
    )

    print("\nSaved:")
    print(
        f"  {MODEL_DIR / 'best_model.pkl'}"
    )
    print(
        f"  {MODEL_DIR / 'label_encoder.pkl'}"
    )
    print(
        f"  {MODEL_DIR / 'type_encoder.pkl'}"
    )
    print(
        f"  {MODEL_DIR / 'best_model_metadata.pkl'}"
    )


# ============================================================
# COMPLETE RETRAINING PIPELINE
# ============================================================

def retrain():
    """
    Complete Optuna retraining pipeline.
    """

    global X_RES
    global Y_RES
    global X_VAL
    global Y_VAL

    # --------------------------------------------------------
    # Load validated data
    # --------------------------------------------------------

    train, current, stress = (
        load_and_validate_data("data")
    )

    # --------------------------------------------------------
    # Prepare training data
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

    X_RES = X_res
    Y_RES = y_res
    X_VAL = X_val
    Y_VAL = y_val

    # --------------------------------------------------------
    # Optuna
    # --------------------------------------------------------

    study = run_optuna()

    # --------------------------------------------------------
    # Train final tuned model
    # --------------------------------------------------------

    (
        model,
        best_params,
        macro_f1,
        weighted_f1,
        accuracy,
        per_class_f1,
    ) = train_best_model(
        study
    )

    # --------------------------------------------------------
    # MLflow logging
    # --------------------------------------------------------

    run_id, model_uri = log_tuned_model(
        model=model,
        best_params=best_params,
        macro_f1=macro_f1,
        weighted_f1=weighted_f1,
        accuracy=accuracy,
        per_class_f1=per_class_f1,
    )

    # --------------------------------------------------------
    # Register model
    # --------------------------------------------------------

    registered_version = register_model(
        model_uri
    )

    # --------------------------------------------------------
    # Save local model
    # --------------------------------------------------------

    metadata = {
        "model_name": "PredMaint_XGBoost",
        "model_version": registered_version,
        "mlflow_run_id": run_id,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "accuracy": accuracy,
        "optuna_trials": N_TRIALS,
        "best_params": best_params,
        "per_class_f1": per_class_f1,
    }

    save_local_artifacts(
        model=model,
        target_encoder=target_encoder,
        type_encoder=type_encoder,
        metadata=metadata,
    )

    print("\n" + "=" * 70)
    print("RETRAINING PIPELINE COMPLETED")
    print("=" * 70)

    print(
        f"Final Macro F1: {macro_f1:.4f}"
    )

    print(
        f"Registered Version: "
        f"{registered_version}"
    )

    return {
        "model": model,
        "study": study,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "accuracy": accuracy,
        "per_class_f1": per_class_f1,
        "registered_version": registered_version,
    }


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":

    retrain()