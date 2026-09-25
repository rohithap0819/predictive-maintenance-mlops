from pathlib import Path

import joblib
import mlflow
import mlflow.sklearn

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from src.data_validation import load_and_validate_data
from src.preprocessing import prepare_training_data
from src.evaluate import (
    evaluate_model,
    print_evaluation,
    compare_models,
    save_comparison,
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

MLFLOW_DB = "sqlite:///mlflow.db"
MLFLOW_EXPERIMENT = "PredMaint_ModelSelection"

MODEL_DIR = Path("models")
REPORT_DIR = Path("reports")

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# MLFLOW SETUP
# ============================================================

mlflow.set_tracking_uri(MLFLOW_DB)
mlflow.set_experiment(MLFLOW_EXPERIMENT)


# ============================================================
# MODEL DEFINITIONS
# ============================================================

def build_models():
    """
    Return the four candidate models required by the project.
    """

    models = {
        "LogisticRegression": LogisticRegression(
            max_iter=2000,
            random_state=RANDOM_STATE,
        ),

        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),

        "XGBoost": XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.1,
            min_child_weight=1,
            subsample=1.0,
            colsample_bytree=1.0,
            gamma=0.0,
            reg_alpha=0.0,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            eval_metric="mlogloss",
            verbosity=0,
            n_jobs=-1,
        ),

        "LightGBM": LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=-1,
            num_leaves=31,
            random_state=RANDOM_STATE,
            verbosity=-1,
            n_jobs=-1,
        ),
    }

    return models


# ============================================================
# TRAIN ONE MODEL + LOG TO MLFLOW
# ============================================================

def train_and_log_model(
    model_name,
    model,
    X_res,
    y_res,
    X_val,
    y_val,
):
    """
    Train one model, evaluate it, and log the complete
    experiment to MLflow.
    """

    print("\n" + "=" * 70)
    print(f"TRAINING: {model_name}")
    print("=" * 70)

    with mlflow.start_run(
        run_name=model_name
    ) as run:

        # ----------------------------------------------------
        # Train
        # ----------------------------------------------------

        model.fit(
            X_res,
            y_res,
        )

        # ----------------------------------------------------
        # Evaluate
        # ----------------------------------------------------

        results = evaluate_model(
            model=model,
            X=X_val,
            y=y_val,
            model_name=model_name,
        )

        print_evaluation(results)

        # ----------------------------------------------------
        # Log model identity
        # ----------------------------------------------------

        mlflow.log_param(
            "model_name",
            model_name,
        )

        mlflow.log_param(
            "random_state",
            RANDOM_STATE,
        )

        mlflow.log_param(
            "smote_k_neighbors",
            3,
        )

        mlflow.log_param(
            "validation_size",
            0.20,
        )

        # ----------------------------------------------------
        # Log model hyperparameters
        # ----------------------------------------------------

        model_params = model.get_params()

        for parameter_name, parameter_value in model_params.items():

            # MLflow params need simple serializable values
            if parameter_value is not None:
                mlflow.log_param(
                    f"model_{parameter_name}",
                    str(parameter_value),
                )

        # ----------------------------------------------------
        # Log aggregate metrics
        # ----------------------------------------------------

        mlflow.log_metric(
            "macro_f1",
            float(results["macro_f1"]),
        )

        mlflow.log_metric(
            "weighted_f1",
            float(results["weighted_f1"]),
        )

        mlflow.log_metric(
            "accuracy",
            float(results["accuracy"]),
        )

        # ----------------------------------------------------
        # Log per-class F1
        # ----------------------------------------------------

        for class_id, score in results[
            "per_class_f1"
        ].items():

            mlflow.log_metric(
                f"f1_class_{class_id}",
                float(score),
            )

        # ----------------------------------------------------
        # Save model artifact to MLflow
        # ----------------------------------------------------

        mlflow.sklearn.log_model(
            model,
            artifact_path="model",
            input_example=X_val.head(5),
        )

        run_id = run.info.run_id

        print(
            f"\nMLflow run ID: {run_id}"
        )

        return model, results, run_id


# ============================================================
# TRAIN ALL FOUR MODELS
# ============================================================

def train_all_models():

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
    ) = prepare_training_data(train)

    # --------------------------------------------------------
    # Build candidate models
    # --------------------------------------------------------

    models = build_models()

    all_results = []
    trained_models = {}

    # --------------------------------------------------------
    # Train each model
    # --------------------------------------------------------

    for model_name, model in models.items():

        trained_model, results, run_id = (
            train_and_log_model(
                model_name=model_name,
                model=model,
                X_res=X_res,
                y_res=y_res,
                X_val=X_val,
                y_val=y_val,
            )
        )

        trained_models[
            model_name
        ] = trained_model

        results["run_id"] = run_id

        all_results.append(
            results
        )

        # Save individual model locally
        joblib.dump(
            trained_model,
            MODEL_DIR / f"{model_name}.pkl",
        )

    # --------------------------------------------------------
    # Compare all models
    # --------------------------------------------------------

    comparison = compare_models(
        all_results
    )

    print("\n\nMODEL COMPARISON")
    print("=" * 80)

    print(
        comparison.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # Save comparison report
    # --------------------------------------------------------

    save_comparison(
        comparison,
        REPORT_DIR / "model_comparison.csv",
    )

    # --------------------------------------------------------
    # Select winner by macro-F1
    # --------------------------------------------------------

    winner_name = comparison.iloc[0]["Model"]

    winner_macro_f1 = comparison.iloc[0]["Macro F1"]

    winner_accuracy = comparison.iloc[0]["Accuracy"]

    winner_model = trained_models[
        winner_name
    ]

    print("\n" + "=" * 80)
    print("BASELINE MODEL SELECTION")
    print("=" * 80)

    print(
        f"Winning model: {winner_name}"
    )

    print(
        f"Validation Macro F1: "
        f"{winner_macro_f1:.4f}"
    )

    print(
        f"Validation Accuracy: "
        f"{winner_accuracy:.4f}"
    )

    print(
        "\nSelection metric: MACRO F1"
    )

    # --------------------------------------------------------
    # Save metadata
    # --------------------------------------------------------

    metadata = {
        "winner_model": winner_name,
        "macro_f1": float(winner_macro_f1),
        "accuracy": float(winner_accuracy),
        "random_state": RANDOM_STATE,
    }

    joblib.dump(
        metadata,
        MODEL_DIR / "baseline_model_metadata.pkl",
    )

    print(
        "\nSaved baseline models to:",
        MODEL_DIR,
    )

    return {
        "comparison": comparison,
        "models": trained_models,
        "winner_name": winner_name,
        "winner_model": winner_model,
        "X_train": X_train,
        "X_val": X_val,
        "y_train": y_train,
        "y_val": y_val,
        "X_res": X_res,
        "y_res": y_res,
        "target_encoder": target_encoder,
        "type_encoder": type_encoder,
    }


# ============================================================
# SCRIPT ENTRY POINT
# ============================================================

if __name__ == "__main__":

    results = train_all_models()

    print("\nTraining pipeline completed successfully.")