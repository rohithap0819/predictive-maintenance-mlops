# Predictive Maintenance MLOps

An end-to-end Machine Learning Operations (MLOps) system for **predictive maintenance classification** using industrial machine sensor data.

The project predicts whether a machine is operating normally or is heading toward one of four failure modes while demonstrating a complete ML lifecycle:

**Data Validation → Feature Engineering → Imbalance Handling → Model Training → Experiment Tracking → Hyperparameter Tuning → Model Registry → Drift Monitoring → Explainability → Retraining**

---

## Project Overview

Unplanned machine failures can cause production downtime, maintenance costs, and operational disruption.

This project builds a predictive maintenance classification workflow using machine sensor measurements such as:

- Air temperature
- Process temperature
- Rotational speed
- Torque
- Tool wear
- Machine type

Two physically meaningful features are engineered:

- **Power_W** — estimated mechanical power from torque and rotational speed
- **Temp_diff** — process temperature minus air temperature

The target contains five classes:

| Class | Failure Type |
|---|---|
| 0 | No Failure |
| 1 | TWF |
| 2 | HDF |
| 3 | PWF |
| 4 | OSF |

The dataset is highly imbalanced, with rare failure classes being much harder to predict.

---

# MLOps Architecture

```text
                    ┌───────────────────┐
                    │    CSV Data       │
                    │ train/current/    │
                    │ stress            │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ Data Validation   │
                    │     Pandera       │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ Feature Engineering│
                    │ Power_W / Temp_diff│
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ Stratified Split  │
                    │      80 / 20      │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │       SMOTE       │
                    │ k_neighbors = 3   │
                    └─────────┬─────────┘
                              │
                              ▼
          ┌────────────────────────────────────┐
          │          Model Comparison          │
          │                                    │
          │ Logistic Regression                │
          │ Random Forest                      │
          │ XGBoost                            │
          │ LightGBM                           │
          └────────────────┬───────────────────┘
                           │
                           ▼
                    ┌───────────────────┐
                    │      MLflow       │
                    │ Experiment Track  │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │  Optuna Tuning    │
                    │   XGBoost         │
                    └─────────┬─────────┘
                              │
                              ▼
                    ┌───────────────────┐
                    │ MLflow Model      │
                    │     Registry      │
                    └─────────┬─────────┘
                              │
               ┌──────────────┴──────────────┐
               ▼                             ▼
      ┌─────────────────┐           ┌─────────────────┐
      │ Drift Monitoring│           │  SHAP           │
      │   Evidently     │           │ Explainability  │
      └────────┬────────┘           └────────┬────────┘
               │                             │
               └──────────────┬──────────────┘
                              ▼
                    ┌───────────────────┐
                    │ Retraining        │
                    │ Decision          │
                    └───────────────────┘
