# Predictive Maintenance MLOps

End-to-end predictive maintenance machine learning project with a local MLOps workflow.

## Workflow

Data Validation
→ Feature Engineering
→ Class Imbalance Handling
→ Model Training
→ MLflow Experiment Tracking
→ Optuna Hyperparameter Tuning
→ MLflow Model Registry
→ Evidently Drift Monitoring
→ SHAP Explainability
→ Retraining

## Models

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM

## MLOps Tools

- MLflow
- Optuna
- Evidently
- SHAP
- Pandera
- Scikit-learn
- XGBoost
- LightGBM

## Project Structure

```text
src/
├── data_validation.py
├── preprocessing.py
├── train.py
├── evaluate.py
├── drift.py
├── explain.py
└── retrain.py

reports/
├── model_comparison.csv
├── shap_per_class.png
├── shap_feature_summary.csv
├── drift_current_features.csv
└── drift_stress_features.csv