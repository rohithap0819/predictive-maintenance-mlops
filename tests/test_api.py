import importlib
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.dummy import DummyClassifier
from sklearn.preprocessing import OrdinalEncoder


@pytest.fixture
def client(monkeypatch):
    # Lightweight model used only for automated testing.
    model = DummyClassifier(strategy="most_frequent")

    X = np.zeros((5, 8))
    y = np.array([0, 1, 2, 3, 4])

    model.fit(X, y)

    # Test version of the Type encoder.
    type_encoder = OrdinalEncoder(
        handle_unknown="use_encoded_value",
        unknown_value=-1,
    )

    type_encoder.fit(
        pd.DataFrame({"Type": ["L", "M", "H"]})
    )

    test_metadata = {
        "model_name": "CI Test Model",
        "version": "test",
    }

    def fake_load(path):
        filename = Path(path).name

        if filename == "best_model.pkl":
            return model

        if filename == "type_encoder.pkl":
            return type_encoder

        if filename == "best_model_metadata.pkl":
            return test_metadata

        raise AssertionError(f"Unexpected model file: {path}")

    # Replace joblib.load during API import.
    monkeypatch.setattr(joblib, "load", fake_load)

    monkeypatch.setenv("TESTING", "1")

    api_main = importlib.import_module("api.main")
    api_main = importlib.reload(api_main)

    return TestClient(api_main.app)


def test_health_endpoint(client):
    response = client.get("/health")

    assert response.status_code == 200

    data = response.json()

    assert data["status"] == "healthy"


def test_model_info_endpoint(client):
    response = client.get("/model-info")

    assert response.status_code == 200

    data = response.json()

    assert isinstance(data, dict)


def test_prediction_endpoint(client):
    payload = {
        "type": "L",
        "air_temperature": 300.0,
        "process_temperature": 315.0,
        "rotational_speed": 1200,
        "torque": 70.0,
        "tool_wear": 240,
    }

    response = client.post(
        "/predict",
        json=payload,
    )

    assert response.status_code == 200

    data = response.json()

    assert "prediction" in data
    assert "class_id" in data
    assert "probabilities" in data
    assert "engineered_features" in data

    assert data["class_id"] in [0, 1, 2, 3, 4]

    probability_sum = sum(
        data["probabilities"].values()
    )

    assert abs(probability_sum - 1.0) < 0.01