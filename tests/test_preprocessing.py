import pandas as pd

from src.preprocessing import engineer_features


def test_engineer_features():
    df = pd.DataFrame(
        {
            "Type": ["L"],
            "Air temperature": [300.0],
            "Process temperature": [315.0],
            "Rotational speed": [1200.0],
            "Torque": [70.0],
            "Tool wear": [240.0],
            "Failure_Type": [4],
        }
    )

    result = engineer_features(df)

    assert "Power_W" in result.columns
    assert "Temp_diff" in result.columns

    expected_power = 70.0 * (2 * 3.141592653589793 * 1200.0 / 60)

    assert abs(result["Power_W"].iloc[0] - expected_power) < 1e-6
    assert result["Temp_diff"].iloc[0] == 15.0