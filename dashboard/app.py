import os
import json
from pathlib import Path

import pandas as pd
import requests
import streamlit as st
import streamlit.components.v1 as components



# ============================================================
# CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Predictive Maintenance MLOps",
    page_icon="⚙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]

API_URL = os.getenv("API_URL", "http://localhost:8000")

DRIFT_CURRENT_HTML = PROJECT_ROOT / "reports" / "drift_current.html"
DRIFT_STRESS_HTML = PROJECT_ROOT / "reports" / "drift_stress.html"

DRIFT_CURRENT_CSV = PROJECT_ROOT / "reports" / "drift_current_features.csv"
DRIFT_STRESS_CSV = PROJECT_ROOT / "reports" / "drift_stress_features.csv"

SHAP_IMAGE = PROJECT_ROOT / "reports" / "shap_per_class.png"
SHAP_CSV = PROJECT_ROOT / "reports" / "shap_feature_summary.csv"


CLASS_NAMES = {
    0: "No Failure",
    1: "TWF",
    2: "HDF",
    3: "PWF",
    4: "OSF",
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def check_api():
    """Check whether FastAPI is running."""
    try:
        response = requests.get(
            f"{API_URL}/health",
            timeout=5,
        )

        if response.status_code == 200:
            return True, response.json()

        return False, f"API returned HTTP {response.status_code}"

    except requests.exceptions.RequestException as e:
        return False, str(e)


def get_model_info():
    """Get model information from FastAPI."""
    try:
        response = requests.get(
            f"{API_URL}/model-info",
            timeout=5,
        )

        if response.status_code == 200:
            return response.json()

        return None

    except requests.exceptions.RequestException:
        return None


def predict_machine(machine_data):
    """Send machine data to FastAPI prediction endpoint."""
    try:
        response = requests.post(
            f"{API_URL}/predict",
            json=machine_data,
            timeout=10,
        )

        if response.status_code == 200:
            return response.json(), None

        try:
            error_detail = response.json()
        except Exception:
            error_detail = response.text

        return None, f"API Error {response.status_code}: {error_detail}"

    except requests.exceptions.RequestException as e:
        return None, f"Could not connect to FastAPI: {e}"


def display_probability_chart(probabilities):
    """Display failure probabilities."""
    prob_df = pd.DataFrame(
        list(probabilities.items()),
        columns=["Failure Type", "Probability"],
    )

    prob_df["Probability (%)"] = prob_df["Probability"] * 100

    prob_df = prob_df.sort_values(
        "Probability (%)",
        ascending=False,
    )

    st.dataframe(
        prob_df.style.format(
            {
                "Probability": "{:.6f}",
                "Probability (%)": "{:.4f}%",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.bar_chart(
        prob_df.set_index("Failure Type")["Probability (%)"]
    )


def display_drift_report(html_path, title):
    """Display an Evidently HTML report."""
    st.subheader(title)

    if not html_path.exists():
        st.warning(f"Report not found: {html_path}")
        return

    try:
        html_content = html_path.read_text(
            encoding="utf-8",
            errors="ignore",
        )

        components.html(
            html_content,
            height=900,
            scrolling=True,
        )

    except Exception as e:
        st.error(f"Could not load drift report: {e}")


# ============================================================
# HEADER
# ============================================================

st.title("⚙️ Predictive Maintenance MLOps Dashboard")

st.markdown(
    """
    End-to-end predictive maintenance system using:

    **Machine Sensor Data → ML Model → FastAPI → Streamlit → Monitoring → Explainability**
    """
)

st.divider()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Select Dashboard Section",
    [
        "🔮 Prediction",
        "📊 Model Information",
        "📈 Drift Monitoring",
        "🧠 Explainability",
    ],
)

st.sidebar.divider()

api_status, api_details = check_api()

if api_status:
    st.sidebar.success("FastAPI: Online")
else:
    st.sidebar.error("FastAPI: Offline")
    st.sidebar.caption(
        "Start FastAPI before using the prediction page."
    )

display_api_url = (
    "http://localhost:8000"
    if API_URL == "http://api:8000"
    else API_URL
)

st.sidebar.caption(f"API: {display_api_url}")

# ============================================================
# PAGE 1: PREDICTION
# ============================================================

if page == "🔮 Prediction":

    st.header("Machine Failure Prediction")

    st.markdown(
        """
        Enter the current sensor readings of a machine.

        The Streamlit dashboard sends the data to the FastAPI
        prediction service.
        """
    )

    if not api_status:
        st.error("FastAPI is not available.")

        if API_URL == "http://api:8000":
            st.info("Start the application with:")
            st.code(
                "docker compose up --build",
                language="bash",
            )
        else:
            st.info("Start FastAPI with:")
            st.code(
                "python -m uvicorn api.main:app --reload",
                language="bash",
            )

        st.stop()

    st.subheader("Machine Sensor Inputs")

    col1, col2 = st.columns(2)

    with col1:
        machine_type = st.selectbox(
            "Machine Type",
            options=["L", "M", "H"],
            index=0,
            help="Machine operating type.",
        )

        air_temperature = st.number_input(
            "Air Temperature (K)",
            min_value=295.0,
            max_value=305.0,
            value=300.0,
            step=0.1,
        )

        process_temperature = st.number_input(
            "Process Temperature (K)",
            min_value=305.0,
            max_value=315.0,
            value=310.0,
            step=0.1,
        )

    with col2:
        rotational_speed = st.number_input(
            "Rotational Speed (rpm)",
            min_value=1000,
            max_value=2900,
            value=1500,
            step=50,
        )

        torque = st.number_input(
            "Torque (Nm)",
            min_value=3.0,
            max_value=80.0,
            value=40.0,
            step=1.0,
        )

        tool_wear = st.number_input(
            "Tool Wear (min)",
            min_value=0,
            max_value=253,
            value=50,
            step=1,
        )

    st.divider()

    # --------------------------------------------------------
    # Example high-risk input button
    # --------------------------------------------------------

    example_col1, example_col2 = st.columns([1, 4])

    with example_col1:

        if st.button(
            "Load Failure Example",
            use_container_width=True,
        ):
            st.session_state.example_loaded = True

    if st.session_state.get("example_loaded", False):

        st.info(
            "High-risk example loaded. Click Predict Machine below."
        )

    st.divider()

    predict_button = st.button(
        "🔍 Predict Machine Status",
        type="primary",
        use_container_width=True,
    )

    if predict_button:

        machine_data = {
            "type": machine_type,
            "air_temperature": air_temperature,
            "process_temperature": process_temperature,
            "rotational_speed": rotational_speed,
            "torque": torque,
            "tool_wear": tool_wear,
        }

        with st.spinner("Sending data to prediction API..."):

            result, error = predict_machine(machine_data)

        if error:

            st.error(error)

        else:

            prediction = result.get("prediction")
            class_id = result.get("class_id")
            probabilities = result.get(
                "probabilities",
                {},
            )
            engineered_features = result.get(
                "engineered_features",
                {},
            )

            st.divider()

            st.subheader("Prediction Result")

            # =================================================
            # IMPORTANT:
            # Use class_id to determine failure/no failure.
            # Do NOT compare prediction with integer 0.
            # =================================================

            if class_id == 0:

                st.success(
                    "✅ NO FAILURE DETECTED"
                )

            else:

                st.error(
                    f"🚨 FAILURE PREDICTED: {prediction}"
                )

            # -------------------------------------------------
            # Main result metrics
            # -------------------------------------------------

            result_col1, result_col2 = st.columns(2)

            with result_col1:

                st.metric(
                    "Predicted Class",
                    prediction,
                )

            with result_col2:

                st.metric(
                    "Class ID",
                    class_id,
                )

            # -------------------------------------------------
            # Probability distribution
            # -------------------------------------------------

            st.subheader("Prediction Probabilities")

            display_probability_chart(
                probabilities
            )

            # -------------------------------------------------
            # Engineered features
            # -------------------------------------------------

            if engineered_features:

                st.subheader(
                    "Engineered Features"
                )

                feature_col1, feature_col2 = st.columns(2)

                with feature_col1:

                    if "Power_W" in engineered_features:

                        st.metric(
                            "Mechanical Power",
                            f"{engineered_features['Power_W']:,.2f} W",
                        )

                with feature_col2:

                    if "Temp_diff" in engineered_features:

                        st.metric(
                            "Temperature Difference",
                            f"{engineered_features['Temp_diff']:.2f} K",
                        )

            # -------------------------------------------------
            # Raw request
            # -------------------------------------------------

            with st.expander("View Input Data"):

                st.json(machine_data)

            # -------------------------------------------------
            # API response
            # -------------------------------------------------

            with st.expander("View Raw API Response"):

                st.json(result)


# ============================================================
# PAGE 2: MODEL INFORMATION
# ============================================================

elif page == "📊 Model Information":

    st.header("Model Information")

    model_info = get_model_info()

    if model_info is None:

        st.warning(
            "Unable to retrieve model information from FastAPI."
        )

    else:

        st.success(
            "Model information successfully retrieved."
        )

        # Display the returned JSON cleanly

        if isinstance(model_info, dict):

            cols = st.columns(2)

            items = list(
                model_info.items()
            )

            for index, (key, value) in enumerate(items):

                with cols[index % 2]:

                    if isinstance(value, (dict, list)):

                        st.write(
                            f"**{key}**"
                        )

                        st.json(value)

                    else:

                        st.metric(
                            str(key).replace(
                                "_",
                                " ",
                            ).title(),
                            str(value),
                        )

        else:

            st.json(model_info)


# ============================================================
# PAGE 3: DRIFT MONITORING
# ============================================================

elif page == "📈 Drift Monitoring":

    st.header("Data Drift Monitoring")

    st.markdown(
        """
        This section compares the production batches against
        the training/reference distribution.

        **Current batch** represents normal post-deployment data.

        **Stress batch** represents the shifted heavy-load scenario.
        """
    )

    tab1, tab2 = st.tabs(
        [
            "Current Batch",
            "Stress Batch",
        ]
    )

    # --------------------------------------------------------
    # CURRENT
    # --------------------------------------------------------

    with tab1:

        st.subheader(
            "Current Production Batch"
        )

        display_drift_report(
            DRIFT_CURRENT_HTML,
            "Evidently Current Batch Report",
        )

        if DRIFT_CURRENT_CSV.exists():

            st.subheader(
                "Feature-Level Drift Results"
            )

            try:

                current_df = pd.read_csv(
                    DRIFT_CURRENT_CSV
                )

                st.dataframe(
                    current_df,
                    use_container_width=True,
                    hide_index=True,
                )

            except Exception as e:

                st.error(
                    f"Could not load current drift CSV: {e}"
                )

    # --------------------------------------------------------
    # STRESS
    # --------------------------------------------------------

    with tab2:

        st.subheader(
            "Stress Production Batch"
        )

        display_drift_report(
            DRIFT_STRESS_HTML,
            "Evidently Stress Batch Report",
        )

        if DRIFT_STRESS_CSV.exists():

            st.subheader(
                "Feature-Level Drift Results"
            )

            try:

                stress_df = pd.read_csv(
                    DRIFT_STRESS_CSV
                )

                st.dataframe(
                    stress_df,
                    use_container_width=True,
                    hide_index=True,
                )

            except Exception as e:

                st.error(
                    f"Could not load stress drift CSV: {e}"
                )


# ============================================================
# PAGE 4: EXPLAINABILITY
# ============================================================

elif page == "🧠 Explainability":

    st.header("Model Explainability")

    st.markdown(
        """
        SHAP is used to understand which features influence
        the model's prediction for each failure class.
        """
    )

    # --------------------------------------------------------
    # SHAP IMAGE
    # --------------------------------------------------------

    if SHAP_IMAGE.exists():

        st.subheader(
            "SHAP Feature Importance by Failure Class"
        )

        st.image(
            str(SHAP_IMAGE),
            use_container_width=True,
        )

    else:

        st.warning(
            f"SHAP image not found: {SHAP_IMAGE}"
        )

    # --------------------------------------------------------
    # SHAP CSV
    # --------------------------------------------------------

    if SHAP_CSV.exists():

        st.subheader(
            "SHAP Feature Summary"
        )

        try:

            shap_df = pd.read_csv(
                SHAP_CSV
            )

            st.dataframe(
                shap_df,
                use_container_width=True,
                hide_index=True,
            )

        except Exception as e:

            st.error(
                f"Could not load SHAP summary: {e}"
            )

    else:

        st.warning(
            f"SHAP summary not found: {SHAP_CSV}"
        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Predictive Maintenance MLOps | FastAPI + Streamlit + XGBoost + MLflow + Evidently + SHAP"
)