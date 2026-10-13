
import streamlit as st
import pandas as pd
import joblib
import numpy as np
from pathlib import Path

# -----------------------------
# PAGE CONFIGURATION
# -----------------------------
st.set_page_config(
    page_title="Nepal FloodGuard",
    page_icon="🌧️",
    layout="wide"
)

# -----------------------------
# LOAD MODEL AND REFERENCE DATA
# -----------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "floodguard_models" / "landslide_model.pkl"
REFERENCE_PATH = BASE_DIR / "floodguard_models" / "X_train_reference.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

@st.cache_data
def load_reference_data():
    return joblib.load(REFERENCE_PATH)

try:
    model = load_model()
    X_reference = load_reference_data()
except FileNotFoundError:
    st.error(
        "Model files not found. Run the preparation cell in your notebook "
        "and make sure the floodguard_models folder is beside app.py."
    )
    st.stop()
except Exception as e:
    st.error(f"Could not load the model files: {e}")
    st.stop()

# -----------------------------
# HEADER
# -----------------------------
st.title("🌧️ Nepal FloodGuard")
st.subheader("Landslide Susceptibility Prediction Dashboard")

st.markdown(
    """
    Explore how environmental and geographical conditions relate to
    landslide susceptibility using a trained Machine Learning model.
    
    **Current scope:** landslide susceptibility only. This is a research
    prototype, not an official warning or emergency-response system.
    """
)

st.divider()

# -----------------------------
# SIDEBAR INPUTS
# -----------------------------
st.sidebar.header("Enter Environmental Conditions")
st.sidebar.caption("Provide values for the model's input features.")

user_input = {}

for column in X_reference.columns:
    series = X_reference[column]
    clean_series = series.dropna()

    # Categorical feature
    if (
        pd.api.types.is_object_dtype(series)
        or isinstance(series.dtype, pd.CategoricalDtype)
        or pd.api.types.is_string_dtype(series)
        or pd.api.types.is_bool_dtype(series)
    ):
        options = clean_series.astype(str).unique().tolist()
        options = sorted(options)

        if options:
            user_input[column] = st.sidebar.selectbox(
                column.replace("_", " ").title(),
                options=options
            )
        else:
            user_input[column] = st.sidebar.text_input(
                column.replace("_", " ").title(),
                value=""
            )

    # Numeric feature
    else:
        numeric_series = pd.to_numeric(clean_series, errors="coerce").dropna()

        if numeric_series.empty:
            default_value = 0.0
            min_value = -1_000_000.0
            max_value = 1_000_000.0
        else:
            default_value = float(numeric_series.median())
            min_value = float(numeric_series.min())
            max_value = float(numeric_series.max())

            if min_value == max_value:
                min_value -= 1.0
                max_value += 1.0

        if pd.api.types.is_integer_dtype(series.dtype):
            user_input[column] = st.sidebar.number_input(
                column.replace("_", " ").title(),
                min_value=int(np.floor(min_value)),
                max_value=int(np.ceil(max_value)),
                value=int(round(default_value)),
                step=1
            )
        else:
            user_input[column] = st.sidebar.number_input(
                column.replace("_", " ").title(),
                min_value=float(min_value),
                max_value=float(max_value),
                value=float(np.clip(default_value, min_value, max_value)),
                format="%.4f"
            )

# -----------------------------
# PREDICTION
# -----------------------------
st.header("Prediction")

if st.button("Predict Landslide Susceptibility", type="primary"):
    input_df = pd.DataFrame([user_input])

    # Match the training column order
    input_df = input_df.reindex(columns=X_reference.columns)

    try:
        prediction = model.predict(input_df)[0]

        # Find the probability corresponding to class 1, if available
        probability = None
        if hasattr(model, "predict_proba"):
            probabilities = model.predict_proba(input_df)[0]
            classes = list(model.classes_)

            if 1 in classes:
                probability = float(probabilities[classes.index(1)])

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Predicted Class")
            if str(prediction) == "1":
                st.error("Class 1 — Landslide susceptibility indicated")
            elif str(prediction) == "0":
                st.success("Class 0 — Landslide susceptibility not indicated")
            else:
                st.info(f"Model output: {prediction}")

        with col2:
            st.subheader("Model Probability")
            if probability is not None:
                st.metric("Class 1 probability", f"{probability:.1%}")
                st.progress(float(np.clip(probability, 0.0, 1.0)))
            else:
                st.info("This model does not provide probability estimates.")

        st.caption(
            "The probability is the model's estimate for class 1, not a "
            "validated real-world probability of a landslide occurring."
        )

        with st.expander("Review the submitted input values"):
            st.dataframe(input_df, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        st.info(
            "Check that the saved model is the complete fitted preprocessing "
            "pipeline and that the input columns match the training columns."
        )

# -----------------------------
# PROJECT INFORMATION
# -----------------------------
st.divider()
st.subheader("About This Project")

left, right = st.columns(2)

with left:
    st.markdown("**Current capabilities**")
    st.markdown(
        "- Machine Learning-based landslide classification\n"
        "- Numerical and categorical feature inputs\n"
        "- Model probability display, when supported\n"
        "- Interactive prediction interface"
    )

with right:
    st.markdown("**Future enhancements**")
    st.markdown(
        "- Nepal flood-risk prediction using suitable flood data\n"
        "- GIS-based susceptibility maps\n"
        "- Road disruption analysis\n"
        "- Rainfall and weather data integration"
    )

st.warning(
    "Do not use this prototype for evacuation, travel, or emergency decisions. "
    "Operational use requires reliable local data, independent validation, "
    "and review by relevant experts."
)
