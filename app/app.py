import os
import sys
import pickle
from datetime import date, timedelta

import pandas as pd
import streamlit as st

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.features import engineer_features

MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "xgb_pipeline.pkl")


@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model


@st.cache_data
def load_raw_data():
    data_path = os.path.join(
        PROJECT_ROOT,
        "data",
        "raw",
        "2022",
        "DataPublication_final",
        "GroundTruth",
        "HYBRID_HIPS_V3.5_ALLPLOTS.csv",
    )
    df = pd.read_csv(data_path)
    df = df.dropna(subset=["yieldPerAcre"]).copy()
    return df


def build_scenario_row(
    base_row: pd.DataFrame,
    planting_date: date,
    nitrogen: float,
    irrigation: int,
    nitrogen_treatment: str,
) -> pd.DataFrame:
    row = base_row.copy()

    if "plantingDate" in row.columns:
        row["plantingDate"] = pd.to_datetime(planting_date)

    if "poundsOfNitrogenPerAcre" in row.columns:
        row["poundsOfNitrogenPerAcre"] = nitrogen

    if "irrigationProvided" in row.columns:
        row["irrigationProvided"] = irrigation

    if "nitrogenTreatment" in row.columns:
        row["nitrogenTreatment"] = nitrogen_treatment

    return row


def predict_yield(model, row: pd.DataFrame) -> float:
    X = row.drop(columns=["yieldPerAcre"], errors="ignore")
    X = engineer_features(X)
    pred = model.predict(X)[0]
    return float(pred)


st.set_page_config(page_title="Corn Yield What-If Tool", layout="wide")
st.title("🌽 Corn Yield What-If Simulator")
st.write("Adjust planting date, nitrogen, and irrigation to estimate predicted yield.")

try:
    model = load_model()
except FileNotFoundError:
    st.error("Model file not found. Save your pipeline first to models/xgb_pipeline.pkl")
    st.stop()

try:
    raw_df = load_raw_data()
except FileNotFoundError:
    st.error("Raw data file not found. Check the training CSV path in load_raw_data().")
    st.stop()

st.sidebar.header("Inputs")

row_index = st.sidebar.number_input(
    "Baseline row index",
    min_value=0,
    max_value=len(raw_df) - 1,
    value=0,
    step=1,
)

planting_date = st.sidebar.date_input(
    "Planting Date",
    value=date(2022, 5, 1),
    min_value=date(2022, 4, 1),
    max_value=date(2022, 6, 30),
)

nitrogen = st.sidebar.slider(
    "Nitrogen (lbs/acre)",
    min_value=0,
    max_value=300,
    value=150,
    step=5,
)

irrigation = st.sidebar.selectbox(
    "Irrigation Provided",
    options=[0, 1],
    format_func=lambda x: "Yes" if x == 1 else "No",
)

nitrogen_treatment = st.sidebar.selectbox(
    "Nitrogen Treatment",
    options=["Low", "Medium", "High"],
)

baseline_row = raw_df.iloc[[row_index]].copy()

baseline_df = build_scenario_row(
    base_row=baseline_row,
    planting_date=planting_date,
    nitrogen=nitrogen,
    irrigation=irrigation,
    nitrogen_treatment=nitrogen_treatment,
)

baseline_pred = predict_yield(model, baseline_df)

st.subheader("Baseline Prediction")
st.metric("Predicted Yield (bushels/acre)", f"{baseline_pred:.2f}")

with st.expander("See baseline inputs"):
    st.dataframe(baseline_df, width="stretch")

st.divider()

st.subheader("What-If Scenarios")

col1, col2, col3 = st.columns(3)

with col1:
    earlier_date = planting_date - timedelta(days=14)
    earlier_df = build_scenario_row(
        base_row=baseline_row,
        planting_date=earlier_date,
        nitrogen=nitrogen,
        irrigation=irrigation,
        nitrogen_treatment=nitrogen_treatment,
    )
    earlier_pred = predict_yield(model, earlier_df)
    st.metric(
        "Plant 2 weeks earlier",
        f"{earlier_pred:.2f}",
        delta=f"{earlier_pred - baseline_pred:.2f}",
    )

with col2:
    higher_n_df = build_scenario_row(
        base_row=baseline_row,
        planting_date=planting_date,
        nitrogen=min(nitrogen + 30, 300),
        irrigation=irrigation,
        nitrogen_treatment=nitrogen_treatment,
    )
    higher_n_pred = predict_yield(model, higher_n_df)
    st.metric(
        "Nitrogen +30 lbs/acre",
        f"{higher_n_pred:.2f}",
        delta=f"{higher_n_pred - baseline_pred:.2f}",
    )

with col3:
    irrigated_df = build_scenario_row(
        base_row=baseline_row,
        planting_date=planting_date,
        nitrogen=nitrogen,
        irrigation=1,
        nitrogen_treatment=nitrogen_treatment,
    )
    irrigated_pred = predict_yield(model, irrigated_df)
    st.metric(
        "Turn irrigation on",
        f"{irrigated_pred:.2f}",
        delta=f"{irrigated_pred - baseline_pred:.2f}",
    )

st.divider()

scenario_df = pd.DataFrame(
    {
        "Scenario": [
            "Baseline",
            "Plant 2 weeks earlier",
            "Nitrogen +30 lbs/acre",
            "Irrigation On",
        ],
        "Predicted Yield": [
            baseline_pred,
            earlier_pred,
            higher_n_pred,
            irrigated_pred,
        ],
        "Change vs Baseline": [
            0.0,
            earlier_pred - baseline_pred,
            higher_n_pred - baseline_pred,
            irrigated_pred - baseline_pred,
        ],
    }
)

st.subheader("Scenario Comparison")
st.dataframe(
    scenario_df.style.format({
        "Predicted Yield": "{:.2f}",
        "Change vs Baseline": "{:.2f}",
    }),
    width="stretch",
)

st.caption("Predictions are based on the trained XGBoost pipeline and engineered agronomic features.")