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


def build_scenario_rows(
    base_rows: pd.DataFrame,
    planting_date: date,
    nitrogen: float,
    irrigation: int,
    nitrogen_treatment: str,
) -> pd.DataFrame:
    rows = base_rows.copy()

    if "plantingDate" in rows.columns:
        rows["plantingDate"] = pd.to_datetime(planting_date)

    if "poundsOfNitrogenPerAcre" in rows.columns:
        rows["poundsOfNitrogenPerAcre"] = nitrogen

    if "irrigationProvided" in rows.columns:
        rows["irrigationProvided"] = irrigation

    if "nitrogenTreatment" in rows.columns:
        rows["nitrogenTreatment"] = nitrogen_treatment

    return rows


def predict_yield(model, rows: pd.DataFrame) -> float:
    X = rows.drop(columns=["yieldPerAcre"], errors="ignore")
    X = engineer_features(X)
    preds = model.predict(X)
    return float(preds.mean())


st.set_page_config(page_title="Corn Yield What-If Tool", layout="wide")
st.title("🌽 Corn Yield What-If Simulator")
st.write("Adjust planting date, nitrogen, irrigation, and plot selection to estimate predicted yield.")

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

location_options = ["All"] + sorted(raw_df["location"].dropna().unique().tolist())
selected_location = st.sidebar.selectbox(
    "Location",
    options=location_options,
)

plot_count_mode = st.sidebar.selectbox(
    "Number of plots",
    options=["1", "10", "50", "100", "All"],
    index=2,
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

# Filter by location
if selected_location == "All":
    filtered_df = raw_df.copy()
else:
    filtered_df = raw_df[raw_df["location"] == selected_location].copy()

if filtered_df.empty:
    st.error("No rows found for the selected location.")
    st.stop()

# Select number of plots
if plot_count_mode == "All":
    baseline_rows = filtered_df.copy()
else:
    n_plots = min(int(plot_count_mode), len(filtered_df))
    baseline_rows = filtered_df.sample(n=n_plots, random_state=42).copy()

st.caption(
    f"Using {len(baseline_rows)} plot(s) from "
    f"{selected_location if selected_location != 'All' else 'all locations'}."
)

baseline_df = build_scenario_rows(
    base_rows=baseline_rows,
    planting_date=planting_date,
    nitrogen=nitrogen,
    irrigation=irrigation,
    nitrogen_treatment=nitrogen_treatment,
)

baseline_pred = predict_yield(model, baseline_df)

st.subheader("Baseline Prediction")
st.metric("Predicted Yield (bushels/acre)", f"{baseline_pred:.2f}")

with st.expander("See baseline inputs"):
    preview_cols = [
        col for col in [
            "location",
            "plantingDate",
            "poundsOfNitrogenPerAcre",
            "irrigationProvided",
            "nitrogenTreatment",
            "yieldPerAcre",
        ]
        if col in baseline_df.columns
    ]
    st.dataframe(baseline_df[preview_cols].head(20), width="stretch")

st.divider()

st.subheader("What-If Scenarios")

col1, col2, col3 = st.columns(3)

with col1:
    earlier_df = build_scenario_rows(
        base_rows=baseline_rows,
        planting_date=planting_date - timedelta(days=14),
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
    higher_n_df = build_scenario_rows(
        base_rows=baseline_rows,
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
    irrigated_df = build_scenario_rows(
        base_rows=baseline_rows,
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

st.divider()
st.subheader("Nitrogen Response Curve")

nitrogen_values = list(range(0, 301, 10))
nitrogen_curve_rows = []

for n in nitrogen_values:
    curve_df = build_scenario_rows(
        base_rows=baseline_rows,
        planting_date=planting_date,
        nitrogen=n,
        irrigation=irrigation,
        nitrogen_treatment=nitrogen_treatment,
    )
    pred = predict_yield(model, curve_df)

    nitrogen_curve_rows.append({
        "Nitrogen": n,
        "Predicted Yield": pred,
    })

nitrogen_curve_df = pd.DataFrame(nitrogen_curve_rows)

st.line_chart(
    nitrogen_curve_df,
    x="Nitrogen",
    y="Predicted Yield",
    width="stretch",
)

best_row = nitrogen_curve_df.loc[nitrogen_curve_df["Predicted Yield"].idxmax()]

st.metric(
    "Best predicted nitrogen level",
    f"{int(best_row['Nitrogen'])} lbs/acre",
    delta=f"Peak predicted yield: {best_row['Predicted Yield']:.2f}"
)

st.divider()
st.subheader("Planting Date Response Curve")

date_curve_rows = []

start_date = date(2022, 4, 1)
end_date = date(2022, 6, 30)
step_days = 7

current_date = start_date
while current_date <= end_date:
    curve_df = build_scenario_rows(
        base_rows=baseline_rows,
        planting_date=current_date,
        nitrogen=nitrogen,
        irrigation=irrigation,
        nitrogen_treatment=nitrogen_treatment,
    )
    pred = predict_yield(model, curve_df)

    date_curve_rows.append({
        "Planting Date": current_date,
        "Predicted Yield": pred,
    })

    current_date += timedelta(days=step_days)

date_curve_df = pd.DataFrame(date_curve_rows)

st.line_chart(
    date_curve_df,
    x="Planting Date",
    y="Predicted Yield",
    width="stretch",
)

best_date_row = date_curve_df.loc[date_curve_df["Predicted Yield"].idxmax()]

st.metric(
    "Best predicted planting date",
    best_date_row["Planting Date"].strftime("%Y-%m-%d"),
    delta=f"Peak predicted yield: {best_date_row['Predicted Yield']:.2f}"
)

st.caption(
    "This curve shows how predicted yield changes as planting date shifts while keeping the other selected conditions fixed."
)

st.caption("Predictions are based on the trained XGBoost pipeline and engineered agronomic features.")