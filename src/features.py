import pandas as pd

def engineer_features(X: pd.DataFrame) -> pd.DataFrame:
    X = X.copy()

    if "plantingDate" in X.columns:
        X["plantingDate"] = pd.to_datetime(X["plantingDate"], errors="coerce")
        X["planting_month"] = X["plantingDate"].dt.month
        X["planting_dayofyear"] = X["plantingDate"].dt.dayofyear
        X["is_early_season"] = (X["planting_dayofyear"] < 150).astype(int)
        X = X.drop(columns=["plantingDate"])

    if "experiment" in X.columns:
        X = X.drop(columns=["experiment"])

    if "poundsOfNitrogenPerAcre" in X.columns and "irrigationProvided" in X.columns:
        X["nitrogen_x_irrigation"] = (
            X["poundsOfNitrogenPerAcre"] * X["irrigationProvided"]
        )

    if "poundsOfNitrogenPerAcre" in X.columns:
        X["nitrogen_squared"] = X["poundsOfNitrogenPerAcre"] ** 2

    if "location" in X.columns:
        X = X.drop(columns=["location"])

    drop_cols = ["index", "row", "range", "block", "plotLength"]
    for col in drop_cols:
        if col in X.columns:
            X = X.drop(columns=[col])

    return X