
import os
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from joblib import dump

from utils import *


def main():

    #Load data 
    hist = pd.read_csv("field_historical_data.csv")
    crops = pd.read_csv("crop_profiles_data.csv")

    #Clear numeric columns in historical data 
    numeric_hist_cols = [
        "soil_ph",
        "soil_moisture_pct",
        "nitrogen_kg_ha",
        "phosphorus_kg_ha",
        "potassium_kg_ha",
        "forecast_rain_mm",
        "avg_temp",
        "precip_history_mm",
        "yield_t_ha",
    ]
    hist = clean_numeric_columns(hist, numeric_hist_cols)
    
    # Validate yield column (ensure values are between 0 and 10)
    hist = validate_yield_column(hist, 'yield_t_ha')
    
    # Drop rows with no yields
    hist = hist.dropna(subset=["yield_t_ha"])

    #Merge with crop profiles
    df = hist.merge(
        crops,
        left_on="crop_planted",
        right_on="crop",
        how="left",
        suffixes=("", "_crop"),
    )

    #Standardize categorical fields 
    if "soil_type" in df.columns:
        df["soil_type"] = df["soil_type"].apply(standardize_soil_type)
    else:
        df["soil_type"] = "unknown"

    if "water_source" in df.columns:
        df["water_source"] = df["water_source"].apply(standardize_water_source)
    else:
        df["water_source"] = "unknown"

    if "previous_crops_last_3_years" in df.columns:
        df["previous_crop"] = df["previous_crops_last_3_years"].apply(extract_last_previous_crop)
    else:
        df["previous_crop"] = "unknown"


    

    #Numeric features (field) 
    numeric_features_field = [
        "soil_ph",
        "soil_moisture_pct",
        "nitrogen_kg_ha",
        "phosphorus_kg_ha",
        "potassium_kg_ha",
        "forecast_rain_mm",
        "avg_temp",
        "precip_history_mm",
    ]

    #Numeric features (crop requirements) 
    numeric_features_crop = [
        "ph_min",
        "ph_max",
        "water_need_mm",
        "nitrogen_need_kg_ha",
        "phosphorus_need_kg_ha",
        "potassium_need_kg_ha",
        "maturity_days",
    ]

    # Ensure numeric
    df = clean_numeric_columns(df, numeric_features_crop)

    # Extra engineered feature: pH mismatch
    df["ph_below_min"] = (df["ph_min"] - df["soil_ph"]).clip(lower=0)
    df["ph_above_max"] = (df["soil_ph"] - df["ph_max"]).clip(lower=0)
    df["ph_out_of_range"] = df["ph_below_min"] + df["ph_above_max"]

    numeric_features_extra = ["ph_below_min", "ph_above_max", "ph_out_of_range"]

    numeric_features = numeric_features_field + numeric_features_crop + numeric_features_extra
    numeric_features = [c for c in numeric_features if c in df.columns]

    # Categorical features 
    categorical_features = [
        "soil_type",
        "water_source",
        "previous_crop",
        "crop_planted",  # which crop is actually grown
    ]
    categorical_features = [c for c in categorical_features if c in df.columns]

    # Prepare X, y 
    X = df[numeric_features + categorical_features].copy()
    y = df["yield_t_ha"].astype(float)

    # Drop rows where all numeric features are NaN
    X = X.replace([np.inf, -np.inf], np.nan)
    mask_valid = X[numeric_features].notna().any(axis=1)
    X = X[mask_valid]
    y = y[mask_valid]

    print(f"Training samples: {len(X)}")

    # Preprocessing pipelines 
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )

    pipeline = Pipeline(
        steps=[
            ("preprocess", preprocessor),
            ("model", model),
        ]
    )

    pipeline.fit(X, y)

    os.makedirs("models", exist_ok=True)
    from joblib import dump

    dump(
        {
            "pipeline": pipeline,
            "numeric_features": numeric_features,
            "categorical_features": categorical_features,
        },
        "models/yield_model.joblib",
    )

    print("Yield model trained and saved to models/yield_model.joblib")


if __name__ == "__main__":
    main()
