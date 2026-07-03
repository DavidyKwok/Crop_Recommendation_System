import numpy as np
import pandas as pd
from joblib import load

from utils import (
    clean_numeric_columns,
    standardize_soil_type,
    standardize_water_source,
    extract_last_previous_crop,
)

# Sustainability water source weights
WATER_SOURCE_SCORE_MAP = {
    "rainfed": 1.0,
    "river": 0.8,
    "well": 0.6,
    "canal": 0.4,
    "unknown": 0.5,
}


def compute_sustainability_score(field_row, crop_row):
    """
    Soft sustainability scoring:
    - water efficiency (field rain vs crop water need)
    - nutrient efficiency (soil NPK vs crop NPK needs)
    - soil type compatibility
    - water source sustainability
    """

    soil_ph = field_row["soil_ph"]
    ph_min = crop_row["ph_min"]
    ph_max = crop_row["ph_max"]

    # pH penalty: distance outside ideal range
    if soil_ph < ph_min:
        ph_penalty = ph_min - soil_ph
    elif soil_ph > ph_max:
        ph_penalty = soil_ph - ph_max
    else:
        ph_penalty = 0.0

    ph_score = np.exp(-ph_penalty)  # in [0,1]

    # Water efficiency
    field_rain = field_row["forecast_rain_mm"]
    crop_water = crop_row["water_need_mm"]
    water_diff = abs(field_rain - crop_water)
    water_eff_score = np.exp(-water_diff / 200.0)

    # Nutrient efficiency
    n_diff = abs(field_row["nitrogen_kg_ha"] - crop_row["nitrogen_need_kg_ha"])
    p_diff = abs(field_row["phosphorus_kg_ha"] - crop_row["phosphorus_need_kg_ha"])
    k_diff = abs(field_row["potassium_kg_ha"] - crop_row["potassium_need_kg_ha"])
    nutrient_diff = n_diff + p_diff + k_diff
    nutrient_eff_score = np.exp(-nutrient_diff / 150.0) * ph_score

    # Soil type match
    soil_type = field_row["soil_type"]
    preferred = str(crop_row["preferred_soil_types"]).lower()
    soil_type_score = 1.0 if soil_type in preferred else 0.0

    # Water source sustainability
    ws = field_row["water_source"]
    water_source_score = WATER_SOURCE_SCORE_MAP.get(ws, 0.5)

    # Combine
    sustainability = (
        0.35 * water_eff_score
        + 0.35 * nutrient_eff_score
        + 0.20 * soil_type_score
        + 0.10 * water_source_score
    )

    return sustainability


def main():
    #Load model and data
    bundle = load("models/yield_model.joblib")
    pipeline = bundle["pipeline"]
    numeric_features = bundle["numeric_features"]
    categorical_features = bundle["categorical_features"]

    crops = pd.read_csv("crop_profiles_data.csv")
    current = pd.read_csv("current_conditions_data.csv")

    # Clean numeric in crops
    numeric_crop_cols = [
        "ph_min",
        "ph_max",
        "water_need_mm",
        "nitrogen_need_kg_ha",
        "phosphorus_need_kg_ha",
        "potassium_need_kg_ha",
        "maturity_days",
    ]
    crops = clean_numeric_columns(crops, numeric_crop_cols)

    # Clean numeric in current conditions
    numeric_current_cols = [
        "soil_ph",
        "soil_moisture_pct",
        "nitrogen_kg_ha",
        "phosphorus_kg_ha",
        "potassium_kg_ha",
        "forecast_rain_mm_next30d",
        "avg_temp_next30d",
        "precip_history_mm_past90d",
    ]
    current = clean_numeric_columns(current, numeric_current_cols)

    # Rename to match dataset
    current = current.rename(
        columns={
            "forecast_rain_mm_next30d": "forecast_rain_mm",
            "avg_temp_next30d": "avg_temp",
            "precip_history_mm_past90d": "precip_history_mm",
        }
    )

    # Standardize categories
    if "soil_type" in current.columns:
        current["soil_type"] = current["soil_type"].apply(standardize_soil_type)
    else:
        current["soil_type"] = "unknown"

    if "water_source" in current.columns:
        current["water_source"] = current["water_source"].apply(standardize_water_source)
    else:
        current["water_source"] = "unknown"

    if "previous_crops_last_3_years" in current.columns:
        current["previous_crop"] = current["previous_crops_last_3_years"].apply(
            extract_last_previous_crop
        )
    else:
        current["previous_crop"] = "unknown"

    # Build (field, crop) pairs
    rows = []
    for _, f in current.iterrows():
        for _, c in crops.iterrows():
            row = {}

            row["field_id"] = f["field_id"]
            row["crop"] = c["crop"]

            # Field numeric
            for col in [
                "soil_ph",
                "soil_moisture_pct",
                "nitrogen_kg_ha",
                "phosphorus_kg_ha",
                "potassium_kg_ha",
                "forecast_rain_mm",
                "avg_temp",
                "precip_history_mm",
            ]:
                row[col] = f.get(col, np.nan)

            # Categorical
            row["soil_type"] = f["soil_type"]
            row["water_source"] = f["water_source"]
            row["previous_crop"] = f["previous_crop"]

            # Crop requirements
            for col in numeric_crop_cols:
                row[col] = c.get(col, np.nan)

            row["preferred_soil_types"] = c["preferred_soil_types"]

            # For the ML model: which crop is being considered
            row["crop_planted"] = c["crop"]

            rows.append(row)

    data = pd.DataFrame(rows)

    # Recompute pH mismatch (for potential future use)
    data["ph_below_min"] = (data["ph_min"] - data["soil_ph"]).clip(lower=0)
    data["ph_above_max"] = (data["soil_ph"] - data["ph_max"]).clip(lower=0)
    data["ph_out_of_range"] = data["ph_below_min"] + data["ph_above_max"]

    #Predict yield for each (field, crop) pair 
    X_for_model = data[numeric_features + categorical_features].copy()
    X_for_model = X_for_model.replace([np.inf, -np.inf], np.nan)
    X_for_model = X_for_model.fillna(0)

    predicted_yield = pipeline.predict(X_for_model)
    data["predicted_yield_t_ha"] = predicted_yield

    #Compute sustainability for each pair
    sustainability_scores = []
    for idx, row in data.iterrows():
        s = compute_sustainability_score(row, row)  # row has both field & crop info
        sustainability_scores.append(s)

    data["sustainability_score"] = sustainability_scores

    # Normalize per field & compute final score 
    results = []

    for field_id, group in data.groupby("field_id"):
        g = group.copy()

        # Normalize yield (0–1)
        y = g["predicted_yield_t_ha"]
        if y.max() > y.min():
            g["yield_score"] = (y - y.min()) / (y.max() - y.min())
        else:
            g["yield_score"] = 0.5

        # Normalize sustainability (0–1)
        s = g["sustainability_score"]
        if s.max() > s.min():
            g["sustainability_score_norm"] = (s - s.min()) / (s.max() - s.min())
        else:
            g["sustainability_score_norm"] = 0.5

        # Dual scoring
        alpha = 0.5  # sustainability weight
        beta = 0.5   # yield weight
        g["final_score"] = alpha * g["sustainability_score_norm"] + beta * g["yield_score"]

        g_sorted = g.sort_values("final_score", ascending=False)

        top = g_sorted.head(3)
        for _, row in top.iterrows():
            results.append(
                {
                    "field_id": field_id,
                    "crop": row["crop"],
                    "final_score": row["final_score"],
                    "sustainability_score": row["sustainability_score_norm"],
                    "yield_score": row["yield_score"],
                    "predicted_yield_t_ha": row["predicted_yield_t_ha"],
                }
            )

    results_df = pd.DataFrame(results)
    print(results_df)

    results_df.to_csv("crop_recommendations.csv", index=False)
    print("Saved crop_recommendations.csv")


if __name__ == "__main__":
    main()
