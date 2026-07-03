import pandas as pd
import numpy as np
import re


#Numeric cleaning 

def clean_numeric(x):
    #Extract numeric value from messy strings
    x = str(x)
    cleaned = re.sub(r'[^0-9.\-]', '', x)
    if cleaned.strip() == "":
        return np.nan
    try:
        return float(cleaned)
    except ValueError:
        return np.nan


def clean_numeric_columns(df: pd.DataFrame, cols):
    #Apply clean_numeric to a list of columns if they exist.
    for col in cols:
        if col in df.columns:
            df[col] = df[col].apply(clean_numeric)
    return df



#Categorical cleaning 
def standardize_soil_type(soil: str) -> str:
    #Map messy soil type descriptions to a few categories.
    if not isinstance(soil, str):
        return "unknown"
    s = soil.strip().lower()
    if "clay" in s:
        return "clay"
    if "loam" in s:
        return "loam"
    if "sand" in s:
        return "sandy"
    if "silt" in s:
        return "silty"
    if "peat" in s:
        return "peat"
    return s or "unknown"


def standardize_water_source(ws: str) -> str:
    #Normalize water_source into one of: 'rainfed', 'well', 'canal', 'river'
    if not isinstance(ws, str):
        return "unknown"
    s = ws.strip().lower()
    if "rain" in s:
        return "rainfed"
    if "well" in s:
        return "well"
    if "canal" in s:
        return "canal"
    if "river" in s:
        return "river"
    return s or "unknown"


def extract_last_previous_crop(prev: str) -> str:
    
   # From a string like 'corn, soy, wheat', return the last crop.
    if not isinstance(prev, str):
        return "unknown"
    parts = [p.strip() for p in prev.split(",") if p.strip()]
    return parts[-1].lower() if parts else "unknown"


def inchesTomm(value):  

    #Convert value from inches to millimeters if it contains 'in'.
    if pd.isna(value):
        return np.nan
    
    value_str = str(value).strip()
    
    # Check if value contains 'in'
    if 'in' in value_str.lower():
        # Extract numeric part
        numeric_value = clean_numeric(value_str)
        if pd.notna(numeric_value):
            return round(numeric_value * 25.4, 1)  # 1 inch = 25.4 mm
    
    # If no 'in' try to return as numeric
    numeric_value = clean_numeric(value_str)
    return numeric_value if pd.notna(numeric_value) else value 


def clean_temperature(value):

   # Remove trailing Fahrenheit/Celsius markers and degree symbols from temperature strings
    if pd.isna(value):
        return np.nan

    s = str(value).strip()
    # Remove degree symbol if present
    s = s.replace('°', '').strip()
    # Remove trailing F or C (case-insensitive)
    s = re.sub(r'(?i)[fc]$', '', s).strip()

    return clean_numeric(s)

def validate_yield_column(df: pd.DataFrame, yield_col: str = 'yield_t_ha') -> pd.DataFrame:

   # Validate and clean the yield_t_ha column to ensure all values are between 0 and 10.

    if yield_col not in df.columns:
        return df
    # First clean the column to ensure it's numeric
    df[yield_col] = pd.to_numeric(df[yield_col], errors='coerce')
    
    # Replace values outside 0-10 range with NaN
    df.loc[(df[yield_col] < 0) | (df[yield_col] > 10), yield_col] = np.nan
    
    return df


def enforce_non_negative_columns(df: pd.DataFrame, cols, how: str = 'clip') -> pd.DataFrame:

 #   Ensure the given numeric columns are not negative.
    for col in cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
            if how == 'clip':
                df[col] = df[col].clip(lower=0)
            else:
                df.loc[df[col] < 0, col] = np.nan
    return df


def validate_soil_ph(df: pd.DataFrame, col: str = 'soil_ph', min_ph: float = 1.0, max_ph: float = 14.0, how: str = 'nan') -> pd.DataFrame:

    #Validate soil pH bounds.
    if col not in df.columns:
        return df

    df[col] = pd.to_numeric(df[col], errors='coerce')
    if how == 'clip':
        df[col] = df[col].clip(lower=min_ph, upper=max_ph)
    else:
        df.loc[(df[col] < min_ph) | (df[col] > max_ph), col] = np.nan

    return df

