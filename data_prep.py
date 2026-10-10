import math
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_FILE = Path(__file__).parent / "ckd-dataset-v2.csv"
LEAKAGE = ["affected", "stage", "grf"]   # these give away the answer

# Data version 2: clinically justified engineered features (PB11)
ENGINEERED = ["comorbidity_count", "urine_sediment_count", "albuminuria_cat"]


def _bin_key(label):
    if label.startswith("<"):
        return (-math.inf, 0)
    if label.startswith("≥"):
        return (float(label[1:]), 1)
    return (float(label.split(" - ")[0]), 0)


def add_engineered_features(df):
    df = df.copy()
    # Number of CKD-related comorbidities: hypertension, diabetes, coronary artery disease (0-3)
    df["comorbidity_count"] = df["htn"] + df["dm"] + df["cad"]
    # Number of abnormal urine sediment findings: red cells, pus cells, pus cell clumps, bacteria (0-4)
    df["urine_sediment_count"] = df["rbc"] + df["pc"] + df["pcc"] + df["ba"]
    # Albuminuria category from the dipstick grade, approximating KDIGO A1/A2/A3 (0 = A1, 1 = A2, 2 = A3)
    df["albuminuria_cat"] = df["al"].clip(upper=2)
    return df


def load_data():
    df = pd.read_csv(DATA_FILE, skiprows=[1, 2], dtype=str)
    df = df.apply(lambda s: s.str.strip())
    df = df.rename(columns={"bp (Diastolic)": "bp"})
    df["grf"] = df["grf"].replace("p", pd.NA)
    raw = df.copy()                      # original range labels, for charts

    for col in df.columns:
        if col in ("class", "stage"):
            continue
        values = df[col].dropna().unique()
        if set(values) <= {"0", "1", "2"}:
            df[col] = df[col].astype(int)
        else:
            order = sorted(values, key=_bin_key)
            df[col] = df[col].map({v: i for i, v in enumerate(order)})

    df["class"] = (df["class"] == "ckd").astype(int)
    df = add_engineered_features(df)
    train, test = train_test_split(df, test_size=0.2,
                                   stratify=df["class"], random_state=42)
    return {"raw": raw, "all": df, "train": train, "test": test}


# FEATURES = the 25 original predictors (data version 1, unchanged)
FEATURES = [c for c in load_data()["all"].columns
            if c not in LEAKAGE + ENGINEERED + ["class"]]

# FEATURES_V2 = original predictors plus the engineered features
FEATURES_V2 = FEATURES + ENGINEERED
