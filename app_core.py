"""Shared, cached logic for the Streamlit app. Same methods as the notebooks in notebooks/."""
import numpy as np
import pandas as pd
import streamlit as st
from scipy.stats import fisher_exact, chi2_contingency, mannwhitneyu
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, cross_val_predict, cross_validate
from sklearn.metrics import confusion_matrix, make_scorer, recall_score
from data_prep import load_data, FEATURES, _bin_key
from viz import apply_style

apply_style()

CHEAP = ["age", "bp", "bp limit", "sg", "al", "su", "htn", "dm", "cad", "appet", "pe"]
NAMES = {
    "age": "Age", "bp": "Diastolic BP flag", "bp limit": "BP category", "sg": "Urine specific gravity",
    "al": "Urine albumin", "su": "Urine sugar", "rbc": "Abnormal red cells (urine)",
    "pc": "Abnormal pus cells (urine)", "pcc": "Pus cell clumps (urine)", "ba": "Bacteria (urine)",
    "bgr": "Random blood glucose", "bu": "Blood urea", "sod": "Sodium", "sc": "Serum creatinine",
    "pot": "Potassium", "hemo": "Haemoglobin", "pcv": "Packed cell volume", "rbcc": "Red blood cell count",
    "wbcc": "White blood cell count", "htn": "Hypertension", "dm": "Diabetes",
    "cad": "Coronary artery disease", "appet": "Poor appetite", "pe": "Pedal oedema", "ane": "Anaemia",
}
UNITS = {"age": "years", "bgr": "mg/dL", "bu": "mg/dL", "sc": "mg/dL", "hemo": "g/dL", "sod": "mEq/L",
         "pot": "mEq/L", "pcv": "%", "wbcc": "cells/µL", "rbcc": "millions/µL"}
BINARY = ["htn", "dm", "cad", "ane", "pe", "appet", "rbc", "pc", "pcc", "ba", "bp"]
ORDERED = ["age", "bp limit", "bgr", "bu", "sc", "sod", "pot", "hemo", "pcv", "rbcc", "wbcc", "sg", "al", "su"]
AL_LABELS = {"< 0": "0", "1 - 1": "1", "2 - 2": "2", "3 - 3": "3", "≥ 4": "4+"}


def show(fig):
    """Display a Matplotlib figure in the app, then free its memory."""
    import matplotlib.pyplot as plt
    st.pyplot(fig)
    plt.close(fig)


def lr():
    return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000))


@st.cache_data
def data():
    d = load_data()
    raw = d["raw"].copy()
    raw["group"] = raw["class"].map({"ckd": "CKD", "notckd": "Non-CKD"})
    return d["all"], d["train"], d["test"], raw


def levels(col):
    """Range labels of a variable in medical order (code 0, 1, 2, ...)."""
    _, _, _, raw = data()
    values = raw[col].dropna().unique()
    if set(values) <= {"0", "1", "2"}:
        return sorted(values)
    return sorted(values, key=_bin_key)


def label(col, code):
    lab = levels(col)[int(code)]
    return AL_LABELS.get(lab, lab) if col in ("al", "su") else lab


def code_key(cols):
    """Reference table: what each group code (0, 1, 2, ...) means for the chosen tests."""
    lv = {c: [label(c, i) for i in range(len(levels(c)))] for c in cols}
    if "bp limit" in lv:
        lv["bp limit"] = ["lowest", "middle", "highest"]
    if "al" in lv:
        lv["al"] = ["none (normal)", "1+", "2+", "3+", "4+"]
    if "su" in lv:
        lv["su"] = ["none (normal)", "1–2+", "2+", "3–4+", "4+", "above 4+"]
    n = max(len(v) for v in lv.values())
    table = {"Code": list(range(n))}
    for c in cols:
        head = NAMES[c] + (f" ({UNITS[c]})" if c in UNITS else "")
        table[head] = lv[c] + [""] * (n - len(lv[c]))
    return pd.DataFrame(table).set_index("Code")


CODE_NOTES = (
    "**How to read the codes.** Every lab value in this dataset was recorded as a range, so each range gets a code "
    "in medical order: 0 is the lowest range, and each step up is the next range. \n"
    "- **Yes/no tests** (hypertension, diabetes, coronary artery disease, poor appetite, pedal oedema, BP flag): "
    "0 = no / normal, 1 = yes / raised.  \n"
    "- **BP category:** 0 = lowest, 1 = middle, 2 = highest. The dataset does not give the mmHg cut-offs; "
    "category 2 was only seen in CKD patients.  \n"
    "- **Urine albumin and sugar:** 0 = none found (normal); 1 or more = found on the dipstick (abnormal).  \n"
    "- **Urine specific gravity:** lower codes = more dilute urine. Codes 0–2 (≤ 1.017) are part of the backup "
    "intake rule.  \n"
    "- **CKD stage** (referral table): s1 = earliest, s5 = kidney failure. Early stage here means s2–s3."
)


def sens_spec(y_true, flagged):
    tn, fp, fn, tp = confusion_matrix(y_true, flagged, labels=[0, 1]).ravel()
    return tp / (tp + fn), tn / (tn + fp), int(tp), int(fn), int(fp), int(tn)


# ---------- descriptive ----------
def range_table(col):
    _, _, _, raw = data()
    lv = levels(col)
    counts = pd.crosstab(raw[col], raw["group"]).reindex(lv, fill_value=0)[["Non-CKD", "CKD"]]
    pct = (100 * counts / counts.sum()).round(1)
    pct.index = [AL_LABELS.get(i, i) if col in ("al", "su") else i for i in pct.index]
    return pct


@st.cache_data
def abnormal_table():
    df, _, _, _ = data()
    rules = {
        "Albumin in urine (1+ or more)": ("al", lambda c: c >= 1),
        "High blood urea (≥ 48.1 mg/dL)": ("bu", lambda c: c >= 1),
        "Low haemoglobin (< 11.3 g/dL)": ("hemo", lambda c: c <= 4),
        "Low urine specific gravity (≤ 1.011)": ("sg", lambda c: c <= 1),
        "High serum creatinine (≥ 3.65 mg/dL)": ("sc", lambda c: c >= 1),
        "High random glucose (≥ 196 mg/dL)": ("bgr", lambda c: c >= 3),
    }
    groups = df["class"].map({1: "CKD", 0: "Non-CKD"})
    t = pd.DataFrame({name: (100 * rule(df[col]).groupby(groups).mean()).round(1)
                      for name, (col, rule) in rules.items()}).T[["Non-CKD", "CKD"]]
    return t.sort_values("CKD", ascending=False)


@st.cache_data
def comorbidities():
    _, _, _, raw = data()
    cols = ["htn", "dm", "cad", "ane", "pe", "appet"]
    return pd.DataFrame({g: [round(100 * raw.loc[raw["group"] == g, c].astype(int).mean(), 1) for c in cols]
                         for g in ["Non-CKD", "CKD"]}, index=[NAMES[c] for c in cols])


# ---------- diagnostic ----------
@st.cache_data
def binary_tests():
    df, _, _, _ = data()
    ckd, non = df[df["class"] == 1], df[df["class"] == 0]
    alpha = 0.05 / (len(BINARY) + len(ORDERED))
    rows = []
    for col in BINARY:
        a = int(ckd[col].sum()); b = len(ckd) - a; c = int(non[col].sum()); dd = len(non) - c
        table = np.array([[a, b], [c, dd]])
        p = fisher_exact(table)[1]
        v = np.sqrt(chi2_contingency(table, correction=False)[0] / table.sum())
        odds = ((a + .5) * (dd + .5)) / ((b + .5) * (c + .5))
        rows.append({"Factor": NAMES[col], "Non-CKD %": round(100 * c / len(non), 1),
                     "CKD %": round(100 * a / len(ckd), 1), "Corrected OR": round(odds, 1),
                     "p-value": float(f"{p:.2g}"), "Cramér's V": round(v, 2),
                     "Significant": "Yes" if p < alpha else "No"})
    return pd.DataFrame(rows).sort_values("Cramér's V", ascending=False).reset_index(drop=True)


@st.cache_data
def marker_tests():
    df, _, _, _ = data()
    ckd, non = df[df["class"] == 1], df[df["class"] == 0]
    alpha = 0.05 / (len(BINARY) + len(ORDERED))
    rows = []
    for col in ORDERED:
        u, p = mannwhitneyu(ckd[col], non[col], alternative="two-sided")
        r = 2 * u / (len(ckd) * len(non)) - 1
        rows.append({"Marker": NAMES[col], "Rank-biserial r": round(r, 2),
                     "Direction in CKD": "higher" if r > 0 else "lower",
                     "p-value": float(f"{p:.2g}"), "Significant": "Yes" if p < alpha else "No"})
    t = pd.DataFrame(rows)
    return t.reindex(t["Rank-biserial r"].abs().sort_values(ascending=False).index).reset_index(drop=True)


@st.cache_data
def combinations():
    df, _, _, _ = data()
    ckd = df[df["class"] == 1]
    combo = {"htn": "Hypertension", "dm": "Diabetes", "cad": "CAD", "ane": "Anaemia"}
    lab = ckd.apply(lambda r: " + ".join(n for c, n in combo.items() if r[c] == 1) or "None of these", axis=1)
    return lab.value_counts()


@st.cache_resource
def tree(depth, cols_key):
    _, train, _, _ = data()
    cols = CHEAP if cols_key == "cheap" else FEATURES
    return DecisionTreeClassifier(max_depth=depth, random_state=42).fit(train[cols], train["class"])


# ---------- predictive ----------
@st.cache_resource
def final_models():
    _, train, _, _ = data()
    return {"Full model": (lr().fit(train[FEATURES], train["class"]), FEATURES),
            "Cheap-test model": (lr().fit(train[CHEAP], train["class"]), CHEAP)}


@st.cache_data(show_spinner="Running cross-validation (about 10 seconds)...")
def cv_compare(set_name, repeats=3):
    _, train, _, _ = data()
    cols = {"Cheap tests (11)": CHEAP, "All routine tests (25)": FEATURES}[set_name]
    models = {"Logistic regression": lr(),
              "Decision tree": DecisionTreeClassifier(max_depth=4, random_state=42),
              "Random forest": RandomForestClassifier(n_estimators=200, random_state=42)}
    scoring = {"Sensitivity": "recall", "Specificity": make_scorer(recall_score, pos_label=0),
               "ROC-AUC": "roc_auc", "F1": "f1"}
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=repeats, random_state=42)
    rows = []
    for name, m in models.items():
        s = cross_validate(m, train[cols], train["class"], cv=cv, scoring=scoring)
        rows.append({"Model": name, **{k: round(s[f"test_{k}"].mean(), 3) for k in scoring}})
    return pd.DataFrame(rows).set_index("Model")


# ---------- prescriptive ----------
@st.cache_data
def cv_risk_train(which, repeats=10):
    """Cross-validated risk for the 160 training patients (used to choose thresholds)."""
    _, train, _, _ = data()
    cols = CHEAP if which == "cheap" else FEATURES
    return np.mean([cross_val_predict(lr(), train[cols], train["class"],
                                      cv=StratifiedKFold(5, shuffle=True, random_state=r),
                                      method="predict_proba")[:, 1] for r in range(repeats)], axis=0)


@st.cache_data
def risk_all():
    """Cross-validated full-model risk for all 200 patients (used for referral slots)."""
    df, _, _, _ = data()
    cv5 = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    return pd.Series(cross_val_predict(lr(), df[FEATURES], df["class"], cv=cv5, method="predict_proba")[:, 1],
                     index=df.index)


def threshold_table(risk, y_true):
    rows = []
    for t in np.round(np.arange(0.05, 0.96, 0.05), 2):
        se, sp, *_ = sens_spec(y_true, (risk >= t).astype(int))
        rows.append({"Threshold": t, "Sensitivity": round(se, 3), "Specificity": round(sp, 3)})
    return pd.DataFrame(rows)


def choose_patients(n_slots, min_early, threshold):
    """OR-Tools CP-SAT: maximise total risk of referred patients (Pasanka's model)."""
    df, _, _, _ = data()
    risk = risk_all()
    stage = df["stage"].str.extract(r"(\d)")[0].astype(int)
    early = stage.isin([2, 3])
    eligible = risk >= threshold
    try:
        from ortools.sat.python import cp_model
    except ImportError:
        cp_model = None
    if cp_model is None:  # fallback: same answer for this simple model
        pool = df.index[eligible]
        order = sorted(pool, key=lambda i: (-risk[i], -stage[i]))
        e = [i for i in order if early[i]][:min_early]
        rest = [i for i in order if i not in e][: max(n_slots - len(e), 0)]
        return e + rest, "fallback (sorted list)"
    model = cp_model.CpModel()
    pick = {i: model.NewBoolVar(f"pick_{i}") for i in df.index}
    for i in df.index:
        if not eligible[i]:
            model.Add(pick[i] == 0)
    model.Add(sum(pick.values()) <= n_slots)
    if min_early:
        model.Add(sum(pick[i] for i in df.index if early[i]) >= min_early)
    model.Maximize(sum((int(round(risk[i] * 10000)) * 10 + int(stage[i])) * pick[i] for i in df.index))
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        return [], solver.StatusName(status)
    return [i for i in df.index if solver.Value(pick[i])], solver.StatusName(status)


@st.cache_resource
def modifiable_model():
    _, train, _, _ = data()
    return LogisticRegression(max_iter=2000).fit(train[["bp limit", "bgr", "htn", "dm"]], train["class"])
