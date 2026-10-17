# Risk Factor Prediction of Chronic Kidney Disease

Group 5 PROHI, Stockholm University. Data: Enam Medical College, Bangladesh ([UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/857/risk+factor+prediction+of+chronic+kidney+disease)).

The project shows how cheap, routine tests can help a low-resource hospital find chronic kidney disease (CKD) early and decide who to refer. It covers four use cases: descriptive, diagnostic, predictive and prescriptive.

**Live app:** https://group5-ckd.streamlit.app/

## Contents

| File | What it is |
|---|---|
| `ckd-dataset-v2.csv` | The dataset (200 patients) |
| `data_prep.py` | Shared data preparation used by every notebook and the app (removes metadata rows, converts ranges to ordered codes, excludes leakage columns, stratified 80/20 split) |
| `viz.py` | Shared chart style (blue = non-CKD, orange = CKD) |
| `notebooks/01_descriptive.ipynb` | Use case 1: baseline patient profile |
| `notebooks/02_diagnostic.ipynb` | Use case 2: factors that distinguish CKD |
| `notebooks/03_predictive.ipynb` | Use case 3: CKD risk models |
| `notebooks/04_prescriptive.ipynb` | Use case 4: referral pathway, triage rules and OR-Tools slot allocation |
| `notebooks/05_feature_engineering.ipynb` | Engineered features with clinical rationale |
| `app.py` and `page_*.py` | Streamlit app (one page per use case); `app_core.py` holds the shared logic |
| `requirements.txt` | Python packages needed |

## How to run the notebooks (Google Colab, nothing to install)

1. Open a notebook on GitHub and click **Open in Colab**, or upload it in Colab with File → Upload notebook.
2. In the first code cell, set `GITHUB_USER = "mhoyor"`.
3. Click Runtime → Run all. Each notebook shows the expected results next to every step.

## How to run the app

**Online:** open the live app link above.

**In GitHub Codespaces (in the browser):**
1. On this repository page, click Code → Codespaces → Create codespace on main.
2. In the terminal, run:
```
pip install -r requirements.txt
streamlit run app.py
```
3. Click "Open in Browser" when the pop-up appears.

**On your own computer (Python 3.10 or newer):** the same two commands in a terminal inside this folder.

## Important limitations

- The non-CKD group has no comorbidities and no abnormal urine results, so results are probably better than in a typical clinic.
- One visit per patient: the models detect current CKD, not future progression.
- All lab values are given as ranges, not exact numbers.
- 200 patients from one hospital. The tool supports clinical judgement; it does not replace it.
