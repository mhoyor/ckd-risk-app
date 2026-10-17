import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from app_core import show, data, final_models, cv_compare, sens_spec, NAMES, CHEAP, label
from data_prep import FEATURES
from viz import MODEL_COLORS, MODEL_STYLES, MUTED

df, train, test, raw = data()
models = final_models()

st.title("3 Predictive: can routine tests detect CKD?")
st.caption("User: Health Informatician · Models trained on 160 patients, checked once on 40 unseen patients. "
           "Detects current CKD, not future progression.")

st.subheader("Compare models (repeated cross-validation on the training patients)")
fs = st.radio("Tests used", ["All routine tests (25)", "Cheap tests (11)"], horizontal=True,
              help="Cheap tests = medical history, urine dipstick, blood pressure and age (no blood test).")
st.dataframe(cv_compare(fs))
st.caption("Baseline to beat: always saying 'CKD' gives sensitivity 1.00 but specificity 0.00 (accuracy 0.64). "
           "The notebook uses 10 repeats; the app uses 3 to stay fast, so the third decimal can differ.")

st.subheader("Final models on the 40 test patients (logistic regression)")
rows, probs = [], {}
for name, (model, cols) in models.items():
    p = model.predict_proba(test[cols])[:, 1]
    probs[name] = p
    se, sp, tp, fn, fp, tn = sens_spec(test["class"], (p >= 0.5).astype(int))
    rows.append({"Model": name, "CKD caught": f"{tp} of {tp + fn}", "False alarms": f"{fp} of {fp + tn}",
                 "Sensitivity": round(se, 3), "Specificity": round(sp, 3),
                 "ROC-AUC": round(roc_auc_score(test["class"], p), 3)})
st.dataframe(pd.DataFrame(rows).set_index("Model"))

c1, c2 = st.columns(2)
with c1:
    fig, axes = plt.subplots(1, 2, figsize=(7, 3.2))
    for ax, name in zip(axes, models):
        cm = confusion_matrix(test["class"], (probs[name] >= 0.5).astype(int))
        ax.imshow(cm, cmap="Greys", vmin=0, vmax=cm.max() * 2.5)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=15, fontweight="bold")
        ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred.\nnon-CKD", "Pred.\nCKD"], fontsize=8)
        ax.set_yticks([0, 1]); ax.set_yticklabels(["Actual\nnon-CKD", "Actual\nCKD"], fontsize=8)
        ax.grid(False); ax.set_title(name, fontsize=10)
    fig.tight_layout()
    show(fig)
with c2:
    fig, ax = plt.subplots(figsize=(4.5, 3.6))
    for name in models:
        fpr, tpr, _ = roc_curve(test["class"], probs[name])
        ax.plot(fpr, tpr, MODEL_STYLES[name], color=MODEL_COLORS[name], linewidth=2, label=name)
    ax.plot([0, 1], [0, 1], ":", color=MUTED, linewidth=1)
    ax.set_xlabel("False alarm rate"); ax.set_ylabel("CKD caught"); ax.legend(loc="lower right", fontsize=8)
    ax.set_title("ROC curves (test set)")
    show(fig)

st.subheader("Which tests drive the full model?")
coefs = pd.Series(models["Full model"][0][-1].coef_[0], index=FEATURES)
top = coefs.reindex(coefs.abs().sort_values(ascending=False).index[:12])[::-1]
fig, ax = plt.subplots(figsize=(7.5, 4.2))
ax.barh([NAMES[f] for f in top.index], top.values, height=0.6,
        color=["#eb6834" if v > 0 else "#2a78d6" for v in top.values])
ax.axvline(0, color=MUTED, linewidth=0.8); ax.set_xlabel("Effect on risk (standardised coefficient)")
ax.grid(axis="x"); ax.grid(axis="y", visible=False)
show(fig)
st.caption("Orange: higher value or 'yes' raises the risk. Blue: higher value lowers the risk.")

st.subheader("Score one test patient")
pid = st.selectbox("Test patient (row number in the dataset)", list(test.index))
row = test.loc[[pid]]
c1, c2, c3 = st.columns(3)
c1.metric("Actual diagnosis", "CKD" if row["class"].iloc[0] == 1 else "Non-CKD")
c2.metric("Full model risk", f"{models['Full model'][0].predict_proba(row[FEATURES])[0, 1]:.2f}")
c3.metric("Cheap-test model risk", f"{models['Cheap-test model'][0].predict_proba(row[CHEAP])[0, 1]:.2f}")
key = ["hemo", "sg", "al", "bu", "sc", "htn", "dm"]
st.dataframe(pd.DataFrame({NAMES[c]: [label(c, row[c].iloc[0]) if c not in ("htn", "dm")
                                      else ("yes" if row[c].iloc[0] == 1 else "no")] for c in key}), hide_index=True)
