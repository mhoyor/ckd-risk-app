import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from app_core import (show, data, final_models, cv_risk_train, risk_all, threshold_table, sens_spec, choose_patients,
                      modifiable_model, levels, label, NAMES, CHEAP)
from data_prep import FEATURES
from viz import MUTED, TEXT

df, train, test, raw = data()
models = final_models()
cheap_model, full_model = models["Cheap-test model"][0], models["Full model"][0]

st.title("4 Prescriptive: who to test, who to refer")
st.caption("User: Clinical Care Coordinator · Two-step pathway: cheap checks for everyone → blood tests only for "
           "flagged patients → referral. Combined work of Pasanka and Victor.")

# ---------------- thresholds and pathway ----------------
st.subheader("Set the two thresholds")
c1, c2 = st.columns(2)
s_t = c1.slider("Step 1: send for blood tests if cheap-test risk is at least", 0.05, 0.95, 0.10, 0.05, key="thr_step1")
r_t = c2.slider("Step 2: refer if full-model risk is at least", 0.05, 0.95, 0.30, 0.05, key="thr_step2")
st.caption("Recommended: 0.10 at step 1 (miss nobody; a false alarm only costs a blood test) and 0.30 at step 2 "
           "(best balance of sensitivity and specificity, Youden's index).")

cheap_cv, full_cv = cv_risk_train("cheap"), cv_risk_train("full")
se1, sp1, *_ = sens_spec(train["class"], (cheap_cv >= s_t).astype(int))
se2, sp2, *_ = sens_spec(train["class"], (full_cv >= r_t).astype(int))
c1.write(f"Training patients (cross-validated): sensitivity **{se1:.3f}**, specificity **{sp1:.3f}**")
c2.write(f"Training patients (cross-validated): sensitivity **{se2:.3f}**, specificity **{sp2:.3f}**")

fig, axes = plt.subplots(1, 2, figsize=(12, 3.6), sharey=True)
for ax, risk, t, title in [(axes[0], cheap_cv, s_t, "Step 1: cheap-test model"),
                           (axes[1], full_cv, r_t, "Step 2: full model")]:
    tab = threshold_table(risk, train["class"])
    ax.plot(tab["Threshold"], tab["Sensitivity"], "-", color="#6250d6", linewidth=2, label="Sensitivity (CKD caught)")
    ax.plot(tab["Threshold"], tab["Specificity"], "--", color="#1baf7a", linewidth=2, label="Specificity (healthy not flagged)")
    ax.axvline(t, color=TEXT, linewidth=1)
    ax.axhline(0.95, color=MUTED, linewidth=0.8, linestyle=":")
    ax.set_title(title); ax.set_xlabel("Threshold"); ax.set_ylim(0.35, 1.03)
axes[0].legend(loc="lower right", fontsize=8)
fig.tight_layout()
show(fig)

p_cheap = cheap_model.predict_proba(test[CHEAP])[:, 1]
p_full = full_model.predict_proba(test[FEATURES])[:, 1]
flagged = p_cheap >= s_t
referred = flagged & (p_full >= r_t)
y = test["class"].values
st.markdown("**Pathway on the 40 test patients with these thresholds**")
m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Screened", len(test))
m2.metric("Sent for blood tests", int(flagged.sum()), f"{int((~flagged).sum())} tests saved", delta_color="off")
m3.metric("Referred", int(referred.sum()))
m4.metric("CKD found", f"{int((referred & (y == 1)).sum())} of {int(y.sum())}")
m5.metric("False referrals", int((referred & (y == 0)).sum()))

# ---------------- intake check ----------------
st.subheader("Check a new patient at intake (cheap tests only)")
defaults = {"age": 5, "bp": 0, "bp limit": 0, "sg": 4, "al": 0, "su": 0}
values = {}
cols = st.columns(3)
for k, col in enumerate(["age", "bp limit", "bp", "sg", "al", "su"]):
    opts = list(range(len(levels(col))))
    values[col] = cols[k % 3].selectbox(NAMES[col], opts, index=defaults[col], format_func=lambda i, c=col: label(c, i),
                                         key=f"intake_{col}")
cols = st.columns(5)
for k, col in enumerate(["htn", "dm", "cad", "appet", "pe"]):
    values[col] = int(cols[k].toggle(NAMES[col], value=False, key=f"intake_{col}"))
patient = pd.DataFrame([values])[CHEAP]
risk = cheap_model.predict_proba(patient)[0, 1]
rule_a = values["al"] >= 1 or values["htn"] == 1 or values["sg"] <= 2
c1, c2 = st.columns(2)
c1.metric("Cheap-test risk", f"{risk:.2f}")
if risk >= s_t:
    c1.error(f"Send for blood tests (risk ≥ {s_t:.2f})")
else:
    c1.success(f"No blood test needed now (risk < {s_t:.2f})")
c2.markdown("**Backup intake rule (if no computer is available)**  \n"
            "Send for blood tests if urine albumin is 1+ or more, OR hypertension, OR urine specific gravity ≤ 1.017.")
c2.write("Rule says: **send for blood tests**" if rule_a else "Rule says: **no blood test needed now**")

# ---------------- referral slots ----------------
st.subheader("Who gets the limited referral slots? (OR-Tools)")
c1, c2 = st.columns(2)
n_slots = c1.slider("Number of referral slots", 1, 30, 10, key="slots_n")
share = c2.slider("Minimum share of slots for early-stage patients (stage 2–3)", 0, 100, 50, 10, key="slots_share",
                  help="Pasanka's rule: early-stage patients can still benefit most from referral. Set to 0 to switch off.")
ids, status = choose_patients(n_slots, int(np.ceil(n_slots * share / 100)), r_t)
ra = risk_all()
eligible = int((ra >= r_t).sum())
if ids:
    chosen = pd.DataFrame({"Risk": ra[ids].round(3), "Stage": df.loc[ids, "stage"],
                           "Actual": df.loc[ids, "class"].map({1: "CKD", 0: "Non-CKD"})}).sort_values(
        ["Risk", "Stage"], ascending=False)
    k1, k2, k3 = st.columns(3)
    k1.metric("Eligible patients (risk ≥ step-2 threshold)", eligible)
    k2.metric("Early-stage patients chosen", int(chosen["Stage"].isin(["s2", "s3"]).sum()))
    k3.metric("Chosen patients with CKD", f"{int((chosen['Actual'] == 'CKD').sum())} of {len(chosen)}")
    st.dataframe(chosen)
    st.caption(f"Solver status: {status}. Objective: maximise total risk; ties broken towards the more advanced stage. "
               "Risk scores are cross-validated, so each patient was scored by a model that did not see them.")
else:
    st.warning(f"No solution ({status}). Lower the early-stage share or the step-2 threshold.")

# ---------------- what-if ----------------
st.subheader("What-if: controlling blood pressure and glucose")
mod = modifiable_model()
c1, c2, c3, c4 = st.columns(4)
bp = c1.selectbox("BP category", list(range(len(levels("bp limit")))), index=1,
                  format_func=lambda i: label("bp limit", i), key="whatif_bp")
glu = c2.selectbox("Random glucose (mg/dL)", list(range(len(levels("bgr")))), index=2,
                   format_func=lambda i: label("bgr", i), key="whatif_glu")
htn = int(c3.toggle("Hypertension", value=False, key="whatif_htn"))
dm = int(c4.toggle("Diabetes", value=False, key="whatif_dm"))
X = pd.DataFrame([[bp, glu, htn, dm]], columns=["bp limit", "bgr", "htn", "dm"])
now = mod.predict_proba(X)[0, 1]
ctrl = X.copy(); ctrl["bp limit"] = 0; ctrl["bgr"] = 0
after = mod.predict_proba(ctrl)[0, 1]
k1, k2 = st.columns(2)
k1.metric("Model risk now", f"{now:.2f}")
k2.metric("If BP category 0 and glucose in the lowest range", f"{after:.2f}", f"{after - now:+.2f}", delta_color="inverse")
odds = pd.Series(np.exp(mod.coef_[0]), index=["BP category (per step)", "Glucose (per range step)",
                                              "Hypertension", "Diabetes"]).round(2)
st.dataframe(odds.rename("Odds ratio"))
st.caption("Associations from one-visit data: this shows how the model's risk responds, not proof that treatment "
           "works. Controlling glucose does not remove a diabetes diagnosis.")
