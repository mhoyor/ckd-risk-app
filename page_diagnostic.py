import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from app_core import show, data, binary_tests, marker_tests, combinations, tree, NAMES, CHEAP
from data_prep import FEATURES
from viz import COLORS, MUTED, plot_readable_tree

df, train, test, raw = data()

st.title("2 Diagnostic: which factors distinguish CKD?")
st.caption("User: Attending Nephrologist · These are associations in one-visit data, not proof of cause.")

b, m = binary_tests(), marker_tests()
c1, c2, c3 = st.columns(3)
c1.metric("Hypertension: CKD vs non-CKD", f"{b.loc[b.Factor == 'Hypertension', 'CKD %'].iloc[0]:.0f}% vs 0%")
c2.metric("Diabetes: CKD vs non-CKD", f"{b.loc[b.Factor == 'Diabetes', 'CKD %'].iloc[0]:.0f}% vs 0%")
c3.metric("Strongest lab difference", "Haemoglobin", f"r = {m.loc[m.Marker == 'Haemoglobin', 'Rank-biserial r'].iloc[0]}",
          delta_color="off")

st.subheader("How strongly does each factor differ? (effect sizes)")
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), gridspec_kw={"width_ratios": [1, 1.15]})
bb = b.sort_values("Cramér's V")
ax1.barh(bb["Factor"], bb["Cramér's V"], height=0.6, edgecolor=COLORS["CKD"],
         color=[COLORS["CKD"] if s == "Yes" else "white" for s in bb["Significant"]])
ax1.set_xlim(0, 1); ax1.set_title("Yes/no factors (Cramér's V)"); ax1.grid(axis="x"); ax1.grid(axis="y", visible=False)
mm = m.sort_values("Rank-biserial r")
cols = [COLORS["CKD"] if r > 0 else COLORS["Non-CKD"] for r in mm["Rank-biserial r"]]
ax2.barh(mm["Marker"], mm["Rank-biserial r"], height=0.6, edgecolor=cols,
         color=[c if s == "Yes" else "white" for c, s in zip(cols, mm["Significant"])])
ax2.axvline(0, color=MUTED, linewidth=0.8); ax2.set_xlim(-1, 1)
ax2.set_title("Lab markers: lower (blue) or higher (orange) in CKD"); ax2.grid(axis="x"); ax2.grid(axis="y", visible=False)
fig.tight_layout()
show(fig)
st.caption("Hollow bars are not significant after Bonferroni correction (25 tests, p < 0.002).")

tab1, tab2 = st.tabs(["Yes/no factors", "Lab markers"])
with tab1:
    st.dataframe(b, hide_index=True)
    st.caption("Corrected OR: 0.5 added to every count, because no non-CKD patient has most of these conditions.")
with tab2:
    st.dataframe(m, hide_index=True)

st.subheader("Which risk-factor combinations are most common in CKD patients?")
combos = combinations().head(10).sort_values()
fig, ax = plt.subplots(figsize=(8, 4.2))
bars = ax.barh(combos.index, combos.values, color=COLORS["CKD"], height=0.6)
for bar, v in zip(bars, combos.values):
    ax.text(bar.get_width() + 0.4, bar.get_y() + bar.get_height() / 2, str(v), va="center", fontsize=8, color=MUTED)
ax.set_xlabel("Number of CKD patients (n = 128)"); ax.grid(axis="x"); ax.grid(axis="y", visible=False)
show(fig)
st.caption("About one in five CKD patients has none of hypertension, diabetes, CAD or anaemia.")

st.subheader("A simple decision tree")
depth = st.slider("Tree depth (number of questions)", 1, 4, 3)
t = tree(depth, "all")
fig, ax = plt.subplots(figsize=(12, 4 + depth))
plot_readable_tree(ax, t, FEATURES, raw=raw, names=NAMES, fontsize=9)
show(fig)
imp = pd.Series(t.feature_importances_, index=[NAMES[f] for f in FEATURES])
st.bar_chart(imp[imp > 0].sort_values(ascending=False))
st.caption("Feature importance: how much each test contributes to the tree's decisions (adds up to 1). "
           "Trained on the 160 training patients.")
