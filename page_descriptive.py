import matplotlib.pyplot as plt
import streamlit as st
from app_core import show, data, range_table, abnormal_table, comorbidities, NAMES, UNITS
from viz import grouped_bar, grouped_barh

df, train, test, raw = data()

st.title("1 Descriptive: baseline patient profile")
st.caption("User: Clinical Data Analyst · Question: what is happening in our patient population?")

c1, c2, c3, c4 = st.columns(4)
abn = abnormal_table()
com = comorbidities()
c1.metric("CKD patients with albumin in urine", f"{abn.loc['Albumin in urine (1+ or more)', 'CKD']:.0f}%")
c2.metric("CKD patients clearly anaemic", f"{abn.loc['Low haemoglobin (< 11.3 g/dL)', 'CKD']:.0f}%")
c3.metric("CKD patients with hypertension", f"{com.loc['Hypertension', 'CKD']:.0f}%")
c4.metric("Non-CKD patients with any of these", "0%")

st.subheader("Share of patients with clearly abnormal results")
fig, ax = plt.subplots(figsize=(8, 3.8))
grouped_barh(ax, abn, "")
show(fig)
st.caption("Cut-offs: WHO anaemia threshold, KDIGO albuminuria, ADA glucose threshold; nearest range boundary on the "
           "abnormal side, so the shares are conservative.")

st.subheader("Explore one variable")
choices = ["hemo", "sg", "al", "bu", "sc", "bgr", "age", "bp limit", "bp"]
col = st.selectbox("Variable", choices, key="desc_var", format_func=lambda c: NAMES[c] + (f" ({UNITS[c]})" if c in UNITS else ""))
table = range_table(col)
fig, ax = plt.subplots(figsize=(8, 3.6))
grouped_bar(ax, table, f"{NAMES[col]} by group", "Range (low → high)")
show(fig)
with st.expander("Show the table (% of patients in each group)"):
    st.dataframe(table)

st.subheader("Comorbidities and symptoms")
fig, ax = plt.subplots(figsize=(8, 3.4))
grouped_bar(ax, com, "None of these conditions occurs in non-CKD patients", "")
ax.set_xticklabels(com.index, rotation=20, ha="right")
show(fig)

st.info("**Headline:** two in three CKD patients have protein in the urine and more than half are clearly anaemic – "
        "both can be found with cheap tests (urine dipstick, haemoglobin).")
