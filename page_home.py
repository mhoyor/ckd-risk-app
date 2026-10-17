import streamlit as st
from app_core import data

df, train, test, raw = data()

st.title("Risk Factor Prediction of Chronic Kidney Disease")
st.caption("Group 5 PROHI · Stockholm University · Data: Enam Medical College, Bangladesh (UCI Machine Learning Repository)")

st.markdown(
    "Chronic kidney disease (CKD) is often silent until late stages, and in low-resource settings dialysis and "
    "transplants are out of reach for most patients. This app shows how **cheap, routine tests** can help a hospital "
    "find CKD early and decide who to refer."
)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Patients", len(df))
c2.metric("With CKD", f"{int(df['class'].sum())} ({100 * df['class'].mean():.0f}%)")
c3.metric("CKD found by the pathway (test set)", "26 of 26")
c4.metric("False referrals (test set)", "0 of 14")

st.subheader("Four users, four questions")
st.markdown(
    "| Page | User | Question |\n"
    "|---|---|---|\n"
    "| 1 Descriptive | Clinical Data Analyst | What does the typical CKD patient look like compared with non-CKD? |\n"
    "| 2 Diagnostic | Attending Nephrologist | Which factors distinguish CKD patients, and how strongly? |\n"
    "| 3 Predictive | Health Informatician | Can routine tests detect CKD, and how reliable is the risk score? |\n"
    "| 4 Prescriptive | Clinical Care Coordinator | Who should be sent for blood tests and referred, and who gets the limited slots? |"
)

st.subheader("Read the results with care")
st.warning(
    "- The non-CKD group has **no comorbidities and no abnormal urine results**, so it looks like healthy volunteers. "
    "Results are probably better than they would be in a normal clinic.\n"
    "- One visit per patient: the models detect **current** CKD; they cannot predict future progression.\n"
    "- All lab values are given as **ranges**, not exact numbers.\n"
    "- 200 patients from one hospital. This tool supports clinical judgement; it does not replace it."
)
st.caption("Code and notebooks: github.com/mhoyor/ckd-risk-app")
