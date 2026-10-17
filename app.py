import streamlit as st
from viz import apply_style

st.set_page_config(page_title="CKD Risk – Group 5", page_icon="🩺", layout="wide")
apply_style()

pages = [
    st.Page("page_home.py", title="Overview", icon="🏠", default=True),
    st.Page("page_descriptive.py", title="1 Descriptive", icon="📊"),
    st.Page("page_diagnostic.py", title="2 Diagnostic", icon="🔍"),
    st.Page("page_predictive.py", title="3 Predictive", icon="🤖"),
    st.Page("page_prescriptive.py", title="4 Prescriptive", icon="🧭"),
]
st.navigation(pages).run()
