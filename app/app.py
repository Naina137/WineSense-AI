import streamlit as st
import joblib
import numpy as np
import os
st.set_page_config(
    page_title="Wine Quality Prediction System",
    layout="centered",
    initial_sidebar_state="collapsed")
st.markdown("""
<style>@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:wght@500;600;700&family=Inter:wght@400;500&display=swap');
.stApp{
    background: linear-gradient(135deg,#0F172A,#1E293 B);
}
[data-testid="stAppViewContainer"]{
    background: linear-gradient(135deg,#0F172A,#1E293 B);
}
[data-testid="stHeader"]{
    background: transparent;
}
.main{
    background: transparent;
}
.block-container{
    max-width:1100px;
    padding-top:2rem;
    padding-bottom:2rem;
    background: transparent;
}
.title{
    font-family:'Cormorant Garamond',serif;
    font-size:56px;
    font-weight:700;
    text-align:center;
    color:#F9FAFB;
    margin-bottom:18px;
}
.subtitle{
    font-family:'Cormorant Garamond',serif;
    font-size:30px;
    font-weight:600;
    text-align:center;
    color:#E5E7EB;
    margin-bottom:25px;
}
.desc{
    font-family:'Inter',sans-serif;
    text-align:center;
    font-size:17px;
    line-height:1.8;
    color:#D1D5DB;
    max-width:900px;
    margin:auto;
    text-align:center;
}
div[data-testid="stNumberInput"] input{
    background:#2A2E39;
    color:white;
    border:1px solid #3B4252;
    border-radius:10px;
    font-size:17px;
    font-weight:500;
}

div[data-testid="stNumberInput"] label{
    color:#F3F4F6;
    font-size:17px;
    font-weight:600;
}
.stButton > button{
    width:100%;
    height:55px;
    background:#B91C1C !important;
    color:#FFFFFF !important;
    border:none !important;
    border-radius:10px;
    font-size:18px;
    font-weight:700;
    transition:0.3s;
}
.stButton > button:hover{
    background:#991B1B !important;
    color:#FFFFFF !important;
}
div[data-baseweb="input"]{
    background-color:#2B3442;
    border-radius:8px;
}
div[data-baseweb="input"] input{
    color:white;
}
</style>
<div class="title">
AI-Powered Wine Quality Prediction System
</div>
<div class="subtitle">
Machine Learning-Based Wine Quality Assessment
</div>
<div class="desc">
Predict the quality of red wine using a trained Random Forest machine learning model.
</div>
<br>
<div class="desc">
Enter the physicochemical properties below and click Predict Wine Quality to estimate the quality score.
</div>""", unsafe_allow_html=True)
model_path = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "models",
        "wine_quality_model.pkl"
    )
)
model = joblib.load(model_path)
st.divider()
fixed_acidity = st.number_input("Fixed Acidity", value=7.40, format="%.2f")
volatile_acidity = st.number_input("Volatile Acidity", value=0.70, format="%.2f")
citric_acid = st.number_input("Citric Acid", value=0.00, format="%.2f")
residual_sugar = st.number_input("Residual Sugar", value=1.90, format="%.2f")
chlorides = st.number_input("Chlorides", value=0.076, format="%.3f")
free_sulfur_dioxide = st.number_input("Free Sulfur Dioxide", value=11)
total_sulfur_dioxide = st.number_input("Total Sulfur Dioxide", value=34)
density = st.number_input("Density", value=0.9978, format="%.4f")
pH = st.number_input("pH", value=3.51, format="%.2f")
sulphates = st.number_input("Sulphates", value=0.56, format="%.2f")
alcohol = st.number_input("Alcohol", value=9.40, format="%.2f")
st.divider()
if st.button("Predict Wine Quality"):
    data = np.array([[
        fixed_acidity,
        volatile_acidity,
        citric_acid,
        residual_sugar,
        chlorides,
        free_sulfur_dioxide,
        total_sulfur_dioxide,
        density,
        pH,
        sulphates,
        alcohol
    ]])

    prediction = model.predict(data)
    quality = int(prediction[0])

    st.divider()

    st.markdown(f"""
    <div style="
    background:#1F2937;
    padding:25px;
    border-radius:15px;
    border-left:6px solid #B91C1C;
    margin-top:30px;
    text-align:center;">
    <h2 style="color:#F9FAFB;">Prediction Result</h2>
    <h1 style="color:#EF4444;font-size:55px;">{quality}</h1>
    </div>
    """, unsafe_allow_html=True)

    if quality >= 7:
        st.success("Classification : High Quality Wine")
    elif quality >= 5:
        st.info("Classification : Medium Quality Wine")
    else:
        st.error("Classification : Low Quality Wine")

    if hasattr(model, "predict_proba"):
        confidence = np.max(model.predict_proba(data)) * 100
        st.progress(int(confidence))
        st.caption(f"Prediction Confidence : {confidence:.2f}%")

    st.subheader("📋 Input Summary")
    st.write("The following values were used for prediction:")

    import pandas as pd

    df = pd.DataFrame({
        "Property": [
            "Fixed Acidity",
            "Volatile Acidity",
            "Citric Acid",
            "Residual Sugar",
            "Chlorides",
            "Free Sulfur Dioxide",
            "Total Sulfur Dioxide",
            "Density",
            "pH",
            "Sulphates",
            "Alcohol"
        ],
        "Value": [
            fixed_acidity,
            volatile_acidity,
            citric_acid,
            residual_sugar,
            chlorides,
            free_sulfur_dioxide,
            total_sulfur_dioxide,
            density,
            pH,
            sulphates,
            alcohol
        ]
    })

    st.dataframe(df, use_container_width=True, hide_index=True)

with st.sidebar:
    st.header("Project Details")
    st.write("**Algorithm** : Random Forest")
    st.write("**Dataset** : Wine Quality Dataset")
    st.write("**Framework** : Streamlit")
    st.write("**Language** : Python")
    st.write("**Purpose** : Wine Quality Prediction")
    st.write("---")
    st.write("Developed as a Machine Learning project.")

st.markdown("""
<div style="text-align:center;color:#9CA3AF;font-size:15px;">
<b>Developed by Naina Kumari</b><br>
WineSense AI • Machine Learning • Streamlit
</div>
""", unsafe_allow_html=True)