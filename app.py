import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import base64

# App Config
st.set_page_config(page_title="Salary Predictor Pro", layout="wide", page_icon="💼")

# Load Model and Data
@st.cache_data

def load_data():
    df = pd.read_csv("Salary_Data.csv")
    return df

df = load_data()
X = df[['YearsExperience']]
y = df['Salary']

# Default Linear Model
model = joblib.load("model.pkl")

# Sidebar Navigation
with st.sidebar:
    if os.path.exists("logo.png"):
        st.image("logo.png", width=120)

    selected = option_menu(
        menu_title="Navigation",
        options=["Home", "Visualize", "Predict", "Info"],
        icons=["house", "bar-chart-line", "search", "upload", "gear", "eye", "info-circle"],
        menu_icon="cast",
        default_index=0,
        styles={
            "container": {"padding": "10px", "background-color": "#111"},
            "icon": {"color": "#00C49A", "font-size": "18px"},
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "5px",
                "--hover-color": "#333",
            },
            "nav-link-selected": {"background-color": "#00C49A", "color": "#000"},
        }
    )

# HOME
if selected == "Home":
    st.title("💼 Salary Predictor Pro")
    st.markdown("Predict salary based on experience using multiple ML models.")

    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Records", len(df))
        st.metric("Avg Experience", f"{df['YearsExperience'].mean():.2f} yrs")
    with col2:
        st.metric("Avg Salary", f"₹{df['Salary'].mean():,.0f}")
        st.metric("Linear R2", f"{model.score(X, y) * 100:.2f}%")

    st.image("https://cdn.dribbble.com/users/1162077/screenshots/3848914/programmer.gif", width=600)

# VISUALIZE
elif selected == "Visualize":
    st.title("Interactive Visualization")
    st.markdown("Choose model and degree to visualize regression.")

    model_type = st.selectbox("Select Model", ["Linear", "Polynomial", "Decision Tree", "Random Forest"])
    degree = st.slider("Polynomial Degree", 1, 5, 2) if model_type == "Polynomial" else None

    x_vals = np.linspace(X.min(), X.max(), 100).reshape(-1, 1)

    if model_type == "Linear":
        pred_model = LinearRegression().fit(X, y)
        y_pred = pred_model.predict(x_vals)
    elif model_type == "Polynomial":
        poly = PolynomialFeatures(degree=degree)
        X_poly = poly.fit_transform(X)
        model_poly = LinearRegression().fit(X_poly, y)
        x_vals_poly = poly.transform(x_vals)
        y_pred = model_poly.predict(x_vals_poly)
    elif model_type == "Decision Tree":
        pred_model = DecisionTreeRegressor().fit(X, y)
        y_pred = pred_model.predict(x_vals)
    elif model_type == "Random Forest":
        pred_model = RandomForestRegressor(n_estimators=100).fit(X, y)
        y_pred = pred_model.predict(x_vals)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df["YearsExperience"], y=df["Salary"], mode='markers', name='Actual'))
    fig.add_trace(go.Scatter(x=x_vals.flatten(), y=y_pred, mode='lines', name='Prediction'))
    fig.update_layout(title="Salary Prediction", xaxis_title="Experience (Years)", yaxis_title="Salary",
                      template="plotly_dark", height=500)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation Heatmap")
    fig_corr, ax = plt.subplots()
    sns.heatmap(df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig_corr)

# PREDICT
elif selected == "Predict":
    st.title("Predict Salary")
    exp = st.slider("Years of Experience", 0.0, 30.0, 3.0, 0.1)

    pred_salary = model.predict(np.array([[exp]]))[0]
    st.success(f"Estimated Salary for {exp} years: ₹{pred_salary:,.2f}")

# INFO
elif selected == "Info":
    st.title("About This App")
    st.markdown("""
    ### 💼 Salary Predictor Pro
    
    This interactive ML web app allows users to:

    - Predict salary based on experience
    - Choose between ML models
    - Visualize model behavior
    - Upload data for batch prediction
    - Train & save custom models
    - View model explainability with SHAP

    #### Tech Stack:
    - Python, Pandas, NumPy
    - Scikit-Learn
    - Plotly, Seaborn, Matplotlib
    - Streamlit
    - SHAP
    
    Dataset: [Kaggle - Salary Data](https://www.kaggle.com/datasets)
    """)
