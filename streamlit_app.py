import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import plotly.express as px
import plotly.graph_objects as go

# Page config
st.set_page_config(
    page_title="🧃 Drink Taste Predictor",
    page_icon="🧃",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .stButton>button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: bold;
        border-radius: 10px;
        padding: 10px 30px;
        font-size: 18px;
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# Train and load models
@st.cache_resource
def load_or_train_models():
    try:
        dt_model = pickle.load(open('models/dt_model.pkl', 'rb'))
        rf_model = pickle.load(open('models/rf_model.pkl', 'rb'))
        svm_model = pickle.load(open('models/svm_model.pkl', 'rb'))
        scaler = pickle.load(open('models/scaler.pkl', 'rb'))
        return dt_model, rf_model, svm_model, scaler
    except:
        with st.spinner("Training models from drink_taste.csv..."):
            df = pd.read_csv('drink_taste.csv')
            
            X = df[['Sweetness (1-10)', 'R', 'G', 'B', 'Temperature (°C)', 'Ingredients_Count']]
            y = df['Liked (1/0)']
            
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(X)
            
            X_train, X_test, y_train, y_test = train_test_split(
                X_scaled, y, test_size=0.2, random_state=42, stratify=y
            )
            
            dt_model = DecisionTreeClassifier(max_depth=5, random_state=42)
            rf_model = RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
            svm_model = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
            
            dt_model.fit(X_train, y_train)
            rf_model.fit(X_train, y_train)
            svm_model.fit(X_train, y_train)
            
            os.makedirs('models', exist_ok=True)
            pickle.dump(dt_model, open('models/dt_model.pkl', 'wb'))
            pickle.dump(rf_model, open('models/rf_model.pkl', 'wb'))
            pickle.dump(svm_model, open('models/svm_model.pkl', 'wb'))
            pickle.dump(scaler, open('models/scaler.pkl', 'wb'))
            
            return dt_model, rf_model, svm_model, scaler

# Load data
@st.cache_data
def load_data():
    return pd.read_csv('drink_taste.csv')

# Main app
def main():
    st.title("🧃 Drink Taste Predictor")
    st.markdown("### Predict if a drink will be liked using Machine Learning")
    
    # Load models and data
    dt_model, rf_model, svm_model, scaler = load_or_train_models()
    df = load_data()
    
    # Sidebar
    with st.sidebar:
        st.header("📊 Dataset Info")
        st.metric("Total Drinks", len(df))
        st.metric("Liked Drinks", int(df['Liked (1/0)'].sum()))
        st.metric("Not Liked", len(df) - int(df['Liked (1/0)'].sum()))
        st.metric("Avg Rating", f"{df['Rating (1-5)'].mean():.2f}/5")
        
        st.markdown("---")
        st.header("🎨 Drink Properties")
        
        sweetness = st.slider("🍯 Sweetness Level", 1, 10, 5)
        
        st.subheader("Color (RGB)")
        col1, col2, col3 = st.columns(3)
        with col1:
            r = st.number_input("R", 0, 255, 255)
        with col2:
            g = st.number_input("G", 0, 255, 180)
        with col3:
            b = st.number_input("B", 0, 255, 100)
        
        # Color preview
        st.markdown(
            f'<div style="background-color: rgb({r},{g},{b}); '
            f'height: 50px; border-radius: 10px; border: 2px solid white;"></div>',
            unsafe_allow_html=True
        )
        
        temperature = st.slider("🌡️ Temperature (°C)", 0, 100, 10)
        ingredients = st.slider("📝 Ingredients Count", 1, 10, 3)
        
        predict_btn = st.button("🔮 Predict Taste", use_container_width=True)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("🎯 Prediction Results")
        
        if predict_btn:
            # Prepare features
            features = pd.DataFrame({
                'Sweetness (1-10)': [sweetness],
                'R': [r],
                'G': [g],
                'B': [b],
                'Temperature (°C)': [temperature],
                'Ingredients_Count': [ingredients]
            })
            
            features_scaled = scaler.transform(features)
            
            # Make predictions
            dt_pred = dt_model.predict(features_scaled)[0]
            rf_pred = rf_model.predict(features_scaled)[0]
            svm_pred = svm_model.predict(features_scaled)[0]
            
            # Get confidence
            rf_proba = rf_model.predict_proba(features_scaled)[0]
            confidence = max(rf_proba) * 100
            
            # Overall prediction
            overall = 1 if (dt_pred + rf_pred + svm_pred) >= 2 else 0
            
            # Display results
            if overall == 1:
                st.success("### 👍 DRINK WILL BE LIKED!")
                st.balloons()
            else:
                st.error("### 👎 DRINK MAY NOT BE LIKED")
            
            st.metric("Confidence", f"{confidence:.1f}%")
            
            st.markdown("---")
            st.subheader("Model Predictions")
            
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("🌳 Decision Tree", "👍" if dt_pred == 1 else "👎")
            with col_b:
                st.metric("🌲 Random Forest", "👍" if rf_pred == 1 else "👎")
            with col_c:
                st.metric("🎯 SVM", "👍" if svm_pred == 1 else "👎")
            
            # Feature display
            st.markdown("---")
            st.subheader("Drink Characteristics")
            feat_col1, feat_col2 = st.columns(2)
            with feat_col1:
                st.metric("Sweetness", f"{sweetness}/10")
                st.metric("Temperature", f"{temperature}°C")
            with feat_col2:
                st.metric("Ingredients", ingredients)
                color_tone = "Vibrant" if (r+g+b)/3 > 150 else "Dark"
                st.metric("Color", color_tone)
    
    with col2:
        st.header("📊 Dataset Insights")
        
        # Sweetness vs Rating scatter
        fig1 = px.scatter(
            df, 
            x='Sweetness (1-10)', 
            y='Rating (1-5)',
            color='Liked (1/0)',
            color_continuous_scale=['red', 'green'],
            title="Sweetness vs Rating"
        )
        st.plotly_chart(fig1, use_container_width=True)
        
        # Temperature distribution
        fig2 = px.histogram(
            df, 
            x='Temperature (°C)',
            nbins=20,
            title="Temperature Distribution",
            color_discrete_sequence=['#667eea']
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    # Show dataset
    with st.expander("📁 View Complete Dataset"):
        st.dataframe(df, use_container_width=True)

if __name__ == "__main__":
    main()