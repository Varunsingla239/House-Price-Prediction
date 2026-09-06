"""
Streamlit Web Application for House Price Prediction
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="House Price Predictor",
    page_icon="🏠",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    color: #1f77b4;
    text-align: center;
    margin-bottom: 2rem;
}
.prediction-box {
    background-color: #f0f2f6;
    padding: 2rem;
    border-radius: 10px;
    text-align: center;
}
.price-display {
    font-size: 3rem;
    color: #2ecc71;
    font-weight: bold;
}
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_scaler():
    """Load the trained model and scaler"""
    try:
        model = joblib.load('../models/house_price_model.pkl')
        scaler = joblib.load('../models/scaler.pkl')
        return model, scaler, True
    except:
        return None, None, False

def main():
    # Header
    st.markdown('<h1 class="main-header">🏠 House Price Prediction System</h1>', unsafe_allow_html=True)
    st.markdown("---")
    
    # Load model
    model, scaler, model_loaded = load_model_and_scaler()
    
    if not model_loaded:
        st.error("⚠️ Model not found! Please run the training pipeline first.")
        st.code("cd src && python main_pipeline.py")
        return
    
    # Sidebar
    st.sidebar.header("📊 About the Model")
    st.sidebar.info("""
    This application uses *Linear Regression* to predict house prices based on:
    - *Size* (square feet)
    - *Number of Bedrooms*
    - *Age of the House*
    
    The model was trained on historical house price data.
    """)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("🏡 Enter House Details")
        
        # Input fields
        size = st.number_input(
            "House Size (sq ft)",
            min_value=500,
            max_value=5000,
            value=1800,
            step=100,
            help="Enter the total living area in square feet"
        )
        
        bedrooms = st.number_input(
            "Number of Bedrooms",
            min_value=1,
            max_value=10,
            value=3,
            step=1,
            help="Enter the number of bedrooms"
        )
        
        age = st.number_input(
            "Age of House (years)",
            min_value=0,
            max_value=100,
            value=10,
            step=1,
            help="Enter the age of the house in years"
        )
        
        predict_button = st.button("🔮 Predict Price", type="primary", use_container_width=True)
    
    with col2:
        st.header("💰 Prediction Result")
        
        if predict_button:
            # Prepare features
            features = np.array([[size, bedrooms, age]])
            features_scaled = scaler.transform(features)
            
            # Make prediction
            prediction = model.predict(features_scaled)[0]
            
            # Display result
            st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
            st.markdown("### Estimated House Price")
            st.markdown(f'<p class="price-display">{prediction:,.2f}</p>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Additional insights
            st.success("✅ Prediction completed successfully!")
            
            # Price breakdown
            st.subheader("📈 Price Insights")
            col_a, col_b, col_c = st.columns(3)
            
            with col_a:
                price_per_sqft = prediction / size
                st.metric("Price per Sq Ft", f"{price_per_sqft:.2f}")
            
            with col_b:
                price_per_bedroom = prediction / bedrooms
                st.metric("Price per Bedroom", f"{price_per_bedroom:,.0f}")
            
            with col_c:
                depreciation = age * 1000  # Simple estimation
                st.metric("Age Depreciation", f"{depreciation:,.0f}")
        
        else:
            st.info("👈 Enter house details and click 'Predict Price' to see the estimated value")
    
    # Footer
    st.markdown("---")
    st.markdown("### 📝 Sample Predictions")
    
    # Sample houses
    samples_display = pd.DataFrame({
        'Size (sq ft)': [1500, 2000, 2500],
        'Bedrooms': [3, 4, 4],
        'Age (years)': [10, 5, 3]
    })
    
    # Create samples with correct feature names for prediction
    samples_for_prediction = pd.DataFrame({
        'size': [1500, 2000, 2500],
        'bedrooms': [3, 4, 4],
        'age': [10, 5, 3]
    })
    
    # Predict for samples
    samples_scaled = scaler.transform(samples_for_prediction)
    samples_display['Predicted Price '] = model.predict(samples_scaled)
    samples_display['Predicted Price '] = samples_display['Predicted Price '].apply(lambda x: f"{x:,.0f}")
    
    st.dataframe(samples_display, use_container_width=True)
    
    # Model information
    with st.expander("ℹ️ Model Information"):
        st.write("*Model Type:* Linear Regression")
        st.write("*Features:* Size, Bedrooms, Age")
        st.write("*Preprocessing:* Standard Scaling")
        st.write("*Training Data:* Historical house price dataset")

if __name__ == "__main__":
    main()
