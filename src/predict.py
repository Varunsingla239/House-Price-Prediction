"""
Standalone Prediction Script
Use this script to predict house prices after training the model
"""
import joblib
import numpy as np
import sys

def load_model_and_scaler():
    """Load the trained model and scaler"""
    try:
        model = joblib.load('../models/house_price_model.pkl')
        scaler = joblib.load('../models/scaler.pkl')
        return model, scaler
    except FileNotFoundError:
        print("❌ Error: Model or scaler not found!")
        print("Please run the training pipeline first: python main_pipeline.py")
        sys.exit(1)

def predict_house_price(size, bedrooms, age):
    """Predict house price given features"""
    model, scaler = load_model_and_scaler()
    
    # Prepare features
    features = np.array([[size, bedrooms, age]])
    features_scaled = scaler.transform(features)
    
    # Make prediction
    price = model.predict(features_scaled)[0]
    return price

def main():
    print("="*60)
    print("🏠 HOUSE PRICE PREDICTION")
    print("="*60)
    
    # Get user input
    print("\nEnter house details:")
    try:
        size = float(input("Size (sq ft): "))
        bedrooms = int(input("Number of bedrooms: "))
        age = int(input("Age (years): "))
    except ValueError:
        print("❌ Invalid input! Please enter numeric values.")
        sys.exit(1)
    
    # Make prediction
    print("\n🔮 Predicting...")
    predicted_price = predict_house_price(size, bedrooms, age)
    
    # Display result
    print("\n" + "="*60)
    print("📊 PREDICTION RESULT")
    print("="*60)
    print(f"\n🏡 House Details:")
    print(f"   - Size: {size:,.0f} sq ft")
    print(f"   - Bedrooms: {bedrooms}")
    print(f"   - Age: {age} years")
    print(f"\n💰 Predicted Price: ${predicted_price:,.2f}")
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
