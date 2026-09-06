"""
Model Training Module
Handles model creation, training, and evaluation
"""
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib

class HousePriceModel:
    def __init__(self):
        """Initialize the Linear Regression model"""
        self.model = LinearRegression()
        self.is_trained = False
    
    def train(self, X_train, y_train):
        """Train the model on training data"""
        print("\n=== Training Model ===")
        print("Fitting Linear Regression model...")
        self.model.fit(X_train, y_train)
        self.is_trained = True
        print("✅ Model trained successfully!")
        
        # Display model coefficients
        print("\n=== Model Coefficients ===")
        print(f"Intercept: {self.model.intercept_:.2f}")
        print(f"Coefficients: {self.model.coef_}")
    
    def predict(self, X):
        """Make predictions"""
        if not self.is_trained:
            raise Exception("Model must be trained before making predictions!")
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance"""
        print("\n=== Model Evaluation ===")
        y_pred = self.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        print(f"Mean Squared Error (MSE): {mse:,.2f}")
        print(f"Root Mean Squared Error (RMSE): {rmse:,.2f}")
        print(f"Mean Absolute Error (MAE): {mae:,.2f}")
        print(f"R² Score: {r2:.4f}")
        
        # Interpretation
        print("\n=== Performance Interpretation ===")
        if r2 > 0.8:
            print("✅ Excellent model performance!")
        elif r2 > 0.6:
            print("✓ Good model performance")
        else:
            print("⚠ Model needs improvement")
        
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'predictions': y_pred
        }
    
    def save_model(self, filepath='../models/house_price_model.pkl'):
        """Save trained model to disk"""
        if not self.is_trained:
            raise Exception("Model must be trained before saving!")
        
        print(f"\nSaving model to {filepath}...")
        joblib.dump(self.model, filepath)
        print("✅ Model saved successfully!")
    
    def load_model(self, filepath='../models/house_price_model.pkl'):
        """Load trained model from disk"""
        print(f"\nLoading model from {filepath}...")
        self.model = joblib.load(filepath)
        self.is_trained = True
        print("✅ Model loaded successfully!")
    
    def predict_single(self, features):
        """Predict price for a single house"""
        if not self.is_trained:
            raise Exception("Model must be trained before making predictions!")
        
        # Reshape for single prediction
        features = np.array(features).reshape(1, -1)
        prediction = self.model.predict(features)[0]
        return prediction

if __name__ == "__main__":
    print("Model Training Module - Ready to use!")
