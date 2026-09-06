"""
Main Training Pipeline
Executes the complete ML workflow
"""
import sys
import os
from data_preprocessing import DataPreprocessor
from model_training import HousePriceModel
from visualization import Visualizer
import joblib

def main():
    print("="*60)
    print("🏠 HOUSE PRICE PREDICTION - ML PIPELINE")
    print("="*60)
    
    # Initialize components
    preprocessor = DataPreprocessor('../data/house_data.csv')
    model_trainer = HousePriceModel()
    visualizer = Visualizer('../outputs')
    
    # Step 1: Load and explore data
    print("\n[STEP 1] Loading and Exploring Data...")
    data = preprocessor.load_data()
    preprocessor.explore_data()
    
    # Step 2: Visualize data
    print("\n[STEP 2] Creating Visualizations...")
    visualizer.plot_data_distribution(data)
    visualizer.plot_correlation_matrix(data)
    
    # Step 3: Preprocess data
    print("\n[STEP 3] Preprocessing Data...")
    preprocessor.handle_missing_values()
    X, y = preprocessor.prepare_features('price')
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)
    
    # Step 4: Train model
    print("\n[STEP 4] Training Model...")
    model_trainer.train(X_train_scaled, y_train)
    
    # Step 5: Evaluate model
    print("\n[STEP 5] Evaluating Model...")
    results = model_trainer.evaluate(X_test_scaled, y_test)
    
    # Step 6: Create evaluation visualizations
    print("\n[STEP 6] Creating Evaluation Plots...")
    visualizer.plot_predictions(y_test, results['predictions'])
    visualizer.plot_residuals(y_test, results['predictions'])
    visualizer.plot_feature_importance(model_trainer.model, X.columns.tolist())
    
    # Step 7: Save model and scaler
    print("\n[STEP 7] Saving Model and Scaler...")
    model_trainer.save_model('../models/house_price_model.pkl')
    joblib.dump(preprocessor.get_scaler(), '../models/scaler.pkl')
    print("✅ Scaler saved to ../models/scaler.pkl")
    
    # Step 8: Test prediction on new data
    print("\n[STEP 8] Testing Prediction on Sample House...")
    sample_house = X_test_scaled[0].reshape(1, -1)
    predicted_price = model_trainer.predict(sample_house)[0]
    actual_price = y_test.iloc[0]
    
    print(f"\nSample House Features: {X_test.iloc[0].to_dict()}")
    print(f"Predicted Price: {predicted_price:,.2f}")
    print(f"Actual Price: {actual_price:,.2f}")
    print(f"Difference: {abs(predicted_price - actual_price):,.2f}")
    
    print("\n" + "="*60)
    print("✅ PIPELINE COMPLETED SUCCESSFULLY!")
    
    print("="*60)
    print(f"\n📊 Model Performance Summary:")
    print(f"   - R² Score: {results['r2']:.4f}")
    print(f"   - RMSE: {results['rmse']:,.2f}")
    print(f"   - MAE: {results['mae']:,.2f}")
    print(f"\n📁 Outputs saved in: ../outputs/")
    print(f"💾 Model saved in: ../models/")

if __name__ == "__main__":
    main()
