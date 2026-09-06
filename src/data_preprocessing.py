"""
Data Preprocessing Module
Handles data loading, cleaning, and feature engineering
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    def __init__(self, filepath):
        """Initialize with dataset filepath"""
        self.filepath = filepath
        self.data = None
        self.scaler = StandardScaler()
    
    def load_data(self):
        """Load data from CSV file"""
        print("Loading data...")
        self.data = pd.read_csv(self.filepath)
        print(f"Data loaded successfully! Shape: {self.data.shape}")
        return self.data
    
    def explore_data(self):
        """Display basic data information"""
        print("\n=== Data Overview ===")
        print(self.data.head())
        print("\n=== Data Info ===")
        print(self.data.info())
        print("\n=== Statistical Summary ===")
        print(self.data.describe())
        print("\n=== Missing Values ===")
        print(self.data.isnull().sum())
    
    def handle_missing_values(self):
        """Handle missing values in the dataset"""
        if self.data.isnull().sum().sum() > 0:
            print("\nHandling missing values...")
            # Fill numeric columns with median
            numeric_cols = self.data.select_dtypes(include=[np.number]).columns
            self.data[numeric_cols] = self.data[numeric_cols].fillna(
                self.data[numeric_cols].median()
            )
            print("Missing values handled!")
        else:
            print("\nNo missing values found!")
    
    def remove_outliers(self, columns, threshold=3):
        """Remove outliers using z-score method"""
        print(f"\nRemoving outliers (z-score threshold: {threshold})...")
        initial_shape = self.data.shape
        
        for col in columns:
            z_scores = np.abs((self.data[col] - self.data[col].mean()) / self.data[col].std())
            self.data = self.data[z_scores < threshold]
        
        print(f"Outliers removed! Shape changed from {initial_shape} to {self.data.shape}")
        return self.data
    
    def prepare_features(self, target_column='price'):
        """Prepare feature matrix and target variable"""
        print(f"\nPreparing features and target variable ('{target_column}')...")
        X = self.data.drop(columns=[target_column])
        y = self.data[target_column]
        
        print(f"Features shape: {X.shape}")
        print(f"Target shape: {y.shape}")
        return X, y
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """Split data into training and testing sets"""
        print(f"\nSplitting data (test_size={test_size})...")
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"Training set size: {X_train.shape[0]}")
        print(f"Testing set size: {X_test.shape[0]}")
        return X_train, X_test, y_train, y_test
    
    def scale_features(self, X_train, X_test):
        """Scale features using StandardScaler"""
        print("\nScaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        print("Features scaled successfully!")
        return X_train_scaled, X_test_scaled
    
    def get_scaler(self):
        """Return the fitted scaler for later use"""
        return self.scaler

if __name__ == "__main__":
    # Example usage
    preprocessor = DataPreprocessor('../data/house_data.csv')
    data = preprocessor.load_data()
    preprocessor.explore_data()
    preprocessor.handle_missing_values()
    X, y = preprocessor.prepare_features()
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y)
    print("\nPreprocessing completed successfully! ✅")
