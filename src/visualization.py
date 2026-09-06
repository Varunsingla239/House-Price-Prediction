"""
Visualization Module
Creates plots and charts for analysis
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)

class Visualizer:
    def __init__(self, output_dir='../outputs'):
        """Initialize visualizer with output directory"""
        self.output_dir = output_dir
    
    def plot_data_distribution(self, data, save=True):
        """Plot distribution of features"""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Feature Distributions', fontsize=16, fontweight='bold')
        
        columns = data.columns
        for idx, col in enumerate(columns):
            row = idx // 2
            col_idx = idx % 2
            axes[row, col_idx].hist(data[col], bins=20, color='skyblue', edgecolor='black')
            axes[row, col_idx].set_title(f'{col.capitalize()} Distribution')
            axes[row, col_idx].set_xlabel(col.capitalize())
            axes[row, col_idx].set_ylabel('Frequency')
        
        plt.tight_layout()
        if save:
            plt.savefig(f'{self.output_dir}/feature_distributions.png', dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {self.output_dir}/feature_distributions.png")
        plt.show()
    
    def plot_correlation_matrix(self, data, save=True):
        """Plot correlation heatmap"""
        plt.figure(figsize=(10, 8))
        correlation = data.corr()
        
        sns.heatmap(correlation, annot=True, cmap='coolwarm', center=0, 
                    square=True, linewidths=1, fmt='.2f')
        plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
        
        if save:
            plt.savefig(f'{self.output_dir}/correlation_matrix.png', dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {self.output_dir}/correlation_matrix.png")
        plt.show()
    
    def plot_predictions(self, y_test, y_pred, save=True):
        """Plot actual vs predicted values"""
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.6, color='blue', edgecolors='black')
        
        # Perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        plt.xlabel('Actual Price ($)', fontsize=12)
        plt.ylabel('Predicted Price ($)', fontsize=12)
        plt.title('Actual vs Predicted House Prices', fontsize=16, fontweight='bold')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save:
            plt.savefig(f'{self.output_dir}/actual_vs_predicted.png', dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {self.output_dir}/actual_vs_predicted.png")
        plt.show()
    
    def plot_residuals(self, y_test, y_pred, save=True):
        """Plot residual analysis"""
        residuals = y_test - y_pred
        
        fig, axes = plt.subplots(1, 2, figsize=(15, 5))
        
        # Residual plot
        axes[0].scatter(y_pred, residuals, alpha=0.6, color='green', edgecolors='black')
        axes[0].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0].set_xlabel('Predicted Price ($)', fontsize=12)
        axes[0].set_ylabel('Residuals ($)', fontsize=12)
        axes[0].set_title('Residual Plot', fontsize=14, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        # Residual distribution
        axes[1].hist(residuals, bins=20, color='orange', edgecolor='black', alpha=0.7)
        axes[1].set_xlabel('Residuals ($)', fontsize=12)
        axes[1].set_ylabel('Frequency', fontsize=12)
        axes[1].set_title('Residual Distribution', fontsize=14, fontweight='bold')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save:
            plt.savefig(f'{self.output_dir}/residual_analysis.png', dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {self.output_dir}/residual_analysis.png")
        plt.show()
    
    def plot_feature_importance(self, model, feature_names, save=True):
        """Plot feature importance (coefficients)"""
        coefficients = model.coef_
        
        plt.figure(figsize=(10, 6))
        bars = plt.barh(feature_names, coefficients, color='teal', edgecolor='black')
        plt.xlabel('Coefficient Value', fontsize=12)
        plt.ylabel('Features', fontsize=12)
        plt.title('Feature Importance (Model Coefficients)', fontsize=16, fontweight='bold')
        plt.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            plt.text(width, bar.get_y() + bar.get_height()/2, 
                    f'{width:.2f}', ha='left' if width > 0 else 'right', 
                    va='center', fontsize=10)
        
        if save:
            plt.savefig(f'{self.output_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
            print(f"✅ Saved: {self.output_dir}/feature_importance.png")
        plt.show()

if __name__ == "__main__":
    print("Visualization Module - Ready to use!")
