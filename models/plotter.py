import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from pathlib import Path
import pandas as pd


def plot_results(y_test, y_pred, model_name='Model', save_path=None):
    """
    Plot 4-subplot visualization of model performance
    
    Parameters:
    -----------
    y_test : array-like
        Actual target values
    y_pred : array-like
        Predicted target values
    model_name : str, optional
        Name of the model (default: 'Model')
    save_path : str, optional
        Path to save figure
    """
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'{model_name} - Performance Metrics', fontsize=16, fontweight='bold', y=0.995)

    residuals = y_test - y_pred
    abs_error = np.abs(residuals)

    # axes[0, 0] - Actual vs Predicted
    axes[0, 0].scatter(y_test, y_pred, alpha=0.5, s=10)
    axes[0, 0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2, label='Perfect Prediction')
    axes[0, 0].set_xlabel('Actual spread_close_pct', fontsize=12)
    axes[0, 0].set_ylabel('Predicted spread_close_pct', fontsize=12)
    axes[0, 0].set_title('Actual vs Predicted', fontsize=14, fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # axes[0, 1] - Residuals
    axes[0, 1].scatter(y_pred, residuals, alpha=0.5, s=10)
    axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Predicted spread_close_pct', fontsize=12)
    axes[0, 1].set_ylabel('Residuals', fontsize=12)
    axes[0, 1].set_title('Residual Plot', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # axes[1, 0] - Residual Distribution
    axes[1, 0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2)
    axes[1, 0].set_xlabel('Residuals', fontsize=12)
    axes[1, 0].set_ylabel('Frequency', fontsize=12)
    axes[1, 0].set_title('Residual Distribution', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # axes[1, 1] - Absolute Error Distribution
    axes[1, 1].hist(abs_error, bins=50, edgecolor='black', alpha=0.7, color='orange')
    axes[1, 1].set_xlabel('Absolute Error', fontsize=12)
    axes[1, 1].set_ylabel('Frequency', fontsize=12)
    axes[1, 1].set_title('Absolute Error Distribution', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Results plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    
def plot_prediction_hist(y_pred, model_name='Model', save_path=None):
    """
    Plot histogram of predictions
    
    Parameters:
    -----------
    y_pred : array-like
        Predicted target values
    model_name : str, optional
        Name of the model (default: 'Model')
    save_path : str, optional
        Path to save figure
    """
    
    plt.figure(figsize=(10, 6))
    plt.hist(y_pred, bins=40, edgecolor='black', alpha=0.75)
    plt.xlabel('Predicted spread_close_pct', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'{model_name} - Prediction Distribution', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Prediction histogram saved to {save_path}")
    else:
        plt.show()
        
    plt.close()
    
    
def plot_training_history(history, model_name='Model', save_path=None):
    """
    Plot training and validation loss over epochs
    
    Parameters:
    -----------
    history : Keras History object
        Training history from model.fit()
    model_name : str, optional
        Name of the model (default: 'Model')
    save_path : str, optional
        Path to save figure
    """
    
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']
    
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='Training Loss', linewidth=2)
    plt.plot(val_loss, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.title(f'{model_name} - Training and Validation Loss Over Epochs', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to {save_path}")
    else:
        plt.show()
        
    plt.close()
    
    
def plot_feature_importance(model, feature_names, model_type, model_name='Model', top_n=20, save_path=None):
    """
    Plot feature importance - handles different model types
    
    Parameters:
    -----------
    model : sklearn model
        Trained model (RandomForest, XGBoost, or LinearRegression)
    feature_names : list
        Names of features
    model_type : str
        Type of model ('rf', 'xgboost', 'linear')
    model_name : str, optional
        Display name of model (default: 'Model')
    top_n : int, optional
        Number of top features to show (default: 20)
    save_path : str, optional
        Path to save figure
    """
    
    if model_type.lower() == 'linear':
        # Linear model - preserve coefficient signs
        coefficients = model.coef_
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients)
        }).sort_values('abs_coefficient', ascending=False).head(top_n)
        
        colors = ['green' if x > 0 else 'red' for x in importance_df['coefficient']]
        values = importance_df['coefficient']
        top_features = importance_df['feature'].tolist()
        xlabel = 'Coefficient'
        
    elif model_type.lower() in ['rf', 'xgboost']:
        # Tree-based models
        importances = model.feature_importances_
        indices = np.argsort(importances)[-top_n:]
        colors = 'steelblue'
        values = importances[indices]
        top_features = [feature_names[i] for i in indices]  # ✓ Store in new variable
        xlabel = 'Importance'
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(values)), values, color=colors, alpha=0.7)
    plt.yticks(range(len(top_features)), top_features)  # ✓ Use top_features, not feature_names
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.title(f'{model_name} - Top {top_n} Feature Importance', fontsize=14, fontweight='bold')
    
    if model_type.lower() == 'linear':
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Feature importance plot saved to {save_path}")
    else:
        plt.show()
    
    plt.close()
    
    
def plot_pr_curve():
    
    
    


def plot_threshold_metrics():
    
    
    
    
    
        
        
        
    
    
    
    










