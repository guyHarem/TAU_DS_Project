"""
Universal plotting module for model evaluation and visualization.

This module provides a centralized set of plotting functions that work across
all regression and classification models (Random Forest, LSTM, GRU, Linear, XGBoost, Transformer).

Functions:
    - plot_results: 4-subplot regression performance visualization
    - plot_prediction_hist: Histogram of predicted values
    - plot_training_history: Training/validation loss over epochs
    - plot_feature_importance: Feature importance for tree and linear models
    - plot_pr_curve: Precision-Recall curve for classification
    - plot_threshold_metrics: Threshold analysis for classification

Author: TAU DS Project | Arbitrage team | Oz & Guy
Date: 2026
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, precision_recall_curve

# ============================================================================
# PLOTTING CONFIGURATION
# ============================================================================

# Standard figure sizes
FIGSIZE_SINGLE = (10, 6)
FIGSIZE_MULTI = (15, 12)
FIGSIZE_FEATURE = (10, 8)

# Standard DPI and quality
DPI = 300
BBOX_INCHES = 'tight'

# Font sizes
FONTSIZE_TITLE = 14
FONTSIZE_LABEL = 12
FONTSIZE_LEGEND = 11
FONTSIZE_TICK = 10

# Colors
COLOR_STEELBLUE = 'steelblue'
COLOR_RED = 'red'
COLOR_GREEN = 'green'
COLOR_ORANGE = 'orange'

# ============================================================================
# GENERAL FUNCTIONS
# ============================================================================

def save_plot(save_path):
    save_path = Path(save_path)
    plt.savefig(save_path, dpi=DPI, bbox_inches=BBOX_INCHES)
    print(f"✓ Results plot saved to {save_path}")

    pdf_path = save_path.with_suffix('.pdf')
    try:
        plt.savefig(pdf_path, bbox_inches=BBOX_INCHES)
        print(f"✓ PDF plot exported to {pdf_path}")
    except Exception as exc:
        print(f"⚠ Could not export PDF plot: {exc}")

# ============================================================================
# REGRESSION PERFORMANCE PLOTS
# ============================================================================

def plot_results(y_test, y_pred, model_name='Model', save_path=None):
    """
    Plot 4-subplot visualization of regression model performance.
    
    Creates a comprehensive figure with:
    1. Actual vs Predicted scatter plot with perfect prediction line
    2. Residuals vs Predicted values scatter plot
    3. Residuals histogram with zero line
    4. Absolute error histogram
    
    Parameters:
    -----------
    y_test : array-like
        Actual target values
    y_pred : array-like
        Predicted target values
    model_name : str, optional
        Display name of the model for title (default: 'Model')
    save_path : str or Path, optional
        Path to save figure. If None, displays in notebook (default: None)
        
    Returns:
    --------
    None
        Displays or saves plot as side effect
        
    Example:
    --------
    >>> from plotter import plot_results
    >>> plot_results(y_test, y_pred, model_name='LSTM', 
    ...              save_path='output/lstm_results.png')
    """
    
    fig, axes = plt.subplots(2, 2, figsize=FIGSIZE_MULTI)
    fig.suptitle(f'{model_name} - Performance Metrics', 
                 fontsize=FONTSIZE_TITLE + 2, fontweight='bold', y=0.995)

    residuals = y_test - y_pred
    abs_error = np.abs(residuals)

    # Subplot 1: Actual vs Predicted
    axes[0, 0].scatter(y_test, y_pred, alpha=0.5, s=10)
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    axes[0, 0].plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    axes[0, 0].set_xlabel('Actual spread_close_pct', fontsize=FONTSIZE_LABEL)
    axes[0, 0].set_ylabel('Predicted spread_close_pct', fontsize=FONTSIZE_LABEL)
    axes[0, 0].set_title('Actual vs Predicted', fontsize=FONTSIZE_TITLE, fontweight='bold')
    axes[0, 0].legend(fontsize=FONTSIZE_LEGEND)
    axes[0, 0].grid(True, alpha=0.3)
    
    # Subplot 2: Residuals
    axes[0, 1].scatter(y_pred, residuals, alpha=0.5, s=10)
    axes[0, 1].axhline(y=0, color=COLOR_RED, linestyle='--', lw=2)
    axes[0, 1].set_xlabel('Predicted spread_close_pct', fontsize=FONTSIZE_LABEL)
    axes[0, 1].set_ylabel('Residuals', fontsize=FONTSIZE_LABEL)
    axes[0, 1].set_title('Residual Plot', fontsize=FONTSIZE_TITLE, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    # Subplot 3: Residual Distribution
    axes[1, 0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[1, 0].axvline(x=0, color=COLOR_RED, linestyle='--', lw=2)
    axes[1, 0].set_xlabel('Residuals', fontsize=FONTSIZE_LABEL)
    axes[1, 0].set_ylabel('Frequency', fontsize=FONTSIZE_LABEL)
    axes[1, 0].set_title('Residual Distribution', fontsize=FONTSIZE_TITLE, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # Subplot 4: Absolute Error Distribution
    axes[1, 1].hist(abs_error, bins=50, edgecolor='black', alpha=0.7, color=COLOR_ORANGE)
    axes[1, 1].set_xlabel('Absolute Error', fontsize=FONTSIZE_LABEL)
    axes[1, 1].set_ylabel('Frequency', fontsize=FONTSIZE_LABEL)
    axes[1, 1].set_title('Absolute Error Distribution', fontsize=FONTSIZE_TITLE, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    plt.tight_layout()

    save_plot(save_path) if save_path else plt.show()
    plt.close()

def plot_prediction_hist(y_pred, model_name='Model', save_path=None):
    """
    Plot histogram of predicted values.
    
    Parameters:
    -----------
    y_pred : array-like
        Predicted target values
    model_name : str, optional
        Display name of the model for title (default: 'Model')
    save_path : str or Path, optional
        Path to save figure. If None, displays in notebook (default: None)
        
    Returns:
    --------
    None
        Displays or saves plot as side effect
        
    Example:
    --------
    >>> from plotter import plot_prediction_hist
    >>> plot_prediction_hist(y_pred, model_name='GRU', 
    ...                      save_path='output/gru_pred_hist.png')
    """
    
    plt.figure(figsize=FIGSIZE_SINGLE)
    plt.hist(y_pred, bins=40, edgecolor='black', alpha=0.75)
    plt.xlabel('Predicted spread_close_pct', fontsize=FONTSIZE_LABEL)
    plt.ylabel('Frequency', fontsize=FONTSIZE_LABEL)
    plt.title(f'{model_name} - Prediction Distribution', fontsize=FONTSIZE_TITLE, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    save_plot(save_path) if save_path else plt.show()   
    plt.close()

# ============================================================================
# DEEP LEARNING PLOTS
# ============================================================================

def plot_training_history(history, model_name='Model', save_path=None):
    """
    Plot training and validation loss over epochs.
    
    Useful for monitoring model learning progress and detecting overfitting.
    Shows both training and validation loss curves side by side.
    
    Parameters:
    -----------
    history : Keras History object or dict
        Training history returned by model.fit() or dict with 'loss'/'val_loss' keys
    model_name : str, optional
        Display name of the model for title (default: 'Model')
    save_path : str or Path, optional
        Path to save figure. If None, displays in notebook (default: None)
        
    Returns:
    --------
    None
        Displays or saves plot as side effect
        
    Example:
    --------
    >>> from plotter import plot_training_history
    >>> plot_training_history(model.history, model_name='LSTM', 
    ...                       save_path='output/lstm_training_history.png')
    """
    
    # Handle both Keras History objects and plain dicts
    if hasattr(history, 'history'):
        train_loss = history.history['loss']
        val_loss = history.history['val_loss']
    else:
        train_loss = history['loss']
        val_loss = history['val_loss']
    
    plt.figure(figsize=FIGSIZE_SINGLE)
    plt.plot(train_loss, label='Training Loss', linewidth=2)
    plt.plot(val_loss, label='Validation Loss', linewidth=2)
    plt.xlabel('Epoch', fontsize=FONTSIZE_LABEL)
    plt.ylabel('Loss', fontsize=FONTSIZE_LABEL)
    plt.title(f'{model_name} - Training and Validation Loss Over Epochs', 
              fontsize=FONTSIZE_TITLE, fontweight='bold')
    plt.legend(fontsize=FONTSIZE_LEGEND)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_plot(save_path) if save_path else plt.show()        
    plt.close()

# ============================================================================
# FEATURE IMPORTANCE PLOTS
# ============================================================================

def plot_feature_importance(model, feature_names, model_type, model_name='Model', 
                           top_n=20, save_path=None):
    """
    Plot feature importance for tree-based and linear models.
    
    Supports the following model families:
    - Linear: 'linear', 'ridge', 'lasso'
    - Tree-based: 'rf', 'randomforest', 'xgboost', 'catboost'
    - Deep learning: 'lstm', 'gru' (estimated from input kernel weights)
    
    Parameters:
    -----------
    model : model object
        Trained model object
    feature_names : list
        Names of all features
    model_type : str
        Type of model (e.g. 'linear', 'ridge', 'lasso', 'rf', 'xgboost',
        'catboost', 'lstm', 'gru')
    model_name : str, optional
        Display name of the model for title (default: 'Model')
    top_n : int, optional
        Number of top features to display (default: 20)
    save_path : str or Path, optional
        Path to save figure. If None, displays in notebook (default: None)
        
    Returns:
    --------
    None
        Displays or saves plot as side effect
        
    Raises:
    -------
    ValueError
        If model_type is unsupported or required attributes are missing
        
    Example:
    --------
    >>> from plotter import plot_feature_importance
    >>> plot_feature_importance(rf_model, feature_names, 'rf', 
    ...                         model_name='Random Forest', 
    ...                         save_path='output/rf_features.png')
    """
    
    model_type_norm = str(model_type).strip().lower()
    linear_types = {'linear', 'ridge', 'lasso'}
    tree_types = {'rf', 'randomforest', 'random_forest', 'random forest', 'xgboost', 'catboost'}
    deep_types = {'lstm', 'gru'}
    model_obj = model.model if hasattr(model, 'model') else model

    if model_type_norm in linear_types:
        # Linear models preserve coefficient sign
        if not hasattr(model_obj, 'coef_'):
            raise ValueError(
                f"Model type '{model_type}' expects a fitted linear model with 'coef_'."
            )

        coefficients = np.asarray(model_obj.coef_)
        if coefficients.ndim > 1:
            coefficients = coefficients[0]
        coefficients = coefficients.ravel()

        if len(coefficients) != len(feature_names):
            raise ValueError(
                f"Feature count mismatch: got {len(feature_names)} feature names "
                f"but {len(coefficients)} coefficients."
            )

        importance_df = pd.DataFrame({
            'feature': feature_names,
            'coefficient': coefficients,
            'abs_coefficient': np.abs(coefficients)
        }).sort_values('abs_coefficient', ascending=False).head(top_n)
        
        colors = [COLOR_GREEN if x > 0 else COLOR_RED for x in importance_df['coefficient']]
        values = importance_df['coefficient']
        top_features = importance_df['feature'].tolist()
        xlabel = 'Coefficient'
        
    elif model_type_norm in tree_types:
        # Tree-based models use built-in feature importances
        if not hasattr(model_obj, 'feature_importances_'):
            raise ValueError(
                f"Model type '{model_type}' expects a fitted tree model with 'feature_importances_'."
            )

        importances = np.asarray(model_obj.feature_importances_).ravel()
        if len(importances) != len(feature_names):
            raise ValueError(
                f"Feature count mismatch: got {len(feature_names)} feature names "
                f"but {len(importances)} importances."
            )

        indices = np.argsort(importances)[-top_n:]
        colors = COLOR_STEELBLUE
        values = importances[indices]
        top_features = [feature_names[i] for i in indices]
        xlabel = 'Importance'

    elif model_type_norm in deep_types:
        # LSTM/GRU: estimate per-feature importance from first recurrent layer input kernel
        if not hasattr(model_obj, 'layers') or not model_obj.layers:
            raise ValueError(
                f"Model type '{model_type}' expects a Keras model with recurrent layers."
            )

        first_layer = model_obj.layers[0]
        if not hasattr(first_layer, 'get_weights'):
            raise ValueError(
                f"Model type '{model_type}' does not expose layer weights for importance extraction."
            )

        layer_weights = first_layer.get_weights()
        if not layer_weights:
            raise ValueError(
                f"Model type '{model_type}' has no learned weights. Train the model first."
            )

        kernel = np.asarray(layer_weights[0])
        if kernel.ndim != 2:
            raise ValueError(
                f"Unexpected recurrent kernel shape {kernel.shape}; expected 2D input kernel."
            )

        importances = np.mean(np.abs(kernel), axis=1)
        if len(importances) != len(feature_names):
            raise ValueError(
                f"Feature count mismatch: got {len(feature_names)} feature names "
                f"but inferred {len(importances)} recurrent input weights."
            )

        indices = np.argsort(importances)[-top_n:]
        colors = COLOR_STEELBLUE
        values = importances[indices]
        top_features = [feature_names[i] for i in indices]
        xlabel = 'Input Weight Importance (abs mean)'

    else:
        raise ValueError(
            f"Unknown model_type: {model_type}. Supported types: "
            f"linear/ridge/lasso, rf/randomforest, xgboost, catboost, lstm, gru"
        )
    
    # Plot
    plt.figure(figsize=FIGSIZE_FEATURE)
    plt.barh(range(len(values)), values, color=colors, alpha=0.7)
    plt.yticks(range(len(top_features)), top_features, fontsize=FONTSIZE_TICK)
    plt.xlabel(xlabel, fontsize=FONTSIZE_LABEL)
    plt.ylabel('Feature', fontsize=FONTSIZE_LABEL)
    plt.title(f'{model_name} - Top {top_n} Feature Importance', 
              fontsize=FONTSIZE_TITLE, fontweight='bold')
    
    if model_type_norm in linear_types:
        plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    
    plt.tight_layout()
    
    save_plot(save_path) if save_path else plt.show()    
    plt.close()

# ============================================================================
# CLASSIFICATION/THRESHOLD ANALYSIS PLOTS
# ============================================================================

def plot_pr_curve(y_true, y_pred, threshold, model_name='Model', save_path=None):
    """
    Plot Precision-Recall curve for classification models.
    
    Shows the trade-off between precision and recall at different thresholds.
    Marks the current threshold with a red dot.
    
    Parameters:
    -----------
    y_true : array-like
        Actual binary labels (0 or 1)
    y_pred : array-like
        Predicted probabilities [0, 1]
    threshold : float
        Current decision threshold to highlight on curve
    model_name : str, optional
        Display name of the model for title (default: 'Model')
    save_path : str or Path, optional
        Path to save figure. If None, displays in notebook (default: None)
        
    Returns:
    --------
    None
        Displays or saves plot as side effect
        
    Example:
    --------
    >>> from plotter import plot_pr_curve
    >>> plot_pr_curve(y_test, y_pred_proba, threshold=0.5, 
    ...               model_name='Logistic Regression',
    ...               save_path='output/pr_curve.png')
    """
    
    # Calculate precision and recall at all thresholds
    precisions, recalls, thresholds_array = precision_recall_curve(y_true, y_pred)
    
    # Find index of closest threshold
    threshold_idx = np.argmin(np.abs(thresholds_array - threshold))
    
    # Plot
    plt.figure(figsize=FIGSIZE_SINGLE)
    plt.plot(recalls, precisions, linewidth=2, label='PR Curve', color=COLOR_STEELBLUE)
    
    # Mark current threshold
    plt.scatter(recalls[threshold_idx], precisions[threshold_idx], 
               color=COLOR_RED, s=150, zorder=5, 
               label=f'Current Threshold ({threshold:.2f})')
    
    plt.xlabel('Recall', fontsize=FONTSIZE_LABEL)
    plt.ylabel('Precision', fontsize=FONTSIZE_LABEL)
    plt.title(f'{model_name} - Precision-Recall Curve', fontsize=FONTSIZE_TITLE, fontweight='bold')
    plt.legend(fontsize=FONTSIZE_LEGEND)
    plt.grid(True, alpha=0.3)
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.tight_layout()
    
    save_plot(save_path) if save_path else plt.show()    
    plt.close()

def plot_threshold_metrics(thresholds, precisions, recalls, f1s, hit_rates, 
                          model_name='Model', save_path=None):
    """
    Plot threshold analysis metrics for classification models.
    
    Shows how different metrics (Precision, Recall, F1, Hit-rate) 
    vary with different decision thresholds.
    
    Parameters:
    -----------
    thresholds : list or array
        List of threshold values
    precisions : list or array
        Precision values at each threshold
    recalls : list or array
        Recall values at each threshold
    f1s : list or array
        F1-score values at each threshold
    hit_rates : list or array
        Hit-rate values at each threshold
    model_name : str, optional
        Display name of the model for title (default: 'Model')
    save_path : str or Path, optional
        Path to save figure. If None, displays in notebook (default: None)
        
    Returns:
    --------
    None
        Displays or saves plot as side effect
        
    Example:
    --------
    >>> from plotter import plot_threshold_metrics
    >>> plot_threshold_metrics(thresholds, precisions, recalls, f1s, hit_rates,
    ...                        model_name='XGBoost',
    ...                        save_path='output/threshold_metrics.png')
    """
    
    plt.figure(figsize=FIGSIZE_SINGLE)
    plt.plot(thresholds, precisions, label='Precision', marker='o', linewidth=2)
    plt.plot(thresholds, recalls, label='Recall', marker='o', linewidth=2)
    plt.plot(thresholds, f1s, label='F1', marker='o', linewidth=2)
    plt.plot(thresholds, hit_rates, label='Hit-rate on true opps', marker='o', linewidth=2)
    plt.xlabel('Prediction Threshold', fontsize=FONTSIZE_LABEL)
    plt.ylabel('Metric Value', fontsize=FONTSIZE_LABEL)
    plt.title(f'{model_name} - Threshold Analysis', fontsize=FONTSIZE_TITLE, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=FONTSIZE_LEGEND)
    plt.tight_layout()
    
    save_plot(save_path) if save_path else plt.show()    
    plt.close()

# ============================================================================
# XGBOOST-SPECIFIC PLOTS
# ============================================================================

def plot_prediction_history(time_index, y_true, y_pred, model_name='XGBoost', save_path=None):
    """
    Plot prediction history over time for chronological evaluation.
    
    Shows actual vs predicted values across time, useful for understanding
    model performance patterns over the trading period.
    
    Parameters:
    -----------
    time_index : pd.Series or array-like
        Time index (usually datetime)
    y_true : array-like
        Actual target values
    y_pred : array-like
        Predicted target values
    model_name : str, optional
        Display name of the model for title (default: 'XGBoost')
    save_path : str or Path, optional
        Path to save figure. If None, displays in notebook (default: None)
        
    Returns:
    --------
    None
        Displays or saves plot as side effect
        
    Example:
    --------
    >>> from plotter import plot_prediction_history
    >>> plot_prediction_history(time_index, y_test, y_pred, 
    ...                         model_name='XGBoost',
    ...                         save_path='output/xgb_prediction_history.png')
    """
    
    plt.figure(figsize=(14, 6))
    plt.plot(time_index, y_true, label='Actual', linewidth=1.5)
    plt.plot(time_index, y_pred, label='Predicted', linewidth=1.5, alpha=0.8)
    plt.xlabel('Time', fontsize=FONTSIZE_LABEL)
    plt.ylabel('spread_close_pct', fontsize=FONTSIZE_LABEL)
    plt.title(f'{model_name} - Prediction History (Chronological)', 
              fontsize=FONTSIZE_TITLE, fontweight='bold')
    plt.legend(fontsize=FONTSIZE_LEGEND)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_plot(save_path) if save_path else plt.show()    
    plt.close()

def plot_xgb_feature_importance(model, feature_names, top_n=30, model_name='XGBoost', save_path=None):
    """
    Plot XGBoost feature importance using gain metric.
    
    Extracts feature importance from XGBoost's booster object using the 'gain' metric,
    which measures the average improvement in loss function each feature provides.
    Maps internal XGBoost feature names (f0, f1, etc.) back to original column names.
    
    Parameters:
    -----------
    model : xgboost.XGBRegressor
        Trained XGBoost model
    feature_names : list or pd.Index
        Names of all features (in the same order as training data)
    top_n : int, optional
        Number of top features to display (default: 30)
    model_name : str, optional
        Display name of the model for title (default: 'XGBoost')
    save_path : str or Path, optional
        Path to save figure. If None, displays in notebook (default: None)
        
    Returns:
    --------
    None
        Displays or saves plot as side effect
        
    Example:
    --------
    >>> from plotter import plot_xgb_feature_importance
    >>> plot_xgb_feature_importance(xgb_model, feature_names, top_n=30,
    ...                             model_name='XGBoost',
    ...                             save_path='output/xgb_features.png')
    """
    
    booster = model.get_booster()
    scores = booster.get_score(importance_type='gain')
    
    if not scores:
        print("No feature importance scores available")
        return
    
    # Convert to dataframe
    rows = []
    for fname, gain in scores.items():
        rows.append((fname, gain))
    imp_df = pd.DataFrame(rows, columns=['feature', 'gain'])

    # Map XGBoost's internal names (f0, f1, etc.) back to original column names
    name_map = {f'f{i}': name for i, name in enumerate(feature_names)}
    imp_df['feature'] = imp_df['feature'].map(name_map).fillna(imp_df['feature'])
    
    # Sort by gain and keep top_n
    imp_df = imp_df.sort_values('gain', ascending=False).head(top_n)

    # Plot
    plt.figure(figsize=(12, 8))
    plt.barh(range(len(imp_df)), imp_df['gain'], color=COLOR_STEELBLUE, alpha=0.8)
    plt.yticks(range(len(imp_df)), imp_df['feature'], fontsize=FONTSIZE_TICK)
    plt.xlabel('Gain', fontsize=FONTSIZE_LABEL)
    plt.title(f'{model_name} - Top {top_n} Feature Importances', 
              fontsize=FONTSIZE_TITLE, fontweight='bold')
    plt.grid(True, axis='x', alpha=0.3)
    plt.tight_layout()
    
    save_plot(save_path) if save_path else plt.show()    
    plt.close()

