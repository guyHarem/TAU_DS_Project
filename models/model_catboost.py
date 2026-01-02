import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, precision_recall_curve, average_precision_score
from catboost import CatBoostRegressor
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')


class CatBoostModel:
    """
    CatBoost Gradient Boosting model to predict spread_close_pct
    """
    
    def __init__(self, iterations=1000, learning_rate=0.03, depth=6, verbose=False):
        """
        Initialize the CatBoost model
        
        Parameters:
        -----------
        iterations : int
            Number of boosting iterations
        learning_rate : float
            Learning rate
        depth : int
            Depth of the trees
        verbose : bool
            Whether to print training progress
        """
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.verbose = verbose
        
        self.model = CatBoostRegressor(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            loss_function='RMSE',
            eval_metric='RMSE',
            random_seed=42,
            verbose=verbose
        )
        
        self.feature_names = None
        self.target_name = 'spread_close_pct'
        self.is_fitted = False
        
    def load_data(self, file_path):
        """
        Load featured data from CSV file
        
        Parameters:
        -----------
        file_path : str
            Path to the CSV file
            
        Returns:
        --------
        pd.DataFrame
            Loaded data
        """
        print(f"Loading data from {file_path}...")
        df = pd.read_csv(file_path)
        print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df
    
    def prepare_features(self, df, exclude_features=None):
        """
        Prepare features for training
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        exclude_features : list
            List of feature names to exclude
            
        Returns:
        --------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
        """
        # Columns to always exclude (to prevent data leakage)
        default_exclude = [
            'time', 
            self.target_name,
            'spread_close_absolute',  # Direct calculation from target
            'is_opportunity',  # Target-related
            'is_opportunity_flag',  # Target-related
            'is_real_opportunity',  # Target-related
            'buy_exchange',  # Categorical
            'sell_exchange',  # Categorical
            'buy_exchange_lag_1',  # Categorical
            'sell_exchange_lag_1',  # Categorical
            'high_exchange',  # Categorical
            'low_exchange',  # Categorical
            'min_close',  # Used to calculate target
            'max_close',  # Used to calculate target
            'price_ratio_buy_sell',  # Directly related to spread
            'opportunity_gap',  # Directly related to spread
            # These features are derived from spread and cause perfect prediction
            'spread_diff_from_lag_1',  # Derived from spread
            'spread_diff_from_lag_5',  # Derived from spread
            'spread_rate_change',  # Derived from spread
            'spread_rate_change_pct',  # Derived from spread
            'spread_rate_acceleration',  # Derived from spread
        ]
        
        if exclude_features:
            default_exclude.extend(exclude_features)
        
        # Remove duplicates
        exclude_cols = list(set(default_exclude))
        
        # Select features
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        
        # Handle missing values
        df_clean = df[feature_cols + [self.target_name]].copy()
        
        # Drop rows with missing target
        df_clean = df_clean.dropna(subset=[self.target_name])
        
        # Check for infinite values and replace with NaN
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
        
        # CatBoost can handle NaN natively, but let's fill for consistency
        for col in feature_cols:
            if df_clean[col].isna().any():
                median_val = df_clean[col].median()
                if pd.isna(median_val):
                    df_clean[col].fillna(0, inplace=True)
                else:
                    df_clean[col].fillna(median_val, inplace=True)
        
        # Final check: drop any remaining rows with NaN
        df_clean = df_clean.dropna()
        
        X = df_clean[feature_cols]
        y = df_clean[self.target_name]
        
        self.feature_names = feature_cols
        
        print(f"\nFeatures prepared: {len(feature_cols)} features")
        print(f"Samples: {X.shape[0]}")
        print(f"Target range: [{y.min():.6f}, {y.max():.6f}]")
        
        return X, y
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the CatBoost model
        
        Parameters:
        -----------
        X_train : pd.DataFrame
            Training features
        y_train : pd.Series
            Training target
        X_val : pd.DataFrame, optional
            Validation features for early stopping
        y_val : pd.Series, optional
            Validation target for early stopping
        """
        print(f"\nTraining CatBoost model (iterations={self.iterations}, lr={self.learning_rate}, depth={self.depth})...")
        
        if X_val is not None and y_val is not None:
            eval_set = (X_val, y_val)
            self.model.fit(X_train, y_train, eval_set=eval_set, early_stopping_rounds=50, verbose=False)
        else:
            self.model.fit(X_train, y_train)
        
        self.is_fitted = True
        
        # Get training score
        train_pred = self.model.predict(X_train)
        train_score = r2_score(y_train, train_pred)
        print(f"Training R² Score: {train_score:.4f}")
        
    def predict(self, X):
        """
        Make predictions
        
        Parameters:
        -----------
        X : pd.DataFrame
            Features for prediction
            
        Returns:
        --------
        np.array
            Predictions
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        
        return self.model.predict(X)
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the model
        
        Parameters:
        -----------
        X_test : pd.DataFrame
            Test features
        y_test : pd.Series
            Test target
            
        Returns:
        --------
        dict
            Dictionary of evaluation metrics
        """
        print("\nEvaluating model...")
        
        y_pred = self.predict(X_test)
        
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'mape': np.mean(np.abs((y_test - y_pred) / (y_test + 1e-9))) * 100
        }
        
        print(f"Test R² Score: {metrics['r2']:.4f}")
        print(f"RMSE: {metrics['rmse']:.6f}")
        print(f"MAE: {metrics['mae']:.6f}")
        print(f"MAPE: {metrics['mape']:.2f}%")
        
        return metrics, y_pred

    def baseline_metrics(self, y_train, y_test):
        """
        Compute baseline metrics using mean and median predictors.
        """
        mean_pred = np.full_like(y_test, y_train.mean())
        median_pred = np.full_like(y_test, np.median(y_train))
        baselines = {}
        for name, pred in [('mean', mean_pred), ('median', median_pred)]:
            baselines[name] = {
                'mae': mean_absolute_error(y_test, pred),
                'rmse': np.sqrt(mean_squared_error(y_test, pred)),
                'r2': r2_score(y_test, pred)
            }
        print("\nBaseline (train mean/median) on test set:")
        for key, vals in baselines.items():
            print(f"  {key.capitalize()} -> MAE: {vals['mae']:.6f}, RMSE: {vals['rmse']:.6f}, R²: {vals['r2']:.4f}")
        return baselines

    def bucket_errors(self, y_true, y_pred, n_bins=5):
        """
        Inspect errors across buckets of the actual target to reveal imbalance effects.
        """
        edges = np.quantile(y_true, np.linspace(0, 1, n_bins + 1))
        edges = np.unique(edges)
        if len(edges) < 2:
            print("Not enough unique values to bucket errors.")
            return None
        bins = pd.cut(y_true, bins=edges, include_lowest=True, duplicates='drop')
        df_bins = pd.DataFrame({'y_true': y_true, 'y_pred': y_pred, 'bin': bins})
        grouped = df_bins.groupby('bin')
        rows = []
        for bin_label, g in grouped:
            mae = mean_absolute_error(g['y_true'], g['y_pred'])
            rmse = np.sqrt(mean_squared_error(g['y_true'], g['y_pred']))
            rows.append((str(bin_label), len(g), mae, rmse, g['y_true'].mean()))
        print("\nPer-bucket errors (by target quantiles):")
        for label, count, mae, rmse, mean_true in rows:
            print(f"  {label} | n={count} | mean_true={mean_true:.4f} | MAE={mae:.6f} | RMSE={rmse:.6f}")
        return rows

    def plot_prediction_hist(self, y_pred, save_path=None):
        """Plot histogram of predictions to detect collapse to a constant."""
        plt.figure(figsize=(10, 6))
        plt.hist(y_pred, bins=40, edgecolor='black', alpha=0.75)
        plt.xlabel('Predicted spread_close_pct', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.title('Prediction Histogram', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Prediction histogram saved to {save_path}")
        else:
            plt.show()
        plt.close()

    def opportunity_detection_metrics(self, y_true, y_pred, opp_thresh=0.1, pred_thresh=None, tol=0.002, verbose=True):
        """
        Evaluate how well the model detects "real opportunities" defined by a threshold on the target.
        """
        if pred_thresh is None:
            pred_thresh = opp_thresh

        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)

        is_opp_true = y_true >= opp_thresh
        is_opp_pred = y_pred >= pred_thresh

        tp = np.sum(is_opp_true & is_opp_pred)
        fp = np.sum(~is_opp_true & is_opp_pred)
        fn = np.sum(is_opp_true & ~is_opp_pred)
        tn = np.sum(~is_opp_true & ~is_opp_pred)

        recall_opp = tp / (tp + fn + 1e-9)
        precision_opp = tp / (tp + fp + 1e-9)
        f1_opp = 2 * precision_opp * recall_opp / (precision_opp + recall_opp + 1e-9)

        hit_rate_on_opps = np.mean((np.abs(y_true - y_pred) <= tol)[is_opp_true]) if np.any(is_opp_true) else np.nan

        if verbose:
            print("\nOpportunity detection (threshold-based):")
            print(f"  opp_thresh (actual): {opp_thresh}")
            print(f"  pred_thresh (prediction): {pred_thresh}")
            print(f"  True opportunities: {is_opp_true.sum()} | Predicted opportunities: {is_opp_pred.sum()}")
            print(f"  Precision: {precision_opp:.3f} | Recall: {recall_opp:.3f} | F1: {f1_opp:.3f}")
            if not np.isnan(hit_rate_on_opps):
                print(f"  Hit-rate on true opportunities within ±{tol}: {hit_rate_on_opps:.3f}")
            else:
                print("  No true opportunities in test set to evaluate hit-rate.")

        return {
            'precision': precision_opp,
            'recall': recall_opp,
            'f1': f1_opp,
            'hit_rate_on_opps': hit_rate_on_opps,
            'tp': tp,
            'fp': fp,
            'fn': fn,
            'tn': tn,
            'n_true_opp': int(is_opp_true.sum()),
            'n_pred_opp': int(is_opp_pred.sum())
        }

    def plot_pr_curve(self, recalls, precisions, ap, save_path=None):
        """Plot Precision-Recall curve with Average Precision annotation."""
        plt.figure(figsize=(8, 6))
        plt.plot(recalls, precisions, label=f'PR curve (AP={ap:.3f})', color='blue')
        plt.xlabel('Recall', fontsize=12)
        plt.ylabel('Precision', fontsize=12)
        plt.title('Precision-Recall Curve', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"PR curve saved to {save_path}")
        else:
            plt.show()
        plt.close()

    def plot_threshold_metrics(self, thresholds, precisions, recalls, f1s, hit_rates, save_path=None):
        """Plot precision, recall, F1, and hit-rate vs threshold."""
        plt.figure(figsize=(10, 6))
        plt.plot(thresholds, precisions, label='Precision', marker='o')
        plt.plot(thresholds, recalls, label='Recall', marker='o')
        plt.plot(thresholds, f1s, label='F1', marker='o')
        plt.plot(thresholds, hit_rates, label='Hit-rate on true opps', marker='o')
        plt.xlabel('Prediction Threshold', fontsize=12)
        plt.ylabel('Metric value', fontsize=12)
        plt.title('Threshold vs Precision/Recall/F1/Hit-rate', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Threshold metrics plot saved to {save_path}")
        else:
            plt.show()
        plt.close()
    
    def get_feature_importance(self, top_n=20):
        """
        Get feature importance from CatBoost
        
        Parameters:
        -----------
        top_n : int
            Number of top features to return
            
        Returns:
        --------
        pd.DataFrame
            Feature importance dataframe
        """
        if not self.is_fitted:
            raise ValueError("Model must be trained first")
        
        importances = self.model.get_feature_importance()
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importances
        })
        
        importance_df = importance_df.sort_values('importance', ascending=False)
        
        print(f"\nTop {top_n} Most Important Features:")
        print(importance_df[['feature', 'importance']].head(top_n).to_string(index=False))
        
        return importance_df.head(top_n)
    
    def plot_results(self, y_test, y_pred, save_path=None):
        """
        Plot prediction results
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Actual vs Predicted
        axes[0, 0].scatter(y_test, y_pred, alpha=0.5, s=10)
        axes[0, 0].plot([y_test.min(), y_test.max()], 
                        [y_test.min(), y_test.max()], 
                        'r--', lw=2, label='Perfect Prediction')
        axes[0, 0].set_xlabel('Actual spread_close_pct', fontsize=12)
        axes[0, 0].set_ylabel('Predicted spread_close_pct', fontsize=12)
        axes[0, 0].set_title('Actual vs Predicted', fontsize=14, fontweight='bold')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. Residuals
        residuals = y_test - y_pred
        axes[0, 1].scatter(y_pred, residuals, alpha=0.5, s=10)
        axes[0, 1].axhline(y=0, color='r', linestyle='--', lw=2)
        axes[0, 1].set_xlabel('Predicted spread_close_pct', fontsize=12)
        axes[0, 1].set_ylabel('Residuals', fontsize=12)
        axes[0, 1].set_title('Residual Plot', fontsize=14, fontweight='bold')
        axes[0, 1].grid(True, alpha=0.3)
        
        # 3. Residual Distribution
        axes[1, 0].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
        axes[1, 0].axvline(x=0, color='r', linestyle='--', lw=2)
        axes[1, 0].set_xlabel('Residuals', fontsize=12)
        axes[1, 0].set_ylabel('Frequency', fontsize=12)
        axes[1, 0].set_title('Residual Distribution', fontsize=14, fontweight='bold')
        axes[1, 0].grid(True, alpha=0.3)
        
        # 4. Prediction Error
        error = np.abs(residuals)
        axes[1, 1].hist(error, bins=50, edgecolor='black', alpha=0.7, color='orange')
        axes[1, 1].set_xlabel('Absolute Error', fontsize=12)
        axes[1, 1].set_ylabel('Frequency', fontsize=12)
        axes[1, 1].set_title('Absolute Error Distribution', fontsize=14, fontweight='bold')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nPlot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()
    
    def plot_feature_importance(self, top_n=20, save_path=None):
        """
        Plot feature importance
        """
        importance_df = self.get_feature_importance(top_n)
        
        fig, ax = plt.subplots(figsize=(12, 8))
        
        ax.barh(range(len(importance_df)), importance_df['importance'], alpha=0.7, color='steelblue')
        ax.set_yticks(range(len(importance_df)))
        ax.set_yticklabels(importance_df['feature'])
        ax.set_xlabel('Importance', fontsize=12)
        ax.set_title(f'Top {top_n} Feature Importances (CatBoost)', 
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"\nFeature importance plot saved to {save_path}")
        else:
            plt.show()
        
        plt.close()


def main():
    """
    Main function to train and evaluate the CatBoost model
    """
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Define paths
    base_path = Path(__file__).parent.parent
    data_path = base_path / 'data' / 'featured_data'
    
    # Choose a cryptocurrency to model
    crypto = 'BTCUSD'
    file_path = data_path / f'featured_{crypto}_data.csv'
    
    # Initialize model
    model = CatBoostModel(iterations=1000, learning_rate=0.03, depth=6, verbose=False)
    
    # Load data
    df = model.load_data(file_path)
    
    # Prepare features
    X, y = model.prepare_features(df)
    
    # Chronological split (no shuffling)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
    
    print(f"\nTrain set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Train model
    model.train(X_train, y_train)
    
    # Evaluate model
    metrics, y_pred = model.evaluate(X_test, y_test)

    # Baselines vs model
    baselines = model.baseline_metrics(y_train, y_test)

    # Bucket errors
    model.bucket_errors(y_test, y_pred, n_bins=5)

    # Opportunity detection metrics
    model.opportunity_detection_metrics(
        y_test,
        y_pred,
        opp_thresh=0.10,
        pred_thresh=0.10,
        tol=0.002
    )

    # Precision-Recall sweep
    scores = y_pred
    y_true_bin = (y_test >= 0.10).astype(int)
    precisions, recalls, thresh = precision_recall_curve(y_true_bin, scores)
    ap = average_precision_score(y_true_bin, scores)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx = int(np.nanargmax(f1s))
    best_thresh = float(thresh[best_idx]) if best_idx < len(thresh) else float(thresh[-1])

    print("\nPrecision–Recall sweep (label opp_thresh=0.10):")
    print(f"  Average Precision: {ap:.3f}")
    print(f"  Best F1: {f1s[best_idx]:.3f} at pred_thresh={best_thresh:.4f}")
    print(f"  Precision@best: {precisions[best_idx]:.3f} | Recall@best: {recalls[best_idx]:.3f}")

    # Recompute opportunity metrics at best threshold
    best_metrics = model.opportunity_detection_metrics(
        y_test,
        y_pred,
        opp_thresh=0.10,
        pred_thresh=best_thresh,
        tol=0.002,
        verbose=False
    )

    # Threshold table
    thresholds_eval = [0.05, 0.075, 0.10, 0.125, 0.15, best_thresh]
    precisions_eval, recalls_eval, f1_eval, hit_eval = [], [], [], []
    for t in thresholds_eval:
        m = model.opportunity_detection_metrics(
            y_test, y_pred, opp_thresh=0.10, pred_thresh=float(t), tol=0.002, verbose=False
        )
        precisions_eval.append(m['precision'])
        recalls_eval.append(m['recall'])
        f1_eval.append(m['f1'])
        hit_eval.append(m['hit_rate_on_opps'])
        print(
            f"  thr={float(t):.4f} | P={m['precision']:.3f} R={m['recall']:.3f} "
            f"F1={m['f1']:.3f} hit@tol={m['hit_rate_on_opps']:.3f}"
        )

    # Output directory for plots
    output_path = base_path / 'models' / 'ds_model' / 'catboost' / crypto
    if not output_path.exists():
        output_path.mkdir(parents=True, exist_ok=True)

    # Save visuals
    pr_curve_path = output_path / f'catboost_{crypto}_pr_curve.png'
    model.plot_pr_curve(recalls, precisions, ap, save_path=pr_curve_path)

    threshold_plot_path = output_path / f'catboost_{crypto}_threshold_metrics.png'
    model.plot_threshold_metrics(
        thresholds_eval,
        precisions_eval,
        recalls_eval,
        f1_eval,
        hit_eval,
        save_path=threshold_plot_path
    )
    
    # Get feature importance
    model.get_feature_importance(top_n=20)
    
    # Plot results
    model.plot_results(
        y_test, 
        y_pred, 
        save_path=output_path / f'catboost_{crypto}_results.png'
    )

    model.plot_prediction_hist(
        y_pred,
        save_path=output_path / f'catboost_{crypto}_prediction_hist.png'
    )
    
    model.plot_feature_importance(
        top_n=20, 
        save_path=output_path / f'catboost_{crypto}_feature_importance.png'
    )
    
    # Cross-validation
    print("\nPerforming time-series cross-validation...")
    tscv = TimeSeriesSplit(n_splits=3)
    cv_scores = []
    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_fold_train, X_fold_val = X.iloc[train_idx], X.iloc[val_idx]
        y_fold_train, y_fold_val = y.iloc[train_idx], y.iloc[val_idx]
        
        fold_model = CatBoostRegressor(
            iterations=model.iterations,
            learning_rate=model.learning_rate,
            depth=model.depth,
            loss_function='RMSE',
            random_seed=42,
            verbose=False
        )
        fold_model.fit(X_fold_train, y_fold_train)
        y_fold_pred = fold_model.predict(X_fold_val)
        fold_r2 = r2_score(y_fold_val, y_fold_pred)
        cv_scores.append(fold_r2)
    
    cv_scores = np.array(cv_scores)
    print(f"Cross-validation R² scores: {cv_scores}")
    
    # Robust CV reporting
    cv_scores_clean = cv_scores[cv_scores > -100]
    if len(cv_scores_clean) > 0:
        print(f"Median CV R² Score: {np.median(cv_scores_clean):.4f}")
        print(f"Mean CV R² Score (cleaned): {cv_scores_clean.mean():.4f} (+/- {cv_scores_clean.std() * 2:.4f})")
    if len(cv_scores_clean) < len(cv_scores):
        print(f"  (Note: {len(cv_scores) - len(cv_scores_clean)} fold(s) had extreme negative R² and were excluded.)")
    
    print("\n" + "="*60)
    print("Model training completed successfully!")
    print("="*60)


if __name__ == '__main__':
    main()
