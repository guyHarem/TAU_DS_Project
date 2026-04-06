"""Train a CatBoost gradient boosting model to predict `is_real_opportunity`.

Usage (example):
    python models/model_catboost.py --symbol BTCUSD --iterations 1000 --seed 42
"""

#region Imports
import argparse
import pandas as pd
import numpy as np
from pathlib import Path

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (
    mean_squared_error,
    precision_recall_curve,
    average_precision_score,
    accuracy_score,
    roc_auc_score,
    f1_score,
)
import joblib
from catboost import CatBoostClassifier
import shutil
import warnings

# Import plotting functions from plotter
from plotter import (
    plot_results,
    plot_prediction_hist,
    plot_feature_importance,
    plot_pr_curve,
    plot_threshold_metrics
)
#endregion

warnings.filterwarnings('ignore')

ROOT_PATH = Path(__file__).resolve().parent.parent
DATA_PATH = ROOT_PATH / 'data' / 'featured_data'
MODEL_PLOT_PATH = ROOT_PATH / 'models' / 'ds_model' / 'catboost'

class CatBoostModel:
    """
    CatBoost classifier to predict next-minute is_real_opportunity
    """
    
    def __init__(self, iterations=1000, learning_rate=0.03, depth=6, random_state=42, verbose=False, decision_threshold=0.5):
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
        random_state : int
            Random seed for reproducibility
        verbose : bool
            Whether to print training progress
        decision_threshold : float
            Default probability threshold for converting probabilities to class labels
        """
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.random_state = random_state
        self.verbose = verbose
        self.decision_threshold = decision_threshold
        
        self.model = CatBoostClassifier(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            loss_function='Logloss',
            eval_metric='AUC',
            random_seed=random_state,
            verbose=verbose
        )
        
        self.feature_names = None
        self.target_name = 'is_real_opportunity'
        self.is_fitted = False
        
    def load_data(self, symbol):
        """
        Load featured data from CSV file
        
        Parameters:
        -----------
        symbol : str
            Cryptocurrency symbol
            
        Returns:
        --------
        pd.DataFrame
            Loaded data
        """
        file_path = DATA_PATH / f'featured_{symbol}_data.csv'
        if not file_path.exists():
            available = sorted(p.name for p in DATA_PATH.glob("featured_*_data.csv"))
            raise FileNotFoundError(
                f"Could not find {file_path.name}. Available files: {available}"
            )
        
        df = pd.read_csv(file_path)
        return df
    
    def prepare_features(self, df, split_ratio=0.6, exclude_features=None):
        """
        Prepare features for training
        
        Parameters:
        -----------
        df : pd.DataFrame
            Input dataframe
        exclude_features : list
            List of feature names to exclude
        include_current_target_feature : bool
            If True, include current minute target as an autoregressive feature
            
        Returns:
        --------
        X : pd.DataFrame
            Feature matrix
        y : pd.Series
            Target variable
        """
        # Columns to always exclude from model inputs.
        # Exchange identifiers are object/string columns and are excluded here
        # because this training path does not pass categorical feature indices to CatBoost.
        default_exclude = [
            'time', 
            'buy_exchange',
            'sell_exchange',
            'buy_exchange_lag_1',
            'sell_exchange_lag_1',
            'high_exchange',
            'low_exchange',
            'num_exchanges_available',
        ]
        
        if exclude_features:
            default_exclude.extend(exclude_features)
        
        # Remove duplicates and never include the target in feature_cols by default.
        exclude_cols = list(set(default_exclude))
        feature_cols = [col for col in df.columns if col not in exclude_cols]

        # Define X, y
        X = df[feature_cols].copy()
        y = df[self.target_name].shift(-1)

        # Keep only rows with available next-minute target.
        valid_y_mask = y.notna()
        X = X.loc[valid_y_mask]
        y = y.loc[valid_y_mask]

        # Fill missing numeric feature values with median.
        for col in feature_cols:
            if X[col].isna().any():
                if pd.api.types.is_numeric_dtype(X[col]):
                    median_val = X[col].median()
                    if pd.isna(median_val):
                        X[col] = X[col].fillna(0)
                    else:
                        X[col] = X[col].fillna(median_val)
                else:
                    mode_vals = X[col].mode(dropna=True)
                    fill_val = mode_vals.iloc[0] if not mode_vals.empty else 'unknown'
                    X[col] = X[col].fillna(fill_val)

        # Drop rows that still have NaN after imputation and keep X/y aligned.
        final_mask = ~X.isna().any(axis=1)
        X = X.loc[final_mask]
        y = y.loc[final_mask]
        y = y.astype(int)

        # Chronological split
        split_idx = int(len(X) * split_ratio)
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]
        
        self.feature_names = feature_cols
        
        print(f"\nFeatures prepared: {len(feature_cols)} features")
        print(f"Samples: {X.shape[0]}")
        print(f"Target range: [{y.min():.6f}, {y.max():.6f}]")
        
        return X_train, X_test, y_train, y_test
    
    def predict_proba(self, X):
        """Return positive-class probabilities for the provided features."""
        if not self.is_fitted:
            raise ValueError("Model must be trained before making predictions")
        return self.model.predict_proba(X)[:, 1]
    
    def predict(self, X, threshold=None):
        """Return binary predictions using a probability threshold."""
        if threshold is None:
            threshold = self.decision_threshold
        y_prob = self.predict_proba(X)
        return (y_prob >= threshold).astype(int)
    
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
            self.model.fit(X_train, y_train, verbose=False)
        
        self.is_fitted = True
        
        # Get training metrics
        train_prob = self.predict_proba(X_train)
        train_pred = self.predict(X_train)
        train_acc = accuracy_score(y_train, train_pred)
        print(f"Training Accuracy: {train_acc:.4f}")
        if len(np.unique(y_train)) > 1:
            train_auc = roc_auc_score(y_train, train_prob)
            print(f"Training ROC-AUC: {train_auc:.4f}")

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

        y_prob = self.predict_proba(X_test)
        y_pred = self.predict(X_test)
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'brier': mean_squared_error(y_test, y_prob),
        }

        if len(np.unique(y_test)) > 1:
            metrics['roc_auc'] = roc_auc_score(y_test, y_prob)
        
        print(f"Test Accuracy: {metrics['accuracy']:.4f}")
        print(f"Test F1: {metrics['f1']:.4f}")
        if 'roc_auc' in metrics:
            print(f"Test ROC-AUC: {metrics['roc_auc']:.4f}")
        print(f"Brier score: {metrics['brier']:.6f}")
        
        return metrics, y_prob

    def baseline_metrics(self, y_train, y_test):
        """Compute baseline metrics using majority class predictor."""
        majority_class = int(np.bincount(y_train).argmax())
        majority_pred = np.full_like(y_test, majority_class)
        
        baselines = {
            'majority_class': {
                'accuracy': accuracy_score(y_test, majority_pred),
                'f1': f1_score(y_test, majority_pred, zero_division=0),
            }
        }
        
        print("\nBaseline (majority class) on test set:")
        print(f"  Majority class: {majority_class}")
        print(f"  Accuracy: {baselines['majority_class']['accuracy']:.4f} | F1: {baselines['majority_class']['f1']:.4f}")
        return baselines

    def bucket_classification_metrics(self, y_true, y_prob, n_bins=5):
        """Inspect calibration across probability buckets. (are scores trustworthy)"""

        y_true = np.asarray(y_true).astype(int)
        y_prob = np.asarray(y_prob)
        if len(y_true) == 0:
            print("No samples available for bucket diagnostics.")
            return None

        # Quantile buckets keep similar sample sizes per bucket.
        probs = pd.Series(y_prob)
        bins = pd.qcut(probs, q=n_bins, duplicates='drop')
        if bins.isna().all():
            print("Not enough probability spread to create buckets.")
            return None

        df_bins = pd.DataFrame({
            'y_true': y_true,
            'y_prob': y_prob,
            'bin': bins,
        })

        grouped = df_bins.groupby('bin', observed=False)
        rows = []
        for bin_label, g in grouped:
            if len(g) == 0:
                continue
            avg_prob = float(g['y_prob'].mean())
            pos_rate = float(g['y_true'].mean())
            rows.append({
                'bucket': str(bin_label),
                'count': int(len(g)),
                'avg_pred_prob': avg_prob,
                'actual_pos_rate': pos_rate,
                'calibration_gap': abs(avg_prob - pos_rate),
            })

        print("\nPer-bucket calibration diagnostics (probability quantiles):")
        for r in rows:
            print(
                f"  {r['bucket']} | n={r['count']} | avg_p={r['avg_pred_prob']:.3f} "
                f"| pos_rate={r['actual_pos_rate']:.3f} | gap={r['calibration_gap']:.3f} "
            )
        return rows

    def opportunity_detection_metrics(self, y_true, y_pred, opp_thresh=0.1, pred_thresh=None, tol=0.002, verbose=True):
        """
        Evaluate how well the model detects "real opportunities" defined by a threshold on the target (is yes/no a good rule).
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

    def save_all_plots(self, symbol, y_test, y_prob, y_true_bin,
                       scores, best_thresh, thresholds_eval, precisions_eval,
                       recalls_eval, f1_eval, hit_eval):
        """Save all analysis plots for the current run."""
        output_path = MODEL_PLOT_PATH / symbol
        output_path.mkdir(parents=True, exist_ok=True)
        print(f"\nSaving results to: {output_path}")

        plot_results(y_test, y_prob, model_name='CatBoost',
                     save_path=output_path / f'catboost_{symbol}_results.png')

        plot_prediction_hist(y_prob, model_name='CatBoost',
                             save_path=output_path / f'catboost_{symbol}_prediction_hist.png')

        plot_feature_importance(self.model, self.feature_names, 'catboost',
                                model_name='CatBoost', top_n=20,
                                save_path=output_path / f'catboost_{symbol}_feature_importance.png')

        plot_pr_curve(y_true_bin, scores, best_thresh,
                      model_name='CatBoost',
                      save_path=output_path / f'catboost_{symbol}_pr_curve.png')

        plot_threshold_metrics(thresholds_eval, precisions_eval, recalls_eval, f1_eval, hit_eval,
                               model_name='CatBoost',
                               save_path=output_path / f'catboost_{symbol}_threshold_metrics.png')

    def save_model(self, symbol):
        """Save trained model artifact to disk."""
        output_path = MODEL_PLOT_PATH / symbol
        output_path.mkdir(parents=True, exist_ok=True)
        model_path = output_path / f"catboost_{symbol}_model.joblib"
        joblib.dump(self, model_path)
        print(f"Model saved to: {model_path}")


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train CatBoost model for crypto spread prediction')
    parser.add_argument('--symbol', type=str, default='BTCUSD',
                        help='Cryptocurrency symbol (default: BTCUSD)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--threshold', type=float, default=0.3,
                        help='Opportunity threshold (default: 0.3)')
    parser.add_argument('--iterations', type=int, default=1000,
                        help='Number of iterations for CatBoost (default: 1000)')
    parser.add_argument('--learning-rate', type=float, default=0.03,
                        help='Learning rate for CatBoost (default: 0.03)')
    parser.add_argument('--depth', type=int, default=6,
                        help='Depth of trees for CatBoost (default: 6)')
    parser.add_argument('--decision-threshold', type=float, default=0.5,
                        help='Probability threshold for class predictions (default: 0.5)')
    
    return parser.parse_args()

def clean_catboost_info():
    catboost_info_path = ROOT_PATH / 'catboost_info'
    if catboost_info_path.exists():
        shutil.rmtree(catboost_info_path, ignore_errors=True)
        print(f"Removed temporary folder: {catboost_info_path}")


def main():
    """
    Main function to train and evaluate the CatBoost model
    """
    args = parse_args()
    
    symbol = args.symbol
    seed = args.seed
    threshold = args.threshold
    iterations = args.iterations
    learning_rate = args.learning_rate
    depth = args.depth
    decision_threshold = args.decision_threshold
    
    np.random.seed(seed)
    
    print(f"\n{'='*60}")
    print(f"Training CatBoost for {symbol}")
    print(f"{'='*60}\n")
    
    print(f"Configuration:")
    print(f"  Iterations: {iterations}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Depth: {depth}")
    print(f"  Seed: {seed}")
    print(f"  Threshold: {threshold}")
    print(f"  Decision threshold: {decision_threshold}\n")
    
    # Initialize model
    model = CatBoostModel(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        random_state=seed,
        verbose=False,
        decision_threshold=decision_threshold
    )
    
    # Load data
    df = model.load_data(symbol)
    
    # Prepare features and chronological split
    X_train, X_test, y_train, y_test = model.prepare_features(df)
    
    print(f"Train set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Train model
    model.train(X_train, y_train)
    
    # Evaluate model
    metrics, y_prob = model.evaluate(X_test, y_test)

    # Baselines vs model
    baselines = model.baseline_metrics(y_train, y_test)

    # Probability-bucket diagnostics
    model.bucket_classification_metrics(y_test, y_prob, n_bins=5)

    # Precision-Recall sweep
    scores = y_prob
    y_true_bin = (y_test >= threshold).astype(int)
    precisions, recalls, thresh = precision_recall_curve(y_true_bin, scores)
    ap = average_precision_score(y_true_bin, scores)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx = int(np.nanargmax(f1s))
    best_thresh = float(thresh[best_idx]) if best_idx < len(thresh) else float(thresh[-1])

    print("\nPrecision–Recall sweep (label opp_thresh={}):")
    print(f"  Average Precision: {ap:.3f}")
    print(f"  Best F1: {f1s[best_idx]:.3f} at pred_thresh={best_thresh:.4f}")
    print(f"  Precision@best: {precisions[best_idx]:.3f} | Recall@best: {recalls[best_idx]:.3f}")

    # Threshold table and arrays for plotting
    thresholds_eval = [
        max(0.01, decision_threshold - 0.1),
        max(0.01, decision_threshold - 0.05),
        decision_threshold,
        min(0.99, decision_threshold + 0.05),
        min(0.99, decision_threshold + 0.1),
        best_thresh,
    ]
    thresholds_eval = sorted(list(set(thresholds_eval)))
    precisions_eval, recalls_eval, f1_eval, hit_eval = [], [], [], []

    print(f"\nThreshold sweep (opp_thresh={threshold}):")
    for t in thresholds_eval:
        m = model.opportunity_detection_metrics(
            y_test, y_prob, opp_thresh=threshold, pred_thresh=float(t), tol=0.002, verbose=False
        )
        precisions_eval.append(m['precision'])
        recalls_eval.append(m['recall'])
        f1_eval.append(m['f1'])
        hit_eval.append(m['hit_rate_on_opps'])
        print(
            f"  thr={float(t):.4f} | P={m['precision']:.3f} R={m['recall']:.3f} "
            f"F1={m['f1']:.3f} hit@tol={m['hit_rate_on_opps']:.3f}"
        )

    model.save_all_plots(
        symbol=symbol,
        y_test=y_test,
        y_prob=y_prob,
        y_true_bin=y_true_bin,
        scores=scores,
        best_thresh=best_thresh,
        thresholds_eval=thresholds_eval,
        precisions_eval=precisions_eval,
        recalls_eval=recalls_eval,
        f1_eval=f1_eval,
        hit_eval=hit_eval,
    )
    
    # Save the model
    model.save_model(symbol)
    
    # Cross-validation
    print("\nPerforming time-series cross-validation...")
    X_full = pd.concat([X_train, X_test], axis=0)
    y_full = pd.concat([y_train, y_test], axis=0)

    tscv = TimeSeriesSplit(n_splits=3)
    cv_scores = []
    for fold_idx, (train_idx, val_idx) in enumerate(tscv.split(X_full)):
        X_fold_train, X_fold_val = X_full.iloc[train_idx], X_full.iloc[val_idx]
        y_fold_train, y_fold_val = y_full.iloc[train_idx], y_full.iloc[val_idx]
        
        fold_model = CatBoostClassifier(
            iterations=iterations,
            learning_rate=learning_rate,
            depth=depth,
            loss_function='Logloss',
            random_seed=seed,
            verbose=False
        )
        fold_model.fit(X_fold_train, y_fold_train, verbose=False)
        y_fold_prob = fold_model.predict_proba(X_fold_val)[:, 1]
        if len(np.unique(y_fold_val)) > 1:
            fold_auc = roc_auc_score(y_fold_val, y_fold_prob)
            cv_scores.append(fold_auc)
        else:
            y_fold_pred = (y_fold_prob >= model.decision_threshold).astype(int)
            fold_acc = accuracy_score(y_fold_val, y_fold_pred)
            cv_scores.append(fold_acc)
    
    cv_scores = np.array(cv_scores)
    print(f"Cross-validation scores (AUC when possible, else Accuracy): {cv_scores}")
    if len(cv_scores) > 0:
        print(f"Median CV score: {np.median(cv_scores):.4f}")
        print(f"Mean CV score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    print(f"\n{'='*60}")
    print("Model training completed successfully!")
    print(f"{'='*60}\n")

    # Clean up CatBoost training metadata directory created in the project root.
    clean_catboost_info()


if __name__ == '__main__':
    main()
