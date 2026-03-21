import argparse
import pandas as pd
import numpy as np
import warnings
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, TimeSeriesSplit
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, precision_recall_curve, average_precision_score

# Import plotting functions from plotter
from models.plotter import (
    plot_results,
    plot_prediction_hist,
    plot_feature_importance,
    plot_pr_curve,
    plot_threshold_metrics
)

# Ignore warnings
warnings.filterwarnings('ignore')


class RandomForestSpreadModel:
    
    def __init__(self, n_estimators=100, max_depth=20, random_state=42):
        """
        Initialize the Random Forest model with specified parameters.
        
        Parameters:
        -----------
        n_estimators : int
            Number of trees in the forest
        max_depth : int
            Maximum depth of trees
        random_state : int
            Random seed for reproducibility
        """
        self.model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)
        self.df = None
        self.X_train = None
        self.y_train = None
        self.X_test = None
        self.y_test = None
        self.feature_names = None
        self.target_name = 'spread_close_pct'
        self.is_fitted = False
              
    def load_data(self, symbol):
        """Load featured data from CSV file"""
        base_path = Path(__file__).parent.parent
        data_path = base_path / 'data' / 'featured_data'
        file_path = data_path / f'featured_{symbol}_data.csv'
        print(f"Loading data from {file_path}...")
        self.df = pd.read_csv(file_path)
        print(f"Data loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        
    def prepare_features(self, exclude_features=None):
        """Prepare features for training"""
        default_exclude = [
            'time', 
            self.target_name,
            'spread_close_absolute',
            'is_opportunity',
            'is_real_opportunity',
            'buy_exchange',
            'sell_exchange',
            'buy_exchange_lag_1',
            'sell_exchange_lag_1',
            'high_exchange',
            'low_exchange',
            'min_close',
            'max_close',
            'price_ratio_buy_sell',
            'opportunity_gap',
            'spread_diff_from_lag_1',
            'spread_diff_from_lag_5',
            'spread_rate_change',
            'spread_rate_change_pct',
            'spread_rate_acceleration'
        ]

        X = self.df.drop(columns=default_exclude, errors='ignore')
        self.feature_names = X.columns.tolist()
        y = self.df[self.target_name]
        
        # 80/20 chronological split
        split_idx = int(len(X) * 0.8)
        self.X_train = X.iloc[:split_idx]
        self.y_train = y.iloc[:split_idx]
        self.X_test = X.iloc[split_idx:]
        self.y_test = y.iloc[split_idx:]
        
        print(f"\nFeatures prepared: {len(self.feature_names)} features")
        print(f"Train samples: {len(self.X_train)}")
        print(f"Test samples: {len(self.X_test)}")

    def train(self):
        """Train the Random Forest model"""
        if self.X_train is None or self.y_train is None:
            raise ValueError("Data not prepared yet, use prepare_features() first.")
        
        print("Training Random Forest Model...")
        self.model.fit(self.X_train, self.y_train)
        self.is_fitted = True
        
        # Training score
        train_score = self.model.score(self.X_train, self.y_train)
        print(f"Training R² Score: {train_score:.4f}")
        
    def predict(self, X):
        """Make predictions"""
        if not self.is_fitted:
            raise ValueError("Model not trained yet, use train() first.")
                             
        predictions = self.model.predict(X)
        return predictions
    
    def evaluate(self):
        """Evaluate the model"""
        y_pred = self.predict(self.X_test)
        
        MSE = mean_squared_error(self.y_test, y_pred)
        MAE = mean_absolute_error(self.y_test, y_pred)
        R2 = r2_score(self.y_test, y_pred)
        
        print(f"\nTest Results:")
        print(f"  MSE: {MSE:.6f}")
        print(f"  MAE: {MAE:.6f}")
        print(f"  R² Score: {R2:.4f}")
        
        return MSE, MAE, R2

    def baseline_metrics(self, y_train, y_test):
        """Compute baseline metrics using mean and median predictors"""
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
        """Inspect errors across buckets of the actual target"""
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

    def opportunity_detection_metrics(self, y_true, y_pred, opp_thresh=0.1, pred_thresh=None, tol=0.002, verbose=True):
        """Evaluate how well the model detects opportunities"""
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
        """Get feature importance from Random Forest"""
        importances = self.model.feature_importances_
        importance_df = pd.DataFrame({'feature': self.feature_names, 'importance': importances})
        importance_df = importance_df.sort_values(by='importance', ascending=False)
        top_features = importance_df.head(top_n)
        
        print(f"\nTop {top_n} Most Important Features:")
        print(top_features.to_string(index=False))
        
        return top_features


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Train Random Forest model for crypto spread prediction')
    parser.add_argument('--symbol', type=str, default='BTCUSD',
                        help='Cryptocurrency symbol (default: BTCUSD)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed for reproducibility (default: 42)')
    parser.add_argument('--threshold', type=float, default=0.3,
                        help='Opportunity threshold (default: 0.3)')
    parser.add_argument('--n-estimators', type=int, default=100,
                        help='Number of trees in the forest (default: 100)')
    parser.add_argument('--max-depth', type=int, default=20,
                        help='Maximum depth of trees (default: 20)')
    return parser.parse_args()

    
def main():
    """Main function to train and evaluate the Random Forest model"""
    args = parse_args()
    
    symbol = args.symbol
    seed = args.seed
    threshold = args.threshold
    n_estimators = args.n_estimators
    max_depth = args.max_depth
    
    np.random.seed(seed)
    
    base_path = Path(__file__).parent.parent
    
    print(f"\n{'='*60}")
    print(f"Training Random Forest for {symbol}")
    print(f"{'='*60}\n")
    
    print(f"Configuration:")
    print(f"  N-estimators: {n_estimators}")
    print(f"  Max-depth: {max_depth}")
    print(f"  Seed: {seed}")
    print(f"  Threshold: {threshold}\n")
    
    # Model initialization
    model = RandomForestSpreadModel(n_estimators=n_estimators, max_depth=max_depth, random_state=seed)
    
    # Load and prepare data
    model.load_data(symbol)
    model.prepare_features()
    
    # Train the model
    model.train()
    
    # Evaluate the model
    model.evaluate()
    
    # Baselines vs model
    baselines = model.baseline_metrics(model.y_train, model.y_test)
    
    # Bucket errors
    model.bucket_errors(model.y_test, model.predict(model.X_test), n_bins=5)
    
    # Make predictions
    y_pred = model.predict(model.X_test)
    
    # Opportunity detection metrics
    model.opportunity_detection_metrics(
        model.y_test,
        y_pred,
        opp_thresh=threshold,
        pred_thresh=threshold,
        tol=0.002
    )
    
    # Precision-Recall sweep
    scores = y_pred
    y_true_bin = (model.y_test >= threshold).astype(int)
    precisions, recalls, thresh = precision_recall_curve(y_true_bin, scores)
    ap = average_precision_score(y_true_bin, scores)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    best_idx = int(np.nanargmax(f1s))
    best_thresh = float(thresh[best_idx]) if best_idx < len(thresh) else float(thresh[-1])
    
    print("\nPrecision–Recall sweep (label opp_thresh={}):")
    print(f"  Average Precision: {ap:.3f}")
    print(f"  Best F1: {f1s[best_idx]:.3f} at pred_thresh={best_thresh:.4f}")
    print(f"  Precision@best: {precisions[best_idx]:.3f} | Recall@best: {recalls[best_idx]:.3f}")
    
    # Threshold table
    thresholds_eval = [max(0.01, threshold - 0.1), threshold - 0.05, threshold, threshold + 0.05, threshold + 0.1, best_thresh]
    thresholds_eval = sorted(list(set(thresholds_eval)))
    precisions_eval, recalls_eval, f1_eval, hit_eval = [], [], [], []
    
    print(f"\nThreshold sweep (opp_thresh={threshold}):")
    for t in thresholds_eval:
        m = model.opportunity_detection_metrics(
            model.y_test, y_pred, opp_thresh=threshold, pred_thresh=float(t), tol=0.002, verbose=False
        )
        precisions_eval.append(m['precision'])
        recalls_eval.append(m['recall'])
        f1_eval.append(m['f1'])
        hit_eval.append(m['hit_rate_on_opps'])
        print(
            f"  thr={float(t):.4f} | P={m['precision']:.3f} R={m['recall']:.3f} "
            f"F1={m['f1']:.3f} hit@tol={m['hit_rate_on_opps']:.3f}"
        )
    
    # Feature importance
    model.get_feature_importance(top_n=20)
    
    # Create output directory
    output_dir = base_path / 'models' / 'ds_model' / 'random-forest' / symbol
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving results to: {output_dir}")
    
    # Plot results using plotter functions
    plot_results(model.y_test, y_pred, model_name='Random Forest',
                 save_path=output_dir / f'rf_{symbol}_results.png')
    
    plot_prediction_hist(y_pred, model_name='Random Forest',
                        save_path=output_dir / f'rf_{symbol}_prediction_hist.png')
    
    plot_feature_importance(model.model, model.feature_names, 'rf',
                           model_name='Random Forest', top_n=20,
                           save_path=output_dir / f'rf_{symbol}_feature_importance.png')
    
    plot_pr_curve(y_true_bin, scores, best_thresh,
                  model_name='Random Forest',
                  save_path=output_dir / f'rf_{symbol}_pr_curve.png')
    
    plot_threshold_metrics(thresholds_eval, precisions_eval, recalls_eval, f1_eval, hit_eval,
                          model_name='Random Forest',
                          save_path=output_dir / f'rf_{symbol}_threshold_metrics.png')
    
    # Cross-validation
    print("\nPerforming time-series cross-validation...")
    tscv = TimeSeriesSplit(n_splits=3)
    cv_model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=seed)
    cv_scores = cross_val_score(cv_model, model.X_train, model.y_train, cv=tscv, scoring='r2')
    print(f"Cross-validation R² scores: {cv_scores}")
    
    cv_scores_clean = cv_scores[cv_scores > -100]
    if len(cv_scores_clean) > 0:
        print(f"Median CV R² Score: {np.median(cv_scores_clean):.4f}")
        print(f"Mean CV R² Score (cleaned): {cv_scores_clean.mean():.4f} (+/- {cv_scores_clean.std() * 2:.4f})")
    
    print(f"\n{'='*60}")
    print("Model training completed successfully!")
    print(f"{'='*60}\n")

    
if __name__ == "__main__":
    main()