"""Train an XGBoost regression model to predict `spread_close_pct`.

The script loads a featured dataset, keeps chronological ordering, trains an
XGBoost regressor, and evaluates classification metrics on the derived
`is_real_opportunity` target (or a threshold on `spread_close_pct`).

Usage (example):
    python models/model_xgboost.py --symbol BTCUSD --threshold 0.6 --seed 42
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
import joblib
from xgboost import XGBRegressor

# Import ALL plotting functions from plotter
from models.plotter import (
    plot_results,
    plot_prediction_hist,
    plot_pr_curve,
    plot_threshold_metrics,
    plot_prediction_history,
    plot_xgb_feature_importance
)


REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data" / "featured_data"


def load_featured(symbol: str) -> pd.DataFrame:
    """Load and time-sort the featured dataset for a trading pair."""

    path = DATA_DIR / f"featured_{symbol}_data.csv"
    if not path.exists():
        available = sorted(p.name for p in DATA_DIR.glob("featured_*_data.csv"))
        raise FileNotFoundError(
            f"Could not find {path.name}. Available files: {available}"
        )

    df = pd.read_csv(path)
    if "time" not in df.columns:
        raise ValueError("Expected a 'time' column for chronological sorting.")

    df["time"] = pd.to_datetime(df["time"], errors="coerce")
    df = df.sort_values("time").reset_index(drop=True)
    return df


def prepare_features(
    df: pd.DataFrame, threshold: float
) -> Tuple[pd.DataFrame, pd.Series, pd.Series, pd.DataFrame]:
    """Build feature matrix and regression/class targets, removing non-finite rows.

    - Drops target and obvious leakage columns (time and exchange labels).
    - Replaces inf/-inf with NaN and drops rows with non-finite feature/target values.
    - Keeps only numeric columns for XGBoost.
    Returns the cleaned feature matrix, regression target, classification target, and
    the filtered dataframe (for time/index alignment downstream).
    """

    df = df.copy()
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    target_col = "spread_close_pct"
    if target_col not in df.columns:
        raise ValueError(f"'{target_col}' not found in dataframe.")

    y_reg = df[target_col].shift(-1).astype(float)
    if "is_real_opportunity" in df.columns:
        y_cls = df["is_real_opportunity"].astype(int)
    else:
        y_cls = (y_reg >= threshold).astype(int)

    # Remove columns that are targets, time, categorical labels, or direct/derived
    # leak paths into the target (mirrors linear-model exclusions to keep fairness).
    drop_cols = {
        target_col,
        "time",
        # Target-related flags
        "is_opportunity",
        "is_real_opportunity",
        # Categorical exchange labels
        "buy_exchange",
        "sell_exchange",
        "high_exchange",
        "low_exchange",
        "buy_exchange_lag_1",
        "sell_exchange_lag_1",
        # Direct target components
        "min_close",
        "max_close",
        "spread_close_absolute",
        "price_ratio_buy_sell",
        "opportunity_gap",
        # Derived-from-target features that can cause leakage
        "spread_diff_from_lag_1",
        "spread_diff_from_lag_5",
        "spread_rate_change",
        "spread_rate_change_pct",
        "spread_rate_acceleration",
    }

    feature_df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    numeric_cols = feature_df.select_dtypes(include=[np.number]).columns.tolist()
    if not numeric_cols:
        raise ValueError("No numeric feature columns remain after filtering.")

    feature_df = feature_df[numeric_cols]

    y_reg = y_reg.fillna(y_reg.median())
    y_cls = y_cls.fillna(y_cls.median()).astype(int)
    df = df.fillna(df.select_dtypes(include=[np.number]).median())

    return feature_df, y_reg, y_cls, df


def chronological_split(
    X: pd.DataFrame,
    y_reg: pd.Series,
    y_cls: pd.Series,
    train_frac: float,
    val_frac: float,
) -> Tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
    pd.Series,
    pd.Series,
    pd.Series,
    pd.Series,
]:
    """Split data chronologically into train/val/test segments."""

    if train_frac <= 0 or val_frac < 0 or train_frac + val_frac >= 1:
        raise ValueError("Fractions must satisfy: train>0, val>=0, train+val<1.")

    n = len(X)
    train_end = int(n * train_frac)
    val_end = train_end + int(n * val_frac)

    X_train, X_val, X_test = X.iloc[:train_end], X.iloc[train_end:val_end], X.iloc[val_end:]
    y_reg_train, y_reg_val, y_reg_test = y_reg.iloc[:train_end], y_reg.iloc[train_end:val_end], y_reg.iloc[val_end:]
    y_cls_train, y_cls_val, y_cls_test = y_cls.iloc[:train_end], y_cls.iloc[train_end:val_end], y_cls.iloc[val_end:]

    if len(X_test) == 0:
        raise ValueError("Not enough rows for the requested split; reduce train_frac/val_frac.")

    # Fit missing-value imputation on train, apply to all splits.
    medians = X_train.median()
    X_train = X_train.fillna(medians)
    X_val = X_val.fillna(medians)
    X_test = X_test.fillna(medians)

    return (
        X_train,
        X_val,
        X_test,
        y_reg_train,
        y_reg_val,
        y_reg_test,
        y_cls_train,
        y_cls_val,
        y_cls_test,
    )


def compute_opportunity_metrics(
    y_true: pd.Series,
    y_pred: np.ndarray,
    opp_thresh: float,
    pred_thresh: float,
    tol: float = 0.002,
) -> Dict[str, float]:
    """Precision/recall/F1/hit-rate on true opportunities."""

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    is_opp_true = y_true_arr >= opp_thresh
    is_opp_pred = y_pred_arr >= pred_thresh

    tp = int(np.sum(is_opp_true & is_opp_pred))
    fp = int(np.sum(~is_opp_true & is_opp_pred))
    fn = int(np.sum(is_opp_true & ~is_opp_pred))
    tn = int(np.sum(~is_opp_true & ~is_opp_pred))

    precision_opp = tp / (tp + fp + 1e-9)
    recall_opp = tp / (tp + fn + 1e-9)
    f1_opp = 2 * precision_opp * recall_opp / (precision_opp + recall_opp + 1e-9)
    hit_rate_on_opps = (
        np.mean((np.abs(y_true_arr - y_pred_arr) <= tol)[is_opp_true])
        if np.any(is_opp_true)
        else np.nan
    )

    return {
        "precision": precision_opp,
        "recall": recall_opp,
        "f1": f1_opp,
        "hit_rate_on_opps": hit_rate_on_opps,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "n_true_opp": int(is_opp_true.sum()),
        "n_pred_opp": int(is_opp_pred.sum()),
    }


def fraction_within_tolerance(y_true: pd.Series, y_pred: np.ndarray, tol: float) -> Tuple[float, int, int]:
    """Return fraction, hits, total for predictions within ±tol of target."""

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    total = len(y_true_arr)
    if total == 0:
        return float("nan"), 0, 0
    hits = int(np.sum(np.abs(y_true_arr - y_pred_arr) <= tol))
    frac = hits / total
    return frac, hits, total


def regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    """Compute MAE, RMSE, R², and MAPE (safe for zeros)."""

    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2 = r2_score(y_true, y_pred)
    with np.errstate(divide="ignore", invalid="ignore"):
        mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return {"mae": mae, "rmse": rmse, "r2": r2, "mape": mape}


def evaluate_classification(
    y_true: pd.Series, y_pred_reg: np.ndarray, threshold: float
) -> Dict[str, float]:
    """Convert regression outputs to opportunity flags and compute metrics."""

    y_pred_cls = (y_pred_reg >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred_cls),
        "precision": precision_score(y_true, y_pred_cls, zero_division=0),
        "recall": recall_score(y_true, y_pred_cls, zero_division=0),
        "f1": f1_score(y_true, y_pred_cls, zero_division=0),
    }
    metrics["confusion_matrix"] = confusion_matrix(y_true, y_pred_cls).tolist()
    return metrics


def summarize_class_balance(name: str, y: pd.Series) -> str:
    """Summarize class balance for binary classification."""
    total = len(y)
    pos = int(y.sum())
    neg = total - pos
    pos_pct = (pos / total * 100) if total else 0.0
    neg_pct = (neg / total * 100) if total else 0.0
    return f"{name}: {pos} pos ({pos_pct:.2f}%), {neg} neg ({neg_pct:.2f}%), total={total}"


def train_xgb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    y_cls_train,
    seed: int,
) -> XGBRegressor:
    """Train an XGBoost regressor with squared-error objective."""

    pos_weight = (len(y_cls_train) - y_cls_train.sum()) / (y_cls_train.sum() + 1e-9)
    sample_weights = np.where(y_cls_train == 1, pos_weight, 1.0)
    
    model = XGBRegressor(
        n_estimators=600,
        learning_rate=0.03,
        max_depth=5,
        min_child_weight=1,
        subsample=0.7,
        colsample_bytree=0.7,
        gamma=0.1,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="reg:squarederror",
        random_state=seed,
        eval_metric="rmse",
        tree_method="hist",
        scale_pos_weight=pos_weight,
    )

    model.fit(
        X_train,
        y_train,
        sample_weight=sample_weights,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=False,
    )
    return model


def run(symbol: str, threshold: float, train_frac: float, val_frac: float, seed: int) -> None:
    """Run the full XGBoost pipeline."""
    
    print(f"\n{'='*60}")
    print(f"Training XGBoost for {symbol}")
    print(f"{'='*60}\n")
    
    print(f"Configuration:")
    print(f"  Threshold: {threshold}")
    print(f"  Train fraction: {train_frac}")
    print(f"  Val fraction: {val_frac}")
    print(f"  Seed: {seed}\n")
    
    df_raw = load_featured(symbol)
    X, y_reg, y_cls, df = prepare_features(df_raw, threshold=threshold)

    n = len(df)
    if n < 3:
        print(f"Not enough rows (n={n}). Need at least 3 rows for train/val/test. Skipping.")
        return

    # Compute split sizes and ensure each split has at least one row
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    n_test = n - n_train - n_val

    if n_train < 1:
        n_train = 1
        n_test = n - n_train - n_val

    if n_val < 1:
        n_val = 1
        n_test = n - n_train - n_val

    if n_test < 1:
        n_test = 1
        if n_train >= n_val and n_train > 1:
            n_train -= 1
        elif n_val > 1:
            n_val -= 1
        n_test = n - n_train - n_val

    if n_train <= 0 or n_val <= 0 or n_test <= 0:
        n_train = max(1, int(n * 0.7))
        n_val = max(1, int(n * 0.15))
        n_test = n - n_train - n_val
        if n_test <= 0:
            n_test = 1
            if n_train > n_val and n_train > 1:
                n_train -= 1
            elif n_val > 1:
                n_val -= 1

    val_end = n_train + n_val
    print(f"Using split sizes -> train: {n_train}, val: {n_val}, test: {n_test} (n={n})")

    (
        X_train,
        X_val,
        X_test,
        y_reg_train,
        y_reg_val,
        y_reg_test,
        y_cls_train,
        y_cls_val,
        y_cls_test,
    ) = chronological_split(X, y_reg, y_cls, train_frac=train_frac, val_frac=val_frac)

    print("Data balance (is_real_opportunity or thresholded spread_close_pct):")
    print("  " + summarize_class_balance("train", y_cls_train))
    print("  " + summarize_class_balance("val", y_cls_val))
    print("  " + summarize_class_balance("test", y_cls_test))

    model = train_xgb(X_train, y_reg_train, X_val, y_reg_val, y_cls_train, seed)

    test_preds = model.predict(X_test)
    
    # Overall regression metrics
    reg_all = regression_metrics(y_reg_test, test_preds)
    print(
        "\nRegression metrics on test: "
        f"MAE={reg_all['mae']:.6f}, RMSE={reg_all['rmse']:.6f}, R2={reg_all['r2']:.4f}, "
        f"MAPE={reg_all['mape']:.2f}%"
    )

    # Regression metrics on is_real_opportunity=1 (if available in test)
    if "is_real_opportunity" in df.columns:
        opp_mask_test = df["is_real_opportunity"].iloc[val_end:] == 1
        opp_total = int(opp_mask_test.sum())
        if opp_total > 0:
            reg_opp = regression_metrics(y_reg_test[opp_mask_test.values], test_preds[opp_mask_test.values])
            print(
                "Regression metrics on is_real_opportunity=1 rows: "
                f"MAE={reg_opp['mae']:.6f}, RMSE={reg_opp['rmse']:.6f}, "
                f"R2={reg_opp['r2']:.4f}, MAPE={reg_opp['mape']:.2f}% (n={opp_total})"
            )
        else:
            print("No is_real_opportunity=1 rows in test split; skipping opportunity-only regression metrics.")

    metrics = evaluate_classification(y_cls_test, test_preds, threshold=threshold)

    print("\nClassification metrics on test (pred >= threshold):")
    print(
        f"  accuracy={metrics['accuracy']:.4f}, "
        f"precision={metrics['precision']:.4f}, "
        f"recall={metrics['recall']:.4f}, "
        f"f1={metrics['f1']:.4f}"
    )
    print(f"  confusion_matrix={metrics['confusion_matrix']}")

    print("\nDetailed report:")
    print(classification_report(y_cls_test, (test_preds >= threshold).astype(int), zero_division=0))

    # Prepare output directory
    out_dir = REPO_ROOT / "models" / "ds_model" / "xgboost" / symbol
    out_dir.mkdir(parents=True, exist_ok=True)

    # Regression tolerance check (overall)
    tol = 0.002
    frac_tol, hits_tol, total_tol = fraction_within_tolerance(y_reg_test, test_preds, tol)
    if total_tol:
        print(f"\nWithin ±{tol} on test: {hits_tol}/{total_tol} ({frac_tol:.3f} fraction)")
    else:
        print("\nNo test samples to evaluate tolerance.")

    # Regression tolerance on rows flagged as real opportunities (if present)
    if "is_real_opportunity" in df.columns:
        opp_mask_test = df["is_real_opportunity"].iloc[val_end:] == 1
        opp_total = int(opp_mask_test.sum())
        if opp_total > 0:
            frac_opp, hits_opp, _ = fraction_within_tolerance(
                y_reg_test[opp_mask_test.values], test_preds[opp_mask_test.values], tol
            )
            print(
                f"Hits within ±{tol} on is_real_opportunity=1 rows: {hits_opp}/{opp_total} "
                f"({frac_opp:.3f} fraction)"
            )
        else:
            print("No is_real_opportunity=1 rows in test split; skipping tolerance hit-rate on opps.")
    
    # Precision-Recall curve - use plotter function
    plot_pr_curve(y_cls_test, test_preds, threshold, model_name='XGBoost',
                  save_path=out_dir / f"xgboost_{symbol}_pr_curve.png")

    # Threshold sweep for detailed analysis
    thr_candidates = sorted({max(0.0, threshold - 0.1), threshold, threshold + 0.1, threshold + 0.2})
    precisions_eval, recalls_eval, f1_eval, hit_eval = [], [], [], []
    
    print("\nThreshold sweep results:")
    for t in thr_candidates:
        m = compute_opportunity_metrics(y_reg_test, test_preds, opp_thresh=threshold, pred_thresh=t, tol=0.002)
        precisions_eval.append(m["precision"])
        recalls_eval.append(m["recall"])
        f1_eval.append(m["f1"])
        hit_eval.append(m["hit_rate_on_opps"])
        print(
            f"  thr={t:.4f} | P={m['precision']:.3f} R={m['recall']:.3f} "
            f"F1={m['f1']:.3f} hit@tol={m['hit_rate_on_opps']:.3f}"
        )
    
    # Use plotter function for threshold metrics
    plot_threshold_metrics(
        list(thr_candidates),
        precisions_eval,
        recalls_eval,
        f1_eval,
        hit_eval,
        model_name='XGBoost',
        save_path=out_dir / f"xgboost_{symbol}_threshold_metrics.png",
    )

    # Results and diagnostics using plotter functions
    plot_results(y_reg_test, test_preds, model_name='XGBoost',
                 save_path=out_dir / f"xgboost_{symbol}_results.png")
    
    plot_prediction_hist(test_preds, model_name='XGBoost',
                         save_path=out_dir / f"xgboost_{symbol}_prediction_hist.png")

    # Prediction history (chronological) - using plotter function
    time_test = df["time"].iloc[val_end:]
    plot_prediction_history(
        time_test,
        y_reg_test,
        test_preds,
        model_name='XGBoost',
        save_path=out_dir / f"xgboost_{symbol}_prediction_history.png",
    )

    # Feature importance - using plotter function
    plot_xgb_feature_importance(
        model,
        feature_names=X_train.columns,
        top_n=30,
        model_name='XGBoost',
        save_path=out_dir / f"xgboost_{symbol}_feature_importance.png",
    )

    # Save the trained model
    model_path = out_dir / f"xgboost_{symbol}_model.joblib"
    joblib.dump(model, model_path)
    print(f"Model saved to: {model_path}")
    
    print(f"\n{'='*60}")
    print(f"All outputs saved to: {out_dir}")
    print(f"{'='*60}\n")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train and evaluate XGBoost on featured spreads")
    parser.add_argument("--symbol", default="BTCUSD", 
                        help="Cryptocurrency symbol (default: BTCUSD)")
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.3,
        help="Opportunity threshold (default: 0.3)",
    )
    parser.add_argument(
        "--train-frac",
        type=float,
        default=0.7,
        help="Fraction of data for training (default: 0.7)",
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.15,
        help="Fraction of data for validation (default: 0.15)",
    )
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility (default: 42)")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(
        symbol=args.symbol,
        threshold=args.threshold,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=args.seed,
    )
