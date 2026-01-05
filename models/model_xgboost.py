"""Train an XGBoost regression model to predict `spread_close_pct`.

The script loads a featured dataset, keeps chronological ordering, trains an
XGBoost regressor, and evaluates classification metrics on the derived
`is_real_opportunity` target (or a threshold on `spread_close_pct`).

Usage (example):
	python models/model_xgboost.py --symbol BTCUSD --threshold 0.6
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
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
from xgboost import XGBRegressor


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

	y_reg = df[target_col].astype(float)
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
		"is_opportunity_flag",
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

	# Drop rows with any non-finite feature or target to avoid XGBoost errors.
	mask_finite = feature_df.notna().all(axis=1) & y_reg.notna()
	feature_df = feature_df[mask_finite].reset_index(drop=True)
	y_reg = y_reg[mask_finite].reset_index(drop=True)
	y_cls = y_cls[mask_finite].reset_index(drop=True)
	df = df[mask_finite].reset_index(drop=True)

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


def plot_results(y_true: pd.Series, y_pred: np.ndarray, save_path: Path) -> None:
	"""Plot actual vs predicted and residual diagnostics."""

	fig, axes = plt.subplots(2, 2, figsize=(15, 12))

	axes[0, 0].scatter(y_true, y_pred, alpha=0.5, s=10)
	axes[0, 0].plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], "r--", lw=2)
	axes[0, 0].set_xlabel("Actual spread_close_pct", fontsize=12)
	axes[0, 0].set_ylabel("Predicted spread_close_pct", fontsize=12)
	axes[0, 0].set_title("Actual vs Predicted", fontsize=14, fontweight="bold")
	axes[0, 0].grid(True, alpha=0.3)

	residuals = y_true - y_pred
	axes[0, 1].scatter(y_pred, residuals, alpha=0.5, s=10)
	axes[0, 1].axhline(y=0, color="r", linestyle="--", lw=2)
	axes[0, 1].set_xlabel("Predicted spread_close_pct", fontsize=12)
	axes[0, 1].set_ylabel("Residuals", fontsize=12)
	axes[0, 1].set_title("Residual Plot", fontsize=14, fontweight="bold")
	axes[0, 1].grid(True, alpha=0.3)

	axes[1, 0].hist(residuals, bins=50, edgecolor="black", alpha=0.7)
	axes[1, 0].axvline(x=0, color="r", linestyle="--", lw=2)
	axes[1, 0].set_xlabel("Residuals", fontsize=12)
	axes[1, 0].set_ylabel("Frequency", fontsize=12)
	axes[1, 0].set_title("Residual Distribution", fontsize=14, fontweight="bold")
	axes[1, 0].grid(True, alpha=0.3)

	abs_err = np.abs(residuals)
	axes[1, 1].hist(abs_err, bins=50, edgecolor="black", alpha=0.7, color="orange")
	axes[1, 1].set_xlabel("Absolute Error", fontsize=12)
	axes[1, 1].set_ylabel("Frequency", fontsize=12)
	axes[1, 1].set_title("Absolute Error Distribution", fontsize=14, fontweight="bold")
	axes[1, 1].grid(True, alpha=0.3)

	plt.tight_layout()
	plt.savefig(save_path, dpi=300, bbox_inches="tight")
	plt.close()


def plot_prediction_hist(y_pred: np.ndarray, save_path: Path) -> None:
	plt.figure(figsize=(10, 6))
	plt.hist(y_pred, bins=40, edgecolor="black", alpha=0.75)
	plt.xlabel("Predicted spread_close_pct", fontsize=12)
	plt.ylabel("Frequency", fontsize=12)
	plt.title("Prediction Histogram", fontsize=14, fontweight="bold")
	plt.grid(True, alpha=0.3)
	plt.tight_layout()
	plt.savefig(save_path, dpi=300, bbox_inches="tight")
	plt.close()


def plot_prediction_history(time_index: pd.Series, y_true: pd.Series, y_pred: np.ndarray, save_path: Path) -> None:
	plt.figure(figsize=(14, 6))
	plt.plot(time_index, y_true, label="Actual", linewidth=1.5)
	plt.plot(time_index, y_pred, label="Predicted", linewidth=1.5, alpha=0.8)
	plt.xlabel("Time", fontsize=12)
	plt.ylabel("spread_close_pct", fontsize=12)
	plt.title("Prediction History (chronological)", fontsize=14, fontweight="bold")
	plt.legend()
	plt.grid(True, alpha=0.3)
	plt.tight_layout()
	plt.savefig(save_path, dpi=300, bbox_inches="tight")
	plt.close()


def plot_feature_importance(model: XGBRegressor, feature_names: pd.Index, top_n: int, save_path: Path) -> None:
	booster = model.get_booster()
	scores = booster.get_score(importance_type="gain")
	if not scores:
		return
	rows = []
	for fname, gain in scores.items():
		rows.append((fname, gain))
	imp_df = pd.DataFrame(rows, columns=["feature", "gain"])

	# Map XGBoost's internal names back to column names when possible
	name_map = {f"f{i}": name for i, name in enumerate(feature_names)}
	imp_df["feature"] = imp_df["feature"].map(name_map).fillna(imp_df["feature"])
	imp_df = imp_df.sort_values("gain", ascending=False).head(top_n)

	plt.figure(figsize=(12, 8))
	plt.barh(range(len(imp_df)), imp_df["gain"], color="steelblue", alpha=0.8)
	plt.yticks(range(len(imp_df)), imp_df["feature"])
	plt.xlabel("Gain", fontsize=12)
	plt.title(f"Top {top_n} Feature Importances (XGBoost)", fontsize=14, fontweight="bold")
	plt.grid(True, axis="x", alpha=0.3)
	plt.tight_layout()
	plt.savefig(save_path, dpi=300, bbox_inches="tight")
	plt.close()


def plot_pr_curve(recalls: np.ndarray, precisions: np.ndarray, ap: float, save_path: Path) -> None:
	plt.figure(figsize=(8, 6))
	plt.plot(recalls, precisions, label=f"PR curve (AP={ap:.3f})", color="blue")
	plt.xlabel("Recall", fontsize=12)
	plt.ylabel("Precision", fontsize=12)
	plt.title("Precision-Recall Curve", fontsize=14, fontweight="bold")
	plt.grid(True, alpha=0.3)
	plt.legend()
	plt.tight_layout()
	plt.savefig(save_path, dpi=300, bbox_inches="tight")
	plt.close()


def plot_threshold_metrics(
	thresholds: list,
	precisions: list,
	recalls: list,
	f1s: list,
	hit_rates: list,
	save_path: Path,
) -> None:
	plt.figure(figsize=(10, 6))
	plt.plot(thresholds, precisions, label="Precision", marker="o")
	plt.plot(thresholds, recalls, label="Recall", marker="o")
	plt.plot(thresholds, f1s, label="F1", marker="o")
	plt.plot(thresholds, hit_rates, label="Hit-rate on true opps", marker="o")
	plt.xlabel("Prediction Threshold", fontsize=12)
	plt.ylabel("Metric value", fontsize=12)
	plt.title("Threshold vs Precision/Recall/F1/Hit-rate", fontsize=14, fontweight="bold")
	plt.grid(True, alpha=0.3)
	plt.legend()
	plt.tight_layout()
	plt.savefig(save_path, dpi=300, bbox_inches="tight")
	plt.close()


def train_xgb(
	X_train: pd.DataFrame,
	y_train: pd.Series,
	X_val: pd.DataFrame,
	y_val: pd.Series,
	seed: int,
) -> XGBRegressor:
	"""Train an XGBoost regressor with the original squared-error setup."""

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
	)

	model.fit(
		X_train,
		y_train,
		eval_set=[(X_train, y_train), (X_val, y_val)],
		verbose=False,
	)
	return model


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
	total = len(y)
	pos = int(y.sum())
	neg = total - pos
	pos_pct = (pos / total * 100) if total else 0.0
	neg_pct = (neg / total * 100) if total else 0.0
	return f"{name}: {pos} pos ({pos_pct:.2f}%), {neg} neg ({neg_pct:.2f}%), total={total}"


def run(symbol: str, threshold: float, train_frac: float, val_frac: float, seed: int) -> None:
	df_raw = load_featured(symbol)
	X, y_reg, y_cls, df = prepare_features(df_raw, threshold=threshold)
	dropped = len(df_raw) - len(df)
	if dropped:
		print(f"Dropped {dropped} rows with non-finite values after cleaning (from {len(df_raw)} to {len(df)})")
	else:
		print("No rows dropped for non-finite values")

	n = len(df)
	if n < 3:
		print(f"Not enough rows after cleaning (n={n}). Need at least 3 rows for train/val/test. Skipping.")
		return

	# Compute split sizes and ensure each split has at least one row. Adjust gently if needed.
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
		# Borrow a row from the largest split to keep totals consistent.
		if n_train >= n_val and n_train > 1:
			n_train -= 1
		elif n_val > 1:
			n_val -= 1
		n_test = n - n_train - n_val

	# Final sanity: if still broken, fall back to 70/15/15 min-1-each.
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

	train_end = n_train
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

	model = train_xgb(X_train, y_reg_train, X_val, y_reg_val, seed)

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
	print(f"\nWithin ±{tol} on test: {hits_tol}/{total_tol} ({frac_tol:.3f} fraction)" if total_tol else "\nNo test samples to evaluate tolerance.")

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
	# Precision-Recall curve
	precisions, recalls, thresh = precision_recall_curve(y_cls_test, test_preds)
	ap = average_precision_score(y_cls_test, test_preds)
	plot_pr_curve(recalls, precisions, ap, save_path=out_dir / f"xgboost_{symbol}_pr_curve.png")

	# Threshold sweep
	base_thr = threshold
	thr_candidates = sorted({max(0.0, base_thr - 0.1), base_thr, base_thr + 0.1, base_thr + 0.2})
	precisions_eval, recalls_eval, f1_eval, hit_eval = [], [], [], []
	for t in thr_candidates:
		m = compute_opportunity_metrics(y_reg_test, test_preds, opp_thresh=base_thr, pred_thresh=t, tol=0.002)
		precisions_eval.append(m["precision"])
		recalls_eval.append(m["recall"])
		f1_eval.append(m["f1"])
		hit_eval.append(m["hit_rate_on_opps"])
		print(
			f"  thr={t:.4f} | P={m['precision']:.3f} R={m['recall']:.3f} "
			f"F1={m['f1']:.3f} hit@tol={m['hit_rate_on_opps']:.3f}"
		)
	plot_threshold_metrics(
		thr_candidates,
		precisions_eval,
		recalls_eval,
		f1_eval,
		hit_eval,
		save_path=out_dir / f"xgboost_{symbol}_threshold_metrics.png",
	)

	# Results and diagnostics
	plot_results(y_reg_test, test_preds, save_path=out_dir / f"xgboost_{symbol}_results.png")
	plot_prediction_hist(test_preds, save_path=out_dir / f"xgboost_{symbol}_prediction_hist.png")

	# Prediction history (chronological)
	time_test = df["time"].iloc[val_end:]
	plot_prediction_history(
		time_test,
		y_reg_test,
		test_preds,
		save_path=out_dir / f"xgboost_{symbol}_prediction_history.png",
	)

	# Feature importance
	plot_feature_importance(
		model,
		feature_names=X_train.columns,
		top_n=30,
		save_path=out_dir / f"xgboost_{symbol}_feature_importance.png",
	)


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(description="Train and evaluate XGBoost on featured spreads")
	parser.add_argument("--symbol", default="BTCUSD", help="Trading pair symbol, e.g., BTCUSD")
	parser.add_argument(
		"--threshold",
		type=float,
		default=0.3,
		help="Spread percentage threshold to flag real opportunities",
	)
	parser.add_argument(
		"--train-frac",
		type=float,
		default=0.7,
		help="Fraction of data for training (chronological split)",
	)
	parser.add_argument(
		"--val-frac",
		type=float,
		default=0.15,
		help="Fraction of data for validation (chronological split)",
	)
	parser.add_argument("--seed", type=int, default=42, help="Random seed")
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
