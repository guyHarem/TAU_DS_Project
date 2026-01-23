import subprocess
import sys
import argparse
from pathlib import Path

# ---- CONFIG ----
PYTHON = sys.executable  # uses the current venv python
BASE_DIR = Path(__file__).parent
MODELS_DIR = BASE_DIR / "models"

SYMBOLS = ["BTCUSD", "DOGEUSD", "ETHUSD", "LINKUSD", "SOLUSD", "XRPUSD"]
MODELS = sorted(p.name for p in MODELS_DIR.glob("model_*.py"))

def arg_parse():
    parser = argparse.ArgumentParser(description="Run all models with specified parameters.")
    parser.add_argument('--symbol', type=str, default='BTCUSD',
                        help='Cryptocurrency to model (default: BTCUSD)')
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility (default: 42)"
    )
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Threshold value (default: 0.3)"
    )
    return parser.parse_args()

def run_model(model_file, symbol, seed, threshold, extra_args):
    cmd = [
        PYTHON,
        str(MODELS_DIR / model_file),
        "--symbol", symbol,
        "--seed", str(seed),
        "--threshold", str(threshold),
    ]

    for k, v in extra_args.items():
        cmd.extend([f"--{k}", str(v)])

    print(" ".join(cmd))
    subprocess.run(cmd, check=True)



def main():
    # NOT READY - DON'T USE YET
    return # safety exit
    args = arg_parse()
    symbol = args.symbol
    seed = args.seed
    threshold = args.threshold


    for model in MODELS:
        for symbol in SYMBOLS:

            if model.name == "model_linear.py":
                model_types = ['linear', 'ridge', 'lasso']
                for model_type in model_types:
                    extra_args = {"model-type": model_type}
                    run_model(model, symbol, seed, threshold, extra_args)

            elif model.name == "model_catboost.py":
                depths = [4, 6, 8]
                for depth in depths:
                    extra_args = {"depth": depth}
                    run_model(model, symbol, seed, threshold, extra_args)

    print("All models finished.")


if __name__ == "__main__":
    main()