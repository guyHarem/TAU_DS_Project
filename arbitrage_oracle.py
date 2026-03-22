"""
Arbitrage Oracle - Main CLI for Cryptocurrency Cross-Exchange Arbitrage Analysis

This is the central orchestration script for the entire project, providing unified
CLI access to all data retrieval, feature engineering, analysis, and ML workflows.

Usage:
    python arbitrage_oracle.py data fetch --symbol BTCUSD --start 2025-12-01 --end 2025-12-02
    python arbitrage_oracle.py data add-features --symbol BTCUSD
    python arbitrage_oracle.py analysis run --symbol BTCUSD
    python arbitrage_oracle.py model train --model xgboost --symbol BTCUSD
    python arbitrage_oracle.py model train --all
"""

import sys
import subprocess
import argparse
from pathlib import Path
from typing import List, Optional
import pandas as pd
import numpy as np

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent
DATA_RETRIEVE_DIR = BASE_DIR / "data_retrieve"
DATA_ANALYSIS_DIR = BASE_DIR / "data_analysis"
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"

PYTHON = sys.executable  # Uses current venv python

# Supported cryptocurrencies
SUPPORTED_SYMBOLS = [
    "BTCUSD", "ETHUSD", "DOGEUSD", "LINKUSD", "SOLUSD", "XRPUSD"
]

# Supported exchanges
SUPPORTED_EXCHANGES = [
    "binance", "bitfinex", "coinbase", "gateio", "kraken"
]

# Available ML models
AVAILABLE_MODELS = {
    "randomforest": "model_randomforest.py",
    "lstm": "model_lstm.py",
    "gru": "model_gru.py",
    "linear": "model_linear.py",
    "xgboost": "model_xgboost.py",
    "transformer": "model_transformer.py",
    "catboost": "model_catboost.py"
}

# Model-specific arguments
MODEL_ARGS = {
    "linear": ["--model-type", "--alpha"],
    "lstm": ["--seq-length", "--units","--epochs", "--batch-size"],
    "gru": ["--seq-length", "--units",  "--epochs", "--batch-size"],
    "transformer": ["--seq-length", "--d-model", "--nhead", "--num-layers",  "--epochs", "--batch-size", "--lr"],
    "catboost": ["--iterations", "--learning-rate", "--depth"],
    "xgboost": ["--train-frac", "--val-frac"],
    "randomforest": ["--n-estimators", "--max-depth"]
}

PLOT_TYPE_PATTERNS = {
    "results": ["results"],
    "prediction_hist": ["prediction_hist"],
    "training_history": ["training_history", "prediction_history"],
    "feature_importance": ["feature_importance"],
    "pr_curve": ["pr_curve"],
    "threshold_metrics": ["threshold_metrics"],
    "opportunity_comparison": ["opportunity_comparison"],
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_header(title: str):
    """Print a formatted header"""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")

def print_success(message: str):
    """Print success message"""
    print(f"✅ {message}")

def print_error(message: str):
    """Print error message"""
    print(f"❌ {message}")

def print_info(message: str):
    """Print info message"""
    print(f"ℹ️  {message}")

def run_command(cmd: List[str], description: str) -> bool:
    """
    Execute a shell command and return success status
    
    Parameters:
    -----------
    cmd : list
        Command and arguments to execute
    description : str
        Human-readable description of the command
        
    Returns:
    --------
    bool
        True if command succeeded, False otherwise
    """
    print_info(f"Executing: {' '.join(cmd)}\n")
    
    try:
        subprocess.run(cmd, check=True, capture_output=False)
        print_success(f"{description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print_error(f"{description} failed with exit code {e.returncode}")
        return False
    except Exception as e:
        print_error(f"{description} failed: {str(e)}")
        return False

def validate_symbol(symbol: str) -> bool:
    """Validate if symbol is supported"""
    if symbol not in SUPPORTED_SYMBOLS:
        print_error(f"Symbol '{symbol}' not supported. Choose from: {', '.join(SUPPORTED_SYMBOLS)}")
        return False
    return True

def validate_exchange(exchange: str) -> bool:
    """Validate if exchange is supported"""
    if exchange not in SUPPORTED_EXCHANGES:
        print_error(f"Exchange '{exchange}' not supported. Choose from: {', '.join(SUPPORTED_EXCHANGES)}")
        return False
    return True

def validate_available_models(model: str) -> bool:
    """Validate if exchange is supported"""
    if model not in AVAILABLE_MODELS:
        print_error(f"Model '{model}' not supported. Choose from: {', '.join(AVAILABLE_MODELS.keys())}")
        return False
    return True

class DataCommands:
    """Data retrieval and management commands"""
    @staticmethod
    def fetch():
        """Fetch data from exchanges"""        
        cmd = [PYTHON, str(DATA_RETRIEVE_DIR / "data_retrieve.py")]      
        success = run_command(cmd, "Data retrieval")
        return success
    
class FeatureCommands:
    """Feature engineering commands"""
    @staticmethod
    def add():
        """Add features to raw data"""        
        cmd = [PYTHON, str(DATA_ANALYSIS_DIR / "data_featuring.py")]        
        success = run_command(cmd, "Feature engineering")        
        return success
    
    @staticmethod
    def list_features():
        """List all available features"""        
        feature_doc = DATA_ANALYSIS_DIR / "FEATURE_LIST.md"
        
        if not feature_doc.exists():
            print_error(f"Feature documentation not found: {feature_doc}")
            return False
        
        print_info("Feature documentation found. Key feature categories:\n")
        
        # Read and display feature categories
        with open(feature_doc, 'r') as f:
            content = f.read()
            # Extract first 100 lines for preview
            lines = content.split('\n')[:100]
            print('\n'.join(lines))
        
        print(f"\nFor complete documentation, see: {feature_doc}")
        return True

class AnalysisCommands:
    """Data analysis commands"""
    @staticmethod
    def run():
        """Run full analysis pipeline"""               
        cmd = [PYTHON, str(DATA_ANALYSIS_DIR / "data_analysis.py")]
        success = run_command(cmd, f"Run Feature Analysis")
        return success

class ModelCommands:
    """Machine learning model commands"""
    
    @staticmethod
    def train(args):
        """Train a model"""
        print_header("MODEL TRAINING")
        model_input = (getattr(args, "model", "") or "").strip().lower()
        
        # Determine which models to train
        if (not model_input) or (model_input == "all"):
            models_to_train = list(AVAILABLE_MODELS.keys())
        else:
            models_to_train = [m.strip() for m in model_input.split(',') if m.strip()]
            if not models_to_train:
                print_error("No model was provided.")
                return False
            invalid_models = [m for m in models_to_train if m not in AVAILABLE_MODELS]
            if invalid_models:
                print_error(f"Invalid model(s): {', '.join(invalid_models)}")
                print_error(f"Available: {', '.join(AVAILABLE_MODELS.keys())}")
                return False
        
        # Train each model
        all_success = True
        for model in models_to_train:
            for symbol in SUPPORTED_SYMBOLS:
                success = ModelCommands._train_single(model, symbol, args)
                if not success:
                    all_success = False
        
        return all_success
    
    @staticmethod
    def _train_single(model_name: str, symbol: str, args) -> bool:
        """Train a single model"""
        model_file = AVAILABLE_MODELS[model_name]
        
        cmd = [
            PYTHON,
            str(MODELS_DIR / model_file),
            "--symbol", symbol,
            "--seed", str(getattr(args, "seed", 42) or 42),
            "--threshold", str(getattr(args, "threshold", 0.3) or 0.3),
        ]

        # Add relevant optional args for this specific model.
        for cli_arg in MODEL_ARGS.get(model_name, []):
            attr_name = cli_arg.lstrip("-").replace("-", "_")
            value = getattr(args, attr_name, None)
            if value is not None and str(value).strip() != "":
                cmd.extend([cli_arg, str(value)])
        
        description = f"Training {model_name.upper()} on {symbol}"
        return run_command(cmd, description)
    
    @staticmethod
    def list_models():
        """List available models"""        
        print_header("AVAILABLE MODELS:\n")
        
        for model_name, model_file in AVAILABLE_MODELS.items():
            print(f"  {model_name.upper()}")
            print(f"     File: {model_file}")
            print()
        
        return True
    
    @staticmethod
    def list_trained():
        """List trained models"""
        print_header("TRAINED MODELS")
        
        ds_model_dir = MODELS_DIR / "ds_model"
        
        if not ds_model_dir.exists():
            print_info("No trained models found")
            return True
        
        print("Trained models by type:\n")
        
        for model_dir in sorted(ds_model_dir.iterdir()):
            if model_dir.is_dir():
                print(f"  {model_dir.name}")
                for symbol_dir in sorted(model_dir.iterdir()):
                    if symbol_dir.is_dir():
                        files = list(symbol_dir.glob("*.*"))
                        print(f"     - {symbol_dir.name} ({len(files)} files)")
        
        return True
    
    @staticmethod
    def evaluate(args):
        """Evaluate trained models by listing matching plot artifacts across all symbols."""
        print_header("MODEL EVALUATION")
        selected_models = parse_model_names(getattr(args, "model", "all"))
        invalid_models = [m for m in selected_models if m not in AVAILABLE_MODELS]
        if invalid_models:
            print_error(f"Invalid model(s): {', '.join(invalid_models)}")
            return False

        selected_plot_types = parse_plot_types(getattr(args, "plot_types", "all"))

        all_success = True
        any_found = False
        allowed_ext = {".png", ".jpg", ".jpeg", ".svg", ".pdf", ".html"}

        for model_name in selected_models:
            model_dirs = ModelCommands._resolve_model_output_dirs(model_name, args)
            if not model_dirs:
                print_error(f"No output directory found for {model_name.upper()}.")
                all_success = False
                continue

            print_info(f"\nModel: {model_name.upper()}")
            for model_dir in model_dirs:
                if not model_dir.exists():
                    continue
                print(f"  Directory: {model_dir}")

                for symbol in SUPPORTED_SYMBOLS:
                    symbol_dir = model_dir / symbol
                    if not symbol_dir.exists() or not symbol_dir.is_dir():
                        continue

                    files = sorted([
                        f for f in symbol_dir.glob("*.*")
                        if f.suffix.lower() in allowed_ext and matches_plot_type(f.name, selected_plot_types)
                    ])

                    if files:
                        any_found = True
                        print(f"    {symbol}:")
                        for file_path in files:
                            print(f"      - {file_path.name}")

        if not any_found:
            print_error("No matching plot artifacts found for the selected models/plot types.")
            all_success = False

        return all_success

    @staticmethod
    def _resolve_model_output_dirs(model_name: str, args) -> List[Path]:
        ds_model_dir = MODELS_DIR / "ds_model"

        if model_name == "linear":
            model_type = (getattr(args, "model_type", "") or "").strip().lower()
            if model_type:
                return [ds_model_dir / f"regression-{model_type}"]
            return sorted([d for d in ds_model_dir.glob("regression-*") if d.is_dir()])

        if model_name == "randomforest":
            return [ds_model_dir / "random-forest"]

        return [ds_model_dir / model_name]

    @staticmethod
    def plot(args):
        """Find and print the path to a specific plot file."""
        print_header("MODEL PLOT")

        if not args.model or not args.symbol or not args.name:
            print_error("--model, --symbol, and --name are required for plot.")
            return False

        if not validate_symbol(args.symbol):
            return False

        if args.model not in AVAILABLE_MODELS:
            print_error(f"Model '{args.model}' not found. Available: {', '.join(AVAILABLE_MODELS.keys())}")
            return False

        model_name = args.model
        symbol = args.symbol
        plot_name = args.name

        # Determine the output directory
        model_dir_name = model_name
        if model_name == 'linear':
            model_type = args.model_type or 'linear'
            model_dir_name = f'regression-{model_type}'
        elif model_name == 'randomforest':
            model_dir_name = 'random-forest'

        output_dir = MODELS_DIR / "ds_model" / model_dir_name / symbol
        plot_file = output_dir / plot_name

        if not plot_file.exists():
            print_error(f"Plot '{plot_name}' not found for {model_name.upper()} on {symbol}.")
            print_info(f"Searched for file: {plot_file}")
            print_info("You can list available plots with the 'evaluate' command.")
            return False

        print_success(f"Found plot for {model_name.upper()} on {symbol}:")
        print(f"  {plot_file}")

        return True

# ============================================================================
# INTERACTIVE CLI
# ============================================================================

class Args:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)

def isQuit(command):
    return (command.lower() in ["quit", "q", "exit"])

def get_choice(prompt, options):
    print(prompt)

    for i, option in enumerate(options, 1):
        print(f"  {i}. {option}")

    while True:
        try:
            choice = int(input("> "))
            if 1 <= choice <= len(options):
                return options[choice - 1]
            elif isQuit(choice):
                sys.exit(0)
            else:
                print("Invalid choice. Please try again.")

        except ValueError:
            print("Invalid input. Please enter a number.")

def get_input(prompt, default=None):
    if default:
        return input(f"{prompt} (default: {default}): ") or default
    else:
        return input(f"{prompt}: ")

def parse_model_names(model_input: str) -> List[str]:
    cleaned = (model_input or "").strip().lower()
    if not cleaned or cleaned == "all":
        return list(AVAILABLE_MODELS.keys())

    model_names = []
    for token in cleaned.split(','):
        name = token.strip()
        if name and name not in model_names:
            model_names.append(name)
    return model_names

def parse_plot_types(plot_input: str) -> List[str]:
    cleaned = (plot_input or "").strip().lower()
    if not cleaned or cleaned == "all":
        return ["all"]

    plot_types = []
    for token in cleaned.split(','):
        name = token.strip()
        if name and name not in plot_types:
            plot_types.append(name)
    return plot_types

def matches_plot_type(filename: str, selected_plot_types: List[str]) -> bool:
    file_lower = filename.lower()
    if "all" in selected_plot_types:
        return True

    for plot_type in selected_plot_types:
        patterns = PLOT_TYPE_PATTERNS.get(plot_type, [plot_type])
        if any(pattern in file_lower for pattern in patterns):
            return True

    return False

def collect_optional_model_args(selected_models: List[str]) -> dict:
    optional_args = {}
    prompted_attrs = set()

    print_info("Leave optional values empty to use model defaults.")
    for model_name in selected_models:
        print(f"\nOptional args for {model_name.upper()}:")
        for cli_arg in MODEL_ARGS.get(model_name, []):
            attr_name = cli_arg.lstrip("-").replace("-", "_")
            if attr_name in prompted_attrs:
                continue
            value = get_input(f"  {cli_arg}", "")
            if str(value).strip() != "":
                optional_args[attr_name] = value
            prompted_attrs.add(attr_name)

    return optional_args

def execute_data():
    DataCommands.fetch()
    return

def feature_menu():
    while True:
        print_header("FEATURE MENU")

        command = get_choice("Choose a feature command:", ["Add Features", "List Features", "Back"])

        if command == "Add Features":
            FeatureCommands.add()

        elif command == "List Features":
            FeatureCommands.list_features()

        elif command == "Back":
            return
        
        elif isQuit(command):
            sys.exit(0)

def execute_analysis():
    AnalysisCommands.run()
    return

def model_menu():
    while True:
        print_header("MODEL MENU")
        
        command = get_choice("Choose a model command:", ["Train", "List Models", "List Trained", "Evaluate", "Back"])

        if command == "Train":
            print_info(f"Available models: {', '.join(AVAILABLE_MODELS.keys())}")
            print_info("Enter one or more models separated by commas, or 'all'.")

            model_input = get_input("Enter model(s)", "all")
            selected_models = parse_model_names(model_input)
            invalid_models = [m for m in selected_models if m not in AVAILABLE_MODELS]
            if invalid_models:
                print_error(f"Invalid model(s): {', '.join(invalid_models)}")
                continue

            seed = get_input("Enter random seed", "42")
            threshold = get_input("Enter threshold", "0.3")
            optional_args = collect_optional_model_args(selected_models)

            args = Args(
                model=','.join(selected_models),
                seed=seed,
                threshold=threshold,
                **optional_args,
            )
            ModelCommands.train(args)

        elif command == "List Models":
            ModelCommands.list_models()

        elif command == "List Trained":
            ModelCommands.list_trained()

        elif command == "Evaluate":
            print_info(f"Available models: {', '.join(AVAILABLE_MODELS.keys())}")
            print_info("Enter one or more models separated by commas, or 'all'.")
            model_input = get_input("Enter model(s)", "all")
            selected_models = parse_model_names(model_input)
            invalid_models = [m for m in selected_models if m not in AVAILABLE_MODELS]
            if invalid_models:
                print_error(f"Invalid model(s): {', '.join(invalid_models)}")
                continue

            print_info("Available plot types:")
            print(f"  {', '.join(PLOT_TYPE_PATTERNS.keys())}")
            print_info("Enter one or more plot types separated by commas, or 'all'.")
            plot_types = get_input("Enter plot type(s)", "all")

            model_type = get_input("Linear model type (optional: linear/ridge/lasso)", "")
            args = Args(model=','.join(selected_models), plot_types=plot_types, model_type=model_type)
            ModelCommands.evaluate(args)

        elif command == "Back":
            return
        
        elif isQuit(command):
            sys.exit(0)


def main():
    """Main entry point for the interactive Arbitrage Oracle."""
    while True:
        print_header("ARBITRAGE ORACLE")
        command = get_choice("Choose a command category:", ["Data", "Feature", "Analysis", "Model", "Exit"])

        if command == "Data":
            execute_data()
        elif command == "Feature":
            feature_menu()
        elif command == "Analysis":
            execute_analysis()
        elif command == "Model":
            model_menu()
        elif isQuit(command):
            sys.exit(0)


if __name__ == "__main__":
    main()

