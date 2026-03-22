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
    "binance", "bitfinex", "coinbase", "gateio", "kraken", "mexc"
]

# Available ML models
AVAILABLE_MODELS = {
    "randomforest": "model_randomforest.py",
    "lstm": "model_lstm.py",
    "gru": "model_gru.py",
    "linear": "model_linear.py",
    "xgboost": "model_xgboost.py",
    "transformer": "model_transformer.py",
}

# Model-specific arguments
MODEL_ARGS = {
    "linear": ["--model-type"],  # linear, ridge, lasso
    "lstm": ["--seq-length", "--units", "--dropout"],
    "gru": ["--seq-length", "--units", "--dropout"],
    "transformer": ["--seq-length", "--d-model", "--nhead", "--num-layers"],
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
    print_info(f"Executing: {description}")
    print(f"   Command: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(cmd, check=True, capture_output=False)
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


def check_data_exists(symbol: str, data_type: str = "raw") -> bool:
    """Check if data file exists"""
    if data_type == "raw":
        file_path = DATA_DIR / "raw_data" / f"combined_{symbol}_data.csv"
    elif data_type == "featured":
        file_path = DATA_DIR / "featured_data" / f"featured_{symbol}_data.csv"
    else:
        return False
    
    return file_path.exists()


def get_data_stats(file_path: Path) -> Optional[dict]:
    """Get basic statistics about a data file"""
    try:
        df = pd.read_csv(file_path, nrows=100)
        full_df = pd.read_csv(file_path)
        return {
            "rows": len(full_df),
            "columns": len(df.columns),
            "date_range": f"{full_df['time'].min()} to {full_df['time'].max()}" 
                         if 'time' in df.columns else "Unknown"
        }
    except Exception as e:
        return None


# ============================================================================
# DATA RETRIEVAL COMMANDS
# ============================================================================

class DataCommands:
    """Data retrieval and management commands"""
    
    @staticmethod
    def fetch(args):
        """Fetch data from exchanges"""
        print_header("DATA RETRIEVAL")
        
        cmd = [PYTHON, str(DATA_RETRIEVE_DIR / "data_retrieve.py")]
        
        # The data_retrieve.py uses interactive prompts, so we just run it
        print_info("Starting interactive data retrieval...")
        print_info("Follow the prompts to select mode, currencies, and date range.\n")
        
        success = run_command(cmd, "Data retrieval")
        
        if success and args.symbol:
            file_path = DATA_DIR / "raw_data" / f"combined_{args.symbol}_data.csv"
            if file_path.exists():
                stats = get_data_stats(file_path)
                if stats:
                    print_info(f"\nData file: {file_path.name}")
                    print_info(f"  Rows: {stats['rows']:,}")
                    print_info(f"  Columns: {stats['columns']}")
                    print_info(f"  Date range: {stats['date_range']}")
        
        return success
    
    @staticmethod
    def list_raw(args):
        """List all raw data files"""
        print_header("RAW DATA FILES")
        
        raw_dir = DATA_DIR / "raw_data"
        if not raw_dir.exists():
            print_error(f"Raw data directory not found: {raw_dir}")
            return False
        
        files = sorted(raw_dir.glob("combined_*.csv"))
        
        if not files:
            print_info("No raw data files found")
            return True
        
        print(f"Found {len(files)} raw data file(s):\n")
        
        for file_path in files:
            stats = get_data_stats(file_path)
            if stats:
                print(f"  📄 {file_path.name}")
                print(f"     Rows: {stats['rows']:,} | Columns: {stats['columns']} | {stats['date_range']}")
            else:
                print(f"  📄 {file_path.name} (unable to read stats)")
        
        return True
    
    @staticmethod
    def list_featured(args):
        """List all featured data files"""
        print_header("FEATURED DATA FILES")
        
        featured_dir = DATA_DIR / "featured_data"
        if not featured_dir.exists():
            print_error(f"Featured data directory not found: {featured_dir}")
            return False
        
        files = sorted(featured_dir.glob("featured_*.csv"))
        
        if not files:
            print_info("No featured data files found. Run 'data add-features' first.")
            return True
        
        print(f"Found {len(files)} featured data file(s):\n")
        
        for file_path in files:
            stats = get_data_stats(file_path)
            if stats:
                print(f"  📊 {file_path.name}")
                print(f"     Rows: {stats['rows']:,} | Columns: {stats['columns']} | {stats['date_range']}")
            else:
                print(f"  📊 {file_path.name} (unable to read stats)")
        
        return True


# ============================================================================
# FEATURE ENGINEERING COMMANDS
# ============================================================================

class FeatureCommands:
    """Feature engineering commands"""
    
    @staticmethod
    def add(args):
        """Add features to raw data"""
        print_header("FEATURE ENGINEERING")
        
        # The data_featuring.py script processes all symbols at once.
        # We can still use the --symbol argument to check if at least one raw data file exists.
        symbol_to_check = args.symbol or "BTCUSD"

        if not validate_symbol(symbol_to_check):
            return False

        if not check_data_exists(symbol_to_check, "raw"):
            print_error(f"Raw data not found for {symbol_to_check}. Please run the data fetch command first.")
            return False
        
        cmd = [PYTHON, str(DATA_ANALYSIS_DIR / "data_featuring.py")]
        
        print_info("Starting feature engineering for all symbols...")
        
        success = run_command(cmd, "Feature engineering")
        
        if success:
            print_success("Feature engineering completed for all symbols.")
            # Optionally, list the created files
            DataCommands.list_featured(args)
        
        return success
    
    @staticmethod
    def list_features(args):
        """List all available features"""
        print_header("FEATURE DOCUMENTATION")
        
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


# ============================================================================
# ANALYSIS COMMANDS
# ============================================================================

class AnalysisCommands:
    """Data analysis commands"""
    
    @staticmethod
    def run(args):
        """Run full analysis pipeline"""
        print_header("DATA ANALYSIS")
        
        symbol = args.symbol or "ALL"
        
        if args.symbol and not validate_symbol(args.symbol):
            return False
        
        if args.symbol and not check_data_exists(args.symbol, "featured"):
            print_error(f"Featured data not found for {args.symbol}")
            print_info(f"Run 'arbitrage_oracle.py feature add --symbol {args.symbol}' first")
            return False
        
        # data_analysis.py uses interactive prompts
        cmd = [PYTHON, str(DATA_ANALYSIS_DIR / "data_analysis.py")]
        
        print_info("Starting interactive analysis...")
        print_info("Select 'ANALYZE' option when prompted.\n")
        
        success = run_command(cmd, f"Analysis for {symbol}")
        
        return success
    
    @staticmethod
    def quick_check(args):
        """Run quick arbitrage check"""
        print_header("QUICK ARBITRAGE CHECK")
        
        if not args.symbol:
            print_error("--symbol is required for quick check")
            return False
        
        if not validate_symbol(args.symbol):
            return False
        
        if not check_data_exists(args.symbol, "featured"):
            print_error(f"Featured data not found for {args.symbol}")
            return False
        
        cmd = [PYTHON, str(DATA_ANALYSIS_DIR / "quick_arbitrage_check.py")]
        
        success = run_command(cmd, f"Quick check for {args.symbol}")
        
        return success
    
    @staticmethod
    def diagnose(args):
        """Diagnose spread data"""
        print_header("SPREAD DIAGNOSIS")
        
        if not args.symbol:
            print_error("--symbol is required for diagnosis")
            return False
        
        if not validate_symbol(args.symbol):
            return False
        
        if not check_data_exists(args.symbol, "raw"):
            print_error(f"Raw data not found for {args.symbol}")
            return False
        
        cmd = [PYTHON, str(DATA_ANALYSIS_DIR / "diagnose_spreads.py")]
        
        success = run_command(cmd, f"Spread diagnosis for {args.symbol}")
        
        return success


# ============================================================================
# MODEL TRAINING COMMANDS
# ============================================================================

class ModelCommands:
    """Machine learning model commands"""
    
    @staticmethod
    def train(args):
        """Train a model"""
        print_header("MODEL TRAINING")
        
        # Determine which models to train
        if args.all:
            models_to_train = list(AVAILABLE_MODELS.keys())
            symbols_to_train = args.symbols or SUPPORTED_SYMBOLS
        else:
            if not args.model:
                print_error("Specify --model or use --all")
                return False
            
            if args.model not in AVAILABLE_MODELS:
                print_error(f"Model '{args.model}' not found. Available: {', '.join(AVAILABLE_MODELS.keys())}")
                return False
            
            models_to_train = [args.model]
            symbols_to_train = args.symbols or [args.symbol or "BTCUSD"]
        
        # Validate symbols
        for symbol in symbols_to_train:
            if not validate_symbol(symbol):
                return False
            
            if not check_data_exists(symbol, "featured"):
                print_error(f"Featured data not found for {symbol}")
                print_info(f"Run 'arbitrage_oracle.py feature add --symbol {symbol}' first")
                return False
        
        # Train each model
        all_success = True
        for model in models_to_train:
            for symbol in symbols_to_train:
                success = ModelCommands._train_single(
                    model, symbol, args
                )
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
            "--seed", str(args.seed or 42),
            "--threshold", str(args.threshold or 0.3),
        ]
        
        # Add model-specific arguments
        if model_name == "linear" and args.model_type:
            cmd.extend(["--model-type", args.model_type])
        
        if model_name in ["lstm", "gru", "transformer"]:
            if args.seq_length:
                cmd.extend(["--seq-length", str(args.seq_length)])
            if model_name == "transformer" and args.d_model:
                cmd.extend(["--d-model", str(args.d_model)])
        
        description = f"Training {model_name.upper()} on {symbol}"
        return run_command(cmd, description)
    
    @staticmethod
    def list_models(args):
        """List available models"""
        print_header("AVAILABLE MODELS")
        
        print("Supported models:\n")
        
        for model_name, model_file in AVAILABLE_MODELS.items():
            print(f"  🤖 {model_name.upper()}")
            print(f"     File: {model_file}")
            
            if model_name == "linear":
                print(f"     Types: linear, ridge, lasso")
            elif model_name in ["lstm", "gru", "transformer"]:
                print(f"     Configurable: seq-length, hidden units, layers")
            print()
        
        return True
    
    @staticmethod
    def list_trained(args):
        """List trained models"""
        print_header("TRAINED MODELS")
        
        ds_model_dir = MODELS_DIR / "ds_model"
        
        if not ds_model_dir.exists():
            print_info("No trained models found")
            return True
        
        print("Trained models by type:\n")
        
        for model_dir in sorted(ds_model_dir.iterdir()):
            if model_dir.is_dir():
                print(f"  📦 {model_dir.name}")
                for symbol_dir in sorted(model_dir.iterdir()):
                    if symbol_dir.is_dir():
                        files = list(symbol_dir.glob("*.*"))
                        print(f"     - {symbol_dir.name} ({len(files)} files)")
        
        return True
    
    @staticmethod
    def evaluate(args):
        """Evaluate a trained model by listing its output artifacts."""
        print_header("MODEL EVALUATION")

        if not args.model or not args.symbol:
            print_error("--model and --symbol are required for evaluation.")
            return False

        if not validate_symbol(args.symbol):
            return False

        if args.model not in AVAILABLE_MODELS:
            print_error(f"Model '{args.model}' not found. Available: {', '.join(AVAILABLE_MODELS.keys())}")
            return False

        model_name = args.model
        symbol = args.symbol

        # Determine the output directory
        model_dir_name = model_name
        if model_name == 'linear':
            model_type = args.model_type or 'linear'
            model_dir_name = f'regression-{model_type}'
        elif model_name == 'randomforest':
            model_dir_name = 'random-forest'

        output_dir = MODELS_DIR / "ds_model" / model_dir_name / symbol

        if not output_dir.exists() or not any(output_dir.iterdir()):
            print_error(f"No evaluation outputs found for {model_name.upper()} on {symbol}.")
            print_info(f"Directory searched: {output_dir}")
            print_info(f"Please train the model first using: python arbitrage_oracle.py model train --model {model_name} --symbol {symbol}")
            return False

        print_info(f"Found evaluation artifacts for {model_name.upper()} on {symbol} in:")
        print(f"  {output_dir}\n")

        files = sorted(output_dir.glob('*.*'))
        for file_path in files:
            print(f"  - {file_path.name}")

        return True

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

def get_choice(prompt, options):
    print(prompt)

    for i, option in enumerate(options, 1):
        print(f"  {i}. {option}")

    while True:
        try:
            choice = int(input("> "))
            if 1 <= choice <= len(options):
                return options[choice - 1]
            else:
                print("Invalid choice. Please try again.")

        except ValueError:
            print("Invalid input. Please enter a number.")

def get_input(prompt, default=None):
    if default:
        return input(f"{prompt} (default: {default}): ") or default
    else:
        return input(f"{prompt}: ")

def data_menu():
    while True:
        print_header("DATA MENU")

        command = get_choice("Choose a data command:", ["Fetch", "List Raw", "List Featured", "Back"])

        if command == "Fetch":
            symbol = get_input("Enter symbol (optional)", "BTCUSD")
            args = Args(symbol=symbol)
            DataCommands.fetch(args)
        elif command == "List Raw":
            DataCommands.list_raw(None)
        elif command == "List Featured":
            DataCommands.list_featured(None)
        elif command == "Back":
            return

def feature_menu():
    while True:
        print_header("FEATURE MENU")

        command = get_choice("Choose a feature command:", ["Add Features", "List Features", "Back"])

        if command == "Add Features":
            symbol = get_input("Enter symbol to check for raw data (optional)", "BTCUSD")
            args = Args(symbol=symbol)
            FeatureCommands.add(args)

        elif command == "List Features":
            FeatureCommands.list_features(None)

        elif command == "Back":
            return

def analysis_menu():
    while True:
        print_header("ANALYSIS MENU")

        command = get_choice("Choose an analysis command:", ["Run Analysis", "Quick Check", "Diagnose Spreads", "Back"])

        if command == "Run Analysis":
            symbol = get_input("Enter symbol (optional)", "ALL")
            args = Args(symbol=symbol)
            AnalysisCommands.run(args)

        elif command == "Quick Check":
            symbol = get_input("Enter symbol", "BTCUSD")
            args = Args(symbol=symbol)
            AnalysisCommands.quick_check(args)

        elif command == "Diagnose Spreads":
            symbol = get_input("Enter symbol", "BTCUSD")
            args = Args(symbol=symbol)
            AnalysisCommands.diagnose(args)

        elif command == "Back":
            return

def model_menu():
    while True:
        print_header("MODEL MENU")

        command = get_choice("Choose a model command:", ["Train", "List Models", "List Trained", "Evaluate", "Plot", "Back"])

        if command == "Train":
            model = get_input("Enter model name")
            symbol = get_input("Enter symbol", "BTCUSD")
            args = Args(model=model, symbol=symbol, all=False, symbols=None, model_type=None, seq_length=None, d_model=None, seed=42, threshold=0.3)
            ModelCommands.train(args)

        elif command == "List Models":
            ModelCommands.list_models(None)

        elif command == "List Trained":
            ModelCommands.list_trained(None)

        elif command == "Evaluate":
            model = get_input("Enter model name")
            symbol = get_input("Enter symbol", "BTCUSD")
            model_type = get_input("Enter model type (for linear model)", "linear")
            args = Args(model=model, symbol=symbol, model_type=model_type)
            ModelCommands.evaluate(args)

        elif command == "Plot":
            model = get_input("Enter model name")
            symbol = get_input("Enter symbol", "BTCUSD")
            plot_name = get_input("Enter plot name")
            model_type = get_input("Enter model type (for linear model)", "linear")
            args = Args(model=model, symbol=symbol, name=plot_name, model_type=model_type)
            ModelCommands.plot(args)

        elif command == "Back":
            return


def main():
    """Main entry point for the interactive Arbitrage Oracle."""
    while True:
        print_header("ARBITRAGE ORACLE")
        command = get_choice("Choose a command category:", ["Data", "Feature", "Analysis", "Model", "Exit"])

        if command == "Data":
            data_menu()
        elif command == "Feature":
            feature_menu()
        elif command == "Analysis":
            analysis_menu()
        elif command == "Model":
            model_menu()
        elif command == "Exit":
            sys.exit(0)


if __name__ == "__main__":
    main()

