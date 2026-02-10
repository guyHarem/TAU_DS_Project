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
        
        symbol = args.symbol or "ALL"
        
        if args.symbol and not validate_symbol(args.symbol):
            return False
        
        if args.symbol and not check_data_exists(args.symbol, "raw"):
            print_error(f"Raw data not found for {args.symbol}")
            print_info(f"Run 'arbitrage_oracle.py data fetch --symbol {args.symbol}' first")
            return False
        
        # data_analysis.py uses interactive prompts
        cmd = [PYTHON, str(DATA_ANALYSIS_DIR / "data_analysis.py")]
        
        print_info("Starting interactive feature engineering...")
        print_info("Select 'ADD' option when prompted.\n")
        
        success = run_command(cmd, f"Feature engineering for {symbol}")
        
        if success and args.symbol:
            file_path = DATA_DIR / "featured_data" / f"featured_{args.symbol}_data.csv"
            if file_path.exists():
                stats = get_data_stats(file_path)
                if stats:
                    print_success(f"\nFeatured data created: {file_path.name}")
                    print_info(f"  Rows: {stats['rows']:,}")
                    print_info(f"  Columns: {stats['columns']}")
        
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
        """Evaluate a trained model"""
        print_header("MODEL EVALUATION")
        
        print_info("Model evaluation not yet implemented")
        print_info("Check generated plots and metrics in ds_model/ directories")
        
        return True


# ============================================================================
# MAIN CLI SETUP
# ============================================================================

def create_parser() -> argparse.ArgumentParser:
    """Create the main argument parser"""
    
    parser = argparse.ArgumentParser(
        description="Arbitrage Oracle - Cryptocurrency Arbitrage Analysis & ML",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Data retrieval
  python arbitrage_oracle.py data fetch
  python arbitrage_oracle.py data list-raw
  python arbitrage_oracle.py data list-featured
  
  # Feature engineering
  python arbitrage_oracle.py feature add --symbol BTCUSD
  python arbitrage_oracle.py feature list
  
  # Analysis
  python arbitrage_oracle.py analysis run --symbol BTCUSD
  python arbitrage_oracle.py analysis quick-check --symbol BTCUSD
  python arbitrage_oracle.py analysis diagnose --symbol BTCUSD
  
  # Model training
  python arbitrage_oracle.py model train --model xgboost --symbol BTCUSD
  python arbitrage_oracle.py model train --model linear --symbol BTCUSD --model-type ridge
  python arbitrage_oracle.py model train --all
  python arbitrage_oracle.py model list
  python arbitrage_oracle.py model list-trained
        """
    )
    
    # Global arguments
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed (default: 42)")
    parser.add_argument("--threshold", type=float, default=0.3,
                        help="Opportunity threshold (default: 0.3)")
    
    # Create subcommands
    subparsers = parser.add_subparsers(dest="command", help="Command category")
    
    # ---- DATA COMMANDS ----
    data_parser = subparsers.add_parser("data", help="Data retrieval and management")
    data_subparsers = data_parser.add_subparsers(dest="data_command")
    
    fetch_parser = data_subparsers.add_parser("fetch", help="Fetch data from exchanges")
    fetch_parser.add_argument("--symbol", type=str, help="Optional: crypto symbol")
    fetch_parser.set_defaults(func=DataCommands.fetch)
    
    list_raw_parser = data_subparsers.add_parser("list-raw", help="List raw data files")
    list_raw_parser.set_defaults(func=DataCommands.list_raw)
    
    list_featured_parser = data_subparsers.add_parser("list-featured", help="List featured data files")
    list_featured_parser.set_defaults(func=DataCommands.list_featured)
    
    # ---- FEATURE COMMANDS ----
    feature_parser = subparsers.add_parser("feature", help="Feature engineering")
    feature_subparsers = feature_parser.add_subparsers(dest="feature_command")
    
    add_parser = feature_subparsers.add_parser("add", help="Add features to data")
    add_parser.add_argument("--symbol", type=str, help="Optional: crypto symbol")
    add_parser.set_defaults(func=FeatureCommands.add)
    
    list_features_parser = feature_subparsers.add_parser("list", help="List available features")
    list_features_parser.set_defaults(func=FeatureCommands.list_features)
    
    # ---- ANALYSIS COMMANDS ----
    analysis_parser = subparsers.add_parser("analysis", help="Data analysis")
    analysis_subparsers = analysis_parser.add_subparsers(dest="analysis_command")
    
    run_parser = analysis_subparsers.add_parser("run", help="Run full analysis")
    run_parser.add_argument("--symbol", type=str, help="Optional: crypto symbol")
    run_parser.set_defaults(func=AnalysisCommands.run)
    
    quick_parser = analysis_subparsers.add_parser("quick-check", help="Quick arbitrage check")
    quick_parser.add_argument("--symbol", type=str, required=True, help="Crypto symbol")
    quick_parser.set_defaults(func=AnalysisCommands.quick_check)
    
    diagnose_parser = analysis_subparsers.add_parser("diagnose", help="Diagnose spreads")
    diagnose_parser.add_argument("--symbol", type=str, required=True, help="Crypto symbol")
    diagnose_parser.set_defaults(func=AnalysisCommands.diagnose)
    
    # ---- MODEL COMMANDS ----
    model_parser = subparsers.add_parser("model", help="Machine learning models")
    model_subparsers = model_parser.add_subparsers(dest="model_command")
    
    train_parser = model_subparsers.add_parser("train", help="Train a model")
    train_parser.add_argument("--model", type=str, help="Model name (randomforest, lstm, gru, linear, xgboost, transformer)")
    train_parser.add_argument("--symbol", type=str, help="Crypto symbol (default: BTCUSD)")
    train_parser.add_argument("--symbols", nargs="+", help="Multiple symbols")
    train_parser.add_argument("--all", action="store_true", help="Train all models on all symbols")
    train_parser.add_argument("--model-type", type=str, help="Linear model type (linear, ridge, lasso)")
    train_parser.add_argument("--seq-length", type=int, help="Sequence length for LSTM/GRU/Transformer")
    train_parser.add_argument("--d-model", type=int, help="Model dimension for Transformer")
    train_parser.set_defaults(func=ModelCommands.train)
    
    list_models_parser = model_subparsers.add_parser("list", help="List available models")
    list_models_parser.set_defaults(func=ModelCommands.list_models)
    
    list_trained_parser = model_subparsers.add_parser("list-trained", help="List trained models")
    list_trained_parser.set_defaults(func=ModelCommands.list_trained)
    
    eval_parser = model_subparsers.add_parser("evaluate", help="Evaluate a model")
    eval_parser.set_defaults(func=ModelCommands.evaluate)
    
    return parser


def main():
    """Main entry point"""
    parser = create_parser()
    args = parser.parse_args()
    
    # Show help if no command
    if not args.command:
        parser.print_help()
        sys.exit(0)
    
    # Execute the command
    if hasattr(args, 'func'):
        success = args.func(args)
        sys.exit(0 if success else 1)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()