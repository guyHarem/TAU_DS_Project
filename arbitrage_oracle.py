"""
ARBITRAGE ORACLE - Main CLI Orchestrator for Cryptocurrency Cross-Exchange Arbitrage

The Arbitrage Oracle is the unified command center for the entire TAU_DS_Project,
providing an interactive menu-driven interface to all workflows:
  ┌─────────────────────────────────────────┐
  │  1. DATA RETRIEVAL                      │
  │     Fetch 1-min OHLCV from 6 exchanges  │
  │                                         │
  │  2. FEATURE ENGINEERING                 │
  │     Generate 160+ features (5 layers)   │
  │                                         │
  │  3. DATA ANALYSIS                       │
  │     11-section statistical analysis     │
  │                                         │
  │  4. ML MODEL TRAINING & EVALUATION      │
  │     7 models: LSTM, GRU, Transformer,   │
  │     Linear, RF, XGBoost, CatBoost       │
  └─────────────────────────────────────────┘

PURPOSE:
  Orchestrate end-to-end cryptocurrency arbitrage analysis pipeline, from raw data
  collection through feature engineering to ML model training and evaluation.

ARCHITECTURE:
  This module provides:
  - Interactive menu-based CLI for casual users
  - Command classes delegating to sub-modules
  - Configuration management (symbols, exchanges, models)
  - Utility functions (printing, validation, command execution)
  - Unified output directory structure (models/ds_model/)

WORKFLOW PIPELINE:

  Step 1: DATA RETRIEVAL
    Command: Data → Fetch
    Input: Cryptocurrencies (BTC, ETH, DOGE, SOL, XRP, LINK)
           Date range (UTC)
    Output: data/raw_data/combined_BTCUSD_data.csv (per cryptocurrency)
    Exchanges: Binance, Bitfinex, Coinbase, Gate.io, Kraken, MEXC
    Note: Each exchange has its own API client with custom pagination/rate limits

  Step 2: FEATURE ENGINEERING
    Command: Feature → Add Features
    Input: Raw data from Step 1
    Output: data/featured_data/featured_BTCUSD_data.csv (~160+ features each)
    Architecture: 5-layer feature engineering
      - Layer 2: Core features (spreads, exchanges, time, price ratios)
      - Layer 3: Volume & volatility metrics
      - Layer 4: Technical analysis (MA, rolling stats, z-scores)
      - Layer 5: Temporal (Bollinger bands, lag features)
    See FEATURE_LIST.md for complete feature documentation

  Step 3: DATA ANALYSIS
    Command: Analysis → Run
    Input: Featured data from Step 2
    Output: data_analysis_results.txt (11-section report)
    Analysis Sections:
      1. Opportunity frequency statistics
      2. Average spreads profitability
      3. Temporal patterns (hour/day effects)
      4. Exchange patterns (best trading pairs)
      5. Volume & liquidity constraints
      6. Risk factors (volatility, gaps)
      7. Profitability estimation (ROI per trade)
      8. Momentum indicators (MA, rate of change)
      9. Bollinger bands patterns
      10. Persistence & autocorrelation
      11. Rolling statistics & volatility

  Step 4: MODEL TRAINING & EVALUATION
    Command: Model → Train / List Trained / Evaluate
    Input: Featured data from Step 2
    Output: models/ds_model/{model_type}/{symbol}/ (plots + artifacts)
    
    Available Models (7 total):
      RNN Regressors (predict spread_close_pct):
        - LSTM: Long Short-Term Memory (TensorFlow)
        - GRU: Gated Recurrent Unit (TensorFlow)
        - Transformer: Self-attention (PyTorch)
      
      Tree/Linear Classifiers (predict is_real_opportunity):
        - Linear: Logistic Regression (scikit-learn)
        - RandomForest: Ensemble trees (scikit-learn)
        - XGBoost: Gradient boosting (XGBoost)
        - CatBoost: Categorical boosting (CatBoost)
    
    Output Plots (300 DPI PNG + PDF):
      - results: Actual vs predicted, residuals, errors
      - training_history: Loss curves over epochs (RNN models)
      - feature_importance: Top-N features by importance
      - pr_curve: Precision-Recall trade-off (classifiers)
      - threshold_metrics: Classification threshold analysis
      - prediction_hist: Prediction distribution

SUPPORTED CONFIGURATIONS:

  Cryptocurrencies: BTC, ETH, DOGE, SOL, XRP, LINK (vs USD)
  Exchanges: Binance, Bitfinex, Coinbase, Gate.io, Kraken, MEXC
  Models: lstm, gru, transformer, linear, randomforest, xgboost, catboost
  
  Model-Specific Hyperparameters:
    Linear: --model-type (linear|ridge|lasso), --alpha
    LSTM/GRU: --lstm-units, --dense-units, --dropout-rate
    Transformer: --d-model, --num-layers, --dropout-rate
    XGBoost: --train-frac, --val-frac
    CatBoost: --iterations, --learning-rate, --depth
    RandomForest: --n-estimators, --max-depth

INTERACTIVE MENU USAGE:

  Start the interactive oracle:
    python arbitrage_oracle.py
    
  Main Menu:
    1. Data        → Fetch cryptocurrency data from exchanges
    2. Feature     → Run feature engineering pipeline
    3. Analysis    → Execute statistical analysis
    4. Model       → Train and evaluate ML models
    5. Exit        → Quit the oracle
  
  Data Menu:
    - Fetch: Collect 1-min OHLCV data from all exchanges
  
  Feature Menu:
    - Add Features: Generate 160+ engineered features
    - List Features: View FEATURE_LIST.md documentation
  
  Analysis Menu:
    - Run: Execute 11-section statistical analysis
  
  Model Menu:
    - Train: Train selected models on all cryptocurrencies
    - List Models: Show available models and their files
    - List Trained: Show trained model outputs
    - Evaluate: View trained model visualizations
  
INTERNAL STRUCTURE:

  Command Classes:
    - DataCommands: Delegate to data_retrieve/data_retrieve.py
    - FeatureCommands: Delegate to data_analysis/data_featuring.py
    - AnalysisCommands: Delegate to data_analysis/data_analysis.py
    - ModelCommands: Delegate to models/model_*.py files
  
  Utility Functions:
    - print_header/success/error/info: Formatted output
    - run_command: Execute subprocess commands
    - validate_symbol/exchange/model: Input validation
    - get_choice/input: User interaction
    - parse_model_names/plot_types: CLI argument parsing

OUTPUT DIRECTORY STRUCTURE:

  data/
    ├── raw_data/
    │   └── combined_{SYMBOL}_data.csv          (6 exchanges merged)
    └── featured_data/
        └── featured_{SYMBOL}_data.csv          (160+ features)

  models/ds_model/
    ├── lstm/
    │   ├── BTCUSD/
    │   │   ├── lstm_BTCUSD_results.png         (300 DPI)
    │   │   ├── lstm_BTCUSD_results.pdf         (vector)
    │   │   ├── lstm_BTCUSD_training_history.* 
    │   │   └── ... (other plots)
    │   └── ... (other symbols)
    ├── gru/ ├── transformer/
    ├── regression-linear/
    ├── regression-ridge/
    ├── regression-lasso/
    ├── random-forest/
    ├── xgboost/
    └── catboost/

INTEGRATION WITH SUBMODULES:

  1. data_retrieve/ module
     - 6 exchange API clients (binance, bitfinex, coinbase, gateio, kraken, mexc)
     - Parallel data fetching with custom pagination per exchange
     - Automatic data merging by timestamp (outer join)
     - Error handling and retry logic

  2. data_analysis/ module
     - data_featuring.py: 5-layer feature engineering pipeline
     - data_analysis.py: 11-section statistical analysis
     - FEATURE_LIST.md: Complete feature documentation
     - trading_costs.md: Trading economics and thresholds

  3. models/ module
     - 7 model implementations (LSTM, GRU, Transformer, Linear, RF, XGB, CatBoost)
     - plotter.py: Universal visualization utility (300 DPI PNG + PDF)
     - Model artifacts saved to ds_model/{model_type}/{symbol}/

KEY CONFIGURATIONS:

  TRADING_COST_PCT = 0.2           # Basic trading costs (0.1-0.2% per exchange)
  SAFETY_MARGIN_PCT = 0.1          # Safety buffer above costs
  REAL_OPPORTUNITY_THRESHOLD = 0.3 # Minimum profitable spread (both included)
  TRADE_AMOUNT_USD = 1000          # Assumed trade size for ROI calculations

TROUBLESHOOTING:

  "No featured data found"
    → Run Feature → Add Features first
  
  "No trained models found"
    → Run Model → Train to create models
  
  "Invalid model specified"
    → Use Model → List Models to see available options
  
  "Data fetch failed"
    → Some exchanges may be down; try later or check individual exchange status

DEPENDENCIES:

  Python: 3.8+
  Major packages: pandas, numpy, tensorflow, torch, scikit-learn, xgboost, catboost
  See requirements.txt for complete list

DOCUMENTATION:

  Main: README.md (comprehensive guide to all modules)
  Features: FEATURE_LIST.md (160+ feature definitions)
  Economics: trading_costs.md (cost breakdown and thresholds)
  Individual modules: See FEATURE_LIST.md in data_analysis/ directory

AUTHOR:  Oz Cabiri & Guy Harem
INSTITUTION: Tel Aviv University
PROJECT: TAU_DS_Project - Cryptocurrency Arbitrage Analysis
DATE: 2026
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
    "lstm": ["--lstm-units","--dense-units", "--dropout-rate"],
    "gru": ["--gru-units","--dense-units", "--dropout-rate"],
    "transformer": ["--d-model", "--num-layers",  "--dropout-rate"],
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
    """Data retrieval and management commands.
    
    Handles fetching cryptocurrency OHLCV data from multiple exchanges.
    Delegates to data_retrieve/data_retrieve.py orchestrator.
    
    Methods:
        fetch(): Fetch 1-minute OHLCV data from all 6 exchanges for specified
                 cryptocurrencies and date range. Merges data by timestamp (outer join).
                 Output: data/raw_data/combined_{SYMBOL}_data.csv
    """
    @staticmethod
    def fetch():
        """Fetch data from exchanges"""        
        cmd = [PYTHON, str(DATA_RETRIEVE_DIR / "data_retrieve.py")]      
        success = run_command(cmd, "Data retrieval")
        return success
    
class FeatureCommands:
    """Feature engineering and feature management commands.
    
    Generates 160+ engineered features organized in 5 dependency layers.
    Delegates to data_analysis/data_featuring.py pipeline.
    
    Methods:
        add(): Execute 5-layer feature engineering pipeline on raw data.
               Generates spreads, volatility, momentum, Bollinger bands, lags.
               Output: data/featured_data/featured_{SYMBOL}_data.csv
        
        list_features(): Display FEATURE_LIST.md documentation showing all
                        features organized by layer with explanations.
    """
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
    """Data analysis and statistical insight commands.
    
    Analyzes engineered features to extract insights about opportunities,
    risks, profitability, and temporal patterns.
    Delegates to data_analysis/data_analysis.py.
    
    Methods:
        run(): Execute 11-section statistical analysis on featured data.
               Generates opportunity frequency, spreads, temporal patterns,
               exchange patterns, volume, risk, profitability, momentum,
               Bollinger bands, persistence, and rolling statistics.
               Output: data_analysis_results.txt
    """
    @staticmethod
    def run():
        """Run full analysis pipeline"""               
        cmd = [PYTHON, str(DATA_ANALYSIS_DIR / "data_analysis.py")]
        success = run_command(cmd, f"Run Feature Analysis")
        return success

class ModelCommands:
    """Machine learning model training and evaluation commands.
    
    Trains and evaluates 7 ML models (LSTM, GRU, Transformer, Linear, RF, XGB, CatBoost)
    for predicting arbitrage opportunities and spread movements.
    Delegates to models/model_*.py files.
    
    Methods:
        train(args): Train selected models on all cryptocurrencies.
                    Supports single model, multiple models, or all models.
                    Output: models/ds_model/{model_type}/{symbol}/ (plots + artifacts)
        
        list_models(): Show available models and their implementation files.
        
        list_trained(): List trained models organized by type and cryptocurrency.
        
        evaluate(args): Display trained model visualization artifacts (PNG, PDF).
                       Supports filtering by model type and plot type.
        
        plot(args): Find and display path to a specific plot file for a model/symbol.
    """
    
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
        """Train a single model on one cryptocurrency.
        
        Parameters:
        -----------
        model_name : str
            Name of model (randomforest, lstm, gru, linear, xgboost, transformer, catboost)
        symbol : str
            Cryptocurrency symbol (BTCUSD, ETHUSD, DOGEUSD, LINKUSD, SOLUSD, XRPUSD)
        args : Args
            Command-line arguments including optional model-specific hyperparameters
            
        Returns:
        --------
        bool
            True if training succeeded, False otherwise
        """
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
        """List all available ML models.
        
        Displays the 7 supported models and their implementation files:
        - randomforest: Scikit-learn Random Forest classifier
        - lstm: TensorFlow/Keras LSTM regressor
        - gru: TensorFlow/Keras GRU regressor
        - linear: Scikit-learn Logistic Regression classifier
        - xgboost: XGBoost classifier
        - transformer: PyTorch Transformer regressor
        - catboost: CatBoost classifier
        
        Returns:
        --------
        bool
            Always returns True
        """        
        print_header("AVAILABLE MODELS:\n")
        
        for model_name, model_file in AVAILABLE_MODELS.items():
            print(f"  {model_name.upper()}")
            print(f"     File: {model_file}")
            print()
        
        return True
    
    @staticmethod
    def list_trained():
        """List all trained models organized by type and cryptocurrency.
        
        Scans models/ds_model/ directory to find trained models grouped by:
        - Model type (randomforest, lstm, gru, linear, xgboost, transformer, catboost)
        - Cryptocurrency symbol (BTCUSD, ETHUSD, DOGEUSD, LINKUSD, SOLUSD, XRPUSD)
        - Artifact count (PNG, PDF, PKL files per model/symbol)
        
        Returns:
        --------
        bool
            True if list completed (even if no models found)
        """
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
        """Display trained model visualization artifacts.
        
        Lists PNG, PDF, SVG, HTML plot files for trained models filtered by:
        - Model type (or all models if --model not specified)
        - Plot type (results, prediction_hist, training_history, feature_importance, 
                     pr_curve, threshold_metrics, opportunity_comparison)
        - Cryptocurrency symbol
        
        Parameters:
        -----------
        args : Args
            --model: Model name(s) or "all" (default: all)
            --model-type: For linear models, selects regression-{type} subdirectory
            --plot-types: Plot type filter
            
        Returns:
        --------
        bool
            True if evaluation succeeded, False if invalid models/no matches found
        """
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
        """Resolve output directory paths for a model.
        
        Handles special naming conventions:
        - linear: regression-{model_type} (linear, lasso, ridge)
        - randomforest: random-forest
        - Others: use model_name directly (lstm, gru, transformer, xgboost, catboost)
        
        Parameters:
        -----------
        model_name : str
            Name of the model
        args : Args
            Arguments including model_type for linear variants
            
        Returns:
        --------
        List[Path]
            List of Path objects pointing to model output directories
        """
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
        """Find and display path to a specific model plot.
        
        Locates a plot file for a given model and cryptocurrency, handling
        directory naming conventions for special models (linear, randomforest).
        
        Parameters:
        -----------
        args : Args
            --model: Model name (required)
            --symbol: Cryptocurrency symbol (required)  
            --name: Plot filename (required, e.g. "lstm_BTCUSD_results.png")
            --model-type: For linear models, selects regression-{type} subdirectory
            
        Returns:
        --------
        bool
            True if plot found, False if file not found or invalid arguments
        """
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

