# User Guide: Trading Event Detection System

This guide provides step-by-step instructions for using the Trading Event Detection System.

## Table of Contents

1. [Getting Started](#getting-started)
2. [Data Preparation](#data-preparation)
3. [Model Training](#model-training)
4. [Backtesting](#backtesting)
5. [Running Strategies](#running-strategies)
6. [Interpreting Results](#interpreting-results)
7. [Best Practices](#best-practices)

## Getting Started

### System Requirements

- Python 3.8 or higher
- 4GB+ RAM (8GB+ recommended for large datasets)
- Windows, Linux, or macOS

### Installation

See the main [README.md](../README.md) for detailed installation instructions.

## Data Preparation

### Data Format Requirements

Your data files should be in Excel (.xlsx) or CSV format with the following columns:

**Required Columns**:
- `date`: DateTime column with timestamps
- `open`: Opening price
- `high`: Highest price
- `low`: Lowest price
- `close`: Closing price
- `volume`: Trading volume

**Example Data Structure**:
```csv
date,open,high,low,close,volume
2023-01-01 08:45:00,100.0,100.5,99.5,100.2,10000
2023-01-01 08:46:00,100.2,100.8,100.0,100.5,12000
```

### Data Quality Checks

Before using your data, ensure:
1. No missing values in required columns
2. Date column is in proper datetime format
3. Price columns are numeric
4. Volume is non-negative
5. High >= Close >= Low (price consistency)

### Placing Data Files

Place your data files in:
- Training data: `data/raw/TX00_training.xlsx` (or specify custom path)
- Validation data: `data/raw/TX00_validation.xlsx` (or specify custom path)

## Model Training

### Basic Training

Train a model with default settings:

```bash
python main.py --mode train --data data/raw/training_data.xlsx --model lightgbm
```

### Advanced Training Options

```bash
python main.py --mode train \
    --data data/raw/training_data.xlsx \
    --model lightgbm \
    --trials 200 \
    --run-id my_experiment_001
```

**Parameters**:
- `--model`: Model type (`lightgbm`, `xgboost`, `randomforest`, `gradientboosting`)
- `--trials`: Number of hyperparameter optimization trials (default: 100)
- `--run-id`: Unique identifier for this training run

### Understanding Training Output

During training, you'll see:
1. Data loading progress
2. Feature calculation progress
3. Event detection statistics
4. Feature engineering results
5. Hyperparameter optimization progress
6. Final model metrics

**Output Files** (saved in `models/`):
- `{model_name}.joblib`: Trained model
- `scaler.joblib`: Feature scaler
- `features.xlsx`: Selected features list
- `params.json`: Best hyperparameters

## Backtesting

### Basic Backtesting

```bash
python main.py --mode backtest \
    --validation-data data/raw/validation_data.xlsx
```

### Advanced Backtesting

```bash
python main.py --mode backtest \
    --validation-data data/raw/validation_data.xlsx \
    --model-path models/lightgbm.joblib \
    --feature-path models/features.xlsx \
    --scaler-path models/scaler.joblib \
    --separate-long-short \
    --enhanced-charts \
    --save-excel
```

**Parameters**:
- `--separate-long-short`: Analyze long and short trades separately
- `--enhanced-charts`: Use enhanced visualization
- `--save-excel`: Save detailed Excel results
- `--no-show-plots`: Don't display plots (for batch processing)

### Understanding Backtest Results

Backtest results include:

**Performance Metrics**:
- Total P&L
- Win Rate
- Profit Factor
- Sharpe Ratio
- Maximum Drawdown
- Separate metrics for long/short trades

**Output Files** (saved in `results/backtests/`):
- `backtest_results_{run_id}.txt`: Detailed text results
- `optimization_results_{run_id}.xlsx`: Parameter optimization results
- `best_params_{run_id}.txt`: Best parameters found
- Figures in `results/backtests/figures/{run_id}/`

## Running Strategies

### MA Strategy

```bash
python main.py --mode ma_strategy \
    --validation-data data/raw/validation_data.xlsx \
    --ma-period 5 \
    --commission 0.0
```

**MA Strategy Parameters**:
- `--ma-period`: Moving average period (3-100, default: 5)
- `--commission`: Commission per trade (default: 0.0)
- `--price-col`: Price column name (default: 'Close')
- `--min-hold-periods`: Minimum holding periods (default: 5)
- `--optimize`: Run parameter optimization

### Real-time Simulation

```bash
python main.py --mode simulate \
    --validation-data data/raw/validation_data.xlsx \
    --long-threshold 0.0026 \
    --short-threshold 0.0026
```

**Simulation Parameters**:
- `--long-threshold`: Probability threshold for long signals (default: 0.0026)
- `--short-threshold`: Probability threshold for short signals (default: 0.0026)

## Interpreting Results

### Performance Metrics Guide

**Sharpe Ratio**:
- > 1.0: Good risk-adjusted returns
- > 2.0: Excellent risk-adjusted returns
- < 1.0: Poor risk-adjusted returns

**Sortino Ratio**:
- Similar to Sharpe but only considers downside risk
- Generally higher than Sharpe for strategies with asymmetric returns

**Calmar Ratio**:
- Annual return / Maximum drawdown
- > 1.0: Good risk-adjusted performance
- Higher is better

**Profit Factor**:
- Gross profit / Gross loss
- > 1.5: Good
- > 2.0: Excellent

**Win Rate**:
- Percentage of profitable trades
- 50%+ is generally good
- Higher win rate with good profit factor is ideal

### Reading Charts

**Cumulative Returns Chart**:
- Shows strategy performance over time
- Compare with buy-and-hold baseline
- Look for consistent upward trend

**Trade Comparison Chart**:
- Shows individual trade results
- Green markers: Winning trades
- Red markers: Losing trades
- Triangle up: Long trades
- Triangle down: Short trades

**Feature Importance Chart**:
- Shows which features the model considers most important
- Higher bars = more important features
- Use to understand model decisions

## Best Practices

### 1. Data Quality

- Always validate your data before training
- Check for outliers and missing values
- Ensure sufficient historical data (at least 1000+ data points recommended)

### 2. Model Training

- Start with default parameters
- Use at least 100 trials for hyperparameter optimization
- Split data into training and validation sets
- Monitor for overfitting (large gap between training and validation performance)

### 3. Backtesting

- Always backtest on out-of-sample data
- Use walk-forward validation for time series
- Consider transaction costs in your analysis
- Be aware of look-ahead bias

### 4. Strategy Development

- Test multiple parameter combinations
- Use separate long/short analysis to understand strategy behavior
- Monitor drawdown periods
- Consider market regime changes

### 5. Risk Management

- Never risk more than you can afford to lose
- Use proper position sizing
- Set stop-loss levels
- Diversify across different strategies

## Common Workflows

### Workflow 1: Complete Model Development

```bash
# Step 1: Train model
python main.py --mode train --data data/raw/training.xlsx --model lightgbm --trials 100

# Step 2: Backtest on validation data
python main.py --mode backtest --validation-data data/raw/validation.xlsx

# Step 3: Run comprehensive backtest (training + validation)
python main.py --mode comprehensive --data data/raw/training.xlsx --validation-data data/raw/validation.xlsx
```

### Workflow 2: Strategy Optimization

```bash
# Step 1: Run MA strategy with optimization
python main.py --mode ma_strategy --validation-data data/raw/validation.xlsx --optimize

# Step 2: Analyze results in results/ma_strategy/
# Step 3: Adjust parameters based on results
```

### Workflow 3: Real-time Simulation

```bash
# Step 1: Train model (if not already done)
python main.py --mode train --data data/raw/training.xlsx

# Step 2: Run simulation with chosen thresholds
python main.py --mode simulate --validation-data data/raw/validation.xlsx --long-threshold 0.0026 --short-threshold 0.0026
```

## Troubleshooting

### Issue: "No events detected"

**Solution**: 
- Adjust event detection parameters (profit thresholds, volume multipliers)
- Check data quality and ensure required technical indicators are present
- Verify data has sufficient history

### Issue: "Model training takes too long"

**Solution**:
- Reduce number of trials (`--trials 50`)
- Use smaller dataset for initial testing
- Reduce feature count
- Use faster model (e.g., LightGBM instead of XGBoost)

### Issue: "Poor backtest performance"

**Solution**:
- Check for data leakage
- Verify model was trained on different data than backtest
- Adjust probability thresholds
- Consider different feature sets
- Check for overfitting

### Issue: "Memory errors"

**Solution**:
- Process data in chunks
- Reduce feature count
- Use data sampling
- Close other applications

## Advanced Usage

### Using Configuration Module

```python
from src.config import Config

# Access paths
data_path = Config.RAW_DATA_DIR / "my_data.xlsx"
model_path = Config.MODELS_DIR / "my_model.joblib"

# Access parameters
event_params = Config.EVENT_DETECTION_PARAMS
strategy_params = Config.TRADING_STRATEGY_PARAMS
```

### Using Logging System

```python
from src.logger import get_logger

logger = get_logger("my_module")
logger.info("Processing data...")
logger.warning("Low data quality detected")
logger.error("Failed to load model")
```

### Using Model Registry

```python
from src.model_registry import ModelRegistry

registry = ModelRegistry()
version_id = registry.register_model(
    model_path="models/lightgbm.joblib",
    model_name="lightgbm",
    model_type="lightgbm",
    parameters={"n_estimators": 100},
    performance_metrics={"accuracy": 0.85}
)

# Get latest version
latest = registry.get_latest_version("lightgbm")

# Compare versions
comparison = registry.compare_models("lightgbm")
```

## Next Steps

- Explore the example scripts in `examples/` directory
- Read API documentation in `docs/API.md`
- Experiment with different models and parameters
- Develop your own alpha factors
- Create custom strategies

For more information, see the main [README.md](../README.md) file.

