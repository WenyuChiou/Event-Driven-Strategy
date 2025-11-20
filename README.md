# Trading Event Detection and Model Prediction System

A comprehensive trading system for analyzing financial data, predicting trading opportunities, and testing various strategies including machine learning models and traditional technical approaches. The system is inspired by the hydraulic jump phenomenon in fluid dynamics, which has striking parallels to market behavior.

## 🌟 System Features

- **Multiple Strategies**: Support for ML models and traditional strategies like Moving Average (MA)
- **Alpha Factor Generation**: Over 200+ behavioral finance and technical alpha factors
- **Signal Generation**: Smart detection of trading events based on price patterns
- **Advanced Backtesting**: Test different strategies with detailed performance analytics
- **Real-time Simulation**: Simulate trading in real-time environments
- **Visualization Tools**: Generate intuitive charts and comprehensive performance reports
- **Model Versioning**: Track and compare different model versions
- **Feature Analysis**: SHAP values and feature importance analysis
- **Comprehensive Metrics**: Financial metrics including Sharpe, Sortino, Calmar ratios

## 🌊 The Hydraulic Jump Concept

This project was inspired by the hydraulic jump phenomenon in fluid dynamics, which has striking parallels to market behavior.

A hydraulic jump occurs when a fast-flowing fluid (supercritical flow) suddenly slows down and rises in height, converting to subcritical flow. This phenomenon is observable in rivers, spillways, and open channels.

<p align="center">
  <img src="Fig/HJ.png" alt="Hydraulic Jump Diagram" width="600"><br>
  <em>Figure 1: Illustration of a hydraulic jump in an open channel.</em>
</p>

In financial markets, similar behavior can be observed:
1. **Slow Price Decline**: Analogous to the gradual flow before the critical point
2. **Rapid Price Decline**: Corresponds to the high-velocity flow
3. **Hydraulic Jump**: The point where the market suddenly reverses direction after a rapid decline

<p align="center">
  <img src="Fig/example.png" alt="Market Hydraulic Jump Example" width="600"><br>
  <em>Figure 2: Example of a market "hydraulic jump" where a rapid price decline is followed by a sudden reversal.</em>
</p>

This natural phenomenon provides the theoretical foundation for our trading strategy, especially for identifying potential reversal points after rapid market declines.

## 💻 Installation Guide

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Step-by-Step Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/WenyuChiou/Event-Driven-Strategy.git
   cd Event-Driven-Strategy
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On Linux/Mac
   source venv/bin/activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Install TA-Lib** (optional, for technical analysis):
   - **Windows**: Download wheel file from [here](https://www.lfd.uci.edu/~gohlke/pythonlibs/#ta-lib) and install:
     ```bash
     pip install TA_Lib‑0.4.24‑cp39‑cp39‑win_amd64.whl
     ```
   - **Linux/Mac**: Follow instructions at [TA-Lib documentation](https://github.com/mrjbq7/ta-lib)

5. **Verify installation**:
   ```bash
   python -c "import pandas, numpy, sklearn, lightgbm; print('Installation successful!')"
   ```

### Alternative: Install as Package

```bash
pip install -e .
```

## 🚀 Quick Start Guide

### 1. Prepare Your Data

Place your data files in the `data/raw/` directory. Required format:
- Excel (.xlsx) or CSV files
- Required columns: `date`, `open`, `high`, `low`, `close`, `volume`
- Date column should be in datetime format

Example data structure:
```
date,open,high,low,close,volume
2023-01-01 08:45:00,100.0,100.5,99.5,100.2,10000
2023-01-01 08:46:00,100.2,100.8,100.0,100.5,12000
...
```

### 2. Basic Usage

**Train a model:**
```bash
python main.py --mode train --data data/raw/TX00_training.xlsx --model lightgbm --trials 100
```

**Run backtesting:**
```bash
python main.py --mode backtest --validation-data data/raw/TX00_validation.xlsx
```

**Run MA strategy:**
```bash
python main.py --mode ma_strategy --validation-data data/raw/TX00_validation.xlsx --ma-period 5
```

**Run all modes (train + backtest):**
```bash
python main.py --mode all
```

### 3. Using the Streamlit Web Interface

```bash
streamlit run app.py
```

Then open your browser to `http://localhost:8501`

## 📁 Project Structure

```
project/
│
├── src/                          # Source code directory
│   ├── __init__.py              
│   ├── config.py                # Configuration management
│   ├── logger.py                # Logging system
│   ├── event_detection.py       # Event detection module
│   ├── feature_engineering.py   # Feature engineering module
│   ├── feature_pipeline.py      # Modular feature pipeline
│   ├── feature_analysis.py     # Feature importance analysis (SHAP)
│   ├── alpha_selector.py        # Alpha factor selection
│   ├── model.py                 # Model training and evaluation
│   ├── model_registry.py        # Model versioning
│   ├── backtesting.py           # Backtesting module
│   ├── evaluation.py            # Financial metrics
│   ├── visualization.py        # Visualization tools
│   ├── ma_strategy.py           # MA strategy implementation
│   └── utils.py                 # Helper functions
│
├── package/                      # External functionality packages
│   ├── alpha_eric.py            # Alpha factor calculations (200+ factors)
│   ├── FE.py                    # Feature engineering tools
│   ├── ModelLoader.py           # Model loading utilities
│   └── FuturesFilter.py         # Futures data filtering
│
├── examples/                     # Example scripts
│   ├── basic_example.py         # Basic usage example
│   ├── model_training.py        # Model training example
│   └── backtesting_example.py   # Backtesting example
│
├── tests/                        # Test suite
│   ├── unit/                    # Unit tests
│   ├── integration/             # Integration tests
│   └── fixtures/                # Test data
│
├── data/                         # Data directory
│   ├── raw/                     # Raw data
│   └── processed/               # Processed data
│
├── models/                       # Saved trained models
│   └── registry/                # Model registry
│
├── results/                      # Results output directory
│   ├── backtests/               # Backtest results
│   ├── visualization/          # Chart outputs
│   └── summaries/               # Analysis summaries
│
├── docs/                         # Documentation
│   ├── USER_GUIDE.md            # User guide
│   └── API.md                   # API documentation
│
├── Fig/                          # Image directory
│   ├── HJ.png                   # Hydraulic jump diagram
│   └── example.png              # Market example
│
├── main.py                       # Main program entry point
├── app.py                        # Streamlit web application
├── imports.py                    # Streamlit dependencies
├── setup.py                      # Package installation script
├── requirements.txt              # Project dependencies
└── README.md                     # This file
```

## 📊 Main Components

### 1. Trading Strategies

- **Machine Learning Models**: Train prediction models using historical data
  - RandomForest, GradientBoosting, XGBoost, LightGBM
  - Bayesian hyperparameter optimization
  - Model versioning and tracking
  
- **Moving Average Strategy**: Classic MA crossover strategy with optimization
  - Multiple MA periods
  - Parameter optimization
  - Performance analytics

- **Signal Generator**: Generate trading signals based on price patterns
  - Probability-based thresholds
  - Time-based filtering
  - Position management

### 2. Data Processing

- **Feature Engineering**: Advanced feature creation and selection
  - 200+ alpha factors based on behavioral finance
  - Technical indicators (TA-Lib)
  - Feature selection using Lasso and variance threshold
  - Modular feature pipeline with caching

- **Alpha Factor Selection**: 
  - Information Coefficient (IC) analysis
  - Statistical significance testing
  - Alpha decay analysis
  - Feature ranking

### 3. Performance Analysis

- **Backtesting Engine**: Evaluate strategies with detailed metrics
  - Separate long/short analysis
  - Parameter optimization
  - Strategy comparison
  - Walk-forward validation

- **Evaluation Metrics**:
  - Sharpe Ratio, Sortino Ratio, Calmar Ratio
  - Information Ratio, Treynor Ratio
  - Maximum drawdown analysis
  - Trade statistics

- **Visualization Tools**: Generate various charts and performance reports
  - Cumulative returns charts
  - Trade comparison visualizations
  - Feature importance plots
  - Performance heatmaps

## 📋 Command-Line Usage

### Training a Model

```bash
python main.py --mode train \
    --data data/raw/training_data.xlsx \
    --model lightgbm \
    --trials 100
```

**Available models**: `randomforest`, `gradientboosting`, `xgboost`, `lightgbm`

### Backtesting

```bash
python main.py --mode backtest \
    --validation-data data/raw/validation_data.xlsx \
    --model-path models/lightgbm.joblib \
    --feature-path models/features.xlsx \
    --scaler-path models/scaler.joblib
```

### Comprehensive Backtesting (Training + Validation)

```bash
python main.py --mode comprehensive \
    --data data/raw/training_data.xlsx \
    --validation-data data/raw/validation_data.xlsx
```

### Real-time Trading Simulation

```bash
python main.py --mode simulate \
    --validation-data data/raw/validation_data.xlsx \
    --long-threshold 0.0026 \
    --short-threshold 0.0026
```

### MA Strategy

```bash
python main.py --mode ma_strategy \
    --validation-data data/raw/validation_data.xlsx \
    --ma-period 5 \
    --commission 0.0
```

**MA Strategy Parameters**:
- `--ma-period`: Moving average period (default: 5)
- `--commission`: Trading commission (default: 0.0)
- `--price-col`: Price column name (default: 'Close')
- `--min-hold-periods`: Minimum holding periods (default: 5)
- `--optimize`: Run parameter optimization

## 🧠 Machine Learning Models

The system supports multiple machine learning algorithms:

- **RandomForest**: Ensemble learning method using multiple decision trees
- **GradientBoosting**: Sequential ensemble technique that combines weak learners
- **XGBoost**: Optimized gradient boosting implementation
- **LightGBM**: High-performance gradient boosting framework by Microsoft

**Model Features**:
- Bayesian hyperparameter optimization using Optuna
- Time series cross-validation
- Feature importance analysis
- Model versioning and registry
- Performance tracking

## 📈 Alpha Factors

The system includes over 200 alpha factors based on behavioral finance theories:

- **Prospect Theory**: Factors based on risk aversion and reference-dependent preferences
- **Herding Behavior**: Factors capturing market herding and sentiment shifts
- **Anchoring Effect**: Factors analyzing anchoring biases in trading behavior
- **Market Microstructure**: Factors based on trading volume, volatility, and price momentum
- **Momentum-based**: Cross-asset momentum, relative strength
- **Volatility-based**: GARCH-based volatility, realized volatility ratios
- **Liquidity-based**: Bid-ask spread factors, volume-price relationships

**Alpha Selection Tools**:
- Information Coefficient (IC) calculation
- Statistical significance testing
- Alpha decay analysis
- Feature ranking and selection

## 📊 Performance Metrics

The system calculates comprehensive performance metrics:

### Return Metrics
- Total Return
- Annualized Return
- Volatility

### Risk-Adjusted Metrics
- **Sharpe Ratio**: Risk-adjusted return using standard deviation
- **Sortino Ratio**: Risk-adjusted return using downside deviation only
- **Calmar Ratio**: Annual return / maximum drawdown
- **Information Ratio**: Active return / tracking error
- **Treynor Ratio**: Excess return / beta

### Risk Metrics
- Maximum Drawdown
- Drawdown Duration
- Recovery Time

### Trade Statistics
- Win Rate
- Profit Factor
- Average Win/Loss
- Largest Win/Loss
- Win/Loss Streaks
- Average Trade Duration

## 🏗️ Architecture Overview

### Data Flow

```
Raw Data → Feature Calculation → Event Detection → Feature Engineering 
    → Model Training → Backtesting → Performance Evaluation
```

### Key Modules

1. **Event Detection** (`src/event_detection.py`): Detects trading events based on price patterns
2. **Feature Engineering** (`src/feature_engineering.py`): Creates and selects features
3. **Model Training** (`src/model.py`): Trains ML models with hyperparameter optimization
4. **Backtesting** (`src/backtesting.py`): Evaluates strategies on historical data
5. **Evaluation** (`src/evaluation.py`): Calculates financial performance metrics
6. **Visualization** (`src/visualization.py`): Creates charts and reports

## 🧪 Testing

Run tests using pytest:

```bash
# Run all tests
pytest

# Run unit tests only
pytest tests/unit/

# Run with coverage
pytest --cov=src --cov-report=html
```

## 📚 Documentation

- **User Guide**: See `docs/USER_GUIDE.md` for detailed usage instructions
- **API Documentation**: See `docs/API.md` for function and class references
- **Examples**: Check `examples/` directory for code examples

## 🔧 Configuration

Configuration is managed through `src/config.py`. Key settings include:

- Paths: All directory paths are automatically configured
- Event Detection Parameters: Profit/loss windows, thresholds
- Feature Engineering Parameters: Variance thresholds, correlation thresholds
- Model Parameters: Default hyperparameters for each model type
- Trading Strategy Parameters: Thresholds, holding periods, excluded times

## 🚨 Troubleshooting

### Common Issues

1. **Import Errors**: Make sure all dependencies are installed:
   ```bash
   pip install -r requirements.txt
   ```

2. **TA-Lib Installation**: See installation guide above for platform-specific instructions

3. **Memory Issues**: For large datasets, consider:
   - Using data sampling
   - Processing data in chunks
   - Reducing feature count

4. **Model Loading Errors**: Ensure model files exist in `models/` directory

## 🤝 Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Make your changes
4. Add tests for new functionality
5. Ensure all tests pass (`pytest`)
6. Commit your changes (`git commit -m 'Add amazing feature'`)
7. Push to the branch (`git push origin feature/amazing-feature`)
8. Open a Pull Request

## 📝 License

This project is for learning and research purposes only and does not constitute investment advice.

## ⚠️ Disclaimer

- This project is for learning and research purposes only
- Does not constitute investment advice
- Please carefully evaluate backtest results and consider model limitations
- Thoroughly test and evaluate strategies before using actual funds for trading
- Past performance does not guarantee future results

## 📞 Support

For issues, questions, or contributions, please open an issue on the repository.

## 🎯 Roadmap

- [ ] Add more advanced alpha factors
- [ ] Implement deep learning models (LSTM, Transformer)
- [ ] Add ensemble methods
- [ ] Improve parallel processing
- [ ] Enhanced visualization dashboard
- [ ] Real-time data integration
- [ ] Paper trading interface

## 📖 Citation

If you use this project in your research, please cite:

```
Trading Event Detection System based on Hydraulic Jump Concept
https://github.com/yourusername/trading-event-detection
```
