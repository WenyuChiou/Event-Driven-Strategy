# API Documentation

This document provides comprehensive API documentation for the Trading Event Detection and Model Prediction System.

## Table of Contents

1. [Event Detection](#event-detection)
2. [Feature Engineering](#feature-engineering)
3. [Model Training](#model-training)
4. [Backtesting](#backtesting)
5. [Evaluation Metrics](#evaluation-metrics)
6. [Validation](#validation)
7. [Alpha Factors](#alpha-factors)

---

## Event Detection

### `detect_trading_events`

Detects trading events based on price patterns and volume analysis.

**Location**: `src/event_detection.py`

**Signature**:
```python
def detect_trading_events(
    data: pd.DataFrame,
    profit_loss_window: int = 3,
    atr_window: int = 14,
    long_profit_threshold: float = 10.0,
    short_loss_threshold: float = -10.0,
    volume_multiplier: float = 2.0,
    use_atr_filter: bool = True
) -> pd.DataFrame
```

**Parameters**:
- `data` (pd.DataFrame): DataFrame containing OHLCV data
- `profit_loss_window` (int): Window size for calculating future profit/loss (default: 3)
- `atr_window` (int): Window size for ATR calculation (default: 14)
- `long_profit_threshold` (float): Profit threshold for long events (default: 10.0)
- `short_loss_threshold` (float): Loss threshold for short events (default: -10.0)
- `volume_multiplier` (float): Volume multiplier for filtering (default: 2.0)
- `use_atr_filter` (bool): Whether to use ATR-based filtering (default: True)

**Returns**:
- `pd.DataFrame`: DataFrame with detected events and labels

**Example**:
```python
from src.event_detection import detect_trading_events

events = detect_trading_events(
    data=df,
    profit_loss_window=3,
    long_profit_threshold=10.0
)
```

---

## Feature Engineering

### `calculate_features`

Calculates technical indicators and alpha factors for the dataset.

**Location**: `src/feature_engineering.py`

**Signature**:
```python
def calculate_features(
    data: pd.DataFrame,
    slope_window: int = 3,
    ema_window: int = 9,
    avg_vol_window: int = 9,
    long_ema_window: int = 13,
    scaler: Optional[MinMaxScaler] = None
) -> Tuple[pd.DataFrame, MinMaxScaler]
```

**Parameters**:
- `data` (pd.DataFrame): DataFrame with OHLCV data
- `slope_window` (int): Window for slope calculation (default: 3)
- `ema_window` (int): Window for EMA calculation (default: 9)
- `avg_vol_window` (int): Window for average volatility (default: 9)
- `long_ema_window` (int): Window for long-term EMA (default: 13)
- `scaler` (MinMaxScaler, optional): Pre-fitted scaler for normalization

**Returns**:
- `Tuple[pd.DataFrame, MinMaxScaler]`: DataFrame with features and scaler object

**Example**:
```python
from src.feature_engineering import calculate_features

df_features, scaler = calculate_features(df)
```

### `FeatureEngineeringWrapper`

Wrapper class for feature selection and engineering.

**Location**: `src/feature_engineering.py`

**Signature**:
```python
class FeatureEngineeringWrapper:
    def __init__(
        self,
        variance_threshold: float = 0.005,
        lasso_eps: float = 1e-4,
        corr_threshold: float = 0.90,
        remove_column_name: Optional[List[str]] = None
    )
    
    def fit(
        self,
        df: pd.DataFrame,
        target_column: str = 'Label'
    ) -> Tuple[pd.DataFrame, StandardScaler, List[str]]
    
    def transform(self, df: pd.DataFrame) -> pd.DataFrame
```

**Example**:
```python
from src.feature_engineering import FeatureEngineeringWrapper

fe = FeatureEngineeringWrapper(
    variance_threshold=0.005,
    corr_threshold=0.90
)
X_final, scaler, features = fe.fit(df_events)
```

---

## Model Training

### `TradingModel`

Main model class for training and prediction.

**Location**: `src/model.py`

**Signature**:
```python
class TradingModel:
    def __init__(
        self,
        model_name: str = 'lightgbm',
        params: Optional[Dict[str, Any]] = None
    )
    
    def fit(self, X: np.ndarray, y: np.ndarray) -> None
    
    def predict(self, X: np.ndarray) -> np.ndarray
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]
```

**Supported Models**:
- `'randomforest'`: Random Forest Classifier
- `'gradientboosting'`: Gradient Boosting Classifier
- `'xgboost'`: XGBoost Classifier
- `'lightgbm'`: LightGBM Classifier
- `'catboost'`: CatBoost Classifier

**Example**:
```python
from src.model import TradingModel

model = TradingModel(model_name='lightgbm', params={'n_estimators': 100})
model.fit(X_train, y_train)
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

### `BayesianOptimizerWrapper`

Wrapper for Bayesian hyperparameter optimization.

**Location**: `src/model.py`

**Signature**:
```python
class BayesianOptimizerWrapper:
    def __init__(self, model_name: str = 'lightgbm')
    
    def optimize(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_trials: int = 100,
        n_splits: int = 5
    ) -> Tuple[Dict[str, Any], float]
    
    def get_best_params(self) -> Dict[str, Any]
    
    def get_best_score(self) -> float
```

**Example**:
```python
from src.model import BayesianOptimizerWrapper

optimizer = BayesianOptimizerWrapper(model_name='lightgbm')
best_params, best_score = optimizer.optimize(X_train, y_train, n_trials=100)
```

---

## Backtesting

### `Backtester`

Main backtesting engine class.

**Location**: `src/backtesting.py`

**Signature**:
```python
class Backtester:
    def __init__(
        self,
        profit_loss_window: int = 3,
        max_profit_loss: float = 50.0
    )
    
    def run(
        self,
        data: pd.DataFrame,
        strategy: TradingStrategy
    ) -> pd.DataFrame
    
    def calculate_metrics(self) -> Dict[str, float]
    
    def plot_results(
        self,
        title: str = "Backtest Results",
        save_path: Optional[str] = None,
        show_plot: bool = True
    ) -> None
```

**Example**:
```python
from src.backtesting import Backtester, ProbabilityThresholdStrategy

strategy = ProbabilityThresholdStrategy(
    long_threshold=0.005,
    short_threshold=0.002,
    holding_period=3
)
backtester = Backtester(profit_loss_window=3)
backtester.run(data, strategy)
metrics = backtester.calculate_metrics()
```

### `filter_and_compare_strategies`

Filter and compare multiple strategies with parallel processing.

**Location**: `src/backtesting.py`

**Signature**:
```python
def filter_and_compare_strategies(
    data: pd.DataFrame,
    threshold_pairs: Optional[List[Tuple[float, float]]] = None,
    filter_criteria: Optional[Dict[str, float]] = None,
    holding_period: int = 3,
    save_dir: Optional[str] = None,
    show_plot: bool = True
) -> Tuple[Dict, pd.DataFrame, pd.DataFrame]
```

**Parameters**:
- `data` (pd.DataFrame): Trading data
- `threshold_pairs` (List[Tuple[float, float]], optional): List of (long_threshold, short_threshold) pairs
- `filter_criteria` (Dict, optional): Filtering criteria with keys:
  - `min_total_pnl`: Minimum total PnL (default: 0)
  - `min_sharpe`: Minimum Sharpe ratio (default: 1.0)
  - `min_pnl_drawdown_ratio`: Minimum PnL/Max Drawdown ratio (default: 3.0)
  - `max_trades`: Maximum number of trades (default: 2000)
  - `min_profit_factor`: Minimum profit factor (default: 1.0)
- `holding_period` (int): Holding period for trades (default: 3)
- `save_dir` (str, optional): Directory to save figures
- `show_plot` (bool): Whether to display plots (default: True)

**Returns**:
- `Tuple[Dict, pd.DataFrame, pd.DataFrame]`: Filtered backtesters, filtered results, cumulative returns

**Example**:
```python
from src.backtesting import filter_and_compare_strategies

filtered_backtesters, filtered_results, cumulative_returns = filter_and_compare_strategies(
    data=df,
    filter_criteria={
        'min_sharpe': 1.5,
        'min_profit_factor': 1.2
    }
)
```

---

## Evaluation Metrics

### `calculate_sharpe_ratio`

Calculate Sharpe ratio for returns.

**Location**: `src/evaluation.py`

**Signature**:
```python
def calculate_sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0
) -> float
```

### `calculate_sortino_ratio`

Calculate Sortino ratio for returns.

**Location**: `src/evaluation.py`

**Signature**:
```python
def calculate_sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0
) -> float
```

### `calculate_calmar_ratio`

Calculate Calmar ratio (return/max drawdown).

**Location**: `src/evaluation.py`

**Signature**:
```python
def calculate_calmar_ratio(
    returns: pd.Series,
    max_drawdown: float
) -> float
```

**Example**:
```python
from src.evaluation import calculate_sharpe_ratio, calculate_sortino_ratio

sharpe = calculate_sharpe_ratio(returns)
sortino = calculate_sortino_ratio(returns)
```

---

## Validation

### `validate_file_path`

Validate file path.

**Location**: `src/validation.py`

**Signature**:
```python
def validate_file_path(
    file_path: str,
    must_exist: bool = True,
    allowed_extensions: Optional[List[str]] = None
) -> Path
```

### `validate_dataframe`

Validate DataFrame structure and content.

**Location**: `src/validation.py`

**Signature**:
```python
def validate_dataframe(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
    min_rows: int = 1,
    check_nulls: bool = True
) -> pd.DataFrame
```

### `validate_ohlcv_data`

Validate OHLCV (Open, High, Low, Close, Volume) data.

**Location**: `src/validation.py`

**Signature**:
```python
def validate_ohlcv_data(df: pd.DataFrame) -> pd.DataFrame
```

**Example**:
```python
from src.validation import validate_file_path, validate_ohlcv_data

# Validate file path
data_path = validate_file_path(
    "data.xlsx",
    must_exist=True,
    allowed_extensions=['.xlsx', '.csv']
)

# Validate OHLCV data
df_validated = validate_ohlcv_data(df)
```

---

## Alpha Factors

### `AlphaFactory`

Factory class for generating alpha factors.

**Location**: `package/alpha_eric.py`

**Signature**:
```python
class AlphaFactory:
    def __init__(self, data: pd.DataFrame)
    
    def add_all_alphas(self, days: List[int]) -> pd.DataFrame
    
    def alpha01(self, days: List[int], par: Optional[Dict] = None, type: Optional[str] = None) -> pd.DataFrame
    
    def alpha02(self, days: List[int], weight: float = 0.5) -> pd.DataFrame
    
    def alpha03(self, days: List[int], risk_aversion: float = 1.5) -> pd.DataFrame
    
    # ... (many more alpha functions)
```

**Example**:
```python
from package.alpha_eric import AlphaFactory

alpha = AlphaFactory(data)
df_with_alphas = alpha.add_all_alphas(days=[3, 9, 20, 60, 120, 240])
```

---

## Logging

### `setup_logger`

Set up logger with file and console handlers.

**Location**: `src/logger.py`

**Signature**:
```python
def setup_logger(
    name: str = "trading_system",
    log_file: Optional[str] = None,
    log_level: int = logging.INFO,
    log_dir: Optional[Path] = None
) -> logging.Logger
```

**Example**:
```python
from src.logger import setup_logger

logger = setup_logger("my_module")
logger.info("This is an info message")
logger.error("This is an error message")
```

---

## Configuration

### `Config`

Centralized configuration class.

**Location**: `src/config.py`

**Usage**:
```python
from src.config import Config

# Access paths
data_path = Config.DEFAULT_TRAINING_DATA
models_dir = Config.MODELS_DIR

# Access parameters
event_params = Config.EVENT_DETECTION_PARAMS
```

---

## Error Handling

All functions raise appropriate exceptions:

- `ValidationError`: Raised by validation functions when validation fails
- `FileNotFoundError`: Raised when required files are not found
- `ValueError`: Raised for invalid parameter values
- `TypeError`: Raised for incorrect data types

**Example**:
```python
from src.validation import ValidationError, validate_file_path

try:
    path = validate_file_path("nonexistent.xlsx", must_exist=True)
except ValidationError as e:
    print(f"Validation error: {e}")
```

---

## Notes

- All DataFrame operations preserve the original index
- All numeric calculations use float64 precision
- Parallel processing is used where applicable (e.g., `filter_and_compare_strategies`)
- Vectorized operations are preferred for performance (e.g., alpha factor calculations)

