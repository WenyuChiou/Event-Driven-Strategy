## Overview
This project focuses on developing a **quantitative trading strategy** for **Taiwan Stock Futures (1-minute interval)** using **Random Forest**. Inspired by the **hydraulic jump** phenomenon in fluid mechanics—where supercritical flow transitions into subcritical flow and creates a sudden increase in water depth—this strategy aims to identify points in the market where **accelerated downward momentum** is followed by a **strong rebound**, or vice versa.

---

## Core Concept: Hydraulic Jump in Financial Markets
- **Hydraulic Jump Analogy**  
  In fluid mechanics, a hydraulic jump occurs when a high-velocity, shallow flow transitions into a lower-velocity, deeper flow, resulting in a sudden, turbulent “jump.” We draw a parallel to the financial markets: after a period of **rapid price decline**, investors’ fear and panic may create conditions for a strong, sudden rebound—akin to the jump in water levels.

- **Application in Trading**  
  By detecting these abrupt shifts in price momentum, the strategy aims to capture **short-term reversals** that can yield profitable opportunities.

---

## Behavioral & Social Psychology Alpha Factors
- **Self-Made Alpha Factors**  
  Beyond conventional technical indicators, this project incorporates **behavioral finance** and **social psychology** theories. We design custom alpha factors to quantify market sentiment, crowd behavior, and emotional biases—elements often overlooked by purely price- or volume-based indicators.

- **Examples**  
  - **Sentiment Index**: Derived from news headlines, social media posts, or community forums.  
  - **Herding Factor**: Measures crowd-driven trading patterns (e.g., abnormally high volume or one-sided trades).  
  - **Fear-Greed Factor**: Gauges overall market mood by combining volatility measures (like ATR) with external signals (e.g., safe-haven asset flows).

---

## Random Forest Model

1. **Why Random Forest?**  
   - Handles high-dimensional data effectively.  
   - Deals with non-linear relationships and outliers robustly.  
   - Provides feature importance, aiding interpretability.

2. **Feature Set**  
   - **Technical Indicators**: EMA, ATR, volatility bands, slope change, etc.  
   - **Behavioral Alpha Factors**: Derived from behavioral finance and social psychology theories.  
   - **Event Labels**:  
     - **Long Event (Event = 1)**: Triggered when a rapid decline meets specific rebound criteria.  
     - **Short Event (Event = -1)**: Triggered when a rapid ascent shows signs of imminent reversal.

3. **Model Calibration**  
   - **Time Series Cross-Validation (TimeSeriesCV)**:  
     Uses a time-based splitting mechanism to respect chronological order and reduce lookahead bias.  
   - **Bayesian Optimization**:  
     Efficiently tunes hyperparameters (e.g., `n_estimators`, `max_depth`) by balancing exploration and exploitation in the hyperparameter space.

---

## Implementation Workflow

1. **Data Collection & Preprocessing**  
   - Gather 1-minute price and volume data for Taiwan Stock Futures.  
   - Clean and align data, handle missing values, and compute key features.

2. **Feature Engineering**  
   - Calculate conventional technical indicators (ATR, moving averages, etc.).  
   - Generate custom alpha factors based on **behavioral** and **social psychology** principles.  
   - Label potential “hydraulic jump” events (long or short).

3. **Model Training & Testing**  
   - **Time Series Cross-Validation**: Use chronological folds to ensure realistic performance estimates.  
   - **Bayesian Optimization**: Optimize hyperparameters (e.g., `n_estimators`, `max_depth`) based on performance metrics such as accuracy or Sharpe ratio.  
   - Evaluate performance using metrics like **accuracy**, **precision**, **recall**, and **strategy returns**.

4. **Strategy Backtesting**  
   - Implement a backtest to simulate trades based on model predictions.  
   - Assess **P&L**, **max drawdown**, **Sharpe ratio**, and other risk/return metrics.

5. **Deployment & Monitoring**  
   - (Optional) Deploy the model in a live trading environment or paper-trading setup.  
   - Continuously monitor model performance and retrain or adjust factors as needed.

---

## Future Directions
- **Deep Learning Extensions**  
  Experiment with LSTM or Transformer models to capture more complex temporal dependencies.

- **Real-Time Sentiment Analysis**  
  Integrate live data feeds (e.g., Twitter, news APIs) for more timely alpha factor updates.

- **Multi-Market Exploration**  
  Apply the hydraulic jump concept and alpha factors to other futures, equities, or crypto markets.

- **Adaptive Parameters**  
  Dynamically adjust model thresholds and alpha factor weightings to evolving market conditions.

---
