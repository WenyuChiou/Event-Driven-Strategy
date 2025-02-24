#%%
import pandas as pd
import matplotlib.pyplot as plt

def calculate_tradingview_kama(prices, period=10, fast_period=2, slow_period=30):
    """
    Calculate KAMA (Kaufman's Adaptive Moving Average) using TradingView-like logic.
    :param prices: Series of closing prices
    :param period: Look-back period for ER calculation
    :param fast_period: Fast smoothing period for SC
    :param slow_period: Slow smoothing period for SC
    :return: Series of KAMA values
    """
    kama = [prices.iloc[0]]  # Initialize KAMA with the first price value
    for t in range(period, len(prices)):
        # Calculate Change and Volatility
        change = abs(prices.iloc[t] - prices.iloc[t - period])
        volatility = sum(abs(prices.iloc[i] - prices.iloc[i - 1]) for i in range(t - period + 1, t + 1))
        # Efficiency Ratio (ER)
        er = change / (volatility + 1e-10)  # Avoid division by zero
        # Smoothing constant (SC)
        sc = (er * (2 / (fast_period + 1)) + (1 - er) * (2 / (slow_period + 1))) ** 2
        # Calculate KAMA
        kama_value = kama[-1] + sc * (prices.iloc[t] - kama[-1])
        kama.append(kama_value)
    # Fill the initial KAMA values with None for alignment
    kama = [None] * (period - 1) + kama
    return pd.Series(kama[:len(prices)], index=prices.index)

# Load the dataset
data = pd.read_excel(r"C:\Users\wenyu\Desktop\trade\investment\python\API\test\TX00\TX00_1_20240924_20241124.xlsx")
data['date'] = pd.to_datetime(data['date'])
data.set_index('date', inplace=True)

# Calculate Short-KAMA using a period of 5
short_period = 5
data['Short_KAMA'] = calculate_tradingview_kama(data['close'], period=short_period)

# Subset data for visualization
subset_length = 100  # Define the number of data points to display
data_subset = data.iloc[:subset_length]

# Plot K-line with Short-KAMA
fig, ax = plt.subplots(figsize=(14, 8))

# Draw K-line (candlestick chart)
for i in range(len(data_subset)):
    color = 'green' if data_subset['close'].iloc[i] >= data_subset['open'].iloc[i] else 'red'
    ax.plot([data_subset.index[i], data_subset.index[i]], 
            [data_subset['low'].iloc[i], data_subset['high'].iloc[i]], color=color, linewidth=0.8)
    ax.plot([data_subset.index[i], data_subset.index[i]], 
            [data_subset['open'].iloc[i], data_subset['close'].iloc[i]], color=color, linewidth=4)

# Plot Short-KAMA
ax.plot(data_subset.index, data_subset['Short_KAMA'], color='blue', linestyle='-', label='Short-KAMA (Window = 5)')

ax.set_title('K-Line with Short-KAMA')
ax.set_xlabel('Datetime')
ax.set_ylabel('Price')
ax.legend()
ax.grid()
plt.show()
#%%
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import talib


# Bollinger Bands 計算函數
def calculate_bbands(data, window=20, num_std=2):
    """計算 Bollinger Bands"""
    data['Middle_Band'] = data['close'].rolling(window=window).mean()
    data['Std_Dev'] = data['close'].rolling(window=window).std()
    data['Upper_Band'] = data['Middle_Band'] + num_std * data['Std_Dev']
    data['Lower_Band'] = data['Middle_Band'] - num_std * data['Std_Dev']
    return data


# MACD 計算函數
def calculate_macd(data, short=12, long=26, signal=9):
    """計算 MACD 和柱體"""
    data['EMA_short'] = data['close'].ewm(span=short).mean()
    data['EMA_long'] = data['close'].ewm(span=long).mean()
    data['MACD'] = data['EMA_short'] - data['EMA_long']
    data['MACD_Hist'] = data['MACD'] - data['MACD'].ewm(span=signal).mean()
    return data


def detect_accelerated_downtrend_events_with_profit_loss(
    data, slope_window=5, ema_window=5, avg_vol_window=9, profit_loss_window=5, long_ema_window=20
):
    """
    Detect events based on specified criteria:
    1. Downtrend (negative slope of Lower_Band).
    2. Accelerated downtrend (slope change becomes steeper).
    3. Rebound above EMA(3).
    4. Profit/Loss points > 3x average volatility of past 9 windows.
    5. 20-day EMA is trending downward.
    """
    # Step 1: Calculate rolling slope of Lower_Band over slope_window
    data['Lower_Band_Slope'] = data['Lower_Band'].diff(slope_window) / data['Lower_Band'].shift(slope_window)
    data['Lower_Band_Slope'] = data['Lower_Band_Slope'].fillna(0)

    # Normalize slope to [-1, 1]
    scaler = MinMaxScaler(feature_range=(-1, 1))
    data['Normalized_Slope'] = scaler.fit_transform(data[['Lower_Band_Slope']])

    # Step 2: Calculate slope change over slope_window
    data['Slope_Change'] = data['Normalized_Slope'] - data['Normalized_Slope'].shift(slope_window)
    data['Accelerated_Downtrend'] = (data['Lower_Band_Slope'] < 0) & (data['Slope_Change'] < 0)

    # Step 3: Calculate EMA(3)
    data['EMA'] = talib.EMA(data['close'], timeperiod=ema_window)
    data['Rebound_Above_EMA'] = data['close'] > data['EMA']

    # Step 4: Calculate long-term EMA (20-day EMA)
    data['Long_EMA'] = talib.EMA(data['close'], timeperiod=long_ema_window)
    data['Long_EMA_Downward'] = data['Long_EMA'].diff() < 0

    # Step 5: Calculate average volatility
    data['Average_Volatility'] = data['close'].diff().abs().rolling(window=avg_vol_window).mean()

    # Step 6: Calculate profit/loss points
    data['Profit_Loss_Points'] = data['close'].shift(-profit_loss_window) - data['close']

    # Step 7: Identify events
    data['Event'] = (
        (data['Lower_Band_Slope'] < 0)
        & data['Accelerated_Downtrend']
        & data['Rebound_Above_EMA']
        & (data['Profit_Loss_Points'] > 3 * data['Average_Volatility'])
        & (data['Profit_Loss_Points'] > 10)
        & data['Long_EMA_Downward']
    )

    return data, scaler


# Main script
file_path = r"C:\Users\wenyu\Desktop\trade\investment\python\API\test\TX00\TX00_1_20240924_20241124.xlsx"

# Load and prepare data
df = pd.ExcelFile(file_path).parse('Sheet1')
df['date'] = pd.to_datetime(df['date'])
df = df.sort_values(by='date').reset_index(drop=True)

# Calculate indicators
df = calculate_macd(df)
df = calculate_bbands(df)

# Detect events
df, scaler = detect_accelerated_downtrend_events_with_profit_loss(df, profit_loss_window=5)
# 标注正样本：满足条件的事件
df['Label'] = 0  # 初始化为负样本
df.loc[df['Event'] & (df['Profit_Loss_Points'] > 0), 'Label'] = 1

# Drop the 'Profit_Loss_Points' column if it exists
if 'Profit_Loss_Points' in df.columns:
    df = df.drop(columns=['Profit_Loss_Points'])
# Filter detected events
detected_events = df[df['Event']]

# Print event information
print(detected_events[['close', 'Lower_Band_Slope', 'Slope_Change', 'EMA', 'Average_Volatility']])
print(f"Total events detected: {len(detected_events)}")
print(f"percentage of positive events: {len(detected_events[detected_events['Label'] == 1]) / len(df)*100} %")




# %%
import matplotlib.pyplot as plt

def plot_kline_with_signals(data):
    """
    Plot K-line chart with detected event signals and EMA.
    
    :param data: DataFrame containing close prices, EMA, Long_EMA, and Event information.
    """
    plt.figure(figsize=(16, 10))
    
    # 绘制 K 线图（用 close 价格代替简化版本）
    plt.plot(data.index, data['close'], label='Close Price', color='blue', linewidth=1.5)
    
    # 绘制短期 EMA
    plt.plot(data.index, data['EMA'], label='EMA (5)', color='purple', linestyle='--', linewidth=1.5)
    
    # 绘制长期 EMA
    plt.plot(data.index, data['Long_EMA'], label='Long EMA (20)', color='orange', linestyle='-', linewidth=1.5)
    
    # 标记事件发生的位置
    event_data = data[data['Event']]  # 筛选出标记为事件的行
    plt.scatter(event_data.index, event_data['close'], color='red', label='Detected Events', zorder=5)
    
    # 添加标题和图例
    plt.title('K-Line Chart with Detected Events and EMA', fontsize=16)
    plt.xlabel('Date', fontsize=14)
    plt.ylabel('Price', fontsize=14)
    plt.legend(fontsize=12)
    
    # 显示图表
    plt.grid()
    plt.show()

# 绘制图表
plot_kline_with_signals(df)
# %%
print(f"Total events detected: {len(detected_events)}")
print(detected_events[['close', 'Average_Volatility']].describe())

# %%
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from model.ModelLoader import *
import numpy as np

features = ['Lower_Band_Slope', 'Slope_Change', 'Average_Volatility', 'close', 'volume']
X = df[features]
y = df['Label']

model_name = 'catboost'
optimizer = BayesianOptimizer(model_name=model_name)
optimizer.fit(X=X.to_numpy(), y=y.to_numpy(),n_splits=10)
# 獲取最佳參數和權重
best_params, weights = optimizer.get_best_params_and_weights()
print("最佳參數:", best_params)
print("參數權重:", weights)

# 獲取特徵重要性
feature_importances = optimizer.get_feature_importances()
print("特徵重要性:", feature_importances)

import json

besthyperparameter_name = f"best_hyperparams_{model_name}.json"
bestweights_name = f"best_weights_{model_name}.json"
# 保存最佳超参数
with open(besthyperparameter_name, "w") as f:
    json.dump(best_params, f)

# 保存权重
with open(bestweights_name, "w") as f:
    json.dump(weights, f)

#%%

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMRegressor
import joblib

# 读取最佳超参数
with open(besthyperparameter_name, "r") as f:
    best_params = json.load(f)

# 读取权重
with open(bestweights_name, "r") as f:
    weights = json.load(f)


# train model
# model_RF = RandomForestClassifier(**best_params, random_state=42)
model = CatBoostClassifier(**best_params, random_state=42)
# model_RF.fit(X, y)  # X_train, y_train 之前已经定义
model.fit(X, y)  # X_train, y_train

# 保存模型到本地文件
model_filename = f"model_{model_name}.joblib"
joblib.dump(model, model_filename)
joblib.dump(scaler, "scaler.pkl")  # 保存标准化器
print(f"模型已保存到 {model_filename}")


# %%
import matplotlib.pyplot as plt
feature_importances = model.feature_importances_
plt.bar(X.columns, feature_importances)
plt.xticks(rotation=45)
plt.show()

# %%
import pandas as pd
import talib
from sklearn.preprocessing import MinMaxScaler

# Load real-time data
real_time_file = r"C:\Users\wenyu\Desktop\trade\investment\python\API\test\TX00\TX00_1_20241224_20250117.xlsx"
real_time_data = pd.ExcelFile(real_time_file).parse('Sheet1')

# Ensure 'date' column is datetime and filter real-time data starting from 2025-01-12
real_time_data['date'] = pd.to_datetime(real_time_data['date'])
# real_time_data = real_time_data[real_time_data['date'] >= '2025-01-12']
real_time_data = real_time_data.sort_values(by='date').reset_index(drop=True)

# Feature Engineering: Apply the same logic as in training
def calculate_realtime_features(data, slope_window=5, ema_window=5, avg_vol_window=9, long_ema_window=20, scaler=None):
    """Calculate features for real-time data."""
    # Calculate Bollinger Bands
    data['Middle_Band'] = data['close'].rolling(window=20).mean()
    data['Std_Dev'] = data['close'].rolling(window=20).std()
    data['Lower_Band'] = data['Middle_Band'] - 2 * data['Std_Dev']
    
    # Calculate slopes
    data['Lower_Band_Slope'] = data['Lower_Band'].diff(slope_window) / data['Lower_Band'].shift(slope_window)
    data['Lower_Band_Slope'] = data['Lower_Band_Slope'].fillna(0)


    if len(data) >= slope_window:  # Apply scaler only when enough data exists
        data['Normalized_Slope'] = scaler.fit_transform(data[['Lower_Band_Slope']])
    else:
        data['Normalized_Slope'] = 0

    # Slope change
    data['Slope_Change'] = data['Normalized_Slope'] - data['Normalized_Slope'].shift(slope_window)
    data['Slope_Change'] = data['Slope_Change'].fillna(0)

    # EMA calculations
    data['EMA'] = talib.EMA(data['close'], timeperiod=ema_window)
    data['Long_EMA'] = talib.EMA(data['close'], timeperiod=long_ema_window)

    # Average volatility
    data['Average_Volatility'] = data['close'].diff().abs().rolling(window=avg_vol_window).mean()

    return data

# Apply feature engineering to real-time data
real_time_data = calculate_realtime_features(real_time_data, scaler=scaler)

# Prepare features for prediction (ensure columns match training features)
X_real_time = real_time_data[features].fillna(0)  # Replace NaNs with 0 for real-time data

# Predict probabilities
real_time_data['Event_Probability'] = model.predict_proba(X_real_time)[:, 1]

# Step 6: Calculate profit/loss points
real_time_data['Profit_Loss_Points'] = real_time_data['close'].shift(-5) - real_time_data['close']

# Filter and display high-probability events
probability_threshold = 0.010  # Threshold for event detection
high_probability_events = real_time_data[real_time_data['Event_Probability'] >= probability_threshold]

# Output high-probability events
print("Real-Time Event Predictions:")
# print(high_probability_events[['date', 'close', 'Event_Probability','Profit_Loss_Points']])
print(f"Total Profit/Loss Points: {high_probability_events['Profit_Loss_Points'].sum()}")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 確保 real_time_data 有需要的欄位
if 'Profit_Loss_Points' not in real_time_data.columns:
    real_time_data['Profit_Loss_Points'] = real_time_data['close'].shift(-5) - real_time_data['close']

def strategy_metrics(trade_pnl):
    """计算策略评估指标（修正 Sharpe Ratio 计算方式）"""
    total_pnl = trade_pnl.sum()  # 总盈亏
    win_trades = trade_pnl[trade_pnl > 0]
    loss_trades = trade_pnl[trade_pnl < 0]

    # 盈亏比 (Profit Factor)
    profit_factor = win_trades.sum() / abs(loss_trades.sum()) if abs(loss_trades.sum()) > 0 else np.nan

    # 胜率 (Win Rate)
    win_rate = len(win_trades) / len(trade_pnl) if len(trade_pnl) > 0 else np.nan

    # 期望盈亏 (Expectancy)
    expectancy = (win_trades.mean() * win_rate) - (loss_trades.mean() * (1 - win_rate))

    # **修正 Sharpe Ratio 计算**
    if trade_pnl.std() > 0:
        annualized_factor = np.sqrt(len(trade_pnl))  # 交易次数决定年化因子
        sharpe_ratio = (trade_pnl.mean() / trade_pnl.std()) * annualized_factor
    else:
        sharpe_ratio = np.nan  # 避免除零错误

    # 最大回撤 (Max Drawdown)
    cumulative_pnl = trade_pnl.cumsum()
    running_max = np.maximum.accumulate(cumulative_pnl)
    drawdown = running_max - cumulative_pnl
    max_drawdown = drawdown.max()

    return {
        'Total PnL': total_pnl,
        'Profit Factor': profit_factor,
        'Win Rate': win_rate,
        'Expectancy': expectancy,
        'Sharpe Ratio': sharpe_ratio,  # 现在的 Sharpe Ratio 不会异常高
        'Max Drawdown': max_drawdown,
    }

# 遍歷不同的機率閾值來找最佳值
# RF: best_threshold = 0.015, best_sharpe = 1.5
# CatBoost: best_threshold = 0.159, best_sharpe = 1.5

probability_thresholds = np.arange(0.17, 0.181, 0.001)
best_threshold = None
best_sharpe = -np.inf
results = []
cumulative_pnl_dict = {}  # 儲存不同閥值的累積盈虧

for threshold in probability_thresholds:
    selected_trades = real_time_data[real_time_data['Event_Probability'] >= threshold]['Profit_Loss_Points']
    
    if len(selected_trades) > 0:
        metrics = strategy_metrics(selected_trades)
        metrics['Threshold'] = threshold
        results.append(metrics)

        # 儲存累積收益
        cumulative_pnl_dict[threshold] = selected_trades.cumsum()

        # 優先選擇最大夏普比率 或 盈虧比 > 1.5 且回撤最小
        if metrics['Sharpe Ratio'] > best_sharpe and metrics['Profit Factor'] > 1.5:
            best_sharpe = metrics['Sharpe Ratio']

# 建立結果 DataFrame
results_df = pd.DataFrame(results)
print(results_df)

# 繪製不同閥值的累積收益曲線
plt.figure(figsize=(12, 6))
for threshold, pnl in cumulative_pnl_dict.items():
    plt.plot(pnl.index, pnl, label=f'Threshold {threshold:.3f}')

plt.xlabel("Time")
plt.ylabel("Cumulative Profit/Loss")
plt.title("Cumulative PnL for Different Probability Thresholds")
plt.legend()
plt.grid()
plt.show()

# 繪製最佳閥值的累積收益曲線
if best_threshold is not None:
    plt.figure(figsize=(12, 6))
    plt.plot(cumulative_pnl_dict[best_threshold].index, cumulative_pnl_dict[best_threshold], label=f'Best Threshold: {best_threshold:.3f}', color='red', linewidth=2)
    
    plt.xlabel("Time")
    plt.ylabel("Cumulative Profit/Loss")
    plt.title("Best Probability Threshold - Cumulative PnL")
    plt.legend()
    plt.grid()
    plt.show()

    print(f"\n最佳閥值: {best_threshold}")
else:
    print("沒有找到符合條件的最佳閥值。")



# %%
# %%
import pandas as pd
import time
import joblib
import talib
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# **1️⃣ 载入模型 & Scaler**
# model = joblib.load("model_random_forest.joblib")  # 载入已训练好的模型
# scaler = joblib.load("scaler.pkl")  # 载入标准化器

# **2️⃣ 读取 Real-Time Data**
real_time_file = r"C:\Users\wenyu\Desktop\trade\investment\python\API\test\TX00\TX00_1_20241224_20250117.xlsx"
real_time_data = pd.ExcelFile(real_time_file).parse('Sheet1')
real_time_data['date'] = pd.to_datetime(real_time_data['date'])
# real_time_data = real_time_data[real_time_data['date'] >= '2025-01-12']
real_time_data = real_time_data.sort_values(by='date').reset_index(drop=True)

# **3️⃣ 设定参数**
features = ['Lower_Band_Slope', 'Slope_Change', 'Average_Volatility', 'close', 'volume']
probability_threshold = 0.015  # 交易信号概率阈值
holding_period = pd.Timedelta(minutes=5)  # **持仓 5 分钟**
last_trade_time = None  # **记录上次交易时间**

# 初始化交易記錄
trade_records = []

# **4️⃣ 定义计算特征的函数**
def calculate_realtime_features(data, scaler, slope_window=5, ema_window=5, avg_vol_window=9, long_ema_window=20):
    """计算实时数据的特征"""
    data['Middle_Band'] = data['close'].rolling(window=20).mean()
    data['Std_Dev'] = data['close'].rolling(window=20).std()
    data['Lower_Band'] = data['Middle_Band'] - 2 * data['Std_Dev']

    # 计算 Lower_Band 斜率
    data['Lower_Band_Slope'] = data['Lower_Band'].diff(slope_window) / data['Lower_Band'].shift(slope_window)
    data['Lower_Band_Slope'] = data['Lower_Band_Slope'].fillna(0)

    # 斜率标准化
    if len(data) >= slope_window:
        data['Normalized_Slope'] = scaler.transform(data[['Lower_Band_Slope']])
    else:
        data['Normalized_Slope'] = 0

    # 计算斜率变化
    data['Slope_Change'] = data['Normalized_Slope'] - data['Normalized_Slope'].shift(slope_window)
    data['Slope_Change'] = data['Slope_Change'].fillna(0)

    # 计算短期 EMA 和 长期 EMA
    data['EMA'] = talib.EMA(data['close'], timeperiod=ema_window)
    data['Long_EMA'] = talib.EMA(data['close'], timeperiod=long_ema_window)

    # 计算平均波动率
    data['Average_Volatility'] = data['close'].diff().abs().rolling(window=avg_vol_window).mean()

    return data

def strategy_metrics(trade_pnl):
    """计算策略评估指标（修正 Sharpe Ratio 计算方式）"""
    total_pnl = trade_pnl.sum()  # 总盈亏
    win_trades = trade_pnl[trade_pnl > 0]
    loss_trades = trade_pnl[trade_pnl < 0]

    # 盈亏比 (Profit Factor)
    profit_factor = win_trades.sum() / abs(loss_trades.sum()) if abs(loss_trades.sum()) > 0 else np.nan

    # 胜率 (Win Rate)
    win_rate = len(win_trades) / len(trade_pnl) if len(trade_pnl) > 0 else np.nan

    # 期望盈亏 (Expectancy)
    expectancy = (win_trades.mean() * win_rate) - (loss_trades.mean() * (1 - win_rate))

    # **修正 Sharpe Ratio 计算**
    if trade_pnl.std() > 0:
        annualized_factor = np.sqrt(len(trade_pnl))  # 交易次数决定年化因子
        sharpe_ratio = (trade_pnl.mean() / trade_pnl.std()) * annualized_factor
    else:
        sharpe_ratio = np.nan  # 避免除零错误

    # 最大回撤 (Max Drawdown)
    cumulative_pnl = trade_pnl.cumsum()
    running_max = np.maximum.accumulate(cumulative_pnl)
    drawdown = running_max - cumulative_pnl
    max_drawdown = drawdown.max()

    return {
        'Total PnL': total_pnl,
        'Profit Factor': profit_factor,
        'Win Rate': win_rate,
        'Expectancy': expectancy,
        'Sharpe Ratio': sharpe_ratio,  # 现在的 Sharpe Ratio 不会异常高
        'Max Drawdown': max_drawdown,
    }



# **5️⃣ 模拟实时逐步数据**
for i in range(len(real_time_data)):
    # **模拟逐步获取数据**
    current_data = real_time_data.iloc[: i + 1].copy()

    # **计算特征**
    current_data = calculate_realtime_features(current_data, scaler)

    # **提取当前最新的数据点**
    latest_row = current_data.iloc[-1:][features].fillna(0)
    current_time = current_data.iloc[-1]['date']

    # **检查是否在持仓期内**
    if last_trade_time is not None and current_time < last_trade_time + holding_period:
        continue  # **5 分钟内不进行新交易**

    # **模型预测**
    event_probability = model.predict_proba(latest_row)[:, 1][0]

    # **检查是否达到交易信号**
    if event_probability >= probability_threshold:
        # **计算盈亏点数**
        if i + 5 < len(real_time_data):
            profit_loss_points = real_time_data.iloc[i + 5]['close'] - real_time_data.iloc[i]['close']
        else:
            profit_loss_points = 0  # 若未来数据不足，则不计算盈亏

        # **记录交易**
        trade_records.append({
            "date": current_time,
            "close": current_data.iloc[-1]['close'],
            "event_probability": event_probability,
            "profit_loss_points": profit_loss_points
        })

        # **更新最近交易时间**
        last_trade_time = current_time

        print(f"📌 交易信号 - 时间: {current_time}, 价格: {current_data.iloc[-1]['close']:.2f}, 事件概率: {event_probability:.4f}, 盈亏: {profit_loss_points:.2f}")


# **6️⃣ 计算累积盈亏并绘制**
trade_df = pd.DataFrame(trade_records)
trade_df['cumulative_pnl'] = trade_df['profit_loss_points'].cumsum()

# 计算策略指标
metrics = strategy_metrics(trade_df['profit_loss_points'])
print("策略评估指标:", metrics)

# **绘制 K 线图 + 累积盈亏曲线**
fig, ax1 = plt.subplots(figsize=(12, 6))

# 累积盈亏曲线
ax1.plot(trade_df['date'], trade_df['cumulative_pnl'], color='green', label="Cumulative PnL")
ax1.set_xlabel("Date")
ax1.set_ylabel("Cumulative PnL", color="green")
ax1.tick_params(axis="y", labelcolor="green")

# 交易信号点
ax1.scatter(trade_df['date'], trade_df['cumulative_pnl'], zorder=5)

# 标题和图例
plt.title("Real-Time Cumulative PnL with Trade Signals")
ax1.legend()

plt.grid()
plt.show()


# %%
# CatBoost: best_threshold = 0.159, best_sharpe = 
# 策略评估指标: {'Total PnL': 1172, 'Profit Factor': 1.35742604452577, 'Win Rate': 0.5219206680584552, 'Expectancy': 16.617612627553505, 'Sharpe Ratio': 1.9316306659907083, 'Max Drawdown': 243}
