import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter
from datetime import timedelta
import time
import itertools
import datetime
import time
import os

class TradingStrategy:
    """
    Base trading strategy class used to define rules for entering and exiting trades.
    """
    def __init__(self, name="Basic Strategy"):
        """
        Initialize trading strategy
        
        Parameters:
        -----------
        name : str, default="Basic Strategy"
            Strategy name
        """
        self.name = name
    
    def generate_signals(self, data):
        """
        Generate trading signals
        
        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame containing prediction probabilities
            
        Returns:
        --------
        pd.DataFrame
            DataFrame with added trading signals
        """
        # Base class doesn't implement specific logic, subclasses should override this method
        return data

class ProbabilityThresholdStrategy(TradingStrategy):
    """
    Trading strategy based on probability thresholds - Optimized version
    """
    def __init__(self, long_threshold=0.5, short_threshold=0.5, 
                 holding_period=3, exclude_times=None, name="Probability Threshold Strategy"):
        """
        Initialize probability threshold strategy
        
        Parameters:
        -----------
        long_threshold : float, default=0.5
            Probability threshold for long signals
        short_threshold : float, default=0.5
            Probability threshold for short signals
        holding_period : int, default=3
            Holding period (in data points)
        exclude_times : set, optional
            Set of time points to exclude
        name : str, default="Probability Threshold Strategy"
            Strategy name
        """
        super().__init__(name)
        self.long_threshold = long_threshold
        self.short_threshold = short_threshold
        self.holding_period = holding_period
        self.exclude_times = exclude_times or {
            "08:45", "08:46", "08:47", "08:48", "08:49",  # First 5 minutes after morning open
            "13:41", "13:42", "13:43", "13:44", "13:45",  # First 5 minutes after afternoon open
            "15:00", "15:01", "15:02", "15:03", "15:04",  # Last 5 minutes before day session close
            "03:55", "03:56", "03:57", "03:58", "03:59"   # Last 5 minutes before night session close
        }
    
    def generate_signals(self, data):
        """
        Generate trading signals based on probability thresholds - Optimized version
        
        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame containing prediction probabilities, must have 'Long_Probability' and 'Short_Probability' columns
                
        Returns:
        --------
        pd.DataFrame
            DataFrame with added trading signals
        """
        import numpy as np
        import pandas as pd
        
        # Use .copy(deep=False) to reduce unnecessary deep copying
        result = data.copy(deep=False)
        
        # Format time as 'HH:MM', only when needed
        if 'date' in result.columns:
            result['hour_minute'] = pd.to_datetime(result['date']).dt.strftime('%H:%M')
            result['Valid_Trading_Time'] = ~result['hour_minute'].isin(self.exclude_times)
        else:
            # Use vectorized operations to set valid trading times
            result['Valid_Trading_Time'] = True
        
        # Use vectorized operations to generate raw signals based on probability thresholds
        result['Long_Signal'] = (result['Long_Probability'] >= self.long_threshold) & result['Valid_Trading_Time']
        result['Short_Signal'] = (result['Short_Probability'] >= self.short_threshold) & result['Valid_Trading_Time']
        
        # Filter consecutive signals in short periods with a rolling window.
        # 只要在一個持倉窗口內有一個訊號，則整個窗口標記為 True
        result['Filtered_Long_Signal'] = result['Long_Signal'].rolling(window=self.holding_period, min_periods=1).max().astype(bool)
        result['Filtered_Short_Signal'] = result['Short_Signal'].rolling(window=self.holding_period, min_periods=1).max().astype(bool)

        # 使用 infer_objects() 确保正确推断对象类型，避免未来版本的下转型问题
        result = result.infer_objects(copy=False)

        # -------- 過濾連續訊號，只保留連續區段中的第一次出現（上升沿） -----------
        # 利用 shift 檢查前一時點是否為 False
        result['New_Long_Signal'] = (result['Filtered_Long_Signal'] & 
                                    (~result['Filtered_Long_Signal'].shift(1).fillna(False))).astype(bool)
        result['New_Short_Signal'] = (result['Filtered_Short_Signal'] & 
                                    (~result['Filtered_Short_Signal'].shift(1).fillna(False))).astype(bool)

        # ------------------------------------------------------------------------------
        
        # Pre-allocate space and initialize positions and trades
        n = len(result)
        result['Position'] = np.zeros(n)
        result['Trade'] = np.zeros(n)
        
        # Convert necessary columns into numpy arrays for faster loop operations
        position_array = result['Position'].values
        trade_array = result['Trade'].values
        # 使用新訊號作為交易入場依據
        long_signal = result['New_Long_Signal'].values
        short_signal = result['New_Short_Signal'].values
        valid_time = result['Valid_Trading_Time'].values
        
        if 'date' in result.columns:
            date_array = pd.to_datetime(result['date']).values
        else:
            date_array = np.arange(n)
        
        # Track active trades (flag), the trade's starting index, and current position value
        active_trade = False
        trade_start_idx = 0
        position_value = 0
        
        # Optimized trading logic: 遍歷每個時間點
        for i in range(n):
            # If there is an active trade, check if holding period has expired or trading time becomes invalid
            if active_trade:
                if i - trade_start_idx >= self.holding_period or not valid_time[i]:
                    # Expire the trade: clear active flag and reset position
                    active_trade = False
                    position_value = 0
                else:
                    # Continue holding the current position until expiration
                    position_array[i] = position_value
                    continue
            
            # If no active trade, check for a new entry signal based on the rising edge signals
            if long_signal[i]:
                position_array[i] = 1
                trade_array[i] = 1
                active_trade = True
                trade_start_idx = i
                position_value = 1
            elif short_signal[i]:
                position_array[i] = -1
                trade_array[i] = 1
                active_trade = True
                trade_start_idx = i
                position_value = -1
        
        # Assign computed numpy arrays back to DataFrame columns
        result['Position'] = position_array
        result['Trade'] = trade_array
        
        return result

# class KDStrategy(TradingStrategy):
#     """
#     Trading strategy based on KD indicator (Stochastic Oscillator)
#     """
#     def __init__(self, k_period=14, d_period=3, overbought=80, oversold=20,
#                  holding_period=3, exclude_times=None, name="KD Strategy"):
#         """
#         Initialize KD strategy
        
#         Parameters:
#         -----------
#         k_period : int, default=14
#             Period for %K calculation
#         d_period : int, default=3
#             Period for %D calculation
#         overbought : float, default=80
#             Overbought level
#         oversold : float, default=20
#             Oversold level
#         holding_period : int, default=3
#             Holding period (in data points)
#         exclude_times : set, optional
#             Set of time points to exclude
#         name : str, default="KD Strategy"
#             Strategy name
#         """
#         super().__init__(name)
#         self.k_period = k_period
#         self.d_period = d_period
#         self.overbought = overbought
#         self.oversold = oversold
#         self.holding_period = holding_period
#         self.exclude_times = exclude_times or {
#             "08:45", "08:46", "08:47", "08:48", "08:49",  # First 5 minutes after morning open
#             "13:41", "13:42", "13:43", "13:44", "13:45",  # First 5 minutes after afternoon open
#             "15:00", "15:01", "15:02", "15:03", "15:04",  # Last 5 minutes before day session close
#             "03:55", "03:56", "03:57", "03:58", "03:59"   # Last 5 minutes before night session close
#         }
    
#     def calculate_kd(self, data):
#         """
#         Calculate K% and D% values for stochastic oscillator
        
#         Parameters:
#         -----------
#         data : pd.DataFrame
#             DataFrame containing 'high', 'low', and 'close' columns
            
#         Returns:
#         --------
#         pandas.DataFrame
#             DataFrame containing 'K' and 'D' columns
#         """
#         # Calculate %K
#         data_len = len(data)
#         k_values = np.zeros(data_len)
        
#         # Calculate %K: (Current Close - Lowest Low) / (Highest High - Lowest Low) * 100
#         for i in range(self.k_period - 1, data_len):
#             period = data.iloc[i-self.k_period+1:i+1]
#             current_close = data.iloc[i]['close']
#             lowest_low = period['low'].min()
#             highest_high = period['high'].max()
            
#             if highest_high == lowest_low:
#                 k_values[i] = 50  # Avoid division by zero
#             else:
#                 k_values[i] = ((current_close - lowest_low) / (highest_high - lowest_low)) * 100
        
#         # Calculate %D (Moving Average of %K)
#         d_values = np.zeros(data_len)
#         for i in range(self.k_period + self.d_period - 2, data_len):
#             d_values[i] = k_values[i-self.d_period+1:i+1].mean()
        
#         return pd.DataFrame({
#             'K': k_values,
#             'D': d_values
#         })
    
#     def generate_signals(self, data):
#         """
#         Generate trading signals based on KD indicator
        
#         Parameters:
#         -----------
#         data : pd.DataFrame
#             DataFrame containing 'high', 'low', and 'close' columns
            
#         Returns:
#         --------
#         pd.DataFrame
#             DataFrame with added trading signals
#         """
#         # Use .copy() to avoid modifying the original DataFrame
#         result = data.copy()
        
#         # Initialize columns
#         result['Long_Signal'] = False
#         result['Short_Signal'] = False
#         result['Filtered_Long_Signal'] = False
#         result['Filtered_Short_Signal'] = False
#         result['Position'] = 0
#         result['Trade'] = 0
        
#         # Format time as 'HH:MM', only when needed
#         if 'date' in result.columns:
#             result['hour_minute'] = pd.to_datetime(result['date']).dt.strftime('%H:%M')
#             result['Valid_Trading_Time'] = ~result['hour_minute'].isin(self.exclude_times)
#         else:
#             # Use vectorized operations to set valid trading times
#             result['Valid_Trading_Time'] = True
        
#         # Calculate KD indicator
#         kd_values = self.calculate_kd(result)
#         result['K'] = kd_values['K']
#         result['D'] = kd_values['D']
        
#         # Generate signals based on KD crossovers
#         for i in range(self.k_period + self.d_period, len(result)):
#             if not result.iloc[i]['Valid_Trading_Time']:
#                 continue
                
#             # Get current and previous K and D values
#             k_curr = result.iloc[i]['K']
#             d_curr = result.iloc[i]['D']
#             k_prev = result.iloc[i-1]['K']
#             d_prev = result.iloc[i-1]['D']
            
#             # Buy signal: K crosses above D in oversold region
#             if k_prev < d_prev and k_curr > d_curr and k_curr < self.oversold:
#                 result.loc[result.index[i], 'Long_Signal'] = True
            
#             # Sell signal: K crosses below D in overbought region
#             elif k_prev > d_prev and k_curr < d_curr and k_curr > self.overbought:
#                 result.loc[result.index[i], 'Short_Signal'] = True
        
#         # Use vectorized operations to filter consecutive signals in short periods
#         result['Filtered_Long_Signal'] = result['Long_Signal'].rolling(window=self.holding_period, min_periods=1).max()
#         result['Filtered_Short_Signal'] = result['Short_Signal'].rolling(window=self.holding_period, min_periods=1).max()
        
#         # Process positions and trades
#         active_trade = False
#         trade_start_idx = 0
#         position_value = 0
        
#         for i in range(len(result)):
#             # If there's an active trade, check if the holding period has expired
#             if active_trade:
#                 # Holding period expired or encountered invalid trading time
#                 if i - trade_start_idx >= self.holding_period or not result.iloc[i]['Valid_Trading_Time']:
#                     active_trade = False
#                     position_value = 0
#                 else:
#                     # Continue holding, don't change position value
#                     result.iloc[i, result.columns.get_loc('Position')] = position_value
#                     continue
            
#             # Check for new signals
#             if result.iloc[i]['Filtered_Long_Signal']:
#                 result.iloc[i, result.columns.get_loc('Position')] = 1
#                 result.iloc[i, result.columns.get_loc('Trade')] = 1
#                 active_trade = True
#                 trade_start_idx = i
#                 position_value = 1
#             elif result.iloc[i]['Filtered_Short_Signal']:
#                 result.iloc[i, result.columns.get_loc('Position')] = -1
#                 result.iloc[i, result.columns.get_loc('Trade')] = 1
#                 active_trade = True
#                 trade_start_idx = i
#                 position_value = -1
        
#         return result

class Backtester:
    """
    Enhanced backtesting engine used to evaluate trading strategy performance.
    Calculates profitability for both long and short operations separately.
    """
    def __init__(self, profit_loss_window=3, max_profit_loss=50):
        """
        Initialize backtesting engine
        
        Parameters:
        -----------
        profit_loss_window : int, default=3
            Window size for calculating future profit/loss
        max_profit_loss : float, default=50
            Maximum profit/loss limit, used to filter extreme values
        """
        self.profit_loss_window = profit_loss_window
        self.max_profit_loss = max_profit_loss
        self.results = None
        self.long_trades = None  # For storing long trades
        self.short_trades = None  # For storing short trades
    
    def run(self, data, strategy):
        """
        Run backtest with specified strategy - Enhanced: separately calculate long and short P&L

        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame, must include 'close' column
        strategy : TradingStrategy
            Trading strategy object
            
        Returns:
        --------
        pd.DataFrame
            DataFrame containing backtest results
        """
        import time
        print("Starting backtest run...")
        data_len = len(data)
        # 每10%输出一次进度（此處只是預留，可根據需要添加進度顯示）
        milestone = max(1, data_len // 10)

        # 记录开始时间
        start_time = time.time()
        
        # 生成信號，這裡使用修改後的 generate_signals，該函數會產生 New_Long_Signal 和 New_Short_Signal
        data_with_signals = strategy.generate_signals(data)
        
        # Calculate future price changes using a fixed holding window (e.g. 3 periods)
        if 'Profit_Loss_Points' not in data_with_signals.columns:
            data_with_signals['Profit_Loss_Points'] = (
                data_with_signals['close'].shift(-self.profit_loss_window) - 
                data_with_signals['close']
            )
        
        # Initialize trade profit/loss fields
        data_with_signals['Trade_PnL'] = 0.0
        data_with_signals['Long_PnL'] = 0.0  # Long trade profit/loss
        data_with_signals['Short_PnL'] = 0.0  # Short trade profit/loss
        data_with_signals['Position'] = 0    # Initialize position field
        
        # Filter long and short signals separately based on the NEW (unique) signals, 
        # ensuring that for a continuous signal only the first occurrence is counted.
        long_signals = data_with_signals[
            (data_with_signals['New_Long_Signal'] == True) & 
            (data_with_signals['Valid_Trading_Time'] == True) &
            (data_with_signals['Profit_Loss_Points'].abs() <= self.max_profit_loss)
        ]
        
        short_signals = data_with_signals[
            (data_with_signals['New_Short_Signal'] == True) & 
            (data_with_signals['Valid_Trading_Time'] == True) &
            (data_with_signals['Profit_Loss_Points'].abs() <= self.max_profit_loss)
        ]
        
        # Set positions and calculate P&L for each valid entry signal.
        if len(long_signals) > 0:
            data_with_signals.loc[long_signals.index, 'Position'] = 1
            data_with_signals.loc[long_signals.index, 'Long_PnL'] = long_signals['Profit_Loss_Points']
            data_with_signals.loc[long_signals.index, 'Trade_PnL'] = long_signals['Profit_Loss_Points']
        
        if len(short_signals) > 0:
            # 如果同一時間點同時出現多空訊號，這裡以多頭為主，但可根據需要調整
            data_with_signals.loc[short_signals.index, 'Position'] = -1
            data_with_signals.loc[short_signals.index, 'Short_PnL'] = -short_signals['Profit_Loss_Points']
            data_with_signals.loc[short_signals.index, 'Trade_PnL'] = -short_signals['Profit_Loss_Points']
        
        # 處理信號重疊的情況：如果在同一時刻既有 New_Long_Signal 又有 New_Short_Signal，
        # 這裡先發出警告，並優先以多頭訊號計算 P&L。
        conflict_indices = data_with_signals[
            (data_with_signals['New_Long_Signal'] == True) & 
            (data_with_signals['New_Short_Signal'] == True)
        ].index
        if len(conflict_indices) > 0:
            print(f"Warning: Found {len(conflict_indices)} time points with both long and short signals.")
            data_with_signals.loc[conflict_indices, 'Position'] = 1
            data_with_signals.loc[conflict_indices, 'Long_PnL'] = data_with_signals.loc[conflict_indices, 'Profit_Loss_Points']
            data_with_signals.loc[conflict_indices, 'Short_PnL'] = -data_with_signals.loc[conflict_indices, 'Profit_Loss_Points']
            data_with_signals.loc[conflict_indices, 'Trade_PnL'] = data_with_signals.loc[conflict_indices, 'Long_PnL']
        
        # 設置交易標記，判斷 Position 是否為非 0
        data_with_signals['Trade'] = (data_with_signals['Position'] != 0).astype(int)
        
        # 累計 P&L 的計算
        data_with_signals['Cumulative_PnL'] = data_with_signals['Trade_PnL'].cumsum()
        data_with_signals['Cumulative_Long_PnL'] = data_with_signals['Long_PnL'].cumsum()
        data_with_signals['Cumulative_Short_PnL'] = data_with_signals['Short_PnL'].cumsum()
        
        # 存储结果到实例属性
        self.results = data_with_signals
        self.long_trades = data_with_signals[data_with_signals['Long_PnL'] != 0]['Long_PnL']
        self.short_trades = data_with_signals[data_with_signals['Short_PnL'] != 0]['Short_PnL']

        print(f"Signals generated in {time.time() - start_time:.2f} seconds, calculating P&L...")
        # 其他处理...
        print(f"Backtest completed in {time.time() - start_time:.2f} seconds")

        return data_with_signals

    
    def calculate_metrics(self, trade_type='all'):
        """
        Calculate strategy performance metrics
        
        Parameters:
        -----------
        trade_type : str, default='all'
            Type of trades to calculate, can be 'all', 'long', or 'short'
            
        Returns:
        --------
        dict
            Dictionary containing performance metrics
        """
        if self.results is None or len(self.results) == 0:
            return {}
        
        # Choose trade records based on trade type
        if trade_type == 'all':
            trades = self.results[self.results['Trade_PnL'] != 0]['Trade_PnL']
            # Also calculate separate metrics for long and short trades
            long_trades = self.long_trades
            short_trades = self.short_trades
        elif trade_type == 'long':
            trades = self.long_trades
            long_trades = self.long_trades
            short_trades = pd.Series()
        elif trade_type == 'short':
            trades = self.short_trades
            long_trades = pd.Series()
            short_trades = self.short_trades
        else:
            raise ValueError(f"Unsupported trade type: {trade_type}. Supported options are: 'all', 'long', 'short'")
        
        if len(trades) == 0:
            empty_metrics = {
                'Total PnL': 0,
                'Win Rate': 0,
                'Profit Factor': 0,
                'Expectancy': 0,
                'Sharpe Ratio': 0,
                'Max Drawdown': 0,
                'Total Trades': 0,
                'Trade Type': trade_type
            }
            
            # Add separate long and short metrics
            if trade_type == 'all':
                empty_metrics.update({
                    'Long Total PnL': 0,
                    'Long Win Rate': 0,
                    'Long Profit Factor': 0,
                    'Long Total Trades': 0,
                    'Short Total PnL': 0,
                    'Short Win Rate': 0,
                    'Short Profit Factor': 0,
                    'Short Total Trades': 0
                })
                
            return empty_metrics
        
        # Calculate basic metrics
        total_pnl = trades.sum()
        win_trades = trades[trades > 0]
        loss_trades = trades[trades < 0]
        
        # Calculate win rate
        win_rate = len(win_trades) / len(trades) if len(trades) > 0 else 0
        
        # Calculate profit factor
        profit_factor = win_trades.sum() / abs(loss_trades.sum()) if abs(loss_trades.sum()) > 0 else float('inf')
        
        # Calculate expectancy
        expectancy = trades.mean() if len(trades) > 0 else 0
        
        # Calculate Sharpe ratio
        if trades.std() > 0:
            annualized_factor = np.sqrt(len(trades))  # Simplified annualization factor
            sharpe_ratio = (trades.mean() / trades.std()) * annualized_factor
        else:
            sharpe_ratio = 0
        
        # Calculate maximum drawdown
        cumulative_pnl = trades.cumsum()
        rolling_max = cumulative_pnl.cummax()
        drawdown = rolling_max - cumulative_pnl
        max_drawdown = drawdown.max() if len(drawdown) > 0 else 0
        
        # Calculate additional metrics
        avg_win = win_trades.mean() if len(win_trades) > 0 else 0
        avg_loss = loss_trades.mean() if len(loss_trades) > 0 else 0
        largest_win = win_trades.max() if len(win_trades) > 0 else 0
        largest_loss = loss_trades.min() if len(loss_trades) > 0 else 0
        win_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')
        
        metrics = {
            'Total PnL': total_pnl,
            'Win Rate': win_rate,
            'Profit Factor': profit_factor,
            'Expectancy': expectancy,
            'Sharpe Ratio': sharpe_ratio,
            'Max Drawdown': max_drawdown,
            'Total Trades': len(trades),
            'Winning Trades': len(win_trades),
            'Losing Trades': len(loss_trades),
            'Average Win': avg_win,
            'Average Loss': avg_loss,
            'Largest Win': largest_win,
            'Largest Loss': largest_loss,
            'Win/Loss Ratio': win_loss_ratio,
            'Trade Type': trade_type
        }
        
        # If calculating all trades, add separate metrics for long and short trades
        if trade_type == 'all':
            # Long trade metrics
            if len(long_trades) > 0:
                long_win_trades = long_trades[long_trades > 0]
                long_loss_trades = long_trades[long_trades < 0]
                long_win_rate = len(long_win_trades) / len(long_trades) if len(long_trades) > 0 else 0
                long_profit_factor = long_win_trades.sum() / abs(long_loss_trades.sum()) if abs(long_loss_trades.sum()) > 0 else float('inf')
                long_total_pnl = long_trades.sum()
                long_avg_win = long_win_trades.mean() if len(long_win_trades) > 0 else 0
                long_avg_loss = long_loss_trades.mean() if len(long_loss_trades) > 0 else 0
                long_largest_win = long_win_trades.max() if len(long_win_trades) > 0 else 0
                long_largest_loss = long_loss_trades.min() if len(long_loss_trades) > 0 else 0
                
                metrics.update({
                    'Long Total PnL': long_total_pnl,
                    'Long Win Rate': long_win_rate,
                    'Long Profit Factor': long_profit_factor,
                    'Long Total Trades': len(long_trades),
                    'Long Winning Trades': len(long_win_trades),
                    'Long Losing Trades': len(long_loss_trades),
                    'Long Average Win': long_avg_win,
                    'Long Average Loss': long_avg_loss,
                    'Long Largest Win': long_largest_win,
                    'Long Largest Loss': long_largest_loss
                })
            else:
                metrics.update({
                    'Long Total PnL': 0,
                    'Long Win Rate': 0,
                    'Long Profit Factor': 0,
                    'Long Total Trades': 0,
                    'Long Winning Trades': 0,
                    'Long Losing Trades': 0,
                    'Long Average Win': 0,
                    'Long Average Loss': 0,
                    'Long Largest Win': 0,
                    'Long Largest Loss': 0
                })
                
            # Short trade metrics
            if len(short_trades) > 0:
                short_win_trades = short_trades[short_trades > 0]
                short_loss_trades = short_trades[short_trades < 0]
                short_win_rate = len(short_win_trades) / len(short_trades) if len(short_trades) > 0 else 0
                short_profit_factor = short_win_trades.sum() / abs(short_loss_trades.sum()) if abs(short_loss_trades.sum()) > 0 else float('inf')
                short_total_pnl = short_trades.sum()
                short_avg_win = short_win_trades.mean() if len(short_win_trades) > 0 else 0
                short_avg_loss = short_loss_trades.mean() if len(short_loss_trades) > 0 else 0
                short_largest_win = short_win_trades.max() if len(short_win_trades) > 0 else 0
                short_largest_loss = short_loss_trades.min() if len(short_loss_trades) > 0 else 0
                
                metrics.update({
                    'Short Total PnL': short_total_pnl,
                    'Short Win Rate': short_win_rate,
                    'Short Profit Factor': short_profit_factor,
                    'Short Total Trades': len(short_trades),
                    'Short Winning Trades': len(short_win_trades),
                    'Short Losing Trades': len(short_loss_trades),
                    'Short Average Win': short_avg_win,
                    'Short Average Loss': short_avg_loss,
                    'Short Largest Win': short_largest_win,
                    'Short Largest Loss': short_largest_loss
                })
            else:
                metrics.update({
                    'Short Total PnL': 0,
                    'Short Win Rate': 0,
                    'Short Profit Factor': 0,
                    'Short Total Trades': 0,
                    'Short Winning Trades': 0,
                    'Short Losing Trades': 0,
                    'Short Average Win': 0,
                    'Short Average Loss': 0,
                    'Short Largest Win': 0,
                    'Short Largest Loss': 0
                })
        
        return metrics
    
    def get_all_metrics(self):
        """
        Get performance metrics for all trade types
        
        Returns:
        --------
        dict
            Dictionary containing performance metrics for each trade type
        """
        return {
            'all': self.calculate_metrics('all'),
            'long': self.calculate_metrics('long'),
            'short': self.calculate_metrics('short')
        }
    
    def export_results_to_excel(self, filepath, include_trades=True):
        """
        Export backtest results to Excel file with enhanced error handling and flexibility
        
        Parameters:
        -----------
        filepath : str
            Path to save Excel file
        include_trades : bool, default=True
            Whether to include detailed trade information
        
        Raises:
        -------
        ValueError: If filepath is invalid or results are empty
        """
        # Validate input and results
        if not filepath:
            raise ValueError("Invalid file path provided")
        
        if self.results is None or len(self.results) == 0:
            raise ValueError("No backtest results available to export")
        
        try:
            # Create Excel writer object with enhanced options
            with pd.ExcelWriter(filepath, engine='xlsxwriter', mode='w') as writer:
                workbook = writer.book
                header_format = workbook.add_format({
                    'bold': True, 
                    'text_wrap': True,
                    'valign': 'top', 
                    'fg_color': '#D7E4BC',
                    'border': 1
                })
                
                # Export full results with improved formatting
                results_sheet = writer.book.add_worksheet('Full Results')
                self.results.to_excel(writer, sheet_name='Full Results', index=False)
                
                # Autosize columns for readability
                for idx, col in enumerate(self.results.columns):
                    max_len = max(
                        self.results[col].astype(str).map(len).max(),
                        len(str(col))
                    ) + 2
                    results_sheet.set_column(idx, idx, max_len)
                
                # Export metrics for all trade types
                all_metrics = self.get_all_metrics()
                for trade_type, metrics in all_metrics.items():
                    metrics_df = pd.DataFrame({
                        'Metric': list(metrics.keys()),
                        'Value': list(metrics.values())
                    })
                    metrics_df.to_excel(writer, sheet_name=f'{trade_type.capitalize()} Metrics', index=False)
                
                # Enhanced trade details export
                if include_trades:
                    # Export long trades with more details
                    if self.long_trades is not None and len(self.long_trades) > 0:
                        long_trades_df = pd.DataFrame(self.long_trades)
                        long_trades_df.columns = ['Profit/Loss']
                        long_trades_df.index.name = 'Trade Number'
                        long_trades_df.to_excel(writer, sheet_name='Long Trades')
                    
                    # Export short trades with more details
                    if self.short_trades is not None and len(self.short_trades) > 0:
                        short_trades_df = pd.DataFrame(self.short_trades)
                        short_trades_df.columns = ['Profit/Loss']
                        short_trades_df.index.name = 'Trade Number'
                        short_trades_df.to_excel(writer, sheet_name='Short Trades')
                
                # Comprehensive trade summary
                trade_summaries = []
                
                # Long trades summary
                if self.long_trades is not None and len(self.long_trades) > 0:
                    long_summary = self.long_trades.describe()
                    long_summary.name = 'Long Trades'
                    trade_summaries.append(long_summary)
                
                # Short trades summary
                if self.short_trades is not None and len(self.short_trades) > 0:
                    short_summary = self.short_trades.describe()
                    short_summary.name = 'Short Trades'
                    trade_summaries.append(short_summary)
                
                # Overall trades summary
                if self.results is not None:
                    trade_pnl = self.results[self.results['Trade_PnL'] != 0]['Trade_PnL']
                    if len(trade_pnl) > 0:
                        all_summary = trade_pnl.describe()
                        all_summary.name = 'All Trades'
                        trade_summaries.append(all_summary)
                
                # Combine and export summaries
                if trade_summaries:
                    summary_df = pd.concat(trade_summaries, axis=1)
                    summary_df.to_excel(writer, sheet_name='Trade Summary')
            
            print(f"Backtest results successfully exported to: {filepath}")
        
        except PermissionError:
            print(f"Error: Unable to write to {filepath}. The file may be open or you lack write permissions.")
        except Exception as e:
            print(f"An unexpected error occurred: {str(e)}")

    def export_results_to_txt(self, filepath, include_trades=True):
        """
        Export backtest results to a single .txt file, including:
        - Full Results DataFrame
        - Metrics for each trade type
        - Optional trade detail (long & short trades)
        - Summary (long/short/all trades)
        
        Parameters
        ----------
        filepath : str
            Path (including filename) where the .txt file will be saved
        include_trades : bool
            Whether to include detailed trades info in the output
        
        Raises
        ------
        ValueError : if no filepath or results are empty
        """
        # 1. 基本檢查
        if not filepath:
            raise ValueError("Invalid file path provided.")
        if self.results is None or len(self.results) == 0:
            raise ValueError("No backtest results available to export.")
        
        # 確保資料夾存在
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # 2. 開啟 txt 檔案進行寫入
        with open(filepath, "w", encoding="utf-8") as f:
            
            # ---- (A) Full Results ----
            f.write("=========== FULL RESULTS ===========\n")
            f.write(self.results.to_string(index=False))
            f.write("\n\n")
            
            # ---- (B) Metrics ----
            all_metrics = self.get_all_metrics()
            f.write("=========== METRICS ===========\n")
            for trade_type, metrics in all_metrics.items():
                f.write(f"--- {trade_type.capitalize()} Metrics ---\n")
                for k, v in metrics.items():
                    f.write(f"{k}: {v}\n")
                f.write("\n")
            
            # ---- (C) Trades (if needed) ----
            if include_trades:
                f.write("=========== TRADE DETAILS ===========\n")
                # Long trades
                if self.long_trades is not None and len(self.long_trades) > 0:
                    f.write(">>> LONG TRADES:\n")
                    f.write(self.long_trades.to_string())
                    f.write("\n\n")
                
                # Short trades
                if self.short_trades is not None and len(self.short_trades) > 0:
                    f.write(">>> SHORT TRADES:\n")
                    f.write(self.short_trades.to_string())
                    f.write("\n\n")
            
            # ---- (D) Summary ----
            f.write("=========== SUMMARY ===========\n")
            trade_summaries = []
            
            # Long summary
            if self.long_trades is not None and len(self.long_trades) > 0:
                long_summary = self.long_trades.describe()
                long_summary.name = 'Long Trades'
                trade_summaries.append(long_summary)
            
            # Short summary
            if self.short_trades is not None and len(self.short_trades) > 0:
                short_summary = self.short_trades.describe()
                short_summary.name = 'Short Trades'
                trade_summaries.append(short_summary)
            
            # All trades summary
            if 'Trade_PnL' in self.results.columns:
                nonzero_mask = (self.results['Trade_PnL'] != 0)
                trade_pnl = self.results.loc[nonzero_mask, 'Trade_PnL']
                if len(trade_pnl) > 0:
                    all_summary = trade_pnl.describe()
                    all_summary.name = 'All Trades'
                    trade_summaries.append(all_summary)
            
            # Combine summaries
            if trade_summaries:
                summary_df = pd.concat(trade_summaries, axis=1)
                f.write(summary_df.to_string())
                f.write("\n\n")
            else:
                f.write("No summary available (no trades?).\n\n")
        
        print(f"Backtest results successfully exported to TXT: {filepath}")

    def export_results_to_txt(self, filepath):
        """
        Export backtest results to a human-readable text file
        
        Parameters:
        -----------
        filepath : str
            Path to save the text file
        
        Returns:
        --------
        bool: True if export successful, False otherwise
        """
        try:
            # Ensure the directory exists
            import os
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            with open(filepath, 'w') as f:
                # Add timestamp to the report
                from datetime import datetime
                f.write(f"Backtest Results Report\n")
                f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*50 + "\n\n")
                
                # Full Results Summary
                if self.results is not None and not self.results.empty:
                    f.write("FULL RESULTS SUMMARY\n")
                    f.write("-"*25 + "\n")
                    f.write(f"Total Trades: {len(self.results)}\n")
                    
                    # Key statistics from results
                    if 'Trade_PnL' in self.results.columns:
                        total_pnl = self.results['Trade_PnL'].sum()
                        avg_pnl = self.results['Trade_PnL'].mean()
                        f.write(f"Total Profit/Loss: {total_pnl:.2f}\n")
                        f.write(f"Average Trade P/L: {avg_pnl:.2f}\n")
                    
                    f.write("\n")
                
                # Metrics
                try:
                    metrics = self.get_all_metrics()
                    if metrics:
                        f.write("PERFORMANCE METRICS\n")
                        f.write("-"*25 + "\n")
                        for trade_type, trade_metrics in metrics.items():
                            f.write(f"{trade_type.upper()} METRICS:\n")
                            for metric, value in trade_metrics.items():
                                f.write(f"{metric}: {value}\n")
                            f.write("\n")
                except Exception as metrics_error:
                    f.write(f"Error extracting metrics: {metrics_error}\n\n")
                
                # Long Trades
                if hasattr(self, 'long_trades') and self.long_trades is not None and len(self.long_trades) > 0:
                    f.write("LONG TRADES\n")
                    f.write("-"*25 + "\n")
                    long_trades_array = np.array(self.long_trades)
                    f.write(f"Total Long Trades: {len(long_trades_array)}\n")
                    f.write(f"Total Long Trade P/L: {long_trades_array.sum():.2f}\n")
                    f.write(f"Average Long Trade P/L: {long_trades_array.mean():.2f}\n")
                    f.write(f"Best Long Trade: {long_trades_array.max():.2f}\n")
                    f.write(f"Worst Long Trade: {long_trades_array.min():.2f}\n\n")
                
                # Short Trades
                if hasattr(self, 'short_trades') and self.short_trades is not None and len(self.short_trades) > 0:
                    f.write("SHORT TRADES\n")
                    f.write("-"*25 + "\n")
                    short_trades_array = np.array(self.short_trades)
                    f.write(f"Total Short Trades: {len(short_trades_array)}\n")
                    f.write(f"Total Short Trade P/L: {short_trades_array.sum():.2f}\n")
                    f.write(f"Average Short Trade P/L: {short_trades_array.mean():.2f}\n")
                    f.write(f"Best Short Trade: {short_trades_array.max():.2f}\n")
                    f.write(f"Worst Short Trade: {short_trades_array.min():.2f}\n\n")
                
                # Fallback if no data
                if (self.results is None or self.results.empty) and \
                (not hasattr(self, 'long_trades') or self.long_trades is None) and \
                (not hasattr(self, 'short_trades') or self.short_trades is None):
                    f.write("NO BACKTEST RESULTS AVAILABLE\n")
            
            print(f"Backtest results exported to text file: {filepath}")
            return True
        
        except PermissionError:
            print(f"Error: Unable to write to {filepath}. The file may be open or you lack write permissions.")
            return False
        except Exception as e:
            print(f"An unexpected error occurred: {str(e)}")
            return False
            
    def plot_results(self, title=None, figsize=(14, 10), show_separate_pnl=True, save_path=None, show_plot=True):
        """
        Plot backtest results charts with enhanced visuals
        
        Parameters:
        -----------
        title : str, optional
            Chart title
        figsize : tuple, default=(14, 10)
            Chart size
        show_separate_pnl : bool, default=True
            Whether to display separate long and short P&L
        save_path : str, optional
            Path to save the figure. If None, figure is not saved.
        show_plot : bool, default=True
            Whether to display the plot (set to False for automated runs)
        """
        if self.results is None or len(self.results) == 0:
            print("No backtest results to visualize")
            return
        
        # Set up aesthetics
        try:
            import seaborn as sns
            sns.set_style("whitegrid")
        except ImportError:
            pass
            
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.titleweight'] = 'bold'
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        
        # Decide how many subplots to create
        n_plots = 4 if show_separate_pnl else 3
        height_ratios = [3, 1, 1, 1] if show_separate_pnl else [3, 1, 1]
        
        fig, axes = plt.subplots(n_plots, 1, figsize=figsize, gridspec_kw={'height_ratios': height_ratios})
        fig.subplots_adjust(hspace=0.4)  # Add more space between subplots
        
        # Price trend and trade signals
        ax1 = axes[0]
        ax1.plot(self.results['date'], self.results['close'], label='Price', color='#1f77b4', linewidth=1.5)
        
        # Mark long and short signals
        long_entries = self.results[self.results['Long_PnL'] != 0]
        short_entries = self.results[self.results['Short_PnL'] != 0]
        
        if len(long_entries) > 0:
            ax1.scatter(long_entries['date'], long_entries['close'], 
                       marker='^', color='green', s=70, label='Long', alpha=0.7, 
                       edgecolors='darkgreen', linewidths=1)
        
        if len(short_entries) > 0:
            ax1.scatter(short_entries['date'], short_entries['close'], 
                       marker='v', color='red', s=70, label='Short', alpha=0.7,
                       edgecolors='darkred', linewidths=1)
        
        ax1.set_title(title or 'Backtest Results', fontweight='bold', fontsize=16)
        ax1.set_ylabel('Price', fontweight='bold')
        ax1.set_xlabel('')  # Remove x-label from top subplot
        
        # Format date on x-axis
        try:
            date_format = DateFormatter('%Y-%m-%d')
            ax1.xaxis.set_major_formatter(date_format)
        except Exception:
            pass
        
        # Create legend with shadow for better visibility
        ax1.legend(loc='upper left', frameon=True, framealpha=0.9, shadow=True)
        ax1.grid(True, alpha=0.3)
        
        # Cumulative P&L
        ax2 = axes[1]
        ax2.plot(self.results['date'], self.results['Cumulative_PnL'], 
                color='#9467bd', label='Cumulative PnL', linewidth=2)
        ax2.fill_between(self.results['date'], 0, self.results['Cumulative_PnL'], 
                         color='#9467bd', alpha=0.2)
        ax2.set_ylabel('Total PnL', fontweight='bold')
        ax2.set_xlabel('')  # Remove x-label from middle subplot
        ax2.xaxis.set_major_formatter(date_format)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper left', frameon=True, framealpha=0.9)
        
        # Add horizontal line at zero
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Trade P&L
        ax3 = axes[2]
        trade_pnl = self.results[self.results['Trade_PnL'] != 0]
        
        if len(trade_pnl) > 0:
            colors = ['#2ca02c' if pnl > 0 else '#d62728' for pnl in trade_pnl['Trade_PnL']]
            ax3.bar(trade_pnl['date'], trade_pnl['Trade_PnL'], color=colors, label='Trade PnL', alpha=0.7)
            ax3.set_ylabel('Trade PnL', fontweight='bold')
            ax3.set_xlabel('')  # Remove x-label if not the last subplot
            ax3.xaxis.set_major_formatter(date_format)
            ax3.grid(True, alpha=0.3)
            
            # Add horizontal line at zero
            ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # If needed, plot separate long and short P&L
        if show_separate_pnl:
            ax4 = axes[3]
            ax4.plot(self.results['date'], self.results['Cumulative_Long_PnL'], 
                    color='#2ca02c', label='Long PnL', linewidth=2)
            ax4.plot(self.results['date'], self.results['Cumulative_Short_PnL'], 
                    color='#d62728', label='Short PnL', linewidth=2)
            ax4.set_ylabel('Classified PnL', fontweight='bold')
            ax4.set_xlabel('Date', fontweight='bold')  # Add x-label to bottom subplot
            ax4.xaxis.set_major_formatter(date_format)
            ax4.grid(True, alpha=0.3)
            ax4.legend(loc='upper left', frameon=True, framealpha=0.9)
            
            # Add horizontal line at zero
            ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        else:
            # Add x-label to the last subplot
            axes[-1].set_xlabel('Date', fontweight='bold')
        
        # Rotate x-axis labels
        for ax in axes:
            plt.setp(ax.get_xticklabels(), rotation=45, ha="right")
        
        # Calculate metrics for each type of trade
        all_metrics = self.get_all_metrics()
        
        # Add performance metrics at the bottom of the chart - Include both All, Long, and Short metrics
        combined_metrics = all_metrics['all']
        
        # Create detailed metrics text with consistent formatting
        all_text = (
            f"Total PnL: {combined_metrics['Total PnL']:.2f} | "
            f"Win Rate: {combined_metrics['Win Rate']:.2%} | "
            f"Profit Factor: {combined_metrics['Profit Factor']:.2f} | "
            f"Sharpe Ratio: {combined_metrics['Sharpe Ratio']:.2f} | "
            f"Max Drawdown: {combined_metrics['Max Drawdown']:.2f} | "
            f"Total Trades: {combined_metrics['Total Trades']}"
        )
        
        long_text = (
            f"Long PnL: {combined_metrics.get('Long Total PnL', 0):.2f} | "
            f"Long Win Rate: {combined_metrics.get('Long Win Rate', 0):.2%} | "
            f"Long Profit Factor: {combined_metrics.get('Long Profit Factor', 0):.2f} | "
            f"Long Trades: {combined_metrics.get('Long Total Trades', 0)}"
        )
        
        short_text = (
            f"Short PnL: {combined_metrics.get('Short Total PnL', 0):.2f} | "
            f"Short Win Rate: {combined_metrics.get('Short Win Rate', 0):.2%} | "
            f"Short Profit Factor: {combined_metrics.get('Short Profit Factor', 0):.2f} | "
            f"Short Trades: {combined_metrics.get('Short Total Trades', 0)}"
        )
        
        combined_text = all_text + "\n" + long_text + "\n" + short_text
        
        plt.figtext(0.5, 0.01, combined_text, ha='center', fontsize=10, 
                   bbox={'facecolor': '#ff9d4d', 'alpha': 0.2, 'pad': 10, 'boxstyle': 'round,pad=0.5'})
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        # Save the figure if save_path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        # Show the plot if requested
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
        
        return fig
        
    def plot_trade_comparison(self, figsize=(14, 8), save_path=None, show_plot=True):
        """
        Plot enhanced comparison chart of long and short trades
        
        Parameters:
        -----------
        figsize : tuple, default=(14, 8)
            Chart size
        save_path : str, optional
            Path to save the figure. If None, figure is not saved.
        show_plot : bool, default=True
            Whether to display the plot (set to False for batch processing)
        """
        if self.results is None or len(self.results) == 0:
            print("No backtest results to visualize")
            return
        
        # Set up aesthetics
        try:
            import seaborn as sns
            sns.set_style("whitegrid")
        except ImportError:
            pass
            
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.titleweight'] = 'bold'
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        
        # Get detailed metrics including long and short breakdowns
        metrics = self.calculate_metrics('all')
        
        # Create figure with 2 rows, 2 columns for detailed comparison
        fig, axes = plt.subplots(2, 2, figsize=(figsize[0], figsize[1]*1.5))
        fig.subplots_adjust(hspace=0.3, wspace=0.3)  # Add more space between subplots
        
        # 1. Trade Count Comparison (top left)
        trade_counts = [
            metrics.get('Long Total Trades', 0), 
            metrics.get('Short Total Trades', 0)
        ]
        
        ax1 = axes[0, 0]
        bars1 = ax1.bar(['Long Trades', 'Short Trades'], trade_counts, color=['#2ca02c', '#d62728'])
        ax1.set_title('Trade Count Comparison', fontweight='bold')
        ax1.set_ylabel('Number of Trades', fontweight='bold')
        
        # Label exact values above bars
        for i, count in enumerate(trade_counts):
            ax1.text(i, count + 0.5, str(count), ha='center', fontweight='bold')
        
        # Add percentage labels
        total_trades = sum(trade_counts)
        if total_trades > 0:
            for i, count in enumerate(trade_counts):
                percentage = (count / total_trades) * 100
                ax1.text(i, count/2, f'{percentage:.1f}%', ha='center', color='white', fontweight='bold')
        
        # 2. P&L Comparison (top right)
        pnl_values = [
            metrics.get('Long Total PnL', 0), 
            metrics.get('Short Total PnL', 0)
        ]
        
        ax2 = axes[0, 1]
        bars2 = ax2.bar(['Long PnL', 'Short PnL'], pnl_values, 
                color=['#2ca02c' if pnl >= 0 else '#ff9999' for pnl in pnl_values])
        ax2.set_title('P&L Comparison', fontweight='bold')
        ax2.set_ylabel('P&L Points', fontweight='bold')
        
        # Add horizontal line at zero
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Label exact values above bars
        for i, pnl in enumerate(pnl_values):
            ax2.text(i, pnl + (5 if pnl >= 0 else -5), f"{pnl:.2f}", ha='center', fontweight='bold')
        
        # 3. Win Rate Comparison (bottom left)
        win_rates = [
            metrics.get('Long Win Rate', 0) * 100, 
            metrics.get('Short Win Rate', 0) * 100
        ]
        
        ax3 = axes[1, 0]
        bars3 = ax3.bar(['Long Win Rate', 'Short Win Rate'], win_rates, color=['#98df8a', '#ff9999'])
        ax3.set_title('Win Rate Comparison', fontweight='bold')
        ax3.set_ylabel('Win Rate (%)', fontweight='bold')
        ax3.set_ylim(0, 100)  # Set y-axis from 0 to 100%
        
        # Add 50% reference line
        ax3.axhline(y=50, color='gray', linestyle='--', alpha=0.7)
        ax3.text(1.1, 50, '50%', ha='left', va='center', color='gray', fontsize=10)
        
        # Label exact values above bars
        for i, rate in enumerate(win_rates):
            ax3.text(i, rate + 2, f"{rate:.1f}%", ha='center', fontweight='bold')
        
        # 4. Profit Factor Comparison (bottom right)
        profit_factors = [
            min(metrics.get('Long Profit Factor', 0), 10),  # Cap at 10 for better visualization
            min(metrics.get('Short Profit Factor', 0), 10)  # Cap at 10 for better visualization
        ]
        
        ax4 = axes[1, 1]
        bars4 = ax4.bar(['Long P/F', 'Short P/F'], profit_factors, color=['#2ca02c', '#d62728'])
        ax4.set_title('Profit Factor Comparison', fontweight='bold')
        ax4.set_ylabel('Profit Factor', fontweight='bold')
        
        # Add reference line at profit factor = 1
        ax4.axhline(y=1, color='gray', linestyle='--', alpha=0.7)
        ax4.text(1.1, 1, '1.0', ha='left', va='center', color='gray', fontsize=10)
        
        # Label exact values above bars
        for i, pf in enumerate(profit_factors):
            actual_pf = metrics.get('Long Profit Factor' if i == 0 else 'Short Profit Factor', 0)
            display_pf = min(actual_pf, 9.99)  # For display purposes
            ax4.text(i, pf + 0.2, f"{display_pf:.2f}" + (" (∞)" if actual_pf > 10 else ""), 
                    ha='center', fontweight='bold')
        
        # Add title to the figure
        fig.suptitle('Long vs Short Trade Performance Analysis', fontsize=16, fontweight='bold', y=0.98)
        
        # Add total P&L information with enhanced styling
        total_pnl = metrics.get('Total PnL', 0)
        combined_win_rate = metrics.get('Win Rate', 0) * 100
        
        # Create a more visually appealing summary box
        summary_text = (
            f"Total P&L: {total_pnl:.2f} points | Overall Win Rate: {combined_win_rate:.1f}% | " + 
            f"Total Trades: {metrics.get('Total Trades', 0)} | Profit Factor: {metrics.get('Profit Factor', 0):.2f}"
        )
        
        plt.figtext(0.5, 0.01, summary_text, ha='center', fontsize=12, fontweight='bold',
                   bbox={'facecolor': '#ff9d4d', 'alpha': 0.2, 'pad': 10, 'boxstyle': 'round,pad=0.5'})
        
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Leave space at the bottom for total P&L message
        
        # Save the figure if save_path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to: {save_path}")
        
        # Show the plot if requested
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
            
        return fig
    
    def plot_advanced_analysis(self, save_path=None, show_plot=True):
        """
        Create advanced analysis charts including trade distribution, drawdown, and trade outcome analysis
        
        Parameters:
        -----------
        save_path : str, optional
            Path to save the figure. If None, figure is not saved.
        show_plot : bool, default=True
            Whether to display the plot (set to False for batch processing)
        """
        if self.results is None or len(self.results) == 0:
            print("No backtest results to visualize")
            return
            
        # Set up aesthetics
        try:
            import seaborn as sns
            sns.set_style("whitegrid")
        except ImportError:
            pass
            
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.titleweight'] = 'bold'
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        
        # Create a figure with 2 rows and 2 columns
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        
        # 1. Trade Distribution Analysis (top left)
        ax1 = axes[0, 0]
        
        # Get trade data
        long_trades = self.results[self.results['Long_PnL'] != 0]['Long_PnL']
        short_trades = self.results[self.results['Short_PnL'] != 0]['Short_PnL']
        
        # Create bins for histogram
        bin_min = min(min(long_trades) if len(long_trades) > 0 else 0, min(short_trades) if len(short_trades) > 0 else 0)
        bin_max = max(max(long_trades) if len(long_trades) > 0 else 0, max(short_trades) if len(short_trades) > 0 else 0)
        bins = np.linspace(bin_min, bin_max, 20)
        
        # Plot histograms
        if len(long_trades) > 0:
            ax1.hist(long_trades, bins=bins, alpha=0.7, color='#2ca02c', label='Long Trades')
        if len(short_trades) > 0:
            ax1.hist(short_trades, bins=bins, alpha=0.7, color='#d62728', label='Short Trades')
            
        ax1.set_title('Trade P&L Distribution', fontweight='bold')
        ax1.set_xlabel('P&L Points', fontweight='bold')
        ax1.set_ylabel('Frequency', fontweight='bold')
        ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # 2. Drawdown Analysis (top right)
        ax2 = axes[0, 1]
        
        # Calculate drawdown
        if 'Cumulative_PnL' in self.results.columns:
            cumulative_pnl = self.results['Cumulative_PnL']
            peak = cumulative_pnl.expanding(min_periods=1).max()
            drawdown = (peak - cumulative_pnl)
            
            # Plot drawdown
            ax2.fill_between(self.results['date'], 0, drawdown, color='#ff7f0e', alpha=0.5)
            ax2.plot(self.results['date'], drawdown, color='#ff7f0e', linewidth=1.5)
            
            # Mark maximum drawdown
            max_dd_idx = drawdown.idxmax()
            max_dd = drawdown[max_dd_idx]
            max_dd_date = self.results.iloc[max_dd_idx]['date']
            
            ax2.scatter([max_dd_date], [max_dd], color='red', s=100, zorder=5)
            ax2.annotate(f'Max DD: {max_dd:.2f}', 
                         xy=(max_dd_date, max_dd),
                         xytext=(max_dd_date, max_dd * 0.7),
                         arrowprops=dict(facecolor='black', shrink=0.05, width=1.5),
                         fontsize=10, fontweight='bold')
            
        ax2.set_title('Drawdown Analysis', fontweight='bold')
        ax2.set_xlabel('Date', fontweight='bold')
        ax2.set_ylabel('Drawdown (Points)', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Format date on x-axis
        try:
            date_format = DateFormatter('%Y-%m-%d')
            ax2.xaxis.set_major_formatter(date_format)
        except Exception:
            pass
            
        plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
        
        # 3. Monthly Returns Heatmap (bottom left)
        ax3 = axes[1, 0]
        
        # Create monthly returns
        if 'date' in self.results.columns and 'Trade_PnL' in self.results.columns:
            # Add month and year columns
            self.results['year'] = pd.to_datetime(self.results['date']).dt.year
            self.results['month'] = pd.to_datetime(self.results['date']).dt.month
            
            # Group by year and month
            monthly_returns = self.results.groupby(['year', 'month'])['Trade_PnL'].sum().unstack()
            
            # Plot heatmap
            if not monthly_returns.empty:
                try:
                    import seaborn as sns
                    sns.heatmap(monthly_returns, annot=True, fmt=".1f", cmap='RdYlGn', center=0,
                                linewidths=.5, ax=ax3, cbar_kws={'label': 'P&L Points'})
                except (ImportError, ValueError):
                    # Fallback if seaborn not available or heatmap fails
                    ax3.imshow(monthly_returns, cmap='RdYlGn', aspect='auto')
                    ax3.set_title('Monthly Returns - Seaborn required for better visualization')
                
        ax3.set_title('Monthly Returns Heatmap', fontweight='bold')
        ax3.set_xlabel('Month', fontweight='bold')
        ax3.set_ylabel('Year', fontweight='bold')
        
        # 4. Trade Outcome Analysis (bottom right)
        ax4 = axes[1, 1]
        
        # Get win/loss counts for long and short trades
        metrics = self.calculate_metrics('all')
        
        # Data preparation
        categories = ['Long', 'Short']
        win_data = [metrics.get('Long Winning Trades', 0), metrics.get('Short Winning Trades', 0)]
        loss_data = [metrics.get('Long Losing Trades', 0), metrics.get('Short Losing Trades', 0)]
        
        # Width of the bars
        width = 0.35
        
        # Position of bars on x-axis
        x = np.arange(len(categories))
        
        # Create bars
        win_bars = ax4.bar(x - width/2, win_data, width, label='Winning Trades', color='#2ca02c')
        loss_bars = ax4.bar(x + width/2, loss_data, width, label='Losing Trades', color='#d62728')
        
        # Add values on top of bars
        for bar in win_bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2, height + 0.1,
                    f'{int(height)}', ha='center', fontweight='bold')
            
        for bar in loss_bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2, height + 0.1,
                    f'{int(height)}', ha='center', fontweight='bold')
        
        # Customize plot
        ax4.set_title('Win/Loss Trade Analysis', fontweight='bold')
        ax4.set_ylabel('Number of Trades', fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(categories)
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Add title to the figure
        fig.suptitle('Advanced Trading Performance Analytics', fontsize=18, fontweight='bold', y=0.98)
        
        # Save the figure if save_path is provided
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Advanced analysis chart saved to: {save_path}")
        
        # Show the plot if requested
        if show_plot:
            plt.show()
        else:
            plt.close(fig)
            
        return fig
    
    def optimize_strategy(self, data, strategy_class, param_grid, metric='sharpe_ratio', 
                         trade_type='all', verbose=True):
        """
        Optimize strategy parameters - Optimized version
        
        Parameters:
        -----------
        data : pd.DataFrame
            DataFrame, must include 'close' column
        strategy_class : class
            Strategy class
        param_grid : dict
            Parameter grid, e.g., {'long_threshold': [0.1, 0.2], 'short_threshold': [0.1, 0.2]}
        metric : str, default='sharpe_ratio'
            Optimization metric, can be 'sharpe_ratio', 'total_pnl', 'profit_factor', 'win_rate'
        trade_type : str, default='all'
            Type of trades to optimize, can be 'all', 'long', 'short'
        verbose : bool, default=True
            Whether to display progress information
            
        Returns:
        --------
        tuple
            (best_params, best_performance, all_results)
        """
        all_results = []
        best_performance = float('-inf')
        best_params = None
        
        # If single parameter optimization, convert to list
        for param_name, param_values in param_grid.items():
            if not isinstance(param_values, (list, tuple, np.ndarray)):
                param_grid[param_name] = [param_values]
        
        # Generate all parameter combinations
        param_names = list(param_grid.keys())
        param_values = [param_grid[name] for name in param_names]
        param_combinations = list(itertools.product(*param_values))
        total_combinations = len(param_combinations)
        
        if verbose:
            print(f"Starting test of {total_combinations} parameter combinations (optimizing for trade type: {trade_type})...")
            start_time = time.time()
        
        # Pre-calculate shared data to avoid repetitive calculations
        # For example, pre-calculate future price changes
        if 'close' in data.columns and 'Profit_Loss_Points' not in data.columns:
            data['Profit_Loss_Points'] = data['close'].shift(-self.profit_loss_window) - data['close']
        
        for idx, params in enumerate(param_combinations):
            params_dict = {name: value for name, value in zip(param_names, params)}
            
            # Create and run strategy
            strategy = strategy_class(**params_dict)
            backtest_results = self.run(data, strategy)
            
            # Calculate performance metrics (based on specified trade type)
            metrics = self.calculate_metrics(trade_type)
            
            # Choose optimization target
            if metric == 'sharpe_ratio':
                performance = metrics.get('Sharpe Ratio', float('-inf'))
            elif metric == 'total_pnl':
                performance = metrics.get('Total PnL', 0)
            elif metric == 'profit_factor':
                performance = metrics.get('Profit Factor', 0)
            elif metric == 'win_rate':
                performance = metrics.get('Win Rate', 0)
            else:
                performance = metrics.get('Sharpe Ratio', float('-inf'))
            
            # Record results
            result = params_dict.copy()
            result.update(metrics)
            result['performance'] = performance
            all_results.append(result)
            
            # Check if this is the best result
            if performance > best_performance:
                best_performance = performance
                best_params = params_dict
            
            # Display progress
            if verbose and (idx % max(1, total_combinations // 10) == 0 or idx == total_combinations - 1):
                progress = (idx + 1) / total_combinations * 100
                elapsed = time.time() - start_time
                est_total = elapsed / (idx + 1) * total_combinations
                remaining = est_total - elapsed
                print(f"Progress: {idx+1}/{total_combinations} ({progress:.1f}%) | "
                    f"Elapsed time: {elapsed:.2f}s | "
                    f"Est. remaining: {remaining:.2f}s")
        
        # Convert results to DataFrame
        results_df = pd.DataFrame(all_results)
        
        if verbose:
            total_time = time.time() - start_time
            print(f"Parameter optimization completed, total time: {total_time:.2f}s")
            print(f"Best parameters: {best_params}")
            print(f"Best performance ({metric}, {trade_type}): {best_performance:.4f}")
        
        return best_params, best_performance, results_df


def filter_and_compare_strategies(data, threshold_pairs=None, filter_criteria=None, holding_period=3, save_dir=None, show_plot=True):
    """
    Filter and compare strategies based on custom criteria
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing trading data
    threshold_pairs : list of tuples, optional
        List of (long_threshold, short_threshold) tuples to test
        If None, will run a grid search to generate pairs
    filter_criteria : dict, optional
        Dictionary of filtering criteria with keys:
        - min_total_pnl: Minimum total PnL (default: 0)
        - min_sharpe: Minimum Sharpe ratio (default: 1.0)
        - min_pnl_drawdown_ratio: Minimum PnL/Max Drawdown ratio (default: 3.0)
        - max_trades: Maximum number of trades (default: 2000)
        - min_profit_factor: Minimum profit factor (default: 1.0)
    holding_period : int, default=3
        Holding period for trades
    save_dir : str, optional
        Directory to save figures. If None, figures are not saved.
    show_plot : bool, default=True
        Whether to display plots (set to False for batch processing)
        
    Returns:
    --------
    tuple
        (filtered_results, filtered_metrics_df, cumulative_returns_df)
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    import time
    
    # Set default filtering criteria if not provided
    if filter_criteria is None:
        filter_criteria = {
            'min_total_pnl': 0,
            'min_sharpe': 1.0,
            'min_pnl_drawdown_ratio': 3.0,
            'max_trades': 2000,
            'min_profit_factor': 1.0
        }
    else:
        # Set defaults for any missing criteria
        default_criteria = {
            'min_total_pnl': 0,
            'min_sharpe': 1.0,
            'min_pnl_drawdown_ratio': 3.0,
            'max_trades': 2000,
            'min_profit_factor': 1.0
        }
        for key, default_value in default_criteria.items():
            if key not in filter_criteria:
                filter_criteria[key] = default_value
    
    print(f"Testing with filter criteria: {filter_criteria}")
    
    # If no threshold pairs provided, generate a grid
    if threshold_pairs is None:
        long_thresholds = np.linspace(0.0001, 0.005, 10)
        short_thresholds = np.linspace(0.0001, 0.005, 10)
        
        threshold_pairs = []
        for l in long_thresholds:
            for s in short_thresholds:
                threshold_pairs.append((l, s))
        
        print(f"Generated {len(threshold_pairs)} threshold combinations to test")
    
    # Test all threshold pairs with parallel processing
    from joblib import Parallel, delayed
    import multiprocessing
    
    def evaluate_strategy(long_threshold, short_threshold):
        """Helper function to evaluate a single strategy (for parallel processing)"""
        # Create and run strategy
        strategy = ProbabilityThresholdStrategy(
            long_threshold=long_threshold,
            short_threshold=short_threshold,
            holding_period=holding_period
        )
        
        backtester = Backtester(profit_loss_window=holding_period)
        backtester.run(data.copy(), strategy)  # Use copy to avoid data sharing issues
        metrics = backtester.calculate_metrics()
        
        # Add threshold values to metrics
        metrics['Long Threshold'] = long_threshold
        metrics['Short Threshold'] = short_threshold
        
        # Calculate PnL/MaxDD ratio if not already present
        if 'PnL/MaxDD' not in metrics and metrics['Max Drawdown'] > 0:
            metrics['PnL/MaxDD'] = metrics['Total PnL'] / metrics['Max Drawdown']
        elif 'PnL/MaxDD' not in metrics:
            metrics['PnL/MaxDD'] = float('inf') if metrics['Total PnL'] > 0 else 0
        
        # Store cumulative PnL for later plotting
        if 'Cumulative_Return' in backtester.results:
            cumulative_return = backtester.results['Cumulative_Return']
        else:
            # Calculate if not already present
            initial_capital = 100000
            cumulative_return = (backtester.results['Cumulative_PnL'] / initial_capital) * 100
        
        return {
            'metrics': metrics,
            'backtester': backtester,
            'key': (long_threshold, short_threshold),
            'cumulative_return': cumulative_return
        }
    
    # Determine number of parallel jobs (use all available CPUs, but cap at number of threshold pairs)
    n_jobs = min(multiprocessing.cpu_count(), len(threshold_pairs))
    
    print(f"Running {len(threshold_pairs)} strategy evaluations in parallel using {n_jobs} cores...")
    start_time = time.time()
    
    # Run parallel evaluation
    results_list = Parallel(n_jobs=n_jobs, verbose=1)(
        delayed(evaluate_strategy)(long_threshold, short_threshold)
        for long_threshold, short_threshold in threshold_pairs
    )
    
    elapsed = time.time() - start_time
    print(f"Completed {len(threshold_pairs)} evaluations in {elapsed:.1f}s")
    
    # Unpack results
    all_results = []
    all_backtesters = {}
    cumulative_pnl_dict = {}
    
    for result in results_list:
        all_results.append(result['metrics'])
        all_backtesters[result['key']] = result['backtester']
        cumulative_pnl_dict[result['key']] = result['cumulative_return']
    
    # Convert to DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Apply filtering criteria
    filtered_results = results_df[
        (results_df['Total PnL'] > filter_criteria['min_total_pnl']) &
        (results_df['Sharpe Ratio'] > filter_criteria['min_sharpe']) &
        (results_df['PnL/MaxDD'] > filter_criteria['min_pnl_drawdown_ratio']) &
        (results_df['Total Trades'] < filter_criteria['max_trades']) &
        (results_df['Profit Factor'] > filter_criteria['min_profit_factor'])
    ]
    
    print(f"\nFound {len(filtered_results)} strategies meeting the criteria")
    
    if len(filtered_results) == 0:
        print("No strategies meet the filtering criteria. Try relaxing your constraints.")
        return {}, pd.DataFrame(), pd.DataFrame()
    
    # Sort by Sharpe ratio
    filtered_results = filtered_results.sort_values('Sharpe Ratio', ascending=False)
    
    # Display top results
    print("\nTop 10 strategies meeting criteria (sorted by Sharpe Ratio):")
    display_columns = ['Long Threshold', 'Short Threshold', 'Total PnL', 'Sharpe Ratio', 
                       'Win Rate', 'Profit Factor', 'Max Drawdown', 'PnL/MaxDD', 'Total Trades']
    display_cols = [col for col in display_columns if col in filtered_results.columns]
    pd.set_option('display.precision', 6)
    print(filtered_results[display_cols].head(10).to_string(index=False))
    
    # Create cumulative returns DataFrame for plotting
    cumulative_returns = pd.DataFrame()
    
    # Add date column
    if len(filtered_results) > 0:
        first_key = (filtered_results.iloc[0]['Long Threshold'], filtered_results.iloc[0]['Short Threshold'])
        if first_key in all_backtesters:
            cumulative_returns['date'] = all_backtesters[first_key].results['date']
    
    # Add columns for each filtered strategy
    for _, row in filtered_results.iterrows():
        key = (row['Long Threshold'], row['Short Threshold'])
        label = f"L:{key[0]:.6f}, S:{key[1]:.6f}"
        
        if key in all_backtesters:
            cumulative_returns[label] = cumulative_pnl_dict[key]
    
    # Plot cumulative returns for filtered strategies
    if show_plot:
        plt.figure(figsize=(16, 8))
        
        # Limit to top 10 strategies to avoid cluttered plot
        plot_limit = min(10, len(filtered_results))
        
        for i, (_, row) in enumerate(filtered_results.iterrows()):
            if i >= plot_limit:
                break
                
            key = (row['Long Threshold'], row['Short Threshold'])
            label = f"L:{key[0]:.6f}, S:{key[1]:.6f} (SR:{row['Sharpe Ratio']:.2f})"
            
            if 'date' in cumulative_returns.columns and label in cumulative_returns.columns:
                plt.plot(cumulative_returns['date'], cumulative_returns[label], 
                        label=label)
        
        plt.title('Cumulative Return Comparison - Filtered Strategies Only (%)', fontsize=14)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Cumulative Return (%)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10)
        plt.tight_layout()
        
        # Save the figure if save_dir is provided
        if save_dir:
            import os
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(save_dir, f"cumulative_returns_comparison_{timestamp}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Cumulative returns comparison figure saved to: {save_path}")
        
        plt.show()
    
    # Create a heatmap of Sharpe ratios
    if len(results_df) >= 20 and show_plot:  # Only create heatmap if we tested enough combinations
        print("\nSharpe Ratio Heatmap:")
        
        # Create unique sorted lists of thresholds
        long_thresholds = sorted(results_df['Long Threshold'].unique())
        short_thresholds = sorted(results_df['Short Threshold'].unique())
        
        # If we have a reasonable number of unique thresholds, create a heatmap
        if len(long_thresholds) > 1 and len(short_thresholds) > 1:
            # Create pivot table
            pivot_table = results_df.pivot_table(
                values='Sharpe Ratio', 
                index='Long Threshold', 
                columns='Short Threshold'
            )
            
            plt.figure(figsize=(12, 10))
            heatmap = plt.pcolormesh(
                pivot_table.columns, 
                pivot_table.index, 
                pivot_table.values, 
                cmap='viridis', 
                shading='auto'
            )
            plt.colorbar(heatmap, label='Sharpe Ratio')
            
            # Mark filtered points on the heatmap
            filtered_points = filtered_results[['Long Threshold', 'Short Threshold']].values
            if len(filtered_points) > 0:
                plt.scatter(
                    filtered_points[:, 1],  # Short threshold (x-axis) 
                    filtered_points[:, 0],  # Long threshold (y-axis)
                    color='red',
                    marker='*',
                    s=100,
                    label='Filtered Strategies'
                )
            
            plt.title('Sharpe Ratio Heatmap by Threshold Values', fontsize=14)
            plt.xlabel('Short Threshold', fontsize=12)
            plt.ylabel('Long Threshold', fontsize=12)
            plt.legend()
            plt.tight_layout()
            
            # Save the figure if save_dir is provided
            if save_dir:
                import os
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(save_dir, f"sharpe_ratio_heatmap_{timestamp}.png")
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                print(f"Sharpe ratio heatmap figure saved to: {save_path}")
            
            plt.show()
    
    # Find the best strategy
    if len(filtered_results) > 0 and show_plot:
        best_row = filtered_results.iloc[0]
        best_key = (best_row['Long Threshold'], best_row['Short Threshold'])
        
        print("\nBest strategy details:")
        for key, value in best_row.items():
            if isinstance(value, float):
                print(f"{key}: {value:.6f}")
            else:
                print(f"{key}: {value}")
        
        # Plot the best strategy in detail
        if best_key in all_backtesters:
            print("\nDetailed plot of best strategy:")
            # Save the figure if save_dir is provided
            save_path = None
            if save_dir:
                import os
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                save_path = os.path.join(save_dir, f"best_strategy_results_{timestamp}.png")
            
            all_backtesters[best_key].plot_results(
                title=f"Best Strategy (L:{best_key[0]:.6f}, S:{best_key[1]:.6f})",
                save_path=save_path,
                show_plot=show_plot
            )
    
    # Return filtered results and backtesters for further analysis
    filtered_backtesters = {}
    for _, row in filtered_results.iterrows():
        key = (row['Long Threshold'], row['Short Threshold'])
        if key in all_backtesters:
            filtered_backtesters[key] = all_backtesters[key]
    
    return filtered_backtesters, filtered_results, cumulative_returns


def create_cumulative_return_comparison(data, filtered_results, holding_period=3, 
                                      max_strategies=10, save_dir=None, show_plot=True):
    """
    Create and plot cumulative return comparison chart for filtered strategies
    
    Parameters:
    -----------
    data : pd.DataFrame
        Original data, including 'date' and 'close' columns
    filtered_results : pd.DataFrame
        Filtered strategy results, including 'Long Threshold', 'Short Threshold' columns
    holding_period : int, default=3
        Holding period
    max_strategies : int, default=10
        Maximum number of strategies to display
    save_dir : str, optional
        Directory to save figures. If None, figures are not saved.
    show_plot : bool, default=True
        Whether to display plots (set to False for batch processing)
    """
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.dates import DateFormatter
    
    # Limit the maximum number of strategies to display to avoid cluttered chart
    if len(filtered_results) > max_strategies:
        print(f"Note: Only showing top {max_strategies} best strategies in cumulative return comparison")
        display_results = filtered_results.head(max_strategies)
    else:
        display_results = filtered_results
    
    # Create cumulative return comparison DataFrame
    cumulative_returns = pd.DataFrame({'date': data['date']})
    
    # Backtest each strategy to get cumulative returns
    for i, row in display_results.iterrows():
        # Create strategy
        strategy = ProbabilityThresholdStrategy(
            long_threshold=row['Long Threshold'],
            short_threshold=row['Short Threshold'],
            holding_period=holding_period
        )
        
        # Run backtest
        backtester = Backtester(profit_loss_window=holding_period)
        results = backtester.run(data, strategy)
        
        # Calculate cumulative return rate rather than cumulative points
        initial_capital = 100000  # Assume initial capital of 100,000
        cumulative_return = (results['Cumulative_PnL'] / initial_capital) * 100
        
        # Add to comparison DataFrame
        label = f"L:{row['Long Threshold']:.6f}, S:{row['Short Threshold']:.6f} (SR:{row['Sharpe Ratio']:.2f})"
        cumulative_returns[label] = cumulative_return.values
    
    # Plot cumulative return comparison chart
    if show_plot:
        plt.figure(figsize=(15, 8))
        
        for col in cumulative_returns.columns:
            if col != 'date':
                plt.plot(cumulative_returns['date'], cumulative_returns[col], label=col)
        
        # Set chart elements
        plt.title('Comparison (%)', fontsize=14)
        plt.xlabel('Data', fontsize=12)
        plt.ylabel('Cumulative Percentage (%)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=10, loc='best')
        
        # Format date
        ax = plt.gca()
        date_format = DateFormatter('%Y-%m-%d')
        ax.xaxis.set_major_formatter(date_format)
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Save the figure if save_dir is provided
        if save_dir:
            import os
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(save_dir, f"cumulative_return_comparison_{timestamp}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Cumulative return comparison figure saved to: {save_path}")
        
        plt.show()
        
        # Plot scatter chart of Sharpe ratio vs profit factor
        plt.figure(figsize=(12, 8))
        plt.scatter(
            filtered_results['Sharpe Ratio'], 
            filtered_results['Profit Factor'],
            c=filtered_results['Total PnL'],
            cmap='viridis',
            s=100,
            alpha=0.7
        )
        
        # Add labels to points
        for i, row in display_results.iterrows():
            plt.annotate(
                f"L:{row['Long Threshold']:.6f}, S:{row['Short Threshold']:.6f}",
                (row['Sharpe Ratio'], row['Profit Factor']),
                textcoords="offset points",
                xytext=(0, 10),
                ha='center',
                fontsize=8
            )
        
        plt.colorbar(label='Total PnL')
        plt.title('Comparison between satisfied groups', fontsize=14)
        plt.xlabel('Sharp ratio', fontsize=12)
        plt.ylabel('P/L ratio', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Save the figure if save_dir is provided
        if save_dir:
            import os
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(save_dir, f"sharpe_vs_profit_scatter_{timestamp}.png")
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Sharpe vs profit factor scatter plot saved to: {save_path}")
        
        plt.show()
    
    return cumulative_returns


def plot_probability_histogram(data, save_path=None, show_plot=True):
    """
    Plot histogram of long and short probabilities
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing 'Long_Probability' and 'Short_Probability' columns
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    show_plot : bool, default=True
        Whether to display the plot (set to False for batch processing)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Check if probability columns exist
    if 'Long_Probability' not in data.columns or 'Short_Probability' not in data.columns:
        print("Error: DataFrame must contain 'Long_Probability' and 'Short_Probability' columns")
        return
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Plot histograms
    plt.hist(data['Long_Probability'], bins=50, alpha=0.5, label='Long Probability', color='green', density=True)
    plt.hist(data['Short_Probability'], bins=50, alpha=0.5, label='Short Probability', color='red', density=True)
    
    # Add vertical lines for typical threshold values
    thresholds = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005]
    for threshold in thresholds:
        plt.axvline(x=threshold, color='gray', linestyle='--', alpha=0.5)
        plt.text(threshold, plt.gca().get_ylim()[1]*0.9, f"{threshold:.4f}", 
                 rotation=90, verticalalignment='top')
    
    # Add annotations showing percentage of events above thresholds
    y_pos = plt.gca().get_ylim()[1] * 0.8
    for threshold in thresholds:
        long_pct = (data['Long_Probability'] >= threshold).mean() * 100
        short_pct = (data['Short_Probability'] >= threshold).mean() * 100
        
        if long_pct > 0.01 or short_pct > 0.01:  # Only show if percentage is meaningful
            annotation = f"L: {long_pct:.2f}%\nS: {short_pct:.2f}%"
            plt.text(threshold, y_pos, annotation, rotation=90, 
                     fontsize=8, ha='left', va='top')
    
    # Add metadata
    plt.xlabel("Probability", fontsize=12)
    plt.ylabel("Density", fontsize=12)
    plt.title("Probability Distribution of Long and Short Events", fontsize=14)
    plt.legend(fontsize=12)
    plt.grid(True, alpha=0.3)
    
    # Add summary statistics
    long_stats = f"Long: mean={data['Long_Probability'].mean():.6f}, median={data['Long_Probability'].median():.6f}"
    short_stats = f"Short: mean={data['Short_Probability'].mean():.6f}, median={data['Short_Probability'].median():.6f}"
    plt.figtext(0.5, 0.01, long_stats + "\n" + short_stats, ha='center', fontsize=10, 
                bbox={'facecolor':'lightgray', 'alpha':0.5, 'pad':5})
    
    plt.tight_layout()
    
    # Save the figure if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Probability histogram saved to: {save_path}")
    
    # Show the plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()


class RealTimeTrader:
    """
    Real-time trading simulator, used to simulate trading decisions in a real-time trading environment.
    """
    def __init__(self, model, strategy, holding_period_minutes=3, exclude_times=None):
        """
        Initialize real-time trading simulator
        
        Parameters:
        -----------
        model : object
            Trained trading model, must have predict_proba method
        strategy : TradingStrategy
            Trading strategy object
        holding_period_minutes : int, default=3
            Holding period (in minutes)
        exclude_times : set, optional
            Set of time points to exclude
        """
        self.model = model
        self.strategy = strategy
        self.holding_period = timedelta(minutes=holding_period_minutes)
        self.exclude_times = exclude_times or {
            "08:45", "08:46", "08:47", "08:48", "08:49",  # First 5 minutes after morning open
            "13:41", "13:42", "13:43", "13:44", "13:45",  # First 5 minutes after afternoon open
            "15:00", "15:01", "15:02", "15:03", "15:04",  # Last 5 minutes before day session close
            "03:55", "03:56", "03:57", "03:58", "03:59"   # Last 5 minutes before night session close
        }
        self.last_trade_time = None
        self.trade_records = []
    
    def simulate_real_time_trading(self, data, feature_engineering):
        """
        Simulate real-time trading environment
        
        Parameters:
        -----------
        data : pd.DataFrame
            Historical data, sorted by time
        feature_engineering : object
            Feature engineering object, must have transform method
            
        Returns:
        --------
        pd.DataFrame
            Trade records
        """
        trade_records = []
        last_trade_time = None
        
        # Simulate real-time data step by step
        for i in range(len(data)):
            # Simulate step-by-step data acquisition
            current_data = data.iloc[:i+1].copy()
            
            # Extract the current latest data point
            latest_row = current_data.iloc[-1:].copy()
            current_time = latest_row['date'].iloc[0]
            hour_minute = current_time.strftime("%H:%M")
            
            # Check if time is in excluded times
            if hour_minute in self.exclude_times:
                continue
            
            # Check if crossing sessions (prevent overnight positions)
            if last_trade_time is not None:
                time_diff = (current_time - last_trade_time).total_seconds() / 3600  # Convert to hours
                if time_diff > 5:  # If last trade was more than 5 hours ago, reset trading state
                    last_trade_time = None
            
            # Check if in holding period
            if last_trade_time is not None and current_time < last_trade_time + self.holding_period:
                continue  # Don't make new trades during holding period
            
            # Process current data through feature engineering
            X = feature_engineering.transform(latest_row)
            
            # Model prediction
            probabilities = self.model.predict_proba(X)
            
            # Add prediction probabilities to current data
            latest_row['Long_Probability'] = probabilities[0][2]  # P(1) long
            latest_row['Short_Probability'] = probabilities[0][0]  # P(-1) short
            latest_row['Neutral_Probability'] = probabilities[0][1]  # P(0) neutral
            
            # Apply strategy to generate signals
            signals = self.strategy.generate_signals(latest_row)
            
            # Check if there are trading signals
            if signals['Long_Signal'].iloc[0] or signals['Short_Signal'].iloc[0]:
                # Calculate future profit/loss (if future data exists)
                if i + 3 < len(data):  # Assuming holding period is 3 minutes
                    future_price = data.iloc[i + 3]['close']
                    profit_loss = future_price - latest_row['close'].iloc[0]
                    
                    # If it's a short trade, reverse the profit/loss
                    if signals['Short_Signal'].iloc[0]:
                        profit_loss = -profit_loss
                else:
                    profit_loss = 0  # If no future data, assume profit/loss is 0
                
                # Record trade
                trade_records.append({
                    "date": current_time,
                    "close": latest_row['close'].iloc[0],
                    "long_probability": latest_row['Long_Probability'].iloc[0],
                    "short_probability": latest_row['Short_Probability'].iloc[0],
                    "trade_type": "Long" if signals['Long_Signal'].iloc[0] else "Short",
                    "profit_loss": profit_loss
                })
                
                # Update last trade time
                last_trade_time = current_time
        
        # Convert trade records to DataFrame
        if trade_records:
            trade_df = pd.DataFrame(trade_records)
            trade_df['cumulative_pnl'] = trade_df['profit_loss'].cumsum()
            self.trade_records = trade_df
            return trade_df
        else:
            return pd.DataFrame()
        

def optimize_visualization(data, long_threshold=0.005971, short_threshold=0.002471, 
                          holding_period=3, title=None, figsize=(14, 12), save_path=None):
    """
    Create an optimized visualization of the backtest results
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing the backtest data
    long_threshold : float, default=0.005971
        Long threshold for probability strategy
    short_threshold : float, default=0.002471
        Short threshold for probability strategy
    holding_period : int, default=3
        Holding period for the strategy
    title : str, optional
        Chart title
    figsize : tuple, default=(14, 12)
        Figure size
    save_path : str, optional
        Path to save the figure
        
    Returns:
    --------
    Backtester
        The backtester instance with results
    """
    # Create strategy
    strategy = ProbabilityThresholdStrategy(
        long_threshold=long_threshold,
        short_threshold=short_threshold,
        holding_period=holding_period,
        name=f"Best Strategy (L:{long_threshold:.6f}, S:{short_threshold:.6f})"
    )
    
    # Create backtester
    backtester = Backtester(profit_loss_window=holding_period)
    
    # Run backtest
    backtester.run(data, strategy)
    
    # Plot results with optimized visualization
    backtester.plot_results(
        title=title or f"Best Strategy (L:{long_threshold:.6f}, S:{short_threshold:.6f})",
        figsize=figsize,
        save_path=save_path
    )
    
    return backtester

def load_and_prepare_data(file_path):
    """
    Load and prepare trading data for backtesting
    
    Parameters:
    -----------
    file_path : str
        Path to data file (CSV or Excel)
        
    Returns:
    --------
    pd.DataFrame
        Processed DataFrame ready for backtesting
    """
    # Determine file type and load data
    if file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    elif file_path.endswith(('.xlsx', '.xls')):
        data = pd.read_excel(file_path)
    else:
        raise ValueError("Unsupported file format. Use CSV or Excel files.")
    
    # Convert date column to datetime
    if 'date' in data.columns:
        data['date'] = pd.to_datetime(data['date'])
    
    # Ensure numeric columns are float64
    numeric_cols = ['open', 'high', 'low', 'close', 'volume', 
                   'Long_Probability', 'Short_Probability']
    
    for col in numeric_cols:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # Fill missing values (if any)
    data = data.fillna(method='ffill').fillna(method='bfill')
    
    # Sort by date if available
    if 'date' in data.columns:
        data = data.sort_values('date').reset_index(drop=True)
    
    return data

def analyze_trading_data(data, output_dir=None, long_threshold=0.005971, short_threshold=0.002471,
                        holding_period=3, compare_kd=True, optimize_kd=False,
                        k_period=14, d_period=3, overbought=80, oversold=20):
    """
    Analyze trading data with both probability and KD strategies
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing trading data
    output_dir : str, optional
        Directory to save output files
    long_threshold : float, default=0.005971
        Long threshold for probability strategy
    short_threshold : float, default=0.002471
        Short threshold for probability strategy
    holding_period : int, default=3
        Holding period for strategies
    compare_kd : bool, default=True
        Whether to compare with KD strategy
    optimize_kd : bool, default=False
        Whether to try different KD parameters and find optimal settings
    k_period : int, default=14
        K period for KD strategy
    d_period : int, default=3
        D period for KD strategy
    overbought : float, default=80
        Overbought level for KD strategy
    oversold : float, default=20
        Oversold level for KD strategy
        
    Returns:
    --------
    dict
        Results dictionary containing backtester instances and metrics
    """
    # Setup output directory
    if output_dir is None:
        output_dir = os.path.join(os.getcwd(), "backtest_results")
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results = {}
    
    # Create and run probability strategy
    print("Running probability threshold strategy backtest...")
    prob_save_path = os.path.join(output_dir, f"probability_strategy_{timestamp}.png")
    
    probability_backtester = optimize_visualization(
        data,
        long_threshold=long_threshold,
        short_threshold=short_threshold,
        holding_period=holding_period,
        title=f"Probability Strategy (L:{long_threshold:.6f}, S:{short_threshold:.6f})",
        save_path=prob_save_path
    )
    
    results['probability_backtester'] = probability_backtester
    results['probability_metrics'] = probability_backtester.calculate_metrics()
    
    # Compare with KD strategy
    if compare_kd:
        print("Comparing with KD strategy...")
        
        if optimize_kd:
            # Try different KD parameters to find optimal settings
            print("Optimizing KD parameters...")
            
            # Define parameter grid for KD optimization
            k_periods = [5, 9, 14, 21]
            d_periods = [3, 5, 9]
            overbought_levels = [70, 75, 80]
            oversold_levels = [30, 25, 20]
            
            best_kd_pnl = -float('inf')
            best_kd_params = None
            best_kd_backtester = None
            
            # Test combinations of parameters
            for k in k_periods:
                for d in d_periods:
                    for ob in overbought_levels:
                        for os in oversold_levels:
                            print(f"Testing KD parameters: K={k}, D={d}, OB={ob}, OS={os}")
                            
                            # Create and run KD strategy