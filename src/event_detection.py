import pandas as pd


def detect_trading_events(data, profit_loss_window=3, atr_window=14, 
                          long_profit_threshold=10.0, short_loss_threshold=-10.0,
                          volume_multiplier=2.0, use_atr_filter=True):
    """
    根據特定條件判定做多和做空事件，並排除特定時間範圍。
    
    Parameters:
    -----------
    data : pd.DataFrame
        包含價格和技術指標的DataFrame
    profit_loss_window : int, default=3
        計算未來盈虧的時間窗口
    atr_window : int, default=14
        計算ATR的時間窗口
    long_profit_threshold : float, default=10.0
        做多事件的最低獲利閾值（點數）
    short_loss_threshold : float, default=-10.0
        做空事件的最大虧損閾值（點數）
    volume_multiplier : float, default=2.0
        相對於平均波動率的成交量倍數閾值
    use_atr_filter : bool, default=True
        是否使用ATR作為額外過濾條件
        
    Returns:
    --------
    pd.DataFrame
        添加了事件檢測結果的DataFrame
    """
    # 深度複製避免修改原始資料
    result = data.copy()
    
    # 確保date欄位是datetime類型
    if not pd.api.types.is_datetime64_any_dtype(result['date']):
        result['date'] = pd.to_datetime(result['date'])
    
    # 提取時間部分
    result['hour_minute'] = result['date'].dt.strftime('%H:%M')

    # 定義要排除的時間範圍
    exclude_times = {
        "08:45", "08:46", "08:47", "08:48", "08:49",  # 早盤開盤後5分鐘
        "13:41", "13:42", "13:43", "13:44", "13:45",  # 午盤開盤後5分鐘 
        "15:00", "15:01", "15:02", "15:03", "15:04",  # 日盤收盤前5分鐘
        "03:55", "03:56", "03:57", "03:58", "03:59"   # 夜盤收盤前5分鐘
    }
    
    # 排除特定時段
    result['Valid_Trading_Time'] = ~result['hour_minute'].isin(exclude_times)

    # 計算未來盈虧
    result['Profit_Loss_Points'] = result['close'].shift(-profit_loss_window) - result['close']

    # 確保數值型別正確
    for col in ['high', 'low', 'close']:
        result[col] = result[col].astype('float64')

    # 計算ATR
    high = data['high']
    low = data['low']
    close = data['close']
    previous_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - previous_close).abs()
    tr3 = (low - previous_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    result['ATR'] = true_range.rolling(window=atr_window).mean()


    # 初始化Event欄位
    result['Event'] = 0
    
    # 添加交易時段標記
    result['Session'] = 'Unknown'
    result.loc[(result['date'].dt.hour >= 8) & (result['date'].dt.hour < 14), 'Session'] = 'Day'
    result.loc[(result['date'].dt.hour >= 15) | (result['date'].dt.hour < 5), 'Session'] = 'Night'
    
    # 添加星期幾標記
    result['Day_Of_Week'] = result['date'].dt.day_name()

    # 做多事件條件
    long_conditions = (
        (result['Lower_Band_Slope'] < 0) &
        (result['Slope_Change'] < 0) &
        (result['Rebound_Above_EMA']) &
        (result['volume'] > volume_multiplier * result['Average_Volatility_long']) &
        (result['Profit_Loss_Points'] > long_profit_threshold)
    )
    
    # 如果啟用ATR過濾
    if use_atr_filter:
        long_conditions &= (result['Profit_Loss_Points'] > result['ATR'])
        
    # 應用交易時間過濾
    long_conditions &= result['Valid_Trading_Time']
    
    # 標記做多事件
    result.loc[long_conditions, 'Event'] = 1

    # 做空事件條件
    short_conditions = (
        (result['Lower_Band_Slope'] > 0) &
        (result['Slope_Change'] > 0) &
        (result['Break_Below_EMA']) &
        (result['volume'] > volume_multiplier * result['Average_Volatility_long']) &
        (result['Profit_Loss_Points'] < short_loss_threshold)
    )
    
    # 如果啟用ATR過濾
    if use_atr_filter:
        short_conditions &= (abs(result['Profit_Loss_Points']) > result['ATR'])
        
    # 應用交易時間過濾
    short_conditions &= result['Valid_Trading_Time']
    
    # 標記做空事件
    result.loc[short_conditions, 'Event'] = -1

    # 設置Label等於Event
    result['Label'] = result['Event']
    
    # 添加事件分類
    result['Event_Type'] = 'None'
    result.loc[result['Event'] == 1, 'Event_Type'] = 'Long'
    result.loc[result['Event'] == -1, 'Event_Type'] = 'Short'
    

    return result

def analyze_trading_events(data):
    """
    分析交易事件並生成統計數據。
    
    Parameters:
    -----------
    data : pd.DataFrame
        包含檢測到的交易事件的DataFrame
        
    Returns:
    --------
    Dict
        包含分析結果的字典
    """
    # 篩選有事件的資料
    events = data[data['Event'] != 0]
    total_events = len(events)
    
    if total_events == 0:
        return {"error": "沒有檢測到事件"}
    
    # 計算做多和做空事件
    long_events = len(data[data['Event'] == 1])
    short_events = len(data[data['Event'] == -1])
    
    # 計算盈虧統計
    long_stats = data[data['Event'] == 1]['Profit_Loss_Points'].describe()
    short_stats = data[data['Event'] == -1]['Profit_Loss_Points'].describe()
    all_stats = events['Profit_Loss_Points'].describe()
    
    # 計算勝率
    wins = len(events[events['Profit_Loss_Points'] > 0])
    losses = len(events[events['Profit_Loss_Points'] < 0])
    win_rate = wins / total_events if total_events > 0 else 0
    
    # 計算盈虧比
    total_profit = events[events['Profit_Loss_Points'] > 0]['Profit_Loss_Points'].sum()
    total_loss = abs(events[events['Profit_Loss_Points'] < 0]['Profit_Loss_Points'].sum())
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
    
    # 計算期望值
    expectancy = events['Profit_Loss_Points'].mean()
    
    # 時間分析
    by_day = events.groupby('Day_Of_Week')['Event'].count().to_dict()
    by_session = events.groupby('Session')['Event'].count().to_dict()
    by_hour = events.groupby(events['date'].dt.hour)['Event'].count().to_dict()
    
    return {
        "total_events": total_events,
        "long_events": long_events,
        "short_events": short_events,
        "long_percentage": (long_events / total_events * 100) if total_events > 0 else 0,
        "short_percentage": (short_events / total_events * 100) if total_events > 0 else 0,
        "win_rate": win_rate,
        "profit_factor": profit_factor,
        "expectancy": expectancy,
        "all_stats": all_stats,
        "long_stats": long_stats,
        "short_stats": short_stats,
        "by_day": by_day,
        "by_session": by_session,
        "by_hour": by_hour
    }