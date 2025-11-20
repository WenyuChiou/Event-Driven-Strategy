import pandas as pd
from typing import Dict, Any


def detect_trading_events(data: pd.DataFrame, profit_loss_window: int = 3, atr_window: int = 14, 
                          long_profit_threshold: float = 10.0, short_loss_threshold: float = -10.0,
                          volume_multiplier: float = 2.0, use_atr_filter: bool = True) -> pd.DataFrame:
    """
    Detect long and short trading events based on specific conditions, excluding certain time ranges.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing price and technical indicators
    profit_loss_window : int, default=3
        Time window for calculating future profit/loss
    atr_window : int, default=14
        Time window for calculating ATR
    long_profit_threshold : float, default=10.0
        Minimum profit threshold for long events (in points)
    short_loss_threshold : float, default=-10.0
        Maximum loss threshold for short events (in points)
    volume_multiplier : float, default=2.0
        Volume multiplier threshold relative to average volatility
    use_atr_filter : bool, default=True
        Whether to use ATR as an additional filter condition
        
    Returns:
    --------
    pd.DataFrame
        DataFrame with event detection results added
    """
    # Deep copy to avoid modifying original data
    result = data.copy()
    
    # Ensure date column is datetime type
    if not pd.api.types.is_datetime64_any_dtype(result['date']):
        result['date'] = pd.to_datetime(result['date'])
    
    # Extract time component
    result['hour_minute'] = result['date'].dt.strftime('%H:%M')

    # Define time ranges to exclude
    exclude_times = {
        "08:45", "08:46", "08:47", "08:48", "08:49",  # First 5 minutes after morning open
        "13:41", "13:42", "13:43", "13:44", "13:45",  # First 5 minutes after afternoon open
        "15:00", "15:01", "15:02", "15:03", "15:04",  # Last 5 minutes before day session close
        "03:55", "03:56", "03:57", "03:58", "03:59"   # Last 5 minutes before night session close
    }
    
    # Exclude specific time periods
    result['Valid_Trading_Time'] = ~result['hour_minute'].isin(exclude_times)

    # Calculate future profit/loss
    result['Profit_Loss_Points'] = result['close'].shift(-profit_loss_window) - result['close']

    # Ensure numeric types are correct
    for col in ['high', 'low', 'close']:
        result[col] = result[col].astype('float64')

    # Calculate ATR
    high = data['high']
    low = data['low']
    close = data['close']
    previous_close = close.shift(1)

    tr1 = high - low
    tr2 = (high - previous_close).abs()
    tr3 = (low - previous_close).abs()

    true_range = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    result['ATR'] = true_range.rolling(window=atr_window).mean()

    # Initialize Event column
    result['Event'] = 0
    
    # Add trading session markers
    result['Session'] = 'Unknown'
    result.loc[(result['date'].dt.hour >= 8) & (result['date'].dt.hour < 14), 'Session'] = 'Day'
    result.loc[(result['date'].dt.hour >= 15) | (result['date'].dt.hour < 5), 'Session'] = 'Night'
    
    # Add day of week marker
    result['Day_Of_Week'] = result['date'].dt.day_name()

    # Long event conditions
    long_conditions = (
        (result['Lower_Band_Slope'] < 0) &
        (result['Slope_Change'] < 0) &
        (result['Rebound_Above_EMA']) &
        (result['volume'] > volume_multiplier * result['Average_Volatility_long']) &
        (result['Profit_Loss_Points'] > long_profit_threshold)
    )
    
    # If ATR filter is enabled
    if use_atr_filter:
        long_conditions &= (result['Profit_Loss_Points'] > result['ATR'])
        
    # Apply trading time filter
    long_conditions &= result['Valid_Trading_Time']
    
    # Mark long events
    result.loc[long_conditions, 'Event'] = 1

    # Short event conditions
    short_conditions = (
        (result['Lower_Band_Slope'] > 0) &
        (result['Slope_Change'] > 0) &
        (result['Break_Below_EMA']) &
        (result['volume'] > volume_multiplier * result['Average_Volatility_long']) &
        (result['Profit_Loss_Points'] < short_loss_threshold)
    )
    
    # If ATR filter is enabled
    if use_atr_filter:
        short_conditions &= (abs(result['Profit_Loss_Points']) > result['ATR'])
        
    # Apply trading time filter
    short_conditions &= result['Valid_Trading_Time']
    
    # Mark short events
    result.loc[short_conditions, 'Event'] = -1

    # Set Label equal to Event
    result['Label'] = result['Event']
    
    # Add event classification
    result['Event_Type'] = 'None'
    result.loc[result['Event'] == 1, 'Event_Type'] = 'Long'
    result.loc[result['Event'] == -1, 'Event_Type'] = 'Short'

    return result

def analyze_trading_events(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Analyze trading events and generate statistics.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing detected trading events
        
    Returns:
    --------
    Dict
        Dictionary containing analysis results
    """
    # Filter data with events
    events = data[data['Event'] != 0]
    total_events = len(events)
    
    if total_events == 0:
        return {"error": "No events detected"}
    
    # Calculate long and short events
    long_events = len(data[data['Event'] == 1])
    short_events = len(data[data['Event'] == -1])
    
    # Calculate profit/loss statistics
    long_stats = data[data['Event'] == 1]['Profit_Loss_Points'].describe()
    short_stats = data[data['Event'] == -1]['Profit_Loss_Points'].describe()
    all_stats = events['Profit_Loss_Points'].describe()
    
    # Calculate win rate
    wins = len(events[events['Profit_Loss_Points'] > 0])
    losses = len(events[events['Profit_Loss_Points'] < 0])
    win_rate = wins / total_events if total_events > 0 else 0
    
    # Calculate profit factor
    total_profit = events[events['Profit_Loss_Points'] > 0]['Profit_Loss_Points'].sum()
    total_loss = abs(events[events['Profit_Loss_Points'] < 0]['Profit_Loss_Points'].sum())
    profit_factor = total_profit / total_loss if total_loss > 0 else float('inf')
    
    # Calculate expectancy
    expectancy = events['Profit_Loss_Points'].mean()
    
    # Time analysis
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