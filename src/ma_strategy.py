import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import re
import seaborn as sns
from datetime import datetime, time


def setup_chart_directory(base_dir=None, custom_path=None):
    """
    Set up a unified directory for saving charts, defaulting to /results/visualization/ma_charts.
    If custom_path is provided, it takes precedence.

    Parameters:
    -----------
    base_dir : str, optional
        Base directory path, default is '/results/visualization'.
    custom_path : str, optional
        Custom save path, highest priority.

    Returns:
    --------
    str
        The full path to the chart directory.
    """
    # Determine the base directory: use custom_path if provided, otherwise base_dir or default
    if custom_path:
        charts_base = custom_path
    else:
        charts_base = base_dir or '/results/visualization'

    # Construct the charts directory path
    charts_dir = os.path.join(charts_base, "ma_charts")
    # Create the directory if it doesn't exist
    os.makedirs(charts_dir, exist_ok=True)
    print(f"All charts will be saved to: {charts_dir}")
    return charts_dir


def calculate_ma_indicator(df, ma_period=5, price_col='Close'):
    """
    計算移動平均線指標
    
    Parameters:
    -----------
    df : pandas.DataFrame
        數據框架，包含價格數據
    ma_period : int, default=5
        移動平均線計算期間
    price_col : str, default='Close'
        價格列名
        
    Returns:
    --------
    pandas.DataFrame
        添加了MA指標的數據框架
    """
    # 複製數據框架以避免修改原始數據
    df_copy = df.copy()

    # 檢查價格列是否存在
    if price_col not in df_copy.columns:
        for col in df_copy.columns:
            if col.lower() == price_col.lower():
                price_col = col
                break
        else:
            # 如果找不到價格列，嘗試使用第一個數值列
            numeric_cols = df_copy.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                price_col = numeric_cols[0]
                print(f"警告：使用{price_col}作為價格列進行計算。")
            else:
                raise ValueError("在數據中找不到合適的價格列。")

    # 計算移動平均線
    df_copy[f'MA{ma_period}'] = df_copy[price_col].rolling(window=ma_period).mean()

    # 打印一些統計信息
    print(f"移動平均線指標統計信息:")
    print(f"{ma_period}日MA 最小值: {df_copy[f'MA{ma_period}'].min():.2f}, 最大值: {df_copy[f'MA{ma_period}'].max():.2f}, 平均值: {df_copy[f'MA{ma_period}'].mean():.2f}")

    return df_copy


def generate_ma_signals(df, ma_col='MA5', price_col='Close', min_hold_periods=5):
    """
    基於MA突破生成交易信號
    當價格突破MA時買入，當價格跌破MA時賣出
    
    Parameters:
    -----------
    df : pandas.DataFrame
        包含MA指標值的數據框架
    ma_col : str
        移動平均線列名
    price_col : str
        價格列名
    min_hold_periods : int, default=5
        最小持倉周期數，防止過短時間的交易
        
    Returns:
    --------
    pandas.DataFrame
        添加了信號列和倉位列的數據框架
    """
    # 複製數據框架以避免修改原始數據
    df_copy = df.copy()

    # 初始化信號列和倉位列
    df_copy['ma_signal'] = 0
    df_copy['ma_position'] = 0

    # 跟蹤倉位
    position = 0
    # 添加持倉計數器
    hold_counter = 0
    # 記錄上次進場時間，用於監控持倉時間
    entry_time = None

    for i in range(1, len(df_copy)):
        # 取得當前記錄的索引並轉換為 datetime 物件（若尚未轉換）
        current_dt = df_copy.index[i]
        if not isinstance(current_dt, datetime):
            try:
                current_dt = pd.to_datetime(current_dt)
            except:
                # 如果無法轉換為datetime，則使用一個替代方案
                current_dt = datetime.now()  # 只是為了避免錯誤
                
        current_time = current_dt.time()

        # 判斷當前是否在允許交易的時段：
        # 第一時段：09:00 ~ 13:30
        in_morning_session = time(9, 0) <= current_time < time(13, 30)
        # 第二時段分為兩段：
        # 從 15:00 到 23:59 (或接近午夜)
        # 以及從 00:00 到 04:30
        in_afternoon_session = (current_time >= time(15, 0)) or (current_time < time(4, 30))

        # 允許交易的條件：必須在其中一個時段內
        allowed = in_morning_session or in_afternoon_session

        # 若當前不允許交易，則強制平倉
        force_exit = not allowed

        # 更新持倉計數器（如果有持倉）
        if position == 1:
            hold_counter += 1
        
        # 檢查進場條件：
        # 當前無持倉且價格突破MA時進場(即當前價格大於MA，但前一天價格小於等於MA)
        if (position == 0 and 
            df_copy[price_col].iloc[i] > df_copy[ma_col].iloc[i] and 
            df_copy[price_col].iloc[i-1] <= df_copy[ma_col].iloc[i-1]):
            position = 1  # 進入做多倉位
            df_copy.loc[df_copy.index[i], 'ma_signal'] = 1
            hold_counter = 1  # 重置持倉計數器
            entry_time = current_dt  # 記錄進場時間

        # 檢查出場條件：
        # 1. 需滿足最小持倉時間
        # 2. 且價格跌破MA線（即當前價格小於等於MA，但前一天價格大於MA），或
        # 3. 當前不在允許交易的時間區間內（force_exit）
        elif (position == 1 and hold_counter >= min_hold_periods and 
             ((df_copy[price_col].iloc[i] <= df_copy[ma_col].iloc[i] and 
               df_copy[price_col].iloc[i-1] > df_copy[ma_col].iloc[i-1]) or force_exit)):
            position = 0  # 退出做多倉位
            hold_counter = 0  # 重置持倉計數器
            
            # 可以在這裡記錄持倉時間（用於調試）
            if entry_time:
                duration_minutes = (current_dt - entry_time).total_seconds() / 60
                # 這裡可以添加記錄持倉時間的代碼，例如打印或保存到列表
                # print(f"持倉時間: {duration_minutes:.1f} 分鐘")
                entry_time = None  # 重置進場時間

        # 更新倉位列
        df_copy.loc[df_copy.index[i], 'ma_position'] = position
        
        # 添加一個列記錄當前持倉周期數（可選）
        df_copy.loc[df_copy.index[i], 'hold_periods'] = hold_counter if position == 1 else 0

    return df_copy


def create_advanced_analysis_chart(df, trade_points, charts_dir, is_datetime_index, date_format, locator, show_plot=False):
    """創建高級分析圖表"""
    # 設置樣式
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    
    # 創建圖表
    fig, axs = plt.subplots(2, 2, figsize=(16, 12))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    
    # 1. 交易盈虧分佈 (左上)
    ax1 = axs[0, 0]
    
    if trade_points:
        bins = min(20, len(trade_points))
        ax1.hist(trade_points, bins=bins, color='#2ca02c', alpha=0.7)
        ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    
    ax1.set_title('Trade P&L Distribution', fontweight='bold')
    ax1.set_xlabel('P&L Points', fontweight='bold')
    ax1.set_ylabel('Frequency', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. 回撤分析 (右上)
    ax2 = axs[0, 1]
    
    if 'drawdown_points' in df.columns:
        ax2.fill_between(df.index, 0, df['drawdown_points'], color='#ff7f0e', alpha=0.5)
        
        # 標記最大回撤
        max_drawdown_points = df['drawdown_points'].max() if not df.empty else 0
        if max_drawdown_points > 0:
            max_dd_idx = df['drawdown_points'].idxmax()
            ax2.scatter([max_dd_idx], [max_drawdown_points], color='red', s=100, zorder=5)
            ax2.annotate(f'Max DD: {max_drawdown_points:.2f}',
                        xy=(max_dd_idx, max_drawdown_points),
                        xytext=(0, 10), textcoords="offset points",
                        ha='center', fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="red", alpha=0.8))
    
    ax2.set_title('Drawdown Analysis', fontweight='bold')
    ax2.set_xlabel('Date', fontweight='bold')
    ax2.set_ylabel('Drawdown (Points)', fontweight='bold')
    
    if is_datetime_index:
        ax2.xaxis.set_major_formatter(mdates.DateFormatter(date_format))
        if locator:
            ax2.xaxis.set_major_locator(locator)
        else:
            ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
    
    ax2.grid(True, alpha=0.3)
    
    # 3. 月度收益熱圖 (左下)
    ax3 = axs[1, 0]
    
    if is_datetime_index and not df.empty:
        # 添加年月列
        df_copy = df.copy()
        df_copy['year'] = df_copy.index.year
        df_copy['month'] = df_copy.index.month
        
        # 計算月度收益
        monthly_returns = df_copy.groupby(['year', 'month'])['ma_daily_points'].sum().unstack()
        
        if not monthly_returns.empty:
            try:
                # 創建熱圖
                sns.heatmap(monthly_returns, annot=True, fmt=".1f", cmap='RdYlGn', center=0,
                           linewidths=.5, ax=ax3, cbar_kws={'label': 'P&L Points'})
                ax3.set_title('Monthly Returns Heatmap', fontweight='bold')
                ax3.set_xlabel('Month', fontweight='bold')
                ax3.set_ylabel('Year', fontweight='bold')
            except:
                ax3.text(0.5, 0.5, "Error creating heatmap", 
                        ha='center', va='center', transform=ax3.transAxes)
        else:
            ax3.text(0.5, 0.5, "Insufficient data for monthly returns", 
                    ha='center', va='center', transform=ax3.transAxes)
    else:
        ax3.text(0.5, 0.5, "Date index required for monthly returns", 
                ha='center', va='center', transform=ax3.transAxes)
    
    # 4. 勝/負交易分析 (右下)
    ax4 = axs[1, 1]
    
    if trade_points:
        # 勝負交易計數
        winning_trades = sum(1 for p in trade_points if p > 0)
        losing_trades = sum(1 for p in trade_points if p <= 0)
        
        # 繪製柱狀圖
        win_loss_data = [winning_trades, losing_trades]
        win_loss_labels = ['Winning Trades', 'Losing Trades']
        win_loss_colors = ['#2ca02c', '#d62728']
        
        bars = ax4.bar(win_loss_labels, win_loss_data, color=win_loss_colors)
        
        # 添加數值標籤
        for bar in bars:
            height = bar.get_height()
            ax4.text(bar.get_x() + bar.get_width()/2, height + 0.1,
                    f'{int(height)}', ha='center', fontweight='bold')
    
    ax4.set_title('Win/Loss Trade Analysis', fontweight='bold')
    ax4.set_ylabel('Number of Trades', fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 添加標題
    fig.suptitle('Advanced Trading Performance Analytics', fontsize=16, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 保存圖表，使用簡短的檔名
    advanced_chart_path = os.path.join(charts_dir, f'advanced_analysis.png')
    plt.savefig(advanced_chart_path, dpi=300, bbox_inches='tight')
    print(f"高級分析圖保存到：{advanced_chart_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    return advanced_chart_path


def create_ma_comparison_chart(results, charts_dir, show_plot=False):
    """創建多個MA週期的累積收益比較圖表"""
    plt.figure(figsize=(16, 10))
    
    # 準備不同策略的資料
    for i, (ma_period, df) in enumerate(results.items()):
        if 'ma_cumulative_points' in df.columns:
            label = f"MA {ma_period}"
            plt.plot(df.index, df['ma_cumulative_points'], label=label, linewidth=1.5)
    
    # 添加買入持有曲線 (使用最後一個DataFrame，假設它們有相同的日期範圍)
    if results and 'buyhold_cumulative_points' in list(results.values())[0].columns:
        last_df = list(results.values())[0]
        plt.plot(last_df.index, last_df['buyhold_cumulative_points'], label='Buy and Hold', 
                 color='black', linestyle='--', linewidth=1.5)
    
    plt.title('Cumulative Profit Comparison (Points)', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Profit (Points)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='lower right', fontsize=10)
    
    # 設置日期格式
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()
    
    # 保存圖表
    comparison_chart_path = os.path.join(charts_dir, f'ma_comparison.png')
    plt.savefig(comparison_chart_path, dpi=300, bbox_inches='tight')
    print(f"MA比較圖保存到：{comparison_chart_path}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    return comparison_chart_path


def backtest_ma_strategy(data_path, ma_period=5, price_col='Close', save_excel=True, 
                        run_id=None, base_dir=None, custom_path=None, enhanced_charts=True, 
                        show_plots=True, commission=0.0, point_value=1.0, contract_size=1,
                        min_hold_periods=5):
    """
    Backtest a 5-day moving average breakout strategy (only cumulative P&L is shown)

    Parameters:
    -----------
    data_path : str
        Path to the data file (CSV or Excel)
    ma_period : int, default=5
        Period for the moving average
    price_col : str, default='Close'
        Column name for price data
    save_excel : bool, default=True
        Whether to save detailed results to Excel
    run_id : str, optional
        Unique identifier for this backtest run
    base_dir : str, optional
        Base directory path for outputs
    custom_path : str, optional
        Custom path for saving outputs (takes precedence)
    enhanced_charts : bool, default=True
        Whether to generate enhanced charts
    show_plots : bool, default=True
        Whether to display plots interactively
    commission : float, default=0.0
        Commission per trade (in points)
    point_value : float, default=1.0
        Monetary value per point
    contract_size : int, default=1
        Number of contracts per trade

    Returns:
    --------
    dict, DataFrame
        A tuple containing the performance metrics dictionary and the result DataFrame
    """
    # 全局變量，用於圖表創建函數
    global trade_points
    
    # 生成run_id（如果未提供）
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    print(f"開始5日均線突破策略回測（運行ID：{run_id}）")
    print(f"參數：MA週期={ma_period}")

    # 加載數據
    print(f"從以下位置加載數據：{data_path}")
    if data_path.endswith('.xlsx') or data_path.endswith('.xls'):
        df = pd.read_excel(data_path)
    elif data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    else:
        raise ValueError("不支持的文件格式。請使用.xlsx或.csv")

    # 處理日期列
    date_columns = ['date', 'Date', 'datetime', 'Datetime', 'time', 'Time']
    date_col = None
    
    for col in date_columns:
        if col in df.columns:
            date_col = col
            break
    
    if date_col:
        print(f"發現日期列：{date_col}")
        # 轉換為datetime - 處理各種格式
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        
        # 檢查轉換是否成功
        if df[date_col].isna().any():
            print(f"警告：某些日期無法解析。原始格式：{df[date_col].iloc[0]}")
            
        # 將日期設為索引以便繪圖
        df.set_index(date_col, inplace=True)
        print(f"日期範圍：{df.index.min()} 至 {df.index.max()}")
    else:
        print("警告：未找到日期列。使用數字索引。")
        # 如果需要，創建合成的datetime索引
        df.index = pd.date_range(start='2000-01-01', periods=len(df), freq='D')
    
    # 如果索引不是datetime，嘗試轉換
    if not isinstance(df.index, pd.DatetimeIndex):
        try:
            df.index = pd.to_datetime(df.index)
            print("索引已轉換為datetime格式")
        except:
            print("警告：無法將索引轉換為datetime。使用原始索引。")

    # 標準化列名 - 不區分大小寫匹配
    ohlc_mapping = {
        'open': 'Open',
        'high': 'High',
        'low': 'Low',
        'close': 'Close',
        'volume': 'Volume'
    }

    for old_name, new_name in ohlc_mapping.items():
        for col in df.columns:
            if re.match(f"^{old_name}$", col, re.IGNORECASE):
                df[new_name] = df[col]
                if col != new_name:
                    print(f"重命名列 '{col}' 為 '{new_name}'")

    # 確保價格列存在
    if price_col not in df.columns:
        for col in df.columns:
            if re.match(f"^{price_col}$", col, re.IGNORECASE):
                price_col = col
                print(f"使用 {price_col} 作為價格列")
                break
        else:  # 未找到匹配的列
            price_col = 'Close' if 'Close' in df.columns else df.select_dtypes(include=['number']).columns[0]
            print(f"價格列 '{price_col}' 未找到。改用 '{price_col}'。")

    # 計算MA指標
    print(f"計算{ma_period}日移動平均線...")
    df = calculate_ma_indicator(df, ma_period=ma_period, price_col=price_col)
    ma_col = f'MA{ma_period}'

    # 生成MA突破交易信號
    print(f"生成MA突破交易信號...")
    df = generate_ma_signals(df, ma_col=ma_col, price_col=price_col, min_hold_periods=min_hold_periods)

    # 計算點變動
    df['daily_point_change'] = df[price_col].diff()

    # 追蹤交易表現
    position = 0
    entry_price = 0
    entry_date = None
    trade_points = []
    trade_durations = []
    total_points = 0
    running_points = []
    longest_winning_streak = 0
    longest_losing_streak = 0
    current_streak = 0
    trade_details = []
    
    print("計算交易指標...")
    for i in range(1, len(df)):
        current_position = df['ma_position'].iloc[i]
        prev_position = df['ma_position'].iloc[i-1]
        current_date = df.index[i]
        
        # 倉位變化 - 記錄交易動作
        if current_position != prev_position:
            # 入場
            if prev_position == 0 and current_position == 1:
                entry_price = df[price_col].iloc[i]
                entry_date = current_date
            # 出場
            elif prev_position == 1 and current_position == 0:
                exit_price = df[price_col].iloc[i]
                exit_date = current_date
                
                # 計算交易盈虧（以點為單位）
                point_change = exit_price - entry_price - commission
                trade_points.append(point_change)
                total_points += point_change
                
                # 計算交易持續時間
                if entry_date:
                    # 計算精確的分鐘級持倉時間
                    if isinstance(exit_date, datetime) and isinstance(entry_date, datetime):
                        # 計算實際分鐘數
                        duration_minutes = (exit_date - entry_date).total_seconds() / 60
                        duration = max(1, int(duration_minutes))  # 最小1分鐘
                    else:
                        # 如果不是datetime對象，則按天計算
                        try:
                            duration = (exit_date - entry_date).days
                            if duration == 0:  # 同一天交易
                                duration = 1
                        except:
                            duration = 1  # 默認為1天
                    trade_durations.append(duration)
                
                # 追蹤連續盈利/虧損
                if point_change > 0:
                    if current_streak < 0:
                        current_streak = 1
                    else:
                        current_streak += 1
                else:
                    if current_streak > 0:
                        current_streak = -1
                    else:
                        current_streak -= 1
                        
                longest_winning_streak = max(longest_winning_streak, current_streak if current_streak > 0 else 0)
                longest_losing_streak = max(longest_losing_streak, -current_streak if current_streak < 0 else 0)
                
                # 記錄詳細交易信息
                trade_details.append({
                    'Entry Date': entry_date,
                    'Exit Date': exit_date,
                    'Entry Price': entry_price,
                    'Exit Price': exit_price,
                    'Direction': 'Long',
                    'Points': point_change,
                    'Duration': duration if 'duration' in locals() else None,
                    'Commission': commission
                })
        
        # 計算策略的每日盈虧（以點為單位）
        if prev_position == 1:
            day_point_change = df['daily_point_change'].iloc[i]
            # 如果今天倉位變化，則應用佣金
            if current_position != prev_position:
                day_point_change -= commission
        else:
            day_point_change = 0
        
        # 存儲策略的每日點數
        df.loc[df.index[i], 'ma_daily_points'] = day_point_change
        
        # 計算點數的累積總和
        running_sum = total_points
        if len(running_points) > 0:
            running_sum = running_points[-1]
        running_sum += day_point_change
        running_points.append(running_sum)
        df.loc[df.index[i], 'ma_cumulative_points'] = running_sum

    # 計算買入持有的點數以進行比較
    df['buyhold_daily_points'] = df['daily_point_change']
    df['buyhold_cumulative_points'] = df['buyhold_daily_points'].cumsum()

    # 移除NaN值
    df.dropna(inplace=True)

    # 計算績效指標
    print("計算績效指標...")
    
    # 總點數
    total_points_gained = df['ma_cumulative_points'].iloc[-1] if not df.empty else 0
    
    # 將點數轉換為貨幣價值
    total_value = total_points_gained * point_value * contract_size
    
    # 計算勝率
    num_trades = len(trade_points)
    winning_trades = sum(1 for p in trade_points if p > 0)
    losing_trades = sum(1 for p in trade_points if p <= 0)
    win_rate = winning_trades / num_trades if num_trades > 0 else 0
    
    # 計算每筆交易的平均點數
    avg_points_per_trade = total_points_gained / num_trades if num_trades > 0 else 0
    
    # 計算回撤（以點為單位）
    df['peak_points'] = df['ma_cumulative_points'].cummax()
    df['drawdown_points'] = df['peak_points'] - df['ma_cumulative_points']
    max_drawdown_points = df['drawdown_points'].max() if not df.empty else 0
    
    # 計算平均交易持續時間
    avg_trade_duration = sum(trade_durations) / len(trade_durations) if trade_durations else 0
    
    # 計算盈利因子和期望值
    winning_points = sum(p for p in trade_points if p > 0)
    losing_points = abs(sum(p for p in trade_points if p <= 0))
    profit_factor = winning_points / losing_points if losing_points > 0 else float('inf')
    
    # 計算平均盈利和平均虧損
    avg_win = winning_points / winning_trades if winning_trades > 0 else 0
    avg_loss = losing_points / losing_trades if losing_trades > 0 else 0
    
    # 計算夏普比率（使用每日收益，假設一年252個交易日）
    if 'ma_daily_points' in df.columns and len(df) > 1:
        daily_returns = df['ma_daily_points'].values
        sharpe_ratio = (np.mean(daily_returns) / np.std(daily_returns)) * np.sqrt(252) if np.std(daily_returns) > 0 else 0
    else:
        sharpe_ratio = 0
    
    # 計算盈虧/最大回撤比率
    pnl_maxdd_ratio = total_points_gained / max_drawdown_points if max_drawdown_points > 0 else float('inf')
    
    # 收集增強型性能指標
    metrics = {
        'Strategy': f'{ma_period}-Day MA Breakout',
        'Parameters': f"MA Period:{ma_period}",
        'Total PnL': total_points_gained,
        'Total Value': total_value,
        'Sharpe Ratio': sharpe_ratio,
        'Win Rate': win_rate,
        'Profit Factor': profit_factor,
        'PnL/MaxDD': pnl_maxdd_ratio,
        'Max Drawdown': max_drawdown_points,
        'Total Trades': num_trades,
        'Winning Trades': winning_trades,
        'Losing Trades': losing_trades,
        'Average Win': avg_win,
        'Average Loss': avg_loss,
        'Average Trade Duration': avg_trade_duration,
        'Longest Winning Streak': longest_winning_streak,
        'Longest Losing Streak': longest_losing_streak
    }

    # 打印關鍵指標
    print("\n===== 回測結果 =====")
    print(f"策略: {ma_period}日均線突破 (MA周期:{ma_period})")
    print(f"總盈虧: {total_points_gained:.2f} 點 (${total_value:.2f})")
    print(f"夏普比率: {sharpe_ratio:.2f}")
    print(f"勝率: {win_rate:.2%}")
    print(f"盈利因子: {profit_factor:.2f}")
    print(f"最大回撤: {max_drawdown_points:.2f} 點")
    print(f"盈虧/最大回撤比率: {pnl_maxdd_ratio:.2f}")
    print(f"總交易次數: {num_trades}")
    print(f"每筆交易平均點數: {avg_points_per_trade:.2f}")

    # 設置統一的圖表資料夾
    charts_dir = setup_chart_directory(base_dir, custom_path)
    
    # 設置結果資料夾（用於Excel和其他非圖表輸出）
    if custom_path:
        results_dir = os.path.join(custom_path, 'results', f'ma_backtest_{run_id}')
    elif base_dir:
        results_dir = os.path.join(base_dir, 'results', f'ma_backtest_{run_id}')
    else:
        results_dir = os.path.join(os.getcwd(), 'results', f'ma_backtest_{run_id}')
    
    os.makedirs(results_dir, exist_ok=True)

    # 設置日期格式 - 改進的日期處理
    is_datetime_index = isinstance(df.index, pd.DatetimeIndex)
    
    if is_datetime_index:
        # 根據日期範圍決定適當的日期格式
        date_range = df.index.max() - df.index.min()
        
        if date_range.days <= 2:  # 少於2天 - 使用小時和分鐘
            date_format = '%Y-%m-%d %H:%M'
            locator = mdates.HourLocator(interval=4)  # 每4小時顯示一次
        elif date_range.days <= 14:  # 少於2週 - 使用天
            date_format = '%Y-%m-%d'
            locator = mdates.DayLocator(interval=1)  # 每天顯示一次
        elif date_range.days <= 180:  # 少於6個月 - 使用每週
            date_format = '%Y-%m-%d'
            locator = mdates.WeekdayLocator(interval=1)  # 每週顯示一次
        else:  # 超過6個月 - 使用每月
            date_format = '%Y-%m'
            locator = mdates.MonthLocator(interval=1)  # 每月顯示一次
    else:
        print("警告：檢測到非datetime索引。使用數字索引進行繪圖。")
        df = df.reset_index(drop=True)
        date_format = None
        locator = None

    print("\n創建可視化...")
    
    # 創建只有累積損益的圖表
    profit_chart_path = os.path.join(charts_dir, f'ma{ma_period}_profit.png')
    plt.figure(figsize=(14, 8))
    
    # 只繪製累積收益線
    plt.plot(df.index, df['ma_cumulative_points'], label=f'MA {ma_period} Strategy', color='blue', linewidth=2)
    plt.plot(df.index, df['buyhold_cumulative_points'], label='Buy & Hold', color='black', linestyle='--', linewidth=1.5)
    
    plt.title('Cumulative Profit Comparison (Points)', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Profit (Points)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=10)
    
    # 調整日期格式
    if is_datetime_index:
        plt.gca().xaxis.set_major_formatter(mdates.DateFormatter(date_format))
        if locator:
            plt.gca().xaxis.set_major_locator(locator)
        else:
            plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gcf().autofmt_xdate()
    
    # 添加績效統計信息
    text_info = (
        f"Total PnL: {total_points_gained:.1f}\n"
        f"Sharpe Ratio: {sharpe_ratio:.2f}\n"
        f"Win Rate: {win_rate:.2%}\n"
        f"Profit Factor: {profit_factor:.2f}\n"
        f"Max Drawdown: {max_drawdown_points:.1f}\n"
        f"PnL/MaxDD: {pnl_maxdd_ratio:.2f}\n"
        f"Trades: {num_trades}"
    )
    
    plt.annotate(text_info, xy=(0.02, 0.02), xycoords='axes fraction',
                 bbox=dict(boxstyle="round,pad=0.5", fc="white", alpha=0.8),
                 fontsize=10, verticalalignment='bottom')
    
    plt.tight_layout()
    plt.savefig(profit_chart_path, dpi=300, bbox_inches='tight')
    print(f"累積損益圖保存到：{profit_chart_path}")
    
    # 創建高級圖表
    if enhanced_charts:
        print("創建增強型可視化...")
        
        # 創建高級分析圖
        advanced_chart_path = create_advanced_analysis_chart(
            df, trade_points, charts_dir,
            is_datetime_index, date_format, locator, show_plot=False
        )
    
    # 保存圖表路徑信息
    chart_info_path = os.path.join(results_dir, f'chart_paths.txt')
    with open(chart_info_path, 'w') as f:
        f.write(f"累積損益圖: {profit_chart_path}\n")
        if enhanced_charts:
            f.write(f"高級分析圖: {advanced_chart_path}\n")
    
    if show_plots:
        plt.show()
    else:
        plt.close('all')

    # 保存結果到Excel
    if save_excel:
        print("保存詳細結果到Excel...")
        excel_path = os.path.join(results_dir, f'ma_backtest_results_{run_id}.xlsx')
        try:
            with pd.ExcelWriter(excel_path) as writer:
                # 保存詳細回測結果
                df.to_excel(writer, sheet_name='Detailed Results')
                
                # 保存摘要指標
                metrics_df = pd.DataFrame.from_dict(metrics, orient='index', columns=['Value'])
                metrics_df.to_excel(writer, sheet_name='Summary Metrics')
                
                # 保存交易列表
                if trade_details:
                    trades_df = pd.DataFrame(trade_details)
                    trades_df.to_excel(writer, sheet_name='Trade List', index=False)
                
                # 保存交易點數以進行詳細分析
                if trade_points:
                    trades_df = pd.DataFrame({
                        'Trade Number': range(1, len(trade_points) + 1),
                        'Points': trade_points,
                        'Result': ['Win' if p > 0 else 'Loss' for p in trade_points]
                    })
                    trades_df.to_excel(writer, sheet_name='Trade Points', index=False)
                
                # 如果有可用的日期索引，保存每月表現
                if is_datetime_index and 'ma_daily_points' in df.columns:
                    monthly_df = df.copy()
                    monthly_df['Year'] = monthly_df.index.year
                    monthly_df['Month'] = monthly_df.index.month
                    monthly_performance = monthly_df.groupby(['Year', 'Month'])['ma_daily_points'].sum().reset_index()
                    monthly_performance.columns = ['Year', 'Month', 'Monthly Points']
                    monthly_performance.to_excel(writer, sheet_name='Monthly Performance', index=False)
                
            print(f"結果已保存到 {excel_path}")
        except Exception as e:
            print(f"保存Excel文件時出錯: {str(e)}")

    print(f"回測完成。所有可視化已保存到 {charts_dir}")
    return metrics, df

def optimize_ma_parameters(data_path, price_col='Close', ma_periods=[5, 10, 20, 30, 50],
                          run_id=None, base_dir=None, save_excel=True, show_plots=False):
    """
    通過測試不同的參數組合來優化MA策略
    
    Parameters:
    -----------
    data_path : str
        數據文件路徑（CSV或Excel）
    price_col : str, default='Close'
        價格列名
    ma_periods : list, default=[5, 10, 20, 30, 50]
        要測試的MA週期列表
    run_id : str, optional
        此優化運行的唯一標識符
    base_dir : str, optional
        基礎目錄路徑
    save_excel : bool, default=True
        是否保存詳細的Excel結果
    show_plots : bool, default=False
        是否顯示圖表（通常在批處理優化中設置為False）
        
    Returns:
    --------
    tuple
        (best_params, all_results) 包含最佳參數和包含所有結果的DataFrame
    """
    # 如果未提供run_id，生成一個
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"開始MA參數優化（運行ID：{run_id}）")
    print(f"測試 {len(ma_periods)} 個參數組合")
    
    # 設置目錄
    if base_dir:
        results_dir = os.path.join(base_dir, 'results', f'ma_optimization_{run_id}')
        os.makedirs(results_dir, exist_ok=True)
    else:
        results_dir = os.path.join(os.getcwd(), 'results', f'ma_optimization_{run_id}')
        os.makedirs(results_dir, exist_ok=True)
    
    # 初始化結果存儲
    all_results = []
    
    # 追蹤按不同指標的最佳參數
    best_params = {
        'total_pnl': {'params': None, 'value': float('-inf')},
        'sharpe': {'params': None, 'value': float('-inf')},
        'profit_factor': {'params': None, 'value': float('-inf')},
        'pnl_maxdd': {'params': None, 'value': float('-inf')}
    }
    
    # 測試每個參數
    for i, ma_period in enumerate(ma_periods):
        print(f"\n測試參數 {i+1}/{len(ma_periods)}: MA期間={ma_period}")
        
        # 使用此參數運行回測
        combo_run_id = f"{run_id}_MA{ma_period}"
        
        try:
            metrics = backtest_ma_strategy(
                data_path=data_path,
                ma_period=ma_period,
                price_col=price_col,
                save_excel=False,  # 優化期間不保存單個Excel文件
                run_id=combo_run_id,
                base_dir=base_dir,
                enhanced_charts=False,  # 優化期間不創建增強型圖表
                show_plots=False  # 優化期間不顯示圖表
            )
            
            # 存儲結果
            result_row = {
                'MA Period': ma_period,
                'Total PnL': metrics['Total PnL'],
                'Sharpe Ratio': metrics['Sharpe Ratio'],
                'Win Rate': metrics['Win Rate'],
                'Profit Factor': metrics['Profit Factor'],
                'Max Drawdown': metrics['Max Drawdown'],
                'PnL/MaxDD': metrics['PnL/MaxDD'],
                'Total Trades': metrics['Total Trades'],
                'Winning Trades': metrics['Winning Trades'],
                'Losing Trades': metrics['Losing Trades']
            }
            
            all_results.append(result_row)
            
            # 如果這個參數更好，則更新最佳參數
            if metrics['Total PnL'] > best_params['total_pnl']['value']:
                best_params['total_pnl']['value'] = metrics['Total PnL']
                best_params['total_pnl']['params'] = ma_period
            
            if metrics['Sharpe Ratio'] > best_params['sharpe']['value']:
                best_params['sharpe']['value'] = metrics['Sharpe Ratio']
                best_params['sharpe']['params'] = ma_period
            
            if metrics['Profit Factor'] > best_params['profit_factor']['value']:
                best_params['profit_factor']['value'] = metrics['Profit Factor']
                best_params['profit_factor']['params'] = ma_period
            
            if metrics['PnL/MaxDD'] > best_params['pnl_maxdd']['value']:
                best_params['pnl_maxdd']['value'] = metrics['PnL/MaxDD']
                best_params['pnl_maxdd']['params'] = ma_period
            
        except Exception as e:
            print(f"測試參數 MA={ma_period} 時出錯: {str(e)}")
    
    # 將結果轉換為DataFrame
    results_df = pd.DataFrame(all_results)
    
    # 按總盈虧降序排序
    results_df = results_df.sort_values('Total PnL', ascending=False)
    
    # 打印優化結果
    print("\n===== MA參數優化結果 =====")
    print(f"測試的總參數數: {len(all_results)}")
    
    # 顯示按不同指標的最佳參數
    print("\n按總盈虧的最佳參數:")
    if best_params['total_pnl']['params']:
        ma = best_params['total_pnl']['params']
        print(f"MA={ma} (盈虧: {best_params['total_pnl']['value']:.2f})")
    
    print("\n按夏普比率的最佳參數:")
    if best_params['sharpe']['params']:
        ma = best_params['sharpe']['params']
        print(f"MA={ma} (夏普比率: {best_params['sharpe']['value']:.2f})")
    
    print("\n按盈利因子的最佳參數:")
    if best_params['profit_factor']['params']:
        ma = best_params['profit_factor']['params']
        print(f"MA={ma} (盈利因子: {best_params['profit_factor']['value']:.2f})")
    
    print("\n按盈虧/最大回撤比率的最佳參數:")
    if best_params['pnl_maxdd']['params']:
        ma = best_params['pnl_maxdd']['params']
        print(f"MA={ma} (盈虧/最大回撤: {best_params['pnl_maxdd']['value']:.2f})")
    
    # 顯示前5名參數
    print("\n按總盈虧排序的前5名參數:")
    if len(results_df) >= 5:
        for i, row in results_df.head(5).iterrows():
            print(f"MA={int(row['MA Period'])}: "
                 f"盈虧={row['Total PnL']:.2f}, 夏普比率={row['Sharpe Ratio']:.2f}, "
                 f"勝率={row['Win Rate']:.2%}, 盈利因子={row['Profit Factor']:.2f}")
    
    # 保存優化結果到Excel
    if save_excel:
        print("\n保存優化結果...")
        excel_path = os.path.join(results_dir, f'ma_optimization_results_{run_id}.xlsx')
        
        try:
            with pd.ExcelWriter(excel_path) as writer:
                results_df.to_excel(writer, sheet_name='All Results', index=False)
                
                # 創建包含最佳參數的摘要表
                summary_data = {
                    'Metric': ['Total PnL', 'Sharpe Ratio', 'Profit Factor', 'PnL/MaxDD'],
                    'MA Period': [best_params['total_pnl']['params'] if best_params['total_pnl']['params'] else None,
                                best_params['sharpe']['params'] if best_params['sharpe']['params'] else None,
                                best_params['profit_factor']['params'] if best_params['profit_factor']['params'] else None,
                                best_params['pnl_maxdd']['params'] if best_params['pnl_maxdd']['params'] else None],
                    'Value': [best_params['total_pnl']['value'],
                             best_params['sharpe']['value'],
                             best_params['profit_factor']['value'],
                             best_params['pnl_maxdd']['value']]
                }
                
                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='Best Parameters', index=False)
            
            print(f"優化結果已保存到 {excel_path}")
        except Exception as e:
            print(f"保存優化結果時出錯: {str(e)}")
    
    # 使用最佳參數執行最終回測（按總盈虧）
    if best_params['total_pnl']['params']:
        ma = best_params['total_pnl']['params']
        print(f"\n使用最佳參數執行最終回測 (MA={ma})...")
        
        final_run_id = f"{run_id}_best"
        final_metrics = backtest_ma_strategy(
            data_path=data_path,
            ma_period=ma,
            price_col=price_col,
            save_excel=save_excel,
            run_id=final_run_id,
            base_dir=base_dir,
            enhanced_charts=True,
            show_plots=show_plots
        )
    
    return best_params, results_df

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import os
import re
from datetime import datetime

def create_cumulative_comparison_chart(data_path, ma_periods=[3, 5, 10, 20, 30, 50], price_col='Close', 
                                       custom_path=r"C:\Users\wenyu\Desktop\trade\investment\python\scrapping\hydraulic jump\project",
                                       run_id=None):
    """
    只創建累積損益比較圖表，不顯示價格和交易信號
    
    Parameters:
    -----------
    data_path : str
        數據文件路徑（CSV或Excel）
    ma_periods : list
        要測試的MA週期列表
    price_col : str
        價格列名
    custom_path : str
        結果保存路徑
    run_id : str, optional
        運行ID，用於创建唯一的文件名
        
    Returns:
    --------
    tuple
        (chart_path, all_results_df)
    """
    # 生成運行ID（如果未提供）
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 創建保存目錄
    charts_dir = os.path.join(custom_path, "ma_charts")
    os.makedirs(charts_dir, exist_ok=True)
    print(f"圖表將保存到：{charts_dir}")
    
    # 設置結果目錄
    results_dir = os.path.join(custom_path, "results", f"ma_compare_{run_id}")
    os.makedirs(results_dir, exist_ok=True)
    
    # 加載數據
    print(f"從以下位置加載數據：{data_path}")
    if data_path.endswith('.xlsx') or data_path.endswith('.xls'):
        df = pd.read_excel(data_path)
    elif data_path.endswith('.csv'):
        df = pd.read_csv(data_path)
    else:
        raise ValueError("不支持的文件格式。請使用.xlsx或.csv")

    # 處理日期列
    date_columns = ['date', 'Date', 'datetime', 'Datetime', 'time', 'Time']
    date_col = None
    
    for col in date_columns:
        if col in df.columns:
            date_col = col
            break
    
    if date_col:
        # 轉換為datetime
        df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
        df.set_index(date_col, inplace=True)
        print(f"使用 '{date_col}' 作為日期索引")
    else:
        # 如果找不到日期列，嘗試將第一列轉換為日期
        try:
            first_col = df.columns[0]
            df[first_col] = pd.to_datetime(df[first_col], errors='coerce')
            df.set_index(first_col, inplace=True)
            print(f"使用第一列 '{first_col}' 作為日期索引")
        except:
            print("警告：未找到日期列。使用數字索引。")
            df = df.reset_index(drop=True)
            df.index = pd.date_range(start='2000-01-01', periods=len(df), freq='D')

    # 確保價格列存在
    if price_col not in df.columns:
        for col in df.columns:
            if col.lower() == price_col.lower():
                price_col = col
                print(f"使用 {price_col} 作為價格列")
                break
        else:
            # 如果找不到匹配的列，使用第一個數值列
            numeric_cols = df.select_dtypes(include=['number']).columns
            if len(numeric_cols) > 0:
                price_col = numeric_cols[0]
                print(f"價格列 '{price_col}' 未找到。改用 '{price_col}'。")
            else:
                raise ValueError("找不到合適的價格列")

    # 計算每日變動
    df['daily_change'] = df[price_col].diff()
    
    # 計算買入持有策略累積收益
    df['buyhold_cumulative'] = df['daily_change'].cumsum()
    
    # 創建一個大的圖表
    plt.figure(figsize=(16, 10))
    
    # 計算並繪製每個MA週期的累積收益
    all_results = {}
    dfs = {}
    
    for ma_period in ma_periods:
        # 計算MA
        df[f'MA{ma_period}'] = df[price_col].rolling(window=ma_period).mean()
        
        # 生成信號
        df[f'ma{ma_period}_position'] = 0
        
        # 交易信號：當價格穿過MA線時
        for i in range(1, len(df)):
            if df[price_col].iloc[i] > df[f'MA{ma_period}'].iloc[i] and df[price_col].iloc[i-1] <= df[f'MA{ma_period}'].iloc[i-1]:
                df.iloc[i, df.columns.get_loc(f'ma{ma_period}_position')] = 1  # 買入信號
            elif df[price_col].iloc[i] <= df[f'MA{ma_period}'].iloc[i] and df[price_col].iloc[i-1] > df[f'MA{ma_period}'].iloc[i-1]:
                df.iloc[i, df.columns.get_loc(f'ma{ma_period}_position')] = 0  # 賣出信號
            else:
                # 保持前一個倉位狀態
                df.iloc[i, df.columns.get_loc(f'ma{ma_period}_position')] = df.iloc[i-1, df.columns.get_loc(f'ma{ma_period}_position')]
        
        # 計算每日收益
        df[f'ma{ma_period}_daily_return'] = df['daily_change'] * df[f'ma{ma_period}_position'].shift(1).fillna(0)
        
        # 計算累積收益
        df[f'ma{ma_period}_cumulative'] = df[f'ma{ma_period}_daily_return'].cumsum()
        
        # 繪製累積收益線
        plt.plot(df.index, df[f'ma{ma_period}_cumulative'], linewidth=1.5, label=f'MA {ma_period}')
        
        # 保存結果用於後續分析
        all_results[ma_period] = df[f'ma{ma_period}_cumulative'].iloc[-1]
        
        # 保存當前的DataFrame
        dfs[ma_period] = df.copy()
    
    # 繪製買入持有策略的累積收益線
    plt.plot(df.index, df['buyhold_cumulative'], color='black', linestyle='--', linewidth=1.5, label='Buy and Hold')
    
    # 添加標題和標籤
    plt.title('Cumulative Profit Comparison (Points)', fontsize=16, fontweight='bold')
    plt.xlabel('Date', fontsize=12)
    plt.ylabel('Cumulative Profit (Points)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(loc='best', fontsize=10)
    
    # 設置日期格式
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    plt.gcf().autofmt_xdate()
    
    # 保存圖表
    comparison_chart_path = os.path.join(charts_dir, f'ma_cumulative_comparison_{run_id}.png')
    plt.savefig(comparison_chart_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"累積損益比較圖保存到：{comparison_chart_path}")
    
    # 創建結果DataFrame
    results_data = []
    for ma_period, final_pnl in all_results.items():
        results_data.append({
            'MA Period': ma_period,
            'Final Cumulative PnL': final_pnl,
            'Buy and Hold PnL': df['buyhold_cumulative'].iloc[-1],
            'Relative Performance': final_pnl - df['buyhold_cumulative'].iloc[-1]
        })
    
    results_df = pd.DataFrame(results_data)
    results_df = results_df.sort_values('Final Cumulative PnL', ascending=False)
    
    # 保存結果到Excel
    excel_path = os.path.join(results_dir, f'ma_comparison_results_{run_id}.xlsx')
    try:
        with pd.ExcelWriter(excel_path) as writer:
            results_df.to_excel(writer, sheet_name='MA Comparison', index=False)
            
            # 保存每個MA期間的最終結果
            for ma_period, ma_df in dfs.items():
                sheet_name = f'MA{ma_period}'
                # 只保存重要列以減小文件大小
                cols_to_save = [price_col, f'MA{ma_period}', f'ma{ma_period}_position', 
                               f'ma{ma_period}_daily_return', f'ma{ma_period}_cumulative']
                ma_df[cols_to_save].to_excel(writer, sheet_name=sheet_name)
        
        print(f"比較結果已保存到 {excel_path}")
    except Exception as e:
        print(f"保存Excel文件時出錯: {str(e)}")
    
    # 打印各MA週期的最終收益
    print("\n各MA週期最終累積收益 (從高到低):")
    for idx, row in results_df.iterrows():
        ma_period = row['MA Period']
        final_pnl = row['Final Cumulative PnL']
        rel_perf = row['Relative Performance']
        print(f"MA {ma_period}: {final_pnl:.2f} 點 (相對買入持有: {rel_perf:+.2f})")
    
    print(f"\n買入持有策略最終收益: {df['buyhold_cumulative'].iloc[-1]:.2f} 點")
    
    # 找出最佳MA週期
    best_ma = results_df.iloc[0]['MA Period']
    print(f"\n最佳MA週期: MA {best_ma} 產生 {results_df.iloc[0]['Final Cumulative PnL']:.2f} 點收益")
    
    return comparison_chart_path, results_df
