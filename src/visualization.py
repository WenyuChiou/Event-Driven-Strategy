import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.dates as mdates


def visualize_event_summary(data, analysis_results, output_path=None, show_plots=True):
    """
    Visualize a summary of event statistics without displaying detailed information for each event.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing the detected trading events.
    analysis_results : dict
        Analysis results from analyze_trading_events.
    output_path : str, optional
        The path where the visualization output will be saved.
    show_plots : bool, default=True
        Whether to display the plots.
    """
    # Filter data with events
    events = data[data['Event'] != 0]
    
    if len(events) == 0:
        print("No events to visualize")
        return
    
    # Create a multi-panel figure
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Distribution of event types (Long vs. Short)
    ax1 = axes[0, 0]
    event_types = ['Long', 'Short']
    event_counts = [analysis_results['long_events'], analysis_results['short_events']]
    event_colors = ['green', 'red']
    
    # Plot a bar chart
    bars = ax1.bar(event_types, event_counts, color=event_colors)
    
    # Add numeric labels on the bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width() / 2., height + 0.1,
                 f'{height}', ha='center', va='bottom')
    
    ax1.set_title("Comparison of Long and Short Event Counts", fontsize=14)
    ax1.set_xlabel("Event Type")
    ax1.set_ylabel("Event Count")
    ax1.grid(alpha=0.3)
    
    # 2. Profit/Loss distribution plot
    ax2 = axes[0, 1]
    sns.histplot(events['Profit_Loss_Points'], kde=True, ax=ax2, bins=20)
    ax2.axvline(0, color='red', linestyle='--')
    ax2.set_title("Profit and Loss Distribution", fontsize=14)
    ax2.set_xlabel("Profit/Loss Points")
    ax2.set_ylabel("Frequency")
    ax2.grid(alpha=0.3)
    
    # 3. Distribution of events by weekday
    ax3 = axes[1, 0]
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    day_translation = {'Monday': 'Monday', 'Tuesday': 'Tuesday', 'Wednesday': 'Wednesday', 
                       'Thursday': 'Thursday', 'Friday': 'Friday'}
    
    if 'by_day' in analysis_results:
        days_data = pd.DataFrame(list(analysis_results['by_day'].items()), columns=['Day', 'Count'])
        days_data = days_data[days_data['Day'].isin(days_order)]
        days_data['Day'] = pd.Categorical(days_data['Day'], categories=days_order, ordered=True)
        days_data = days_data.sort_values('Day')
        
        # Translate weekday names (if needed)
        days_data['Day_EN'] = days_data['Day'].map(day_translation)
        
        sns.barplot(x='Day_EN', y='Count', data=days_data, ax=ax3)
        ax3.set_title("Event Counts by Weekday", fontsize=14)
        ax3.set_xlabel("Weekday")
        ax3.set_ylabel("Event Count")
        
        # Add numeric labels
        for i, v in enumerate(days_data['Count']):
            ax3.text(i, v + 0.5, str(v), ha='center')
        
        ax3.grid(alpha=0.3)
    
    # 4. Distribution of events by hour
    ax4 = axes[1, 1]
    
    if 'by_hour' in analysis_results:
        hour_data = pd.DataFrame(list(analysis_results['by_hour'].items()), columns=['Hour', 'Count'])
        hour_data = hour_data.sort_values('Hour')
        
        sns.barplot(x='Hour', y='Count', data=hour_data, ax=ax4)
        ax4.set_title("Event Counts by Hour", fontsize=14)
        ax4.set_xlabel("Hour")
        ax4.set_ylabel("Event Count")
        
        # Add numeric labels
        for i, v in enumerate(hour_data['Count']):
            ax4.text(i, v + 0.1, str(v), ha='center')
        
        ax4.grid(alpha=0.3)
    
    # Add key metrics as a figure text
    plt.figtext(0.5, 0.01, 
                f"Total events: {analysis_results['total_events']} | Win rate: {analysis_results['win_rate']:.2%} | "
                f"Profit Factor: {analysis_results['profit_factor']:.2f} | Average Profit: {analysis_results['expectancy']:.2f}",
                ha="center", fontsize=14, bbox={"facecolor": "orange", "alpha": 0.2, "pad": 5})
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    
    # Save the plot if an output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    # Show the plot if requested
    if show_plots:
        plt.show()
    else:
        plt.close()
    return fig

def plot_price_with_event_markers(data, output_path=None, show_plots=True):
    """
    Plot price trends along with event markers without showing detailed information for each event.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing the detected trading events.
    output_path : str, optional
        The path where the visualization output will be saved.
    show_plots : bool, default=True
        Whether to display the plots.
    """
    fig, ax = plt.subplots(figsize=(14, 7))
    
    # Plot the closing price
    ax.plot(data['date'], data['close'], label="Close", color="blue", linewidth=1.2)
    
    # Plot the EMAs
    ax.plot(data['date'], data['EMA'], label=f"EMA (9)", color="purple", linestyle="--")
    ax.plot(data['date'], data['Long_EMA'], label=f"Long EMA (13)", color="orange")
    
    # Mark Long events
    long_events = data[data['Event'] == 1]
    if len(long_events) > 0:
        ax.scatter(long_events['date'], long_events['close'], color='green', 
                   marker='^', s=80, zorder=5, label="Long Event")
    
    # Mark Short events
    short_events = data[data['Event'] == -1]
    if len(short_events) > 0:
        ax.scatter(short_events['date'], short_events['close'], color='red', 
                   marker='v', s=80, zorder=5, label="Short Event")
    
    # Set title and labels
    ax.set_title("Price Trend with Event Markers", fontsize=16)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Prices", fontsize=12)
    ax.legend(loc='best')
    ax.grid(True, alpha=0.3)
    
    # Format the x-axis to display dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=5))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45)
    
    plt.tight_layout()
    
    # Save the plot if an output path is provided
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
    
    # Show the plot if requested
    if show_plots:
        plt.show()
    else:
        plt.close()

def create_summary_plot(feature_counts, top_features, event_stats):
    """
    Create a summary plot for feature selection and event selection.
    
    Parameters:
    -----------
    feature_counts : dict 
        A dictionary with the number of features at each stage (e.g., {'Original Features': 100, 'Final Features': 25}).
    top_features : list of tuples
        The top N important features and their importance scores (e.g., [(feature1, score1), (feature2, score2)]).
    event_stats : dict
        Event statistics including total events, long/short event counts, win rate, etc.
    """
    fig, axes = plt.subplots(2, 1, figsize=(12, 10))
    
    # Upper panel: Feature selection summary
    ax1 = axes[0]
    ax1.bar(feature_counts.keys(), feature_counts.values(), color=['blue', 'green', 'red'])
    ax1.set_title('Summary of Feature Selection', fontsize=14)
    ax1.set_ylabel('Feature Count')
    ax1.set_xlabel('Feature Selection Stage')
    
    # Add top features on the right side using a secondary axis
    ax1_right = ax1.twinx()
    feature_names = [f[0] for f in top_features]
    feature_scores = [f[1] for f in top_features]
    y_pos = range(len(feature_names))
    ax1_right.barh(y_pos, feature_scores, color='orange', alpha=0.7)
    ax1_right.set_yticks(y_pos)
    ax1_right.set_yticklabels(feature_names)
    ax1_right.set_ylabel('Top 10 Features')
    ax1_right.set_xlabel('Importance Score')
    
    # Lower panel: Event selection summary
    ax2 = axes[1]
    event_types = ['Long events', 'Short events']
    event_counts = [event_stats['long_events'], event_stats['short_events']]
    ax2.bar(event_types, event_counts, color=['green', 'red'])
    ax2.set_title('Summary of Event Selection', fontsize=14)
    ax2.set_ylabel('Number of Events')
    ax2.set_xlabel('Event Type')
    
    # Add a text annotation with key metrics
    text_info = (f"Total events: {event_stats['total_events']}\n"
                 f"Win rate: {event_stats['win_rate']:.2%}\n"
                 f"Profit factor: {event_stats['profit_factor']:.2f}\n"
                 f"Expectancy: {event_stats['expectancy']:.2f} points")
    
    plt.figtext(0.7, 0.35, text_info, fontsize=12, 
                bbox=dict(facecolor='lightyellow', alpha=0.5))
    
    plt.tight_layout()
    return fig







def create_monthly_performance_heatmap(backtester, save_path=None, show_plot=True):
    """
    Create a visualization with the Advanced Trading Performance Analytics
    showing trade distribution, drawdown, monthly returns, and win/loss analysis.
    
    Parameters:
    -----------
    backtester : Backtester
        Backtester instance with results
    save_path : str, optional
        Path to save the figure
    show_plot : bool, default=True
        Whether to display the plot
    
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from matplotlib.dates import DateFormatter
    import pandas as pd
    
    # Set up aesthetics
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    
    # Get the results and metrics
    results = backtester.results
    all_metrics = backtester.get_all_metrics()
    metrics = all_metrics['all']
    
    # Make sure we have the data we need
    if results is None or len(results) == 0:
        print("No results available in the backtester")
        return None
    
    # Create figure for Advanced Trading Performance Analytics
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    
    # 1. Trade P&L Distribution (top left)
    ax1 = axes[0, 0]
    
    # Get trade data
    long_trades = results[results['Long_PnL'] != 0]['Long_PnL']
    short_trades = results[results['Short_PnL'] != 0]['Short_PnL']
    
    # Create bins for histogram
    all_pnl = pd.concat([long_trades, short_trades])
    
    if len(all_pnl) > 0:
        min_pnl = all_pnl.min()
        max_pnl = all_pnl.max()
        
        # Create reasonable bin size
        range_pnl = max_pnl - min_pnl
        bin_size = max(1, range_pnl / 20)  # Use at least 20 bins
        
        # Create bins with range from min to max
        bins = np.arange(min_pnl - bin_size/2, max_pnl + bin_size, bin_size)
        
        # Plot histograms
        if len(long_trades) > 0:
            ax1.hist(long_trades, bins=bins, alpha=0.7, color='#2ca02c', label='Long Trades')
        
        if len(short_trades) > 0:
            ax1.hist(short_trades, bins=bins, alpha=0.7, color='#d62728', label='Short Trades')
    
    # Set labels and title
    ax1.set_title('Trade P&L Distribution', fontweight='bold')
    ax1.set_xlabel('P&L Points', fontweight='bold')
    ax1.set_ylabel('Frequency', fontweight='bold')
    ax1.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 2. Drawdown Analysis (top right)
    ax2 = axes[0, 1]
    
    # Calculate drawdown
    if 'date' in results.columns and 'Cumulative_PnL' in results.columns:
        cumulative_pnl = results['Cumulative_PnL']
        peak = cumulative_pnl.expanding(min_periods=1).max()
        drawdown = (peak - cumulative_pnl)
        
        # Plot drawdown
        ax2.fill_between(results['date'], 0, drawdown, color='#ff7f0e', alpha=0.5)
        
        # Mark maximum drawdown
        max_dd_idx = drawdown.idxmax() if not drawdown.empty else None
        
        if max_dd_idx is not None:
            max_dd = drawdown[max_dd_idx]
            max_dd_date = results.iloc[max_dd_idx]['date']
            
            ax2.scatter([max_dd_date], [max_dd], color='red', s=100, zorder=5)
            ax2.annotate(f'Max DD: {max_dd:.2f}', 
                        xy=(max_dd_date, max_dd),
                        xytext=(0, 10),
                        textcoords="offset points",
                        ha='center', fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.3', fc='white', ec='red', alpha=0.8))
    
    # Set labels and title
    ax2.set_title('Drawdown Analysis', fontweight='bold')
    ax2.set_xlabel('Date', fontweight='bold')
    ax2.set_ylabel('Drawdown (Points)', fontweight='bold')
    
    # Format date on x-axis
    try:
        date_format = DateFormatter('%Y-%m-%d')
        ax2.xaxis.set_major_formatter(date_format)
        plt.setp(ax2.get_xticklabels(), rotation=45, ha="right")
    except Exception:
        pass
        
    ax2.grid(True, alpha=0.3)
    
    # 3. Monthly Returns Heatmap (bottom left)
    ax3 = axes[1, 0]
    
    # Create monthly returns heatmap
    if 'date' in results.columns and 'Trade_PnL' in results.columns:
        # Add year and month columns
        results_copy = results.copy()
        results_copy['year'] = pd.to_datetime(results_copy['date']).dt.year
        results_copy['month'] = pd.to_datetime(results_copy['date']).dt.month
        
        # Group by year and month and sum the P&L
        monthly_pnl = results_copy.groupby(['year', 'month'])['Trade_PnL'].sum().unstack()
        
        # Create heatmap
        if not monthly_pnl.empty:
            try:
                # Use seaborn for heatmap
                sns.heatmap(monthly_pnl, annot=True, fmt=".1f", cmap='RdYlGn', center=0,
                           linewidths=.5, ax=ax3, cbar_kws={'label': 'P&L Points'})
                ax3.set_title('Monthly Returns Heatmap', fontweight='bold')
                ax3.set_xlabel('Month', fontweight='bold')
                ax3.set_ylabel('Year', fontweight='bold')
            except Exception as e:
                # Fallback if heatmap fails
                ax3.text(0.5, 0.5, f"Could not create heatmap: {str(e)}", 
                        ha='center', va='center', transform=ax3.transAxes)
    else:
        ax3.text(0.5, 0.5, "Insufficient data for monthly returns", 
                ha='center', va='center', transform=ax3.transAxes)
    
    # 4. Win/Loss Trade Analysis (bottom right)
    ax4 = axes[1, 1]
    
    # Get win/loss counts for long and short trades
    long_win = metrics.get('Long Winning Trades', 0)
    long_loss = metrics.get('Long Losing Trades', 0)
    short_win = metrics.get('Short Winning Trades', 0)
    short_loss = metrics.get('Short Losing Trades', 0)
    
    # Prepare data for grouped bar chart
    categories = ['Long', 'Short']
    win_data = [long_win, short_win]
    loss_data = [long_loss, short_loss]
    
    # Position of bars on x-axis
    x = np.arange(len(categories))
    width = 0.35
    
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
    
    # Set labels and title
    ax4.set_title('Win/Loss Trade Analysis', fontweight='bold')
    ax4.set_ylabel('Number of Trades', fontweight='bold')
    ax4.set_xticks(x)
    ax4.set_xticklabels(categories)
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    # Add main title
    fig.suptitle('Advanced Trading Performance Analytics', fontsize=16, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved Advanced Trading Performance Analytics to: {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def create_strategy_comparison_visualization(filtered_strategies, filtered_backtesters, 
                                            save_path=None, show_plot=True):
    """
    Create a visualization comparing multiple filtered strategies.
    
    Parameters:
    -----------
    filtered_strategies : pd.DataFrame
        DataFrame with information about filtered strategies
    filtered_backtesters : dict
        Dictionary of backtester instances keyed by (long_threshold, short_threshold)
    save_path : str, optional
        Path to save the figure
    show_plot : bool, default=True
        Whether to display the plot
    
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from matplotlib.dates import DateFormatter
    
    # Set up aesthetics
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    
    # Check if we have enough data
    if filtered_strategies is None or len(filtered_strategies) == 0:
        print("No filtered strategies to compare")
        return None
    
    # Limit to top 5 strategies
    top_n = min(5, len(filtered_strategies))
    top_strategies = filtered_strategies.head(top_n)
    
    # Create figure with 2x2 layout
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    
    # 1. Sharpe Ratio Comparison (top left)
    ax1 = axes[0, 0]
    
    # Create bars for Sharpe ratio
    sharpe_values = top_strategies['Sharpe Ratio'].values
    
    # Create labels with strategy parameters
    strategy_labels = [
        f"L:{row['Long Threshold']:.6f}\nS:{row['Short Threshold']:.6f}"
        for _, row in top_strategies.iterrows()
    ]
    
    # Create bars with colormap
    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0.1, 0.9, top_n))
    
    sharpe_bars = ax1.barh(range(top_n), sharpe_values, color=colors)
    
    # Add value labels
    for bar in sharpe_bars:
        width = bar.get_width()
        ax1.text(width + 0.02, bar.get_y() + bar.get_height()/2,
                f'{width:.2f}', va='center', fontweight='bold')
    
    # Set labels and title
    ax1.set_yticks(range(top_n))
    ax1.set_yticklabels(strategy_labels)
    ax1.set_title('Sharpe Ratio Comparison', fontweight='bold')
    ax1.set_xlabel('Sharpe Ratio', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. Total P&L Comparison (top right)
    ax2 = axes[0, 1]
    
    # Create bars for Total P&L
    pnl_values = top_strategies['Total PnL'].values
    
    # Create bars
    pnl_bars = ax2.barh(range(top_n), pnl_values, 
                       color=[plt.cm.RdYlGn(0.7) if v >= 0 else plt.cm.RdYlGn(0.3) for v in pnl_values])
    
    # Add value labels
    for bar in pnl_bars:
        width = bar.get_width()
        ax2.text(width + (0.02 * np.sign(width)), bar.get_y() + bar.get_height()/2,
                f'{width:.2f}', va='center', fontweight='bold')
    
    # Set labels and title
    ax2.set_yticks(range(top_n))
    ax2.set_yticklabels(strategy_labels)
    ax2.set_title('Total P&L Comparison', fontweight='bold')
    ax2.set_xlabel('Total P&L (Points)', fontweight='bold')
    ax2.axvline(x=0, color='black', linestyle='--', alpha=0.5)
    ax2.grid(True, alpha=0.3)
    
    # 3. Win Rate Comparison (bottom left)
    ax3 = axes[1, 0]
    
    # Create bars for Win Rate
    win_rate_values = top_strategies['Win Rate'].values * 100  # Convert to percentage
    
    # Create bars
    win_rate_bars = ax3.barh(range(top_n), win_rate_values,
                           color=plt.cm.RdYlGn(np.clip(win_rate_values/100, 0.3, 0.7)))
    
    # Add value labels
    for bar in win_rate_bars:
        width = bar.get_width()
        ax3.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                f'{width:.1f}%', va='center', fontweight='bold')
    
    # Set labels and title
    ax3.set_yticks(range(top_n))
    ax3.set_yticklabels(strategy_labels)
    ax3.set_title('Win Rate Comparison', fontweight='bold')
    ax3.set_xlabel('Win Rate (%)', fontweight='bold')
    
    # Add reference line at 50%
    ax3.axvline(x=50, color='gray', linestyle='--', alpha=0.7)
    ax3.text(50, -0.5, '50%', ha='center', va='top', color='gray', fontsize=10)
    
    ax3.grid(True, alpha=0.3)
    
    # 4. Profit Factor Comparison (bottom right)
    ax4 = axes[1, 1]
    
    # Create bars for Profit Factor
    profit_factor_values = np.minimum(top_strategies['Profit Factor'].values, 5)  # Cap at 5 for visualization
    actual_pf_values = top_strategies['Profit Factor'].values
    
    # Create bars
    pf_bars = ax4.barh(range(top_n), profit_factor_values,
                      color=plt.cm.RdYlGn(np.clip(profit_factor_values/5, 0.3, 0.7)))
    
    # Add value labels
    for i, bar in enumerate(pf_bars):
        width = bar.get_width()
        display_text = f'{actual_pf_values[i]:.2f}'
        if actual_pf_values[i] > 5:
            display_text += f' (capped)'
        ax4.text(width + 0.1, bar.get_y() + bar.get_height()/2,
                display_text, va='center', fontweight='bold')
    
    # Set labels and title
    ax4.set_yticks(range(top_n))
    ax4.set_yticklabels(strategy_labels)
    ax4.set_title('Profit Factor Comparison', fontweight='bold')
    ax4.set_xlabel('Profit Factor', fontweight='bold')
    
    # Add reference line at 1.0
    ax4.axvline(x=1.0, color='gray', linestyle='--', alpha=0.7)
    ax4.text(1.0, -0.5, '1.0', ha='center', va='top', color='gray', fontsize=10)
    
    ax4.grid(True, alpha=0.3)
    
    # Add title
    fig.suptitle('Filtered Strategies Comparison', fontsize=16, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved Strategy Comparison to: {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def create_cumulative_returns_plot(filtered_results, backtester, df_features, 
                                  save_path=None, show_plot=True):
    """
    Create an enhanced cumulative returns plot for the top filtered strategies.
    
    Parameters:
    -----------
    filtered_results : pd.DataFrame
        DataFrame with filtered results
    backtester : Backtester
        Backtester instance
    df_features : pd.DataFrame
        DataFrame with feature data (must include 'date')
    save_path : str, optional
        Path to save the figure
    show_plot : bool, default=True
        Whether to display the plot
    
    Returns:
    --------
    matplotlib.figure.Figure
        The figure object
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    from matplotlib.dates import DateFormatter
    
    # Set up aesthetics
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    
    # Check if we have enough data
    if filtered_results is None or len(filtered_results) == 0:
        print("No filtered results to plot")
        return None
    
    if 'date' not in df_features.columns:
        print("No date column in feature data")
        return None
    
    # Create figure
    fig, ax = plt.subplots(figsize=(16, 8))
    
    # Limit to top 5 strategies
    top_n = min(5, len(filtered_results))
    top_strategies = filtered_results.head(top_n)
    
    # Create a colormap
    cmap = plt.cm.viridis
    colors = cmap(np.linspace(0.1, 0.9, top_n))
    
    # Plot cumulative returns for each strategy
    for i, (_, row) in enumerate(top_strategies.iterrows()):
        # Get parameters
        long_threshold = row['Long Threshold']
        short_threshold = row['Short Threshold']
        sharpe_ratio = row['Sharpe Ratio']