"""
Enhanced version of backtesting_example.py that maintains the original structure
but adds improved functionality for long/short analysis and automatic file saving.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.dates import DateFormatter
from src.visualization import *
import joblib
from datetime import datetime
import logging
import sys
from src.feature_engineering import calculate_features, FeatureEngineeringWrapper
from src.backtesting import Backtester, ProbabilityThresholdStrategy, filter_and_compare_strategies

def configure_logging(log_file="backtest.log", log_level=logging.INFO):
    """
    Configure logging system
    
    Parameters:
    -----------
    log_file : str
        Path to log file
    log_level : logging level
        Logging level (default: logging.INFO)
        
    Returns:
    --------
    logger : logging.Logger
        Configured logger instance
    """
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

# Initialize logger
logger = logging.getLogger(__name__)

def setup_directories(base_dir=None):
    """
    Setup directory structure for backtesting
    
    Parameters:
    -----------
    base_dir : str, optional
        Base directory path. If None, uses directory structure relative to this file.
        
    Returns:
    --------
    dict
        Dictionary containing paths to all directories
    """
    if base_dir is None:
        # Default: Use directory of this file as base
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    directories = {
        'BASE_DIR': base_dir,
        'DATA_DIR': os.path.join(base_dir, 'data'),
        'MODELS_DIR': os.path.join(base_dir, 'models'),
        'RESULTS_DIR': os.path.join(base_dir, 'results'),
        'BACKTEST_RESULTS_DIR': os.path.join(base_dir, 'results', 'backtests'),
        'FIGURES_DIR': os.path.join(base_dir, 'results', 'backtests', 'figures')
    }
    
    # Ensure directories exist
    for directory in directories.values():
        os.makedirs(directory, exist_ok=True)
    
    return directories

def backtest_model(model_path, feature_path, scaler_path, data_path, run_id=None, 
                  base_dir=None, save_excel=True, separate_long_short=True,
                  enhanced_charts=True, show_plots=False):
    """
    Enhanced version of backtest_model with improved long/short analysis and automatic file saving
    
    Parameters:
    -----------
    model_path : str
        Path to model file
    feature_path : str
        Path to feature list file
    scaler_path : str
        Path to scaler file
    data_path : str
        Path to data file
    run_id : str, optional
        Unique identifier for this backtest run, used in file naming
    base_dir : str, optional
        Base directory path
    save_excel : bool, default=True
        Whether to save detailed Excel results
    separate_long_short : bool, default=True
        Whether to perform separate analysis for long and short trades
    enhanced_charts : bool, default=True
        Whether to use enhanced chart visualization
    show_plots : bool, default=True
        Whether to display plots (set to False for batch processing)
        
    Returns:
    --------
    dict
        Backtest metrics
    """
    # Configure logging if not already configured
    global logger
    if not logger.handlers:
        logger = configure_logging()
    
    # Setup directories
    dirs = setup_directories(base_dir)
    
    # If no run_id provided, generate one based on timestamp
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create run-specific figures directory
    run_figures_dir = os.path.join(dirs['FIGURES_DIR'], run_id)
    os.makedirs(run_figures_dir, exist_ok=True)
    
    # 1. Load model and feature engineering components
    logger.info(f"Loading model: {model_path}")
    model = joblib.load(model_path)
    
    logger.info(f"Loading scaler: {scaler_path}")
    scaler = joblib.load(scaler_path)
    
    logger.info(f"Loading feature list: {feature_path}")
    remove_column_name = ['date', 'Profit_Loss_Points', 'Event', 'Label']
    feature_engineering = FeatureEngineeringWrapper.load_features(
        feature_path, remove_column_name=remove_column_name, scaler=scaler
    )
    
    # 2. Read backtest data
    logger.info(f"Reading backtest data: {data_path}")
    file_ext = os.path.splitext(data_path)[1].lower()
    
    if file_ext == '.xlsx' or file_ext == '.xls':
        df = pd.read_excel(data_path)
    elif file_ext == '.csv':
        df = pd.read_csv(data_path)
    else:
        raise ValueError(f"Unsupported file extension: {file_ext}")
        
    df['date'] = pd.to_datetime(df['date'])
    
    # Ensure numeric columns are float64 type
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = df[col].astype('float64')
    
    df = df.sort_values(by='date').reset_index(drop=True)
    
    # 3. Calculate features
    logger.info("Calculating technical indicators...")
    df_features, _ = calculate_features(df, scaler=scaler)
    
    # 4. Feature engineering transformation
    logger.info("Transforming features...")
    X = feature_engineering.transform(df_features)
    
    # 5. Model prediction
    logger.info("Running model predictions...")
    probabilities = model.predict_proba(X)
    
    # Add prediction probabilities to data
    df_features['Long_Probability'] = probabilities[:, 2]  # P(1) long
    df_features['Short_Probability'] = probabilities[:, 0]  # P(-1) short
    df_features['Neutral_Probability'] = probabilities[:, 1]  # P(0) neutral
    
    # Plot probability distribution
    logger.info("Plotting probability distribution...")
    probability_histogram_path = os.path.join(run_figures_dir, f"probability_histogram_{run_id}.png")
    plot_probability_histogram(df_features, save_path=probability_histogram_path, show_plot=show_plots)

    # 6. Parameter optimization
    logger.info("\nStarting backtest parameter optimization...")
    backtester = Backtester()

    # Define parameter grid
    param_grid = {
        'long_threshold': np.linspace(0.00, 0.01, 50),
        'short_threshold': np.linspace(0.00, 0.005, 50),
    }

    best_params, best_performance, results_df = backtester.optimize_strategy(
        df_features, ProbabilityThresholdStrategy, param_grid, metric='sharpe_ratio'
    )

    logger.info(f"Best parameters: {best_params}")
    logger.info(f"Best performance (Sharpe Ratio): {best_performance:.4f}")

    # 7. Run backtest with best parameters
    logger.info("\nRunning backtest with best parameters...")
    best_strategy = ProbabilityThresholdStrategy(**best_params)
    backtester.run(df_features, best_strategy)

    # 8. Calculate backtest metrics - ENHANCED to include separate long/short metrics
    if separate_long_short:
        all_metrics = backtester.get_all_metrics()
        
        # Display metrics for all trade types
        for trade_type, metrics in all_metrics.items():
            logger.info(f"\n=== {trade_type.upper()} TRADE METRICS ===")
            for metric, value in metrics.items():
                if isinstance(value, (int, float)):
                    logger.info(f"  {metric}: {value:.4f}")
                else:
                    logger.info(f"  {metric}: {value}")
                    
        # Use combined metrics for overall results
        metrics = all_metrics['all']
    else:
        # Use standard metrics calculation for backward compatibility
        metrics = backtester.calculate_metrics()
        logger.info("Backtest metrics:")
        for metric, value in metrics.items():
            if isinstance(value, (int, float)):
                logger.info(f"  {metric}: {value:.4f}")
            else:
                logger.info(f"  {metric}: {value}")

    # 9. Export results to Excel if requested
    if save_excel:
        # Create path for Excel results
        excel_path = os.path.join(dirs['BACKTEST_RESULTS_DIR'], f"backtest_results_{run_id}.txt")
        
        # Export using the standard format for backward compatibility
        if hasattr(backtester, 'export_results_to_txt'):
            # Use enhanced export function if available
            backtester.export_results_to_txt(excel_path)
        else:
            # Fallback to basic Excel export
            with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
                # Save results DataFrame
                backtester.results.to_excel(writer, sheet_name='Results', index=False)
                
                # Save metrics
                metrics_df = pd.DataFrame({
                    'Metric': list(metrics.keys()),
                    'Value': list(metrics.values())
                })
                metrics_df.to_excel(writer, sheet_name='Metrics', index=False)
                
                # Save optimization results
                results_df.to_excel(writer, sheet_name='Optimization', index=False)
        
        logger.info(f"Results exported to: {excel_path}")

    # 10. Filter and compare strategy combinations
    logger.info("\nFiltering and comparing strategy combinations based on criteria...")

    # Calculate PnL/MaxDD ratio if not already in results
    if 'PnL/MaxDD' not in results_df.columns and 'Max Drawdown' in results_df.columns:
        results_df['PnL/MaxDD'] = results_df.apply(
            lambda x: x['Total PnL'] / x['Max Drawdown'] if x['Max Drawdown'] > 0 else float('inf'),
            axis=1
        )

    # Define filtering criteria
    filter_criteria = {
        'min_total_pnl': 0
        ,             # Total PnL > 0
        'min_sharpe': 0,                # Sharpe ratio > 0
        'min_pnl_drawdown_ratio': 0,    # PnL/Max drawdown > 0
        'max_trades': 5000,             # Total trades < 5000
        'min_profit_factor': 1          # Profit factor > 1.0
    }

    # Extract parameter combinations from results DataFrame
    threshold_pairs = []
    for _, row in results_df.iterrows():
        threshold_pairs.append((row['long_threshold'], row['short_threshold']))

    # Run filter_and_compare_strategies function
    filtered_backtesters, positive_results, cumulative_returns = filter_and_compare_strategies(
        df_features,
        threshold_pairs=threshold_pairs,
        filter_criteria=filter_criteria,
        holding_period=backtester.profit_loss_window,
        save_dir=run_figures_dir
    )

    logger.info(f"Found {len(positive_results)} parameter combinations meeting the criteria")

    # Create a filtered strategies directory for the filtered results
    filtered_dir = os.path.join(run_figures_dir, "filtered_strategies")
    os.makedirs(filtered_dir, exist_ok=True)

    if len(positive_results) > 0:
        # Display top 10 best results
        logger.info("\nTop 10 best parameter combinations:")
        display_columns = ['Long Threshold', 'Short Threshold', 'Total PnL', 'Sharpe Ratio', 
                       'Win Rate', 'Profit Factor', 'Max Drawdown', 'PnL/MaxDD', 'Total Trades']
        pd.set_option('display.precision', 6)
        logger.info("\n" + positive_results[display_columns].head(10).to_string(index=False))
        
        # Create visualizations for filtered strategies
        logger.info("\nCreating visualizations for filtered strategies...")
        visualization_paths = filter_and_visualize_strategies(
            positive_results,
            backtester,
            df_features,
            filtered_dir,
            show_plots=show_plots
        )
        
        logger.info(f"Filtered strategy visualizations saved to: {filtered_dir}")
        
        # Create cumulative return comparison chart
        create_cumulative_return_comparison(
            df_features, 
            positive_results, 
            backtester.profit_loss_window,
            save_dir=run_figures_dir,
            show_plot=show_plots
        )
    else:
        logger.info("No parameter combinations meet the filtering criteria. Consider adjusting the filter.")

    # 11. Plot backtest results - ENHANCED with better visuals if requested
    backtest_title = f"Backtest Results (Best Parameters: Long Threshold={best_params['long_threshold']:.6f}, Short Threshold={best_params['short_threshold']:.6f})"
    results_plot_path = os.path.join(run_figures_dir, f"backtest_results_{run_id}.png")
    comparison_plot_path = os.path.join(run_figures_dir, f"trade_comparison_{run_id}.png")
    
    if enhanced_charts and hasattr(backtester, 'plot_results'):
        # Use enhanced plotting if available
        backtester.plot_results(
            title=backtest_title, 
            save_path=results_plot_path,
            show_plot=show_plots
        )
        
        # Plot trade comparison with enhanced visuals
        if hasattr(backtester, 'plot_trade_comparison'):
            backtester.plot_trade_comparison(
                save_path=comparison_plot_path,
                show_plot=show_plots
            )
            
        # Add advanced analysis chart if available
        if hasattr(backtester, 'plot_advanced_analysis'):
            advanced_chart_path = os.path.join(run_figures_dir, f"advanced_analysis_{run_id}.png")
            backtester.plot_advanced_analysis(
                save_path=advanced_chart_path,
                show_plot=show_plots
            )
    else:
        # Fall back to standard plotting
        backtester.plot_results(title=backtest_title, save_path=results_plot_path)
        backtester.plot_trade_comparison(save_path=comparison_plot_path)

    # 12. Save backtest results
    results_path = os.path.join(dirs['BACKTEST_RESULTS_DIR'], f"optimization_results_{run_id}.xlsx")
    results_df.to_excel(results_path, index=False)
    logger.info(f"Parameter optimization results saved to: {results_path}")

    # 13. Save best parameters and metrics
    params_path = os.path.join(dirs['BACKTEST_RESULTS_DIR'], f"best_params_{run_id}.txt")
    with open(params_path, 'w') as f:
        f.write(f"Long Threshold: {best_params['long_threshold']}\n")
        f.write(f"Short Threshold: {best_params['short_threshold']}\n")
        f.write("\nBacktest Metrics:\n")
        for metric, value in metrics.items():
            f.write(f"{metric}: {value}\n")
        
        # Add information about filtered strategies
        f.write(f"\nStrategies meeting criteria: {len(positive_results)}\n")
        if len(positive_results) > 0:
            f.write("\nTop three strategy parameters:\n")
            for i, row in positive_results.head(3).iterrows():
                f.write(f"Rank {i+1}: Long Threshold={row['Long Threshold']:.6f}, Short Threshold={row['Short Threshold']:.6f}, "
                      f"Sharpe Ratio={row['Sharpe Ratio']:.4f}, Profit Factor={row['Profit Factor']:.4f}\n")

    logger.info(f"Best parameters saved to: {params_path}")

    # 14. Interactive mode only: offer to apply second-best parameters 
    # (Keep this feature from the original function)
    if len(positive_results) > 1 and sys.stdout.isatty() and show_plots:
        second_best = positive_results.iloc[1]
        logger.info(f"\nSecond-best parameter combination: Long Threshold={second_best['Long Threshold']:.6f}, Short Threshold={second_best['Short Threshold']:.6f}")
        logger.info(f"Sharpe Ratio: {second_best['Sharpe Ratio']:.4f}, Profit Factor: {second_best['Profit Factor']:.4f}")
        
        try:
            choice = input("Apply second-best parameters for backtest? (y/n): ")
            if choice.lower() == 'y':
                second_best_params = {
                    'long_threshold': second_best['Long Threshold'],
                    'short_threshold': second_best['Short Threshold'],
                    'holding_period': best_params.get('holding_period', 3)
                }
                second_best_strategy = ProbabilityThresholdStrategy(**second_best_params)
                backtester.run(df_features, second_best_strategy)
                
                second_best_title = f"Second-Best Strategy Results (Long Threshold={second_best_params['long_threshold']:.6f}, Short Threshold={second_best_params['short_threshold']:.6f})"
                second_best_path = os.path.join(run_figures_dir, f"second_best_results_{run_id}.png")
                
                if enhanced_charts and hasattr(backtester, 'plot_results'):
                    backtester.plot_results(
                        title=second_best_title, 
                        save_path=second_best_path,
                        show_plot=show_plots
                    )
                else:
                    backtester.plot_results(title=second_best_title, save_path=second_best_path)
        except EOFError:
            # Handle non-interactive execution
            pass
        
    return metrics


def run_comprehensive_backtest(model_path, feature_path, scaler_path, training_data_path, 
                              validation_data_path, base_dir=None, save_excel=True,
                              separate_long_short=True, enhanced_charts=True, show_plots=True):
    """
    Enhanced comprehensive backtest on both training and validation data
    
    Parameters:
    -----------
    model_path : str
        Path to model file
    feature_path : str
        Path to feature list file
    scaler_path : str
        Path to scaler file
    training_data_path : str
        Path to training data file
    validation_data_path : str
        Path to validation data file
    base_dir : str, optional
        Base directory path
    save_excel : bool, default=True
        Whether to save detailed Excel results
    separate_long_short : bool, default=True
        Whether to perform separate analysis for long and short trades
    enhanced_charts : bool, default=True
        Whether to use enhanced chart visualization
    show_plots : bool, default=True
        Whether to display plots (set to False for batch processing)
    
    Returns:
    --------
    tuple
        (training_metrics, validation_metrics)
    """
    # Configure logging if not already configured
    global logger
    if not logger.handlers:
        logger = configure_logging()
    
    # Setup directories
    dirs = setup_directories(base_dir)
    
    # Generate run ID for this comprehensive backtest
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"Starting comprehensive backtest (Run ID: {run_id})")
    
    # Backtest on training data
    logger.info("\n===== BACKTEST ON TRAINING DATA =====")
    training_metrics = backtest_model(
        model_path, 
        feature_path, 
        scaler_path, 
        training_data_path,
        run_id=f"{run_id}_training",
        base_dir=base_dir,
        save_excel=save_excel,
        separate_long_short=separate_long_short,
        enhanced_charts=enhanced_charts,
        show_plots=show_plots
    )
    
    # Backtest on validation data
    logger.info("\n===== BACKTEST ON VALIDATION DATA =====")
    validation_metrics = backtest_model(
        model_path, 
        feature_path, 
        scaler_path, 
        validation_data_path,
        run_id=f"{run_id}_validation",
        base_dir=base_dir,
        save_excel=save_excel,
        separate_long_short=separate_long_short,
        enhanced_charts=enhanced_charts,
        show_plots=show_plots
    )
    
    # Compare metrics - Create a comparison chart
    logger.info("\n===== COMPARISON OF TRAINING VS VALIDATION METRICS =====")
    
    # Convert metrics to DataFrames for easier comparison
    train_df = pd.DataFrame({
        'Metric': list(training_metrics.keys()),
        'Training': [training_metrics[m] for m in training_metrics.keys()]
    })
    
    valid_df = pd.DataFrame({
        'Metric': list(validation_metrics.keys()),
        'Validation': [validation_metrics[m] for m in validation_metrics.keys()]
    })
    
    # Merge and calculate differences
    metrics_comparison = pd.merge(train_df, valid_df, on='Metric', how='outer')
    metrics_comparison['Difference'] = metrics_comparison['Validation'] - metrics_comparison['Training']
    metrics_comparison['% Change'] = (metrics_comparison['Difference'] / metrics_comparison['Training']) * 100
    
    # Format numeric columns
    for col in ['Training', 'Validation', 'Difference']:
        metrics_comparison[col] = metrics_comparison[col].apply(
            lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x
        )
    
    metrics_comparison['% Change'] = metrics_comparison['% Change'].apply(
        lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) else "N/A"
    )
    
    # Display and save comparison
    logger.info("\n" + metrics_comparison.to_string(index=False))
    
    comparison_path = os.path.join(dirs['BACKTEST_RESULTS_DIR'], f"metrics_comparison_{run_id}.xlsx")
    metrics_comparison.to_excel(comparison_path, index=False)
    logger.info(f"Metrics comparison saved to: {comparison_path}")
    
    # Create a visual comparison chart for key metrics
    if enhanced_charts:
        create_comparison_chart(training_metrics, validation_metrics, 
                               save_path=os.path.join(dirs['FIGURES_DIR'], f"metrics_comparison_{run_id}.png"),
                               show_plot=show_plots)
    
    return training_metrics, validation_metrics


def simulate_real_time_trading(model_path, feature_path, scaler_path, data_path, 
                               long_threshold, short_threshold, run_id=None, base_dir=None, 
                               save_excel=True, enhanced_charts=True, show_plots=True):
    """
    Enhanced version of simulate_real_time_trading with improved visualization and analysis
    
    Parameters:
    -----------
    model_path : str
        Path to model file
    feature_path : str
        Path to feature list file
    scaler_path : str
        Path to scaler file
    data_path : str
        Path to data file
    long_threshold : float
        Long probability threshold
    short_threshold : float
        Short probability threshold
    run_id : str, optional
        Unique identifier for this simulation run, used in file naming
    base_dir : str, optional
        Base directory path
    save_excel : bool, default=True
        Whether to save detailed Excel results
    enhanced_charts : bool, default=True
        Whether to use enhanced chart visualization
    show_plots : bool, default=True
        Whether to display plots (set to False for batch processing)
        
    Returns:
    --------
    pd.DataFrame
        Trade records
    """
    # Configure logging if not already configured
    global logger
    if not logger.handlers:
        logger = configure_logging()
    
    # Setup directories
    dirs = setup_directories(base_dir)
    
    # If no run_id provided, generate one based on timestamp
    if run_id is None:
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create run-specific figures directory
    run_figures_dir = os.path.join(dirs['FIGURES_DIR'], f"simulation_{run_id}")
    os.makedirs(run_figures_dir, exist_ok=True)
    
    # 1. Load model and feature engineering components
    logger.info(f"Loading model: {model_path}")
    model = joblib.load(model_path)
    
    logger.info(f"Loading scaler: {scaler_path}")
    scaler = joblib.load(scaler_path)
    
    logger.info(f"Loading feature list: {feature_path}")
    remove_column_name = ['date', 'Profit_Loss_Points', 'Event', 'Label']
    feature_engineering = FeatureEngineeringWrapper.load_features(
        feature_path, remove_column_name=remove_column_name, scaler=scaler
    )
    
    # 2. Read backtest data
    logger.info(f"Reading simulation data: {data_path}")
    file_ext = os.path.splitext(data_path)[1].lower()
    
    if file_ext == '.xlsx' or file_ext == '.xls':
        df = pd.read_excel(data_path)
    elif file_ext == '.csv':
        df = pd.read_csv(data_path)
    else:
        raise ValueError(f"Unsupported file extension: {file_ext}")
        
    df['date'] = pd.to_datetime(df['date'])
    
    # Ensure numeric columns are float64 type
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = df[col].astype('float64')
    
    df = df.sort_values(by='date').reset_index(drop=True)
    
    # 3. Calculate features
    logger.info("Calculating technical indicators...")
    df_features, _ = calculate_features(df, scaler=scaler)
    
    # 4. Create strategy and trading simulator
    logger.info(f"Creating trading strategy (Long Threshold: {long_threshold}, Short Threshold: {short_threshold})...")
    strategy = ProbabilityThresholdStrategy(
        long_threshold=long_threshold,
        short_threshold=short_threshold,
        holding_period=3
    )
    
    from src.backtesting import RealTimeTrader
    
    trader = RealTimeTrader(
        model=model,
        strategy=strategy,
        holding_period_minutes=3
    )
    
    # 5. Simulate real-time trading
    logger.info("Starting real-time trading simulation...")
    trade_df = trader.simulate_real_time_trading(df_features, feature_engineering)
    
    if len(trade_df) == 0:
        logger.warning("No trade records generated")
        return None
    
    # 6. Calculate trade statistics
    total_trades = len(trade_df)
    winning_trades = len(trade_df[trade_df['profit_loss'] > 0])
    losing_trades = len(trade_df[trade_df['profit_loss'] < 0])
    win_rate = winning_trades / total_trades if total_trades > 0 else 0
    
    total_profit = trade_df['profit_loss'].sum()
    avg_profit = trade_df['profit_loss'].mean()
    
    # Calculate separate statistics for long and short trades
    long_trades = trade_df[trade_df['trade_type'] == 'Long']
    short_trades = trade_df[trade_df['trade_type'] == 'Short']
    
    long_count = len(long_trades)
    long_win_count = len(long_trades[long_trades['profit_loss'] > 0])
    long_win_rate = long_win_count / long_count if long_count > 0 else 0
    long_pnl = long_trades['profit_loss'].sum()
    
    short_count = len(short_trades)
    short_win_count = len(short_trades[short_trades['profit_loss'] > 0])
    short_win_rate = short_win_count / short_count if short_count > 0 else 0
    short_pnl = short_trades['profit_loss'].sum()
    
    total_win = trade_df[trade_df['profit_loss'] > 0]['profit_loss'].sum()
    total_loss = abs(trade_df[trade_df['profit_loss'] < 0]['profit_loss'].sum())
    profit_factor = total_win / total_loss if total_loss > 0 else float('inf')
    
    # Calculate maximum drawdown
    cumulative_pnl = trade_df['cumulative_pnl']
    peak = cumulative_pnl.expanding(min_periods=1).max()
    drawdown = (peak - cumulative_pnl)
    max_drawdown = drawdown.max()
    
    # 7. Plot trade results with enhanced styling if requested
    if enhanced_charts:
        # Set up aesthetics
        sns.set_style("whitegrid")
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.titleweight'] = 'bold'
        plt.rcParams['axes.titlesize'] = 14
        plt.rcParams['axes.labelsize'] = 12
        
        # Create figure with better styling
        plt.figure(figsize=(14, 9))
        
        # Plot cumulative P&L curve with enhanced styling
        plt.plot(trade_df['date'], trade_df['cumulative_pnl'], color='#1f77b4', 
                linewidth=2.5, label='Cumulative PnL')
        
        # Add area shading below the line
        plt.fill_between(trade_df['date'], 0, trade_df['cumulative_pnl'], 
                        color='#1f77b4', alpha=0.2)
        
        # Mark trade points with distinct colors and better markers
        # Long winning trades
        long_win = trade_df[(trade_df['trade_type'] == 'Long') & (trade_df['profit_loss'] > 0)]
        if len(long_win) > 0:
            plt.scatter(long_win['date'], long_win['cumulative_pnl'], 
                       marker='^', color='#2ca02c', s=120, label='Long Win', alpha=0.8,
                       edgecolors='darkgreen', linewidths=1.5)
        
        # Long losing trades
        long_loss = trade_df[(trade_df['trade_type'] == 'Long') & (trade_df['profit_loss'] <= 0)]
        if len(long_loss) > 0:
            plt.scatter(long_loss['date'], long_loss['cumulative_pnl'], 
                       marker='^', color='#d62728', s=120, label='Long Loss', alpha=0.8,
                       edgecolors='darkred', linewidths=1.5)
        
        # Short winning trades
        short_win = trade_df[(trade_df['trade_type'] == 'Short') & (trade_df['profit_loss'] > 0)]
        if len(short_win) > 0:
            plt.scatter(short_win['date'], short_win['cumulative_pnl'], 
                       marker='v', color='#9467bd', s=120, label='Short Win', alpha=0.8,
                       edgecolors='rebeccapurple', linewidths=1.5)
        
        # Short losing trades
        short_loss = trade_df[(trade_df['trade_type'] == 'Short') & (trade_df['profit_loss'] <= 0)]
        if len(short_loss) > 0:
            plt.scatter(short_loss['date'], short_loss['cumulative_pnl'], 
                       marker='v', color='#ff7f0e', s=120, label='Short Loss', alpha=0.8,
                       edgecolors='darkorange', linewidths=1.5)
        
        # Highlight drawdown periods
        max_dd_idx = drawdown.idxmax()
        if not pd.isna(max_dd_idx):
            max_dd_date = trade_df.iloc[max_dd_idx]['date']
            peak_date = trade_df.iloc[peak.iloc[max_dd_idx:max_dd_idx+1].idxmax()]['date']
            
            # Shade the max drawdown area
            plt.axvspan(peak_date, max_dd_date, alpha=0.2, color='red', label='Max Drawdown')
            
            # Add annotation for max drawdown
            dd_y_pos = (cumulative_pnl.iloc[max_dd_idx] + peak.iloc[max_dd_idx]) / 2
            plt.annotate(f'Max DD: {max_drawdown:.2f}',
                        xy=((peak_date + max_dd_date)/2, dd_y_pos),
                        xytext=(0, 30),
                        textcoords="offset points",
                        arrowprops=dict(arrowstyle="->", color='red'),
                        ha='center', va='bottom',
                        fontsize=10, fontweight='bold',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="red", alpha=0.8))
        
        # Enhance chart appearance
        plt.title(f'Real-Time Trading Simulation Results (L:{long_threshold:.6f}, S:{short_threshold:.6f})',
                 fontsize=16, fontweight='bold')
        plt.xlabel('Date', fontsize=12, fontweight='bold')
        plt.ylabel('Cumulative PnL (Points)', fontsize=12, fontweight='bold')
        
        # Create better legend
        plt.legend(loc='best', frameon=True, framealpha=0.9, shadow=True)
        
        # Add reference line at zero
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3, linewidth=1.5)
        
        # Format date axis
        date_format = DateFormatter('%Y-%m-%d %H:%M')
        plt.gca().xaxis.set_major_formatter(date_format)
        plt.xticks(rotation=45, ha='right')
        
        # Add grid
        plt.grid(True, alpha=0.3)
        
        # Add trade statistics as text box with enhanced styling
        stats_text = (
            f"Total Trades: {total_trades} | Win Rate: {win_rate:.2%} | Profit Factor: {profit_factor:.2f}\n"
            f"Total PnL: {total_profit:.2f} | Average PnL: {avg_profit:.2f} | Max Drawdown: {max_drawdown:.2f}\n"
            f"Long: {long_count} trades, {long_win_rate:.2%} win rate, {long_pnl:.2f} PnL | "
            f"Short: {short_count} trades, {short_win_rate:.2%} win rate, {short_pnl:.2f} PnL"
        )
        
        plt.figtext(0.5, 0.01, stats_text, ha='center', fontsize=11, fontweight='bold',
                   bbox={'facecolor': '#ff9d4d', 'alpha': 0.2, 'pad': 10, 'boxstyle': 'round,pad=0.5'})
        
        plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    else:
        # Use original plot style for backward compatibility
        plt.figure(figsize=(12, 8))
        
        # Plot cumulative P&L curve
        plt.plot(trade_df['date'], trade_df['cumulative_pnl'], color='blue', label='Cumulative PnL')
        
        # Mark trade points
        long_trades = trade_df[trade_df['trade_type'] == 'Long']
        short_trades = trade_df[trade_df['trade_type'] == 'Short']
        
        # Use different markers for winning and losing trades
        for trades, marker, color, label in [
            (long_trades[long_trades['profit_loss'] > 0], '^', 'lime', 'Long Win'),
            (long_trades[long_trades['profit_loss'] <= 0], '^', 'darkgreen', 'Long Loss'),
            (short_trades[short_trades['profit_loss'] > 0], 'v', 'red', 'Short Win'),
            (short_trades[short_trades['profit_loss'] <= 0], 'v', 'darkred', 'Short Loss')
        ]:
            if len(trades) > 0:
                plt.scatter(
                    trades['date'], 
                    trades['cumulative_pnl'], 
                    marker=marker, 
                    color=color, 
                    s=100, 
                    label=label
                )
        
        plt.title('Real-Time Trading Simulation Results')
        plt.xlabel('Date')
        plt.ylabel('Cumulative PnL (Points)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Add trade statistics
        stats_text = (
            f"Total Trades: {total_trades} | Win Rate: {win_rate:.2%} | Profit Factor: {profit_factor:.2f}\n"
            f"Total PnL: {total_profit:.2f} | Average PnL: {avg_profit:.2f} | Max Drawdown: {max_drawdown:.2f}"
        )
        plt.figtext(0.5, 0.01, stats_text, ha='center', fontsize=12, 
                   bbox={'facecolor': 'orange', 'alpha': 0.2, 'pad': 5})
        
        plt.tight_layout(rect=[0, 0.08, 1, 0.95])
    
    # 8. Save results
    results_path = os.path.join(dirs['BACKTEST_RESULTS_DIR'], f"realtime_simulation_{run_id}.xlsx")
    if save_excel:
        # Create enhanced Excel file with multiple sheets
        with pd.ExcelWriter(results_path, engine='openpyxl') as writer:
            # Save main trade records
            trade_df.to_excel(writer, sheet_name='Trade Records', index=False)
            
            # Create and save summary statistics
            summary_data = {
                'Metric': [
                    'Total Trades', 'Winning Trades', 'Losing Trades', 'Win Rate',
                    'Total PnL', 'Average PnL', 'Profit Factor', 'Max Drawdown',
                    'Long Trades', 'Long Win Rate', 'Long PnL', 'Long Average PnL',
                    'Short Trades', 'Short Win Rate', 'Short PnL', 'Short Average PnL'
                ],
                'Value': [
                    total_trades, winning_trades, losing_trades, win_rate,
                    total_profit, avg_profit, profit_factor, max_drawdown,
                    long_count, long_win_rate, long_pnl, long_trades['profit_loss'].mean() if long_count > 0 else 0,
                    short_count, short_win_rate, short_pnl, short_trades['profit_loss'].mean() if short_count > 0 else 0
                ]
            }
            
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_excel(writer, sheet_name='Summary', index=False)
            
            # Add hourly analysis if time data is available
            if 'date' in trade_df.columns:
                trade_df['hour'] = pd.to_datetime(trade_df['date']).dt.hour
                
                # Hourly trade count
                hourly_counts = trade_df.groupby(['hour', 'trade_type']).size().unstack(fill_value=0)
                hourly_counts.to_excel(writer, sheet_name='Hourly Counts')
                
                # Hourly PnL
                hourly_pnl = trade_df.groupby(['hour', 'trade_type'])['profit_loss'].sum().unstack(fill_value=0)
                hourly_pnl.to_excel(writer, sheet_name='Hourly PnL')
                
                # Win rate by hour
                hourly_wins = {}
                
                for hour in trade_df['hour'].unique():
                    hour_data = trade_df[trade_df['hour'] == hour]
                    win_rate = len(hour_data[hour_data['profit_loss'] > 0]) / len(hour_data) if len(hour_data) > 0 else 0
                    hourly_wins[hour] = win_rate
                    
                hourly_win_rates = pd.DataFrame({
                    'Hour': list(hourly_wins.keys()),
                    'Win Rate': list(hourly_wins.values())
                }).sort_values('Hour')
                
                hourly_win_rates.to_excel(writer, sheet_name='Hourly Win Rates', index=False)
    else:
        # Save basic trade records for backward compatibility
        trade_df.to_excel(results_path, index=False)
    
    chart_path = os.path.join(run_figures_dir, f"realtime_simulation_{run_id}.png")
    plt.savefig(chart_path, dpi=300, bbox_inches='tight')
    
    logger.info(f"Trade records saved to: {results_path}")
    logger.info(f"Trade chart saved to: {chart_path}")
    
    # Show the plot if requested
    if show_plots:
        plt.show()
    else:
        plt.close()
    
    # 9. Output trade statistics
    logger.info("\n===== Trade Statistics =====")
    logger.info(f"Total Trades: {total_trades}")
    logger.info(f"Win Rate: {win_rate:.2%}")
    logger.info(f"Profit Factor: {profit_factor:.2f}")
    logger.info(f"Total PnL: {total_profit:.2f}")
    logger.info(f"Average PnL: {avg_profit:.2f}")
    logger.info(f"Maximum Drawdown: {max_drawdown:.2f}")
    logger.info(f"Long Trades: {long_count} (Win Rate: {long_win_rate:.2%}, PnL: {long_pnl:.2f})")
    logger.info(f"Short Trades: {short_count} (Win Rate: {short_win_rate:.2%}, PnL: {short_pnl:.2f})")
    
    # 10. Create additional analysis charts if enhanced_charts is enabled
    if enhanced_charts:
        # Create trade distribution by type chart
        plt.figure(figsize=(18, 12))
        
        # Create a 2x2 subplot grid
        fig, axes = plt.subplots(2, 2, figsize=(18, 12))
        fig.subplots_adjust(hspace=0.3, wspace=0.3)
        
        # 1. Hourly Trade Distribution (top left)
        ax1 = axes[0, 0]
        if 'hour' in trade_df.columns:
            # Count trades by hour and type
            hour_counts = trade_df.groupby(['hour', 'trade_type']).size().unstack(fill_value=0)
            
            # Plot stacked bar chart
            hour_counts.plot(kind='bar', stacked=True, ax=ax1, colormap='viridis')
            
            ax1.set_title('Trade Distribution by Hour', fontsize=14, fontweight='bold')
            ax1.set_xlabel('Hour of Day', fontsize=12, fontweight='bold')
            ax1.set_ylabel('Number of Trades', fontsize=12, fontweight='bold')
            
            # Add data labels on top of stacked bars
            for i, hour in enumerate(hour_counts.index):
                total = hour_counts.loc[hour].sum()
                ax1.text(i, total + 0.1, str(int(total)), 
                        ha='center', fontsize=9, fontweight='bold')
        else:
            ax1.text(0.5, 0.5, 'No hourly data available', 
                    ha='center', va='center', transform=ax1.transAxes,
                    fontsize=14, fontweight='bold')
        
        ax1.grid(True, alpha=0.3)
        
        # 2. PnL by Trade Type (top right)
        ax2 = axes[0, 1]
        
        # Calculate PnL by trade type
        type_pnl = trade_df.groupby('trade_type')['profit_loss'].sum()
        
        # Define colors based on positive/negative values
        colors = ['#2ca02c' if p >= 0 else '#d62728' for p in type_pnl.values]
        
        # Create bar chart
        bars = ax2.bar(type_pnl.index, type_pnl.values, color=colors, alpha=0.9)
        
        ax2.set_title('P&L by Trade Type', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Trade Type', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Total P&L (Points)', fontsize=12, fontweight='bold')
        
        # Add data labels
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2, height + (0.1 if height >= 0 else -0.1),
                    f'{height:.2f}', ha='center', fontsize=11, fontweight='bold')
        
        # Add reference line at zero
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax2.grid(True, alpha=0.3)
        
        # 3. Win/Loss Distribution (bottom left)
        ax3 = axes[1, 0]
        
        # Create data for grouped bar chart
        categories = ['Long', 'Short']
        win_counts = [len(long_trades[long_trades['profit_loss'] > 0]), 
                     len(short_trades[short_trades['profit_loss'] > 0])]
        loss_counts = [len(long_trades[long_trades['profit_loss'] <= 0]), 
                      len(short_trades[short_trades['profit_loss'] <= 0])]
        
        # Set positions and width for bars
        x = np.arange(len(categories))
        width = 0.35
        
        # Create bars
        win_bars = ax3.bar(x - width/2, win_counts, width, label='Wins', color='#2ca02c', alpha=0.9)
        loss_bars = ax3.bar(x + width/2, loss_counts, width, label='Losses', color='#d62728', alpha=0.9)
        
        # Add titles and labels
        ax3.set_title('Win/Loss Distribution by Type', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Trade Type', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Number of Trades', fontsize=12, fontweight='bold')
        
        # Set x-ticks
        ax3.set_xticks(x)
        ax3.set_xticklabels(categories)
        
        # Add data labels
        for bars in [win_bars, loss_bars]:
            for bar in bars:
                height = bar.get_height()
                ax3.text(bar.get_x() + bar.get_width()/2, height + 0.1,
                        str(int(height)), ha='center', fontsize=9, fontweight='bold')
        
        # Add win rates as text
        if long_count > 0:
            ax3.text(x[0], max(win_counts[0], loss_counts[0]) + 0.5,
                    f'Win Rate: {long_win_rate:.1%}', ha='center', fontsize=10, fontweight='bold')
        
        if short_count > 0:
            ax3.text(x[1], max(win_counts[1], loss_counts[1]) + 0.5,
                    f'Win Rate: {short_win_rate:.1%}', ha='center', fontsize=10, fontweight='bold')
        
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Profit Loss Distribution (bottom right)
        ax4 = axes[1, 1]
        
        # Create separate histograms for long and short trade P&L
        if long_count > 0:
            sns.histplot(long_trades['profit_loss'], bins=15, kde=True, 
                        color='#1f77b4', label='Long Trades', ax=ax4, alpha=0.6)
        
        if short_count > 0:
            sns.histplot(short_trades['profit_loss'], bins=15, kde=True, 
                        color='#ff7f0e', label='Short Trades', ax=ax4, alpha=0.6)
        
        # Add reference line at zero
        ax4.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        
        # Add titles and labels
        ax4.set_title('Profit/Loss Distribution', fontsize=14, fontweight='bold')
        ax4.set_xlabel('P&L per Trade (Points)', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
        
        # Add title to the figure
        fig.suptitle(f'Real-Time Trading Simulation Analysis (L:{long_threshold:.6f}, S:{short_threshold:.6f})', 
                     fontsize=16, fontweight='bold')
        
        # Add summary statistics
        summary_text = (
            f"Total P&L: {total_profit:.2f} points | Win Rate: {win_rate:.1%} | "
            f"Profit Factor: {profit_factor:.2f} | Max Drawdown: {max_drawdown:.2f}"
        )
        
        plt.figtext(0.5, 0.01, summary_text, ha='center', fontsize=12, fontweight='bold',
                   bbox={'facecolor': '#ff9d4d', 'alpha': 0.2, 'pad': 10, 'boxstyle': 'round,pad=0.5'})
        
        # Adjust layout
        plt.tight_layout(rect=[0, 0.05, 1, 0.95])
        
        # Save the figure
        detailed_chart_path = os.path.join(run_figures_dir, f"detailed_analysis_{run_id}.png")
        plt.savefig(detailed_chart_path, dpi=300, bbox_inches='tight')
        
        logger.info(f"Detailed analysis chart saved to: {detailed_chart_path}")
        
        # Show the plot if requested
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    return trade_df


# If run directly, provide a simple command-line interface
if __name__ == "__main__":
    logger = configure_logging()
    dirs = setup_directories()
    
    # Simple CLI interface when run directly
    print("\n===== ENHANCED BACKTESTING SYSTEM =====")
    print("1. Run enhanced backtest on validation data")
    print("2. Run enhanced backtest on training data")
    print("3. Run comprehensive backtest (training + validation)")
    print("4. Run real-time trading simulation with specified thresholds")
    print("Q. Quit")
    
    choice = input("\nSelect an option: ").strip().upper()
    
    # Set default paths
    model_path = os.path.join(dirs['MODELS_DIR'], "lightgbm.joblib")
    feature_path = os.path.join(dirs['MODELS_DIR'], "features.xlsx")
    scaler_path = os.path.join(dirs['MODELS_DIR'], "scaler.joblib")
    training_data_path = os.path.join(dirs['DATA_DIR'], 'raw', "TX00_training.xlsx")
    validation_data_path = os.path.join(dirs['DATA_DIR'], 'raw', "TX00_validation.xlsx")
    
    if choice == '1':
        logger.info("\n===== RUNNING ENHANCED BACKTEST ON VALIDATION DATA =====")
        backtest_model(model_path, feature_path, scaler_path, validation_data_path,
                     save_excel=True, separate_long_short=True, enhanced_charts=True)
    
    elif choice == '2':
        logger.info("\n===== RUNNING ENHANCED BACKTEST ON TRAINING DATA =====")
        backtest_model(model_path, feature_path, scaler_path, training_data_path,
                     save_excel=True, separate_long_short=True, enhanced_charts=True)
    
    elif choice == '3':
        logger.info("\n===== RUNNING COMPREHENSIVE BACKTEST =====")
        training_metrics, validation_metrics = run_comprehensive_backtest(
            model_path, feature_path, scaler_path, training_data_path, validation_data_path,
            save_excel=True, separate_long_short=True, enhanced_charts=True
        )
    
    elif choice == '4':
        logger.info("\n===== RUNNING REAL-TIME TRADING SIMULATION =====")
        # Ask for threshold values
        long_threshold = float(input("Enter long probability threshold (e.g., 0.0026): ").strip())
        short_threshold = float(input("Enter short probability threshold (e.g., 0.0026): ").strip())
        
        trade_df = simulate_real_time_trading(
            model_path, feature_path, scaler_path, validation_data_path, 
            long_threshold, short_threshold, save_excel=True, enhanced_charts=True
        )
    
    elif choice == 'Q':
        logger.info("Exiting...")
    
    else:
        logger.error("Invalid option selected.")
    
    logger.info("\nBacktesting completed!")

################# Visualization Functions for Backtesting Results #################
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
        
        # Create and run strategy
        strategy = ProbabilityThresholdStrategy(
            long_threshold=long_threshold,
            short_threshold=short_threshold,
            holding_period=backtester.profit_loss_window
        )
        
        temp_backtester = Backtester(profit_loss_window=backtester.profit_loss_window)
        temp_backtester.run(df_features, strategy)
        
        # Get cumulative P&L
        if 'Cumulative_PnL' in temp_backtester.results:
            cumulative_pnl = temp_backtester.results['Cumulative_PnL']
            
            # Plot with strategy label
            label = f"L:{long_threshold:.6f}, S:{short_threshold:.6f} (SR:{sharpe_ratio:.2f})"
            ax.plot(temp_backtester.results['date'], cumulative_pnl, 
                   label=label, linewidth=2, color=colors[i])
    
    # Add zero line
    ax.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    
    # Add grid
    ax.grid(True, alpha=0.3)
    
    # Format date on x-axis
    date_format = DateFormatter('%Y-%m-%d')
    ax.xaxis.set_major_formatter(date_format)
    plt.xticks(rotation=45, ha='right')
    
    # Add labels and title
    ax.set_title('Cumulative P&L Comparison - Filtered Strategies', fontsize=16, fontweight='bold')
    ax.set_xlabel('Date', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cumulative P&L (Points)', fontsize=12, fontweight='bold')
    
    # Add legend
    ax.legend(loc='best', frameon=True, framealpha=0.9, shadow=True)
    
    # Add filter criteria to the plot
    filter_text = (
        "Filter Criteria: Total PnL > 0, Sharpe Ratio > 0, "
        "PnL/MaxDD > 0, Trades < 5000, Profit Factor > 1.0"
    )
    plt.figtext(0.5, 0.01, filter_text, ha='center', fontsize=10,
               bbox={'facecolor': 'lightgray', 'alpha': 0.7, 'pad': 5, 'boxstyle': 'round'})
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved Cumulative Returns Plot to: {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close(fig)
    
    return fig


def filter_and_visualize_strategies(filtered_results, backtester, df_features, output_dir, show_plots=True):
    """
    Filter and visualize strategies meeting specific criteria.
    
    Parameters:
    -----------
    filtered_results : pd.DataFrame
        DataFrame with filtered results
    backtester : Backtester
        Backtester instance
    df_features : pd.DataFrame
        DataFrame with feature data
    output_dir : str
        Directory to save visualizations
    show_plots : bool, default=True
        Whether to display plots
    
    Returns:
    --------
    dict
        Dictionary with paths to saved visualizations
    """
    import os
    
    # Check if we have results to visualize
    if filtered_results is None or len(filtered_results) == 0:
        print("No filtered results to visualize")
        return {}
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Save paths for visualizations
    visualization_paths = {}
    
    # 1. Save the filtered results to CSV for reference
    results_csv_path = os.path.join(output_dir, "filtered_strategies.csv")
    filtered_results.to_csv(results_csv_path, index=False)
    visualization_paths['results_csv'] = results_csv_path
    
    # 2. Create summary text file with filter criteria
    summary_path = os.path.join(output_dir, "filter_criteria.txt")
    with open(summary_path, 'w') as f:
        f.write("Filtered Strategies Summary\n")
        f.write("=========================\n\n")
        f.write("Filter Criteria:\n")
        f.write("- Total PnL > 0\n")
        f.write("- Sharpe Ratio > 0\n")
        f.write("- PnL/MaxDD > 0\n")
        f.write("- Total Trades < 5000\n")
        f.write("- Profit Factor > 1.0\n\n")
        f.write(f"Total strategies meeting criteria: {len(filtered_results)}\n\n")
        
        # Add top 5 strategies
        top_n = min(5, len(filtered_results))
        top_strategies = filtered_results.head(top_n)
        
        f.write(f"Top {top_n} strategies by Sharpe Ratio:\n\n")
        for i, (_, row) in enumerate(top_strategies.iterrows()):
            f.write(f"Rank {i+1}:\n")
            f.write(f"  Long Threshold: {row['Long Threshold']:.6f}\n")
            f.write(f"  Short Threshold: {row['Short Threshold']:.6f}\n")
            f.write(f"  Sharpe Ratio: {row['Sharpe Ratio']:.4f}\n")
            f.write(f"  Total PnL: {row['Total PnL']:.2f}\n")
            f.write(f"  Win Rate: {row['Win Rate']*100:.2f}%\n")
            f.write(f"  Profit Factor: {row['Profit Factor']:.4f}\n")
            f.write(f"  Max Drawdown: {row['Max Drawdown']:.2f}\n\n")
    
    visualization_paths['summary'] = summary_path
    
    # 3. Create cumulative returns plot
    returns_plot_path = os.path.join(output_dir, "cumulative_returns_comparison.png")
    create_cumulative_returns_plot(
        filtered_results,
        backtester,
        df_features,
        save_path=returns_plot_path,
        show_plot=show_plots
    )
    visualization_paths['returns_plot'] = returns_plot_path
    
    # 4. Create visualizations for the best strategy
    if len(filtered_results) > 0:
        best_row = filtered_results.iloc[0]
        best_long = best_row['Long Threshold']
        best_short = best_row['Short Threshold']
        
        # Create strategy
        best_strategy = ProbabilityThresholdStrategy(
            long_threshold=best_long,
            short_threshold=best_short,
            holding_period=backtester.profit_loss_window
        )
        
        # Create backtester
        best_backtester = Backtester(profit_loss_window=backtester.profit_loss_window)
        best_backtester.run(df_features, best_strategy)
        
        # A. Create Long vs Short Trade Performance Analysis
        trade_comparison_path = os.path.join(output_dir, "best_strategy_trade_comparison.png")
        create_custom_performance_visualization(
            best_backtester,
            save_path=trade_comparison_path,
            show_plot=show_plots
        )
        visualization_paths['trade_comparison'] = trade_comparison_path
        
        # B. Create Advanced Trading Performance Analytics
        advanced_analysis_path = os.path.join(output_dir, "best_strategy_advanced_analysis.png")
        create_monthly_performance_heatmap(
            best_backtester,
            save_path=advanced_analysis_path,
            show_plot=show_plots
        )
        visualization_paths['advanced_analysis'] = advanced_analysis_path
        
        # C. Save metrics to text file
        metrics = best_backtester.calculate_metrics()
        metrics_path = os.path.join(output_dir, "best_strategy_metrics.txt")
        
        with open(metrics_path, 'w') as f:
            f.write(f"Best Strategy Metrics (Long={best_long:.6f}, Short={best_short:.6f})\n")
            f.write("=".ljust(80, "=") + "\n\n")
            
            # Overall metrics
            f.write("OVERALL METRICS:\n")
            f.write("-".ljust(20, "-") + "\n")
            for key, value in metrics.items():
                if not key.startswith('Long') and not key.startswith('Short'):
                    if isinstance(value, float):
                        f.write(f"{key}: {value:.6f}\n")
                    else:
                        f.write(f"{key}: {value}\n")
            
            # Long trade metrics
            f.write("\nLONG TRADE METRICS:\n")
            f.write("-".ljust(20, "-") + "\n")
            for key, value in metrics.items():
                if key.startswith('Long'):
                    if isinstance(value, float):
                        f.write(f"{key}: {value:.6f}\n")
                    else:
                        f.write(f"{key}: {value}\n")
            
            # Short trade metrics
            f.write("\nSHORT TRADE METRICS:\n")
            f.write("-".ljust(20, "-") + "\n")
            for key, value in metrics.items():
                if key.startswith('Short'):
                    if isinstance(value, float):
                        f.write(f"{key}: {value:.6f}\n")
                    else:
                        f.write(f"{key}: {value}\n")
        
        visualization_paths['metrics'] = metrics_path
    
    # 5. Create strategy comparison visualization if there are multiple strategies
    if len(filtered_results) > 1:
        comparison_path = os.path.join(output_dir, "strategy_comparison.png")
        create_strategy_comparison_visualization(
            filtered_results,
            None,  # No need for backtesters here
            save_path=comparison_path,
            show_plot=show_plots
        )
        visualization_paths['comparison'] = comparison_path
    
    return visualization_paths

def run_comprehensive_backtest(model_path, feature_path, scaler_path, training_data_path, 
                              validation_data_path, base_dir=None, save_excel=True,
                              separate_long_short=True, enhanced_charts=True, show_plots=True):
    """
    Enhanced comprehensive backtest on both training and validation data
    
    Parameters:
    -----------
    model_path : str
        Path to model file
    feature_path : str
        Path to feature list file
    scaler_path : str
        Path to scaler file
    training_data_path : str
        Path to training data file
    validation_data_path : str
        Path to validation data file
    base_dir : str, optional
        Base directory path
    save_excel : bool, default=True
        Whether to save detailed Excel results
    separate_long_short : bool, default=True
        Whether to perform separate analysis for long and short trades
    enhanced_charts : bool, default=True
        Whether to use enhanced chart visualization
    show_plots : bool, default=True
        Whether to display plots (set to False for batch processing)
    
    Returns:
    --------
    tuple
        (training_metrics, validation_metrics)
    """
    # Configure logging if not already configured
    global logger
    if not logger.handlers:
        logger = configure_logging()
    
    # Setup directories
    dirs = setup_directories(base_dir)
    
    # Generate run ID for this comprehensive backtest
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"Starting comprehensive backtest (Run ID: {run_id})")
    
    # Backtest on training data
    logger.info("\n===== BACKTEST ON TRAINING DATA =====")
    training_metrics = backtest_model(
        model_path, 
        feature_path, 
        scaler_path, 
        training_data_path,
        run_id=f"{run_id}_training",
        base_dir=base_dir,
        save_excel=save_excel,
        separate_long_short=separate_long_short,
        enhanced_charts=enhanced_charts,
        show_plots=show_plots
    )
    
    # Backtest on validation data
    logger.info("\n===== BACKTEST ON VALIDATION DATA =====")
    validation_metrics = backtest_model(
        model_path, 
        feature_path, 
        scaler_path, 
        validation_data_path,
        run_id=f"{run_id}_validation",
        base_dir=base_dir,
        save_excel=save_excel,
        separate_long_short=separate_long_short,
        enhanced_charts=enhanced_charts,
        show_plots=show_plots
    )
    
    # Compare metrics - Create a comparison chart
    logger.info("\n===== COMPARISON OF TRAINING VS VALIDATION METRICS =====")
    
    # Convert metrics to DataFrames for easier comparison
    train_df = pd.DataFrame({
        'Metric': list(training_metrics.keys()),
        'Training': [training_metrics[m] for m in training_metrics.keys()]
    })
    
    valid_df = pd.DataFrame({
        'Metric': list(validation_metrics.keys()),
        'Validation': [validation_metrics[m] for m in validation_metrics.keys()]
    })
    
    # Merge and calculate differences
    metrics_comparison = pd.merge(train_df, valid_df, on='Metric', how='outer')
    metrics_comparison['Difference'] = metrics_comparison['Validation'] - metrics_comparison['Training']
    metrics_comparison['% Change'] = (metrics_comparison['Difference'] / metrics_comparison['Training']) * 100
    
    # Format numeric columns
    for col in ['Training', 'Validation', 'Difference']:
        metrics_comparison[col] = metrics_comparison[col].apply(
            lambda x: f"{x:.4f}" if isinstance(x, (int, float)) else x
        )
    
    metrics_comparison['% Change'] = metrics_comparison['% Change'].apply(
        lambda x: f"{x:.2f}%" if isinstance(x, (int, float)) else "N/A"
    )
    
    # Display and save comparison
    logger.info("\n" + metrics_comparison.to_string(index=False))
    
    comparison_path = os.path.join(dirs['BACKTEST_RESULTS_DIR'], f"metrics_comparison_{run_id}.xlsx")
    metrics_comparison.to_excel(comparison_path, index=False)
    logger.info(f"Metrics comparison saved to: {comparison_path}")
    
    # Create a visual comparison chart for key metrics
    if enhanced_charts:
        create_comparison_chart(training_metrics, validation_metrics, 
                               save_path=os.path.join(dirs['FIGURES_DIR'], f"metrics_comparison_{run_id}.png"),
                               show_plot=show_plots)
    
    return training_metrics, validation_metrics


def plot_probability_histogram(data, save_path=None, show_plot=True):
    """
    Enhanced function to plot histogram of long and short probabilities
    
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
    import seaborn as sns
    
    # Check if probability columns exist
    if 'Long_Probability' not in data.columns or 'Short_Probability' not in data.columns:
        print("Error: DataFrame must contain 'Long_Probability' and 'Short_Probability' columns")
        return
    
    # Set up aesthetics
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    
    # Create figure
    plt.figure(figsize=(12, 8))
    
    # Plot histograms with enhanced appearance
    sns.histplot(data['Long_Probability'], bins=50, alpha=0.6, label='Long Probability', 
                color='green', kde=True, stat='density')
    sns.histplot(data['Short_Probability'], bins=50, alpha=0.6, label='Short Probability', 
                color='red', kde=True, stat='density')
    
    # Add vertical lines for typical threshold values with better design
    thresholds = [0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005]
    for threshold in thresholds:
        plt.axvline(x=threshold, color='#555555', linestyle='--', alpha=0.5, linewidth=1.5)
        plt.text(threshold, plt.gca().get_ylim()[1]*0.95, f"{threshold:.4f}", 
                 rotation=90, verticalalignment='top', fontsize=9, fontweight='bold')
    
    # Add annotations showing percentage of events above thresholds
    y_pos = plt.gca().get_ylim()[1] * 0.8
    for threshold in thresholds:
        long_pct = (data['Long_Probability'] >= threshold).mean() * 100
        short_pct = (data['Short_Probability'] >= threshold).mean() * 100
        
        if long_pct > 0.01 or short_pct > 0.01:  # Only show if percentage is meaningful
            annotation = f"L: {long_pct:.2f}%\nS: {short_pct:.2f}%"
            plt.text(threshold*1.1, y_pos, annotation, rotation=0, 
                     fontsize=8, ha='left', va='top', 
                     bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=2))
    
    # Add metadata with enhanced styling
    plt.xlabel("Probability", fontsize=12, fontweight='bold')
    plt.ylabel("Density", fontsize=12, fontweight='bold')
    plt.title("Probability Distribution of Long and Short Events", fontsize=14, fontweight='bold')
    plt.legend(fontsize=12, frameon=True, framealpha=0.9, shadow=True)
    plt.grid(True, alpha=0.3)
    
    # Add summary statistics with enhanced box styling
    long_stats = f"Long: mean={data['Long_Probability'].mean():.6f}, median={data['Long_Probability'].median():.6f}"
    short_stats = f"Short: mean={data['Short_Probability'].mean():.6f}, median={data['Short_Probability'].median():.6f}"
    plt.figtext(0.5, 0.01, long_stats + "\n" + short_stats, ha='center', fontsize=10, fontweight='bold', 
                bbox={'facecolor':'lightgray', 'alpha':0.7, 'pad':10, 'boxstyle':'round,pad=0.5'})
    
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


def create_cumulative_return_comparison(data, filtered_results, holding_period=3, 
                                      max_strategies=10, save_dir=None, show_plot=True):
    """
    Create and plot cumulative return comparison chart for filtered strategies with buy and hold
    
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
        
        # Add to comparison DataFrame
        label = f"L:{row['Long Threshold']:.6f}, S:{row['Short Threshold']:.6f} (SR:{row['Sharpe Ratio']:.2f})"
        cumulative_returns[label] = results['Cumulative_PnL'].values
    
    # Calculate buy and hold performance
    first_close = data['close'].iloc[0]
    buyhold_points = data['close'] - first_close
    cumulative_returns['Buy and Hold'] = buyhold_points.values
    
    # Plot cumulative return comparison chart
    if show_plot:
        plt.figure(figsize=(15, 8))
        
        # Plot strategies
        for col in cumulative_returns.columns:
            if col != 'date':
                if col == 'Buy and Hold':
                    plt.plot(cumulative_returns['date'], cumulative_returns[col], 
                            label=col, color='black', linewidth=2, linestyle='--')
                else:
                    plt.plot(cumulative_returns['date'], cumulative_returns[col], label=col)
        
        # Set chart elements
        plt.title('Cumulative Profit Comparison (Points)', fontsize=14)
        plt.xlabel('Date', fontsize=12)
        plt.ylabel('Cumulative Profit (Points)', fontsize=12)
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
        
    return cumulative_returns

def create_comparison_chart(training_metrics, validation_metrics, save_path=None, show_plot=True):
    """
    Create a visual comparison chart between training and validation metrics
    
    Parameters:
    -----------
    training_metrics : dict
        Training data metrics dictionary
    validation_metrics : dict
        Validation data metrics dictionary
    save_path : str, optional
        Path to save the figure. If None, figure is not saved.
    show_plot : bool, default=True
        Whether to display the plot (set to False for batch processing)
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import seaborn as sns
    
    # Set up aesthetics
    sns.set_style("whitegrid")
    plt.rcParams['font.family'] = 'DejaVu Sans'
    plt.rcParams['axes.titleweight'] = 'bold'
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    
    # Select key metrics to compare
    key_metrics = ['Total PnL', 'Win Rate', 'Profit Factor', 'Sharpe Ratio', 
                  'Max Drawdown', 'Total Trades']
    
    # Filter metrics that exist in both dictionaries
    common_metrics = [m for m in key_metrics if m in training_metrics and m in validation_metrics]
    
    if not common_metrics:
        print("No common metrics found for comparison")
        return
        
    # Extract values for each metric
    train_values = [training_metrics[m] for m in common_metrics]
    valid_values = [validation_metrics[m] for m in common_metrics]
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Set up bar chart
    x = np.arange(len(common_metrics))
    width = 0.35
    
    # Create bars with enhanced styling
    train_bars = ax.bar(x - width/2, train_values, width, label='Training', color='#1f77b4', alpha=0.9)
    valid_bars = ax.bar(x + width/2, valid_values, width, label='Validation', color='#ff7f0e', alpha=0.9)
    
    # Add value labels to bars
    for bars in [train_bars, valid_bars]:
        for bar in bars:
            height = bar.get_height()
            if abs(height) > 1000:
                # Format large numbers with k suffix
                value_str = f'{height/1000:.1f}k'
            elif abs(height) < 0.01:
                # Format very small numbers
                value_str = f'{height:.4f}'
            elif isinstance(height, float):
                if height > 0 and height < 1:  # For ratios like win rate
                    value_str = f'{height:.2f}'
                else:
                    value_str = f'{height:.2f}'
            else:
                value_str = str(height)
                
            ax.annotate(value_str,
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),  # 3 points vertical offset
                       textcoords="offset points",
                       ha='center', va='bottom',
                       fontsize=9, fontweight='bold')
    
    # Customize chart appearance
    ax.set_title('Training vs Validation Performance Metrics', fontsize=16, fontweight='bold')
    ax.set_ylabel('Values', fontsize=12, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(common_metrics, fontsize=10, fontweight='bold')
    ax.legend(fontsize=12, frameon=True, framealpha=0.9, shadow=True)
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Add summary note
    total_pnl_diff = validation_metrics.get('Total PnL', 0) - training_metrics.get('Total PnL', 0)
    pnl_change_pct = (total_pnl_diff / training_metrics.get('Total PnL', 1)) * 100 if training_metrics.get('Total PnL', 0) != 0 else 0
    
    diff_note = (
        f"Total PnL Difference: {total_pnl_diff:.2f} points ({pnl_change_pct:.1f}%)\n"
        f"Win Rate Change: {(validation_metrics.get('Win Rate', 0) - training_metrics.get('Win Rate', 0)):.2%}\n"
        f"Profit Factor Change: {validation_metrics.get('Profit Factor', 0) - training_metrics.get('Profit Factor', 0):.2f}"
    )
    
    plt.figtext(0.5, 0.01, diff_note, ha='center', fontsize=10, fontweight='bold',
               bbox={'facecolor':'lightgray', 'alpha':0.7, 'pad':10, 'boxstyle':'round,pad=0.5'})
    
    plt.subplots_adjust(bottom=0.15)  # Make room for the note
    
    # Save the figure if save_path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Comparison chart saved to: {save_path}")
    
    # Show plot if requested
    if show_plot:
        plt.show()
    else:
        plt.close()

def create_custom_performance_visualization(backtester, save_path=None, show_plot=True):
    """
    Create a custom performance visualization for a filtered strategy
    matching the format shown in the examples (Long vs Short Trade Performance Analysis).
    
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
    
    # Create figure for Long vs Short Trade Performance Analysis
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.subplots_adjust(hspace=0.3, wspace=0.3)
    
    # 1. Trade Count Comparison (top left)
    ax1 = axes[0, 0]
    
    # Get trade counts
    long_trades_count = metrics.get('Long Total Trades', 0)
    short_trades_count = metrics.get('Short Total Trades', 0)
    total_trades = long_trades_count + short_trades_count
    
    # Calculate percentages
    long_pct = (long_trades_count / total_trades * 100) if total_trades > 0 else 0
    short_pct = (short_trades_count / total_trades * 100) if total_trades > 0 else 0
    
    # Create bars
    trade_types = ['Long Trades', 'Short Trades']
    counts = [long_trades_count, short_trades_count]
    colors = ['#2ca02c', '#d62728']  # Green for long, red for short
    
    bars = ax1.bar(trade_types, counts, color=colors)
    
    # Add count labels on top of bars
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2, height + 0.1,
                f'{int(height)}', ha='center', fontweight='bold')
    
    # Add percentage labels in middle of bars
    ax1.text(bars[0].get_x() + bars[0].get_width()/2, bars[0].get_height()/2,
            f'{long_pct:.1f}%', ha='center', color='white', fontweight='bold')
    ax1.text(bars[1].get_x() + bars[1].get_width()/2, bars[1].get_height()/2,
            f'{short_pct:.1f}%', ha='center', color='white', fontweight='bold')
    
    # Set labels and title
    ax1.set_title('Trade Count Comparison', fontweight='bold')
    ax1.set_ylabel('Number of Trades', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    
    # 2. P&L Comparison (top right)
    ax2 = axes[0, 1]
    
    # Get P&L values
    long_pnl = metrics.get('Long Total PnL', 0)
    short_pnl = metrics.get('Short Total PnL', 0)
    
    # Create bars for P&L
    pnl_types = ['Long PnL', 'Short PnL']
    pnl_values = [long_pnl, short_pnl]
    pnl_colors = ['#2ca02c' if long_pnl >= 0 else '#d62728', 
                 '#2ca02c' if short_pnl >= 0 else '#d62728']
    
    pnl_bars = ax2.bar(pnl_types, pnl_values, color=pnl_colors)
    
    # Add value labels on top of bars
    for bar in pnl_bars:
        height = bar.get_height()
        y_pos = height + 0.1 if height >= 0 else height - 0.1
        va = 'bottom' if height >= 0 else 'top'
        ax2.text(bar.get_x() + bar.get_width()/2, y_pos,
                f'{height:.2f}', ha='center', va=va, fontweight='bold')
    
    # Set labels and title
    ax2.set_title('P&L Comparison', fontweight='bold')
    ax2.set_ylabel('P&L Points', fontweight='bold')
    ax2.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    ax2.grid(True, alpha=0.3)
    
    # 3. Win Rate Comparison (bottom left)
    ax3 = axes[1, 0]
    
    # Get win rates
    long_win_rate = metrics.get('Long Win Rate', 0) * 100
    short_win_rate = metrics.get('Short Win Rate', 0) * 100
    
    # Create bars for win rates
    win_rate_types = ['Long Win Rate', 'Short Win Rate']
    win_rates = [long_win_rate, short_win_rate]
    win_rate_colors = ['#98df8a', '#ff9999']  # Lighter green and red
    
    win_rate_bars = ax3.bar(win_rate_types, win_rates, color=win_rate_colors)
    
    # Add win rate percentage labels on top of bars
    for bar in win_rate_bars:
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2, height + 1,
                f'{height:.1f}%', ha='center', fontweight='bold')
    
    # Set labels and title
    ax3.set_title('Win Rate Comparison', fontweight='bold')
    ax3.set_ylabel('Win Rate (%)', fontweight='bold')
    ax3.set_ylim(0, 100)  # Set y-axis from 0 to 100%
    
    # Add 50% reference line
    ax3.axhline(y=50, color='gray', linestyle='--', alpha=0.7)
    ax3.text(1.1, 50, '50%', ha='left', va='center', color='gray', fontsize=10)
    
    ax3.grid(True, alpha=0.3)
    
    # 4. Profit Factor Comparison (bottom right)
    ax4 = axes[1, 1]
    
    # Get profit factors (cap at 10 for better visualization)
    long_pf = min(metrics.get('Long Profit Factor', 0), 10)
    short_pf = min(metrics.get('Short Profit Factor', 0), 10)
    
    # Create bars for profit factors
    pf_types = ['Long P/F', 'Short P/F']
    pfs = [long_pf, short_pf]
    pf_colors = ['#2ca02c', '#d62728']
    
    pf_bars = ax4.bar(pf_types, pfs, color=pf_colors)
    
    # Add profit factor labels on top of bars
    for i, bar in enumerate(pf_bars):
        height = bar.get_height()
        actual_pf = metrics.get('Long Profit Factor' if i == 0 else 'Short Profit Factor', 0)
        display_pf = min(actual_pf, 9.99)  # For display purposes
        ax4.text(bar.get_x() + bar.get_width()/2, height + 0.1,
                f'{display_pf:.2f}' + (" (∞)" if actual_pf > 10 else ""), 
                ha='center', fontweight='bold')
    
    # Set labels and title
    ax4.set_title('Profit Factor Comparison', fontweight='bold')
    ax4.set_ylabel('Profit Factor', fontweight='bold')
    
    # Add reference line at profit factor = 1
    ax4.axhline(y=1, color='gray', linestyle='--', alpha=0.7)
    ax4.text(1.1, 1, '1.0', ha='left', va='center', color='gray', fontsize=10)
    
    ax4.grid(True, alpha=0.3)
    
    # Add title and summary statistics
    fig.suptitle('Long vs Short Trade Performance Analysis', fontsize=16, fontweight='bold')
    
    # Create summary text
    total_pnl = metrics.get('Total PnL', 0)
    overall_win_rate = metrics.get('Win Rate', 0) * 100
    total_trades_count = metrics.get('Total Trades', 0)
    profit_factor = metrics.get('Profit Factor', 0)
    
    summary_text = (
        f"Total P&L: {total_pnl:.2f} points | Overall Win Rate: {overall_win_rate:.1f}% | "
        f"Total Trades: {total_trades_count} | Profit Factor: {profit_factor:.2f}"
    )
    
    # Add text box with summary
    plt.figtext(0.5, 0.01, summary_text, ha='center', fontsize=12, fontweight='bold',
               bbox={'facecolor': 'lightgray', 'alpha': 0.7, 'pad': 5, 'boxstyle': 'round'})
    
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])
    
    # Save figure if path provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Saved Long vs Short Trade Performance Analysis to: {save_path}")
    
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