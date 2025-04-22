import os
import sys
import argparse
from datetime import datetime

# Add the project root directory to PATH
src_path = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, src_path)

from examples.model_training import train_model
from examples.backtesting_example import backtest_model, simulate_real_time_trading, run_comprehensive_backtest
# 直接导入 MA 策略回测函数
from src.ma_strategy import *


def parse_arguments():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='Trading Event Detection and Model Training Project')
    
    # Main execution mode
    parser.add_argument('--mode', type=str, default='train', 
                      choices=['train', 'backtest', 'comprehensive', 'simulate', 'all', 'ma_strategy'],  # kd_long 改为 ma_strategy
                      help='Execution mode: train (train model), backtest (backtest model), '
                           'comprehensive (training+validation backtest), simulate (simulate trading), '
                           'ma_strategy (run MA strategy), all (all modes except MA)')
    
    # Data paths
    parser.add_argument('--data', type=str, help='Data file path')
    parser.add_argument('--validation-data', type=str, help='Validation data file path')
    
    # Model settings
    parser.add_argument('--model', type=str, default='lightgbm', 
                      choices=['randomforest', 'gradientboosting', 'xgboost', 'lightgbm', 'catboost'],
                      help='Model type')
    parser.add_argument('--model-path', type=str, help='Trained model path')
    parser.add_argument('--feature-path', type=str, help='Feature list path')
    parser.add_argument('--scaler-path', type=str, help='Scaler path')
    
    # Backtesting parameters
    parser.add_argument('--long-threshold', type=float, default=0.0026, help='Long position probability threshold')
    parser.add_argument('--short-threshold', type=float, default=0.0026, help='Short position probability threshold')
    parser.add_argument('--save-excel', action='store_true', help='Save detailed Excel results (default: True)')
    parser.add_argument('--no-excel', action='store_true', help='Disable saving Excel results')
    parser.add_argument('--run-id', type=str, help='Unique identifier for this run (default: timestamp)')
    
    # Enhanced visualization options
    parser.add_argument('--separate-long-short', action='store_true', default=True, 
                      help='Perform separate analysis for long and short trades')
    parser.add_argument('--no-separate-analysis', action='store_true', 
                      help='Disable separate long/short analysis')
    parser.add_argument('--enhanced-charts', action='store_true', default=True, 
                      help='Use enhanced chart visualization')
    parser.add_argument('--simple-charts', action='store_true', 
                      help='Use simple chart visualization')
    parser.add_argument('--no-show-plots', action='store_true', 
                      help='Do not display plots (useful for batch processing)')
    
    # Training parameters
    parser.add_argument('--trials', type=int, default=100, help='Number of hyperparameter optimization trials')
    
    # MA Strategy parameters
    parser.add_argument('--ma-period', type=int, default=5, help='Period for Moving Average calculation')
    parser.add_argument('--commission', type=float, default=0.0, help='Trading commission cost per trade')
    parser.add_argument('--price-col', type=str, default='Close', help='Price column name for MA strategy')
    parser.add_argument('--min-hold-periods', type=int, default=5, help='Minimum holding periods for MA strategy')
    parser.add_argument('--optimize', action='store_true', help='Run parameter optimization for MA strategy')
    
    return parser.parse_args()

def main():
    """Main program entry point"""
    args = parse_arguments()
    
    # Set data directory and model directory
    data_dir = os.path.join(src_path, 'data', 'raw')
    models_dir = os.path.join(src_path, 'models')
    
    # Ensure directories exist
    for directory in [data_dir, models_dir]:
        os.makedirs(directory, exist_ok=True)
    
    # Process argument overrides
    save_excel = not args.no_excel if hasattr(args, 'no_excel') else args.save_excel
    separate_long_short = not args.no_separate_analysis if hasattr(args, 'no_separate_analysis') else args.separate_long_short
    enhanced_charts = not args.simple_charts if hasattr(args, 'simple_charts') else args.enhanced_charts
    show_plots = not args.no_show_plots if hasattr(args, 'no_show_plots') else True
    
    # Generate run_id if not provided
    run_id = args.run_id if args.run_id else datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # If no data path is provided, use default path
    if args.data is None:
        args.data = os.path.join(data_dir, "TX00_training.xlsx")
    
    if args.validation_data is None:
        args.validation_data = os.path.join(data_dir, "TX00_validation.xlsx")
    
    # 替換現有的 MA 策略處理代碼
    if args.mode == 'ma_strategy':
        print("\n===== Starting MA Strategy Backtest =====")
        
        # 設置自定義保存路徑
        custom_path = r"C:\Users\wenyu\Desktop\trade\investment\python\scrapping\hydraulic jump\project\results\ma_strategy"
        if not os.path.exists(custom_path):
            os.makedirs(custom_path, exist_ok=True)
        
        try:
            # 運行MA策略回測，解包返回的元組
            metrics_result, df_result = backtest_ma_strategy(
                data_path=args.validation_data,
                ma_period=args.ma_period,
                price_col=args.price_col,
                save_excel=save_excel,
                run_id=run_id,
                base_dir=src_path,
                custom_path=custom_path,
                enhanced_charts=enhanced_charts,
                show_plots=show_plots,
                commission=args.commission
            )
            
            # 打印MA策略指標
            print("\nMA Strategy Results:")
            print("-" * 50)
            
            for key, value in metrics_result.items():
                if isinstance(value, float):
                    print(f"{key}: {value:.4f}")
                else:
                    print(f"{key}: {value}")
            
        except Exception as e:
            print(f"\n執行MA策略時發生錯誤: {str(e)}")
            print("使用簡化版累積收益比較...")
            

            
            ma_periods = [3, 5, 10, 15, 20, 30, 50]
            chart_path, results = create_cumulative_comparison_chart(
                args.validation_data, 
                ma_periods=ma_periods,
                price_col=args.price_col,
                custom_path=custom_path
            )
            
            print(f"\n累積損益比較圖已保存至: {chart_path}")
        
        print("\nMA Strategy backtest completed successfully!")
        print("\nProgram execution completed!")
        return
    
    # Execute selected mode for ML model
    if args.mode in ['train', 'all']:
        print("\n===== Starting Model Training =====")
        model, feature_engineering, metrics = train_model(args.data, args.model, args.trials)
        
        # If 'all' mode, save model path for subsequent steps
        if args.mode == 'all':
            args.model_path = os.path.join(models_dir, f"{args.model}.joblib")
            args.feature_path = os.path.join(models_dir, f"features.xlsx")
            args.scaler_path = os.path.join(models_dir, f"scaler.joblib")
    
    if args.mode in ['backtest', 'all']:
        print("\n===== Starting Model Backtesting =====")
        
        # Set default paths if not specified
        if args.model_path is None:
            args.model_path = os.path.join(models_dir, f"{args.model}.joblib")
        if args.feature_path is None:
            args.feature_path = os.path.join(models_dir, f"features.xlsx")
        if args.scaler_path is None:
            args.scaler_path = os.path.join(models_dir, f"scaler.joblib")
        
        # Use the enhanced backtest_model function with new parameters
        metrics = backtest_model(
            args.model_path, 
            args.feature_path, 
            args.scaler_path, 
            args.validation_data,
            run_id=run_id,
            base_dir=src_path,
            save_excel=save_excel,
            separate_long_short=separate_long_short,
            enhanced_charts=enhanced_charts,
            show_plots=show_plots
        )
        
        print("\nBacktest completed successfully!")
    
    if args.mode in ['comprehensive', 'all']:
        print("\n===== Starting Comprehensive Backtesting (Training + Validation) =====")
        
        # Set default paths if not specified
        if args.model_path is None:
            args.model_path = os.path.join(models_dir, f"{args.model}.joblib")
        if args.feature_path is None:
            args.feature_path = os.path.join(models_dir, f"features.xlsx")
        if args.scaler_path is None:
            args.scaler_path = os.path.join(models_dir, f"scaler.joblib")
        
        # Run comprehensive backtest
        training_metrics, validation_metrics = run_comprehensive_backtest(
            args.model_path, 
            args.feature_path, 
            args.scaler_path, 
            args.data,  # Training data
            args.validation_data,
            base_dir=src_path,
            save_excel=save_excel,
            separate_long_short=separate_long_short,
            enhanced_charts=enhanced_charts,
            show_plots=show_plots
        )
        
        print("\nComprehensive backtest completed successfully!")
    
    if args.mode in ['simulate', 'all']:
        print("\n===== Starting Real-time Trading Simulation =====")
        
        # Set default paths if not specified
        if args.model_path is None:
            args.model_path = os.path.join(models_dir, f"{args.model}.joblib")
        if args.feature_path is None:
            args.feature_path = os.path.join(models_dir, f"features.xlsx")
        if args.scaler_path is None:
            args.scaler_path = os.path.join(models_dir, f"scaler.joblib")
        
        # Run real-time trading simulation with enhanced parameters
        trade_df = simulate_real_time_trading(
            args.model_path, 
            args.feature_path, 
            args.scaler_path, 
            args.validation_data, 
            args.long_threshold, 
            args.short_threshold,
            run_id=run_id,
            base_dir=src_path,
            save_excel=save_excel,
            enhanced_charts=enhanced_charts,
            show_plots=show_plots
        )
        
        if trade_df is not None and len(trade_df) > 0:
            print("\nSimulation completed successfully!")
        else:
            print("\nSimulation completed, but no trades were generated with the specified thresholds.")
    
    print("\nProgram execution completed!")

if __name__ == "__main__":
    main()