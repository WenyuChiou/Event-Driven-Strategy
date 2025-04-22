import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import json
from datetime import datetime
import textwrap

# 引入自定義模組
from src.event_detection import detect_trading_events, analyze_trading_events
from src.feature_engineering import calculate_features, FeatureEngineeringWrapper
from src.model import *
from src.visualization import visualize_event_summary, plot_price_with_event_markers
from src.utils import export_event_summary

# 設定路徑
data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data')
models_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models')
results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'results')
visualization_dir = os.path.join(results_dir, 'visualization')
summaries_dir = os.path.join(results_dir, 'summaries')

# 確保目錄存在
for directory in [data_dir, models_dir, results_dir, visualization_dir, summaries_dir]:
    os.makedirs(directory, exist_ok=True)

def train_model(data_path, model_name='lightgbm', n_trials=100):
    """
    訓練交易模型
    
    Parameters:
    -----------
    data_path : str
        數據文件路徑
    model_name : str, default='lightgbm'
        模型名稱
    n_trials : int, default=100
        超參數優化的嘗試次數
        
    Returns:
    --------
    tuple
        (訓練好的模型, 特徵工程對象, 評估指標)
    """
    # 1. 讀取數據
    print(f"讀取數據: {data_path}")
    df = pd.read_excel(data_path)
    df['date'] = pd.to_datetime(df['date'])
    
    # 確保數值列是 float64 類型
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in df.columns:
            df[col] = df[col].astype('float64')
    
    df = df.sort_values(by='date').reset_index(drop=True)
    
    # 2. 計算特徵
    print("計算技術指標...")
    df, scaler = calculate_features(df)
    
    # Detect events
    print("Detecting trading events...")
    df_events = detect_trading_events(df)

    # Analyze events
    print("Analyzing trading events...")
    analysis = analyze_trading_events(df_events)
     
    # 輸出基本統計
    print("\n===== 交易事件分析 =====")
    print(f"總事件數: {analysis['total_events']}")
    print(f"  做多事件: {analysis['long_events']} ({analysis['long_percentage']:.2f}%)")
    print(f"  做空事件: {analysis['short_events']} ({analysis['short_percentage']:.2f}%)")
    print(f"勝率: {analysis['win_rate']:.2%}")
    print(f"盈虧比: {analysis['profit_factor']:.2f}")
    print(f"平均獲利: {analysis['expectancy']:.2f} 點")

    #  Remove unnecessary columns
    df_events.drop(columns=['Session', 'Day_Of_Week','Valid_Trading_Time','Event_Type','hour_minute'], inplace=True)
   
    # 5. 特徵工程
    print("\n特徵選擇...")
    remove_column_name = [ 'date','Profit_Loss_Points', 'Event', 'Label']
    
    # 檢查所有要移除的列是否存在
    for col in remove_column_name:
        if col not in df_events.columns:
            print(f"警告: 欲移除的列 '{col}' 不存在於資料中")
    
    print(df_events)
    
    feature_engineering = FeatureEngineeringWrapper(remove_column_name=remove_column_name)
    X_final, scaler, selected_features = feature_engineering.fit(df_events)
    
    # 打印特徵信息以便調試
    print(f"特徵工程完成，共 {len(X_final.columns)} 個特徵")
    
    # 6. 超參數優化
    print(f"\n開始 {model_name} 的超參數優化 (嘗試次數: {n_trials})...")
    X = X_final
    y = df_events['Label']
    
    optimizer = BayesianOptimizerWrapper(model_name=model_name)
    optimizer.fit(X=X.to_numpy(), y=y.to_numpy(), n_splits=5, n_trials=n_trials)
    
    # 獲取最佳參數和權重
    best_params, weights = optimizer.get_best_params_and_weights()
    print("最佳參數:", best_params)
    print("參數權重:", weights)
    
    # 獲取特徵重要性
    feature_importances = optimizer.get_feature_importances()
    
    # 7. 訓練最終模型
    print("\n使用最佳參數訓練最終模型...")
    model = TradingModel(model_name=model_name, params=best_params)
    model.fit(X, y)
    
    # 8. 評估模型
    metrics = model.evaluate(X, y)
    print("訓練集評估指標:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # 9. 保存模型和特徵工程 - 移除時間戳記
    model_filename = os.path.join(models_dir, f"{model_name}.joblib")
    scaler_filename = os.path.join(models_dir, f"scaler.joblib")
    features_filename = os.path.join(models_dir, f"features.xlsx")
    params_filename = os.path.join(models_dir, f"params.json")
    
    # 保存模型
    joblib.dump(model, model_filename)
    
    # 保存 scaler
    joblib.dump(scaler, scaler_filename)
    
    # 保存特徵列表
    feature_engineering.save_features(features_filename)
    
    # 保存模型參數
    with open(params_filename, 'w') as f:
        json.dump(best_params, f)
    
    print(f"\n模型訓練完成。")
    print(f"模型保存於: {model_filename}")
    print(f"Scaler 保存於: {scaler_filename}")
    print(f"特徵列表保存於: {features_filename}")
    print(f"參數保存於: {params_filename}")
    
    # 10. Plot Top 10 Feature Importances with wrapped labels
    # Get indices of the top 10 features by importance
    top_idx = np.argsort(feature_importances)[::-1][:10]
    top_features = X.columns[top_idx]
    top_importances = feature_importances[top_idx]

    # Wrap feature names at 20 characters
    wrapped_labels = [
        "\n".join(textwrap.wrap(f, width=20))
        for f in top_features
    ]

    plt.figure(figsize=(10, 6))
    plt.barh(wrapped_labels, top_importances)
    plt.xlabel("Importance", fontsize=12)
    plt.title("Top 10 Feature Importances", fontsize=14, fontweight='bold')
    plt.gca().invert_yaxis()  # Show the highest importance on top
    plt.grid(axis='x', linestyle='--', alpha=0.7)

    # Increase left margin to accommodate wrapped labels
    plt.subplots_adjust(left=0.35)

    plt.tight_layout()
    importance_plot_path = os.path.join(visualization_dir, "feature_importance_top10.png")
    plt.savefig(importance_plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # 11. 輸出事件分析結果
    event_output_path = os.path.join(summaries_dir, f"event_analysis.xlsx")
    export_event_summary(analysis, event_output_path)
    
    # 12. 繪製事件視覺化
    visualization_path = os.path.join(summaries_dir, f"event_summary.png")
    visualize_event_summary(df_events, analysis, visualization_path, show_plots=False)
    
    price_chart_path = os.path.join(visualization_dir, f"price_with_events.png")
    plot_price_with_event_markers(df_events, price_chart_path, show_plots=False)
    
    return model, feature_engineering, metrics

if __name__ == "__main__":
    # 設定參數
    data_path = os.path.join(data_dir,'raw', "TX00_training.xlsx")
    model_name = 'lightgbm'  # 可選: 'randomforest', 'gradientboosting', 'xgboost', 'lightgbm', 'catboost'
    n_trials = 10  # 實際使用時建議設為 100 或更高
    
    # 訓練模型
    model, feature_engineering, metrics = train_model(data_path, model_name, n_trials)
    
    print("\n訓練完成！")