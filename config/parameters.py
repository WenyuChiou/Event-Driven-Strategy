"""
交易事件檢測與模型訓練專案的參數配置文件。
包含所有可配置的參數，包括事件檢測參數、特徵工程參數、模型訓練參數和回測參數。
"""

# 事件檢測參數
EVENT_DETECTION_PARAMS = {
    # 盈虧計算窗口
    'profit_loss_window': 3,  
    # ATR 計算窗口
    'atr_window': 14,
    # 做多事件的最低獲利閾值（點數）
    'long_profit_threshold': 10.0,
    # 做空事件的最大虧損閾值（點數）
    'short_loss_threshold': -10.0,
    # 相對於平均波動率的成交量倍數閾值
    'volume_multiplier': 2.0,
    # 是否使用 ATR 作為額外過濾條件
    'use_atr_filter': True
}

# 特徵計算參數
FEATURE_CALCULATION_PARAMS = {
    # 計算斜率的窗口大小
    'slope_window': 3,
    # 計算 EMA 的窗口大小
    'ema_window': 9,
    # 計算平均波動率的窗口大小
    'avg_vol_window': 9,
    # 計算長期 EMA 的窗口大小
    'long_ema_window': 13
}

# 特徵工程參數
FEATURE_ENGINEERING_PARAMS = {
    # 變異數閾值，用於移除低變異數特徵
    'variance_threshold': 0.005,
    # Lasso 參數，影響特徵選擇數量
    'lasso_eps': 1e-4,
    # 高相關性特徵閾值，用於移除高度相關特徵
    'corr_threshold': 0.9,
    # 需要移除的特徵名稱
    'remove_column_name': ['date', 'Profit_Loss_Points', 'Event', 'Label']
}

# 模型訓練參數 - RandomForest
RANDOM_FOREST_PARAMS = {
    # 決策樹數量
    'n_estimators': 100,
    # 最大深度
    'max_depth': 10,
    # 最小分割樣本數
    'min_samples_split': 5,
    # 最小葉節點樣本數
    'min_samples_leaf': 2,
    # 隨機種子
    'random_state': 42
}

# 模型訓練參數 - GradientBoosting
GRADIENT_BOOSTING_PARAMS = {
    # 決策樹數量
    'n_estimators': 100,
    # 學習率
    'learning_rate': 0.1,
    # 最大深度
    'max_depth': 5,
    # 子樣本比例
    'subsample': 0.8,
    # 隨機種子
    'random_state': 42
}

# 模型訓練參數 - XGBoost
XGBOOST_PARAMS = {
    # 決策樹數量
    'n_estimators': 100,
    # 學習率
    'learning_rate': 0.1,
    # 最大深度
    'max_depth': 5,
    # 子樣本比例
    'subsample': 0.8,
    # 列樣本比例
    'colsample_bytree': 0.8,
    # 隨機種子
    'random_state': 42
}

# 模型訓練參數 - LightGBM
LIGHTGBM_PARAMS = {
    # 決策樹數量
    'n_estimators': 100,
    # 學習率
    'learning_rate': 0.1,
    # 最大深度
    'max_depth': 5,
    # 葉節點數量
    'num_leaves': 31,
    # 子樣本比例
    'subsample': 0.8,
    # 列樣本比例
    'colsample_bytree': 0.8,
    # 隨機種子
    'random_state': 42
}

# 模型訓練參數 - CatBoost
CATBOOST_PARAMS = {
    # 迭代次數
    'iterations': 100,
    # 學習率
    'learning_rate': 0.1,
    # 樹深度
    'depth': 5,
    # L2 正則化係數
    'l2_leaf_reg': 3,
    # 隨機種子
    'random_state': 42
}

# 回測參數
BACKTESTING_PARAMS = {
    # 計算未來盈虧的窗口大小
    'profit_loss_window': 3,
    # 最大盈虧限制，用於過濾極端值
    'max_profit_loss': 50
}

# 交易策略參數
TRADING_STRATEGY_PARAMS = {
    # 做多概率閾值
    'long_threshold': 0.0026,
    # 做空概率閾值
    'short_threshold': 0.0026,
    # 持倉期限
    'holding_period': 3,
    # 要排除的時間點集合
    'exclude_times': {
        "08:45", "08:46", "08:47", "08:48", "08:49",  # 早盤開盤後5分鐘
        "13:41", "13:42", "13:43", "13:44", "13:45",  # 午盤開盤後5分鐘 
        "15:00", "15:01", "15:02", "15:03", "15:04",  # 日盤收盤前5分鐘
        "03:55", "03:56", "03:57", "03:58", "03:59"   # 夜盤收盤前5分鐘
    }
}

# 參數優化空間
OPTIMIZATION_SPACE = {
    'randomforest': {
        'n_estimators': (50, 300),
        'max_depth': (3, 15),
        'min_samples_split': (2, 20),
        'min_samples_leaf': (1, 10)
    },
    'gradientboosting': {
        'n_estimators': (50, 300),
        'learning_rate': (0.01, 0.3),
        'max_depth': (3, 10),
        'subsample': (0.5, 1.0)
    },
    'xgboost': {
        'n_estimators': (50, 300),
        'learning_rate': (0.01, 0.3),
        'max_depth': (3, 10),
        'subsample': (0.5, 1.0),
        'colsample_bytree': (0.5, 1.0)
    },
    'lightgbm': {
        'n_estimators': (50, 300),
        'learning_rate': (0.01, 0.3),
        'max_depth': (3, 10),
        'num_leaves': (20, 150),
        'subsample': (0.5, 1.0),
        'colsample_bytree': (0.5, 1.0)
    },
    'catboost': {
        'iterations': (50, 300),
        'learning_rate': (0.01, 0.3),
        'depth': (3, 10),
        'l2_leaf_reg': (1, 10)
    }
}

# 策略參數優化空間
STRATEGY_OPTIMIZATION_SPACE = {
    'long_threshold': (0.0001, 0.01),
    'short_threshold': (0.0001, 0.01),
    'holding_period': (1, 5)
}

# 輸出路徑配置
OUTPUT_PATHS = {
    'models_dir': 'models',
    'results_dir': 'results',
    'visualizations_dir': 'results/visualizations',
    'backtests_dir': 'results/backtests',
    'summaries_dir': 'results/summaries'
}

# 數據路徑配置
DATA_PATHS = {
    'raw_data_dir': 'data/raw',
    'processed_data_dir': 'data/processed'
}