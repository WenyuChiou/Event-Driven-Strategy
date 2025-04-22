import pandas as pd
import numpy as np

class SignalGenerator:
    def __init__(self, return_threshold=0.001, volatility_threshold=1.5, rolling_window=6, filter_std=3):
        """
        初始化 SignalGenerator 類。
        Args:
            return_threshold: 收益率閾值，用於判斷趨勢信號。
            volatility_threshold: 波動率閾值，用於判斷震盪信號。
            rolling_window: 計算移動平均成交量的窗口大小。
            filter_std: 修正異常值的標準差範圍
        """
        self.return_threshold = return_threshold
        self.volatility_threshold = volatility_threshold
        self.rolling_window = rolling_window
        self.filter_std = filter_std

    def generate_signal(self, row):
        """
        根據趨勢、波動率和成交量變化百分比生成交易信號，並使用 exp 進行加權。
        Args:
            row: 每行數據（DataFrame 的行）。
        Returns:
            float: 指數加權後的交易信號。
        """
        base_signal = 0

        # 判斷基礎信號
        if row['future_return'] > self.return_threshold:
            base_signal = 1
        elif row['future_return'] < -self.return_threshold:
            base_signal = -1
        elif row['future_range'] < self.volatility_threshold:
            base_signal = 0.5  # 震盪盤信號

        # 計算成交量變化百分比
        volume_ma = row['volume_rolling'] if 'volume_rolling' in row else row['volume']
        volume_change_percentage = (row['volume'] - volume_ma) / volume_ma if volume_ma != 0 else 0

        # 計算趨勢權重
        if row['price_change'] > 0:
            trend_weight = row['trend_weight'] + 1
        elif row['price_change'] < 0:
            trend_weight = row['trend_weight'] - 1
        else:
            trend_weight = 1

        # 指數加權公式
        exp_weight = np.exp(trend_weight * volume_change_percentage)
        weighted_signal = base_signal * exp_weight

        return weighted_signal

    def label_signal(self, signal, mean, std):
        """
        根據均值和標準差為信號標籤。
        Args:
            signal: 信號值。
            mean: 信號均值。
            std: 信號標準差。
        Returns:
            int: 信號標籤。
        """
        if signal > mean + 2 * std:
            return 3
        elif mean + 1.5 * std < signal <= mean + 2 * std:
            return 2
        elif mean + std < signal <= mean + 1.5 * std:
            return 1
        elif mean - std <= signal <= mean + std:
            return 0
        elif mean - 1.5 * std <= signal < mean - std:
            return -1
        elif mean - 2 * std <= signal < mean - 1.5 * std:
            return -2
        elif signal < mean - 2 * std:
            return -3

    def correct_outliers(self, data, column):
        """
        將指定列中的異常值修正到合理範圍。
        Args:
            data: 包含交易數據的 DataFrame。
            column: 要進行修正的列名。
        Returns:
            DataFrame: 修正後的 DataFrame。
        """
        # 計算均值和標準差
        mean = data[column].mean()
        std = data[column].std()

        # 計算上下限
        lower_bound = mean - self.filter_std * std
        upper_bound = mean + self.filter_std * std

        # 修正異常值
        data[column] = data[column].apply(
            lambda x: lower_bound if x < lower_bound else (upper_bound if x > upper_bound else x)
        )
        return data
    
    def process(self, data):
        """
        在 DataFrame 中生成交易信號並根據標準差進行標籤。
        Args:
            data: 包含交易數據的 DataFrame。
        Returns:
            DataFrame: 包含信號和標籤的新數據框。
        """
        # 計算移動平均成交量
        data['volume_rolling'] = data['volume'].rolling(window=self.rolling_window).mean()

        # 初始化價格變化與趨勢權重
        data['price_change'] = data['close'].diff()
        data['trend_weight'] = 1

        # 計算未來的收益率與波動率
        data['future_return'] = (data['close'].shift(-3) - data['close']) / data['close']

        # 計算當前波動範圍與未來波動範圍
        data['current_range'] = data['high'] - data['low']
        data['future_range'] = data['high'].shift(-self.rolling_window) - data['low'].shift(-self.rolling_window)

        # 應用函數生成信號
        data['predict_5min_signal'] = data.apply(self.generate_signal, axis=1)

        # 計算信號的均值和標準差
        mean_signal = data['predict_5min_signal'].mean()
        std_signal = data['predict_5min_signal'].std()

        # 添加信號標籤
        data['signal_label'] = data['predict_5min_signal'].apply(
            lambda x: self.label_signal(x, mean_signal, std_signal)
        )

        # 移除基於未來數據計算的欄位
        columns_to_remove = ['future_return', 'future_volatility', 'price_change', 'trend_weight', 'volume_rolling','signal_label',
                             'future_range']
        data = data.drop(columns=columns_to_remove, errors='ignore')

        return self.correct_outliers(data, 'predict_5min_signal')






