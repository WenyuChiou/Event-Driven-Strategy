import numpy as np
import pandas as pd
import talib
from sklearn.preprocessing import MinMaxScaler
from ta import add_all_ta_features  # 引入 `ta` 技術指標庫
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LassoCV
from package.alpha_eric import AlphaFactory

def calculate_realtime_features(
    data, slope_window=3, ema_window=9, avg_vol_window=9, long_ema_window=13, scaler=None
):
    """
    **即時特徵計算**
    1. 計算 `ta` 技術指標
    2. 計算 Bollinger Bands
    3. 計算 `Lower_Band_Slope` 和 `Normalized_Slope`
    4. 計算 `Slope_Change`
    5. 計算 `EMA`, `Long_EMA`
    6. 計算 `Rebound_Above_EMA`（突破短期均線）
    7. 計算 `Long_EMA_Downward`（長期趨勢下降）
    8. 計算 `Average_Volatility`
    9. **不判斷 `Event`，適用於即時數據**
    
    :param data: `pd.DataFrame`，包含即時市場數據
    :param scaler: `MinMaxScaler`，如果提供則使用相同標準化
    :return: 具有新特徵的 `pd.DataFrame`, `scaler`
    """

    # 確保數據足夠計算
    if len(data) < max(slope_window, ema_window, avg_vol_window, long_ema_window):
        raise ValueError("數據不足，請提供更多歷史數據！")

    # 📌 **Step 1: 加入 `ta` 技術指標 與自製alpha**
    data = add_all_ta_features(
        data, open="open", high="high", low="low", close="close", volume="volume", fillna=True
    )

    alpha = AlphaFactory(data)
    data = alpha.add_all_alphas(days=[3, 9, 20, 60, 120, 240])

    # 📌 **Step 2: 計算 Bollinger Bands**
    data['Middle_Band'] = data['close'].rolling(window=20).mean()
    data['Std_Dev'] = data['close'].rolling(window=20).std()
    data['Upper_Band'] = data['Middle_Band'] + 2 * data['Std_Dev']
    data['Lower_Band'] = data['Middle_Band'] - 2 * data['Std_Dev']

    # 📌 **Step 3: 計算 `Lower_Band_Slope`**
    data['Lower_Band_Slope'] = data['Lower_Band'].diff(slope_window) / data['Lower_Band'].shift(slope_window)
    data['Lower_Band_Slope'] = data['Lower_Band_Slope'].fillna(0)

    # 📌 **Step 4: 標準化 `Lower_Band_Slope`**
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(-1, 1))
    


    data['Normalized_Slope'] = scaler.fit_transform(data[['Lower_Band_Slope']])

    # 📌 **Step 5: 計算 `Slope_Change`**
    data['Slope_Change'] = data['Normalized_Slope'] - data['Normalized_Slope'].shift(slope_window)
    data['Slope_Change'] = data['Slope_Change'].fillna(0)

    # 📌 **Step 6: 計算短期 & 長期 `EMA`**
    data['EMA'] = talib.EMA(data['close'], timeperiod=ema_window)
    data['Long_EMA'] = talib.EMA(data['close'], timeperiod=long_ema_window)

    # 📌 **Step 7: 計算 `Rebound_Above_EMA, Break_Below_EMA`**
    data['Rebound_Above_EMA'] = data['EMA'] > data['Long_EMA']
    data['Break_Below_EMA'] = data['EMA'] < data['Long_EMA']

    # 📌 **Step 8: 計算 `Long_EMA_Downward`**
    data['Long_EMA_Downward'] = data['Long_EMA'].diff() < 0

    # 📌 **Step 9: 計算短期平均波動率(利用成交量衡量)**
    data['Average_Volatility_short'] = data['volume'].diff().abs().rolling(window=3).mean()

    # 📌 **Step 10: 計算中期平均波動率(利用成交量衡量)**
    data['Average_Volatility_long'] = data['volume'].diff().abs().rolling(window=9).mean()

    # **確保 data 是 DataFrame**
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    return data, scaler

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LassoCV

class FeatureEngineering:
    def __init__(self, variance_threshold=0.005, lasso_eps=1e-4, corr_threshold=0.90, remove_column_name=None,
                 selected_features=None, scaler=None):
        """
        初始化 Feature Engineering
        :param variance_threshold: 變異數閾值 (越低保留特徵越多)
        :param lasso_eps: Lasso 參數 (影響特徵選擇數量)
        :param corr_threshold: 高相關性特徵閾值
        :param remove_column_name: 需要移除的特徵名稱
        """
        self.variance_threshold = variance_threshold
        self.lasso_eps = lasso_eps
        self.corr_threshold = corr_threshold
        self.remove_column_name = remove_column_name or ['date', 'Profit_Loss_Points', 'Event', 'Label']
        
        self.var_thresh = None
        self.scaler = StandardScaler() if scaler is None else scaler
        self.lasso_model = None  
        self.selected_features = selected_features or []  

    def fit(self, df, target_column='Label'):
        """
        訓練特徵選擇，適用於歷史數據。
        :param df: DataFrame，包含所有計算後的技術指標特徵
        :param target_column: 目標變數名稱（預設為 'Label'）
        :return: DataFrame，僅保留最佳特徵
        """
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)

        # 1️⃣ **移除非特徵列**
        features = df.columns.difference(self.remove_column_name)
        X = df[features]
        y = df[target_column]

        # 2️⃣ **處理 NaN 和 Inf**
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.dropna(axis=1, inplace=True)

        # 3️⃣ **VarianceThreshold 過濾特徵**
        self.var_thresh = VarianceThreshold(threshold=self.variance_threshold)
        print(self.variance_threshold)
        X_filtered = self.var_thresh.fit_transform(X)

        # 4️⃣ **確保 `features` 長度與 `X_filtered` 一致**
        filtered_features = X.columns[self.var_thresh.get_support()]
        X_filtered = pd.DataFrame(X_filtered, columns=filtered_features)

        print(f"📌 變異數過濾後的特徵數量: {X_filtered.shape[1]}")

        # 3️⃣ **Lasso 特徵選擇**
        X_scaled = self.scaler.fit_transform(X_filtered)
        self.lasso_model = LassoCV(cv=10, random_state=42, eps=self.lasso_eps, max_iter=1000)
        self.lasso_model.fit(X_scaled, y)

        selected_features = np.array(filtered_features)[self.lasso_model.coef_ != 0]
        X_selected = X_filtered[selected_features]

        # 4️⃣ **過濾高相關性特徵**
        corr_matrix = X_selected.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > self.corr_threshold)]
        X_final = X_selected.drop(columns=to_drop, errors='ignore')

        self.selected_features = X_final.columns.tolist()
        print(f"📌 最終保留的特徵數量: {len(self.selected_features)}")
        print(f"📌 最終保留的特徵名稱: {self.selected_features}")

        return X_final, self.scaler, self.selected_features

    def transform(self, df):
        """ 應用變異數篩選、Lasso 選擇和高相關性過濾到即時數據 """
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
        
        #  **確保特徵順序與 `fit()` 時一致**
        X = df[self.selected_features]

        #  **標準化處理**
        X_scaled = self.scaler.fit_transform(X)
        return pd.DataFrame(X_scaled, columns=self.selected_features)