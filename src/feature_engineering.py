import pandas as pd
import numpy as np
import talib
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LassoCV
from ta import add_all_ta_features
# 導入外部套件
from package.alpha_eric import AlphaFactory
from package.FE import FeatureEngineering as PackageFE, calculate_realtime_features

def calculate_features(data, slope_window=3, ema_window=9, 
                       avg_vol_window=9, long_ema_window=13, 
                       scaler=None):
    """
    使用外部套件的 calculate_realtime_features 函數計算交易特徵。
    
    Parameters:
    -----------
    data : pd.DataFrame
        包含 OHLCV 數據的 DataFrame
    slope_window : int, default=3
        計算斜率的窗口大小
    ema_window : int, default=9
        計算 EMA 的窗口大小
    avg_vol_window : int, default=9
        計算平均波動率的窗口大小
    long_ema_window : int, default=13
        計算長期 EMA 的窗口大小
    scaler : MinMaxScaler, optional
        已訓練的 scaler，如果提供則使用相同的縮放方式
        
    Returns:
    --------
    tuple
        (添加了特徵的 DataFrame, 使用的 scaler)
    """
    # 調用外部套件的函數計算特徵
    return calculate_realtime_features(
        data, 
        slope_window=slope_window, 
        ema_window=ema_window, 
        avg_vol_window=avg_vol_window, 
        long_ema_window=long_ema_window, 
        scaler=scaler
    )

class FeatureEngineeringWrapper:
    """
    特徵工程包裝類，使用外部 PackageFE 並提供額外的功能。
    """
    def __init__(self, variance_threshold=0.005, lasso_eps=1e-4, 
                 corr_threshold=0.9, remove_column_name=None,
                 selected_features=None, scaler=None):
        """
        初始化特徵工程包裝類
        
        Parameters:
        -----------
        variance_threshold : float, default=0.005
            變異數閾值，用於移除低變異數特徵
        lasso_eps : float, default=1e-4
            Lasso 參數，影響特徵選擇數量
        corr_threshold : float, default=0.9
            高相關性特徵閾值，用於移除高度相關特徵
        remove_column_name : list, optional
            需要移除的特徵名稱
        selected_features : list, optional
            已經選定的特徵列表
        scaler : StandardScaler, optional
            已訓練的 scaler
        """
        self.remove_column_name = remove_column_name or ['date', 'Profit_Loss_Points', 'Event', 'Label']
        self.selected_features = selected_features or []
        self.scaler = scaler
        
        # 初始化外部套件的 FeatureEngineering 類
        self.fe_instance = PackageFE(
            variance_threshold=variance_threshold,
            lasso_eps=lasso_eps,
            corr_threshold=corr_threshold,
            remove_column_name=self.remove_column_name,
            selected_features=self.selected_features,
            scaler=self.scaler
        )

    def fit(self, df, target_column='Label'):
        """
        使用外部套件的 FeatureEngineering 訓練特徵選擇，適用於歷史數據。
        
        Parameters:
        -----------
        df : pd.DataFrame
            包含所有計算後的技術指標特徵的 DataFrame
        target_column : str, default='Label'
            目標變數名稱
            
        Returns:
        --------
        tuple
            (僅保留最佳特徵的 DataFrame, 標準化器, 選擇的特徵列表)
        """
        X_final, self.scaler, self.selected_features = self.fe_instance.fit(df, target_column)
        
        print(f"📌 最終保留的特徵數量: {len(self.selected_features)}")
        print(f"📌 最終保留的特徵名稱: {self.selected_features}")
        
        return X_final, self.scaler, self.selected_features

    def transform(self, df):
        """
        使用外部套件的 FeatureEngineering 應用特徵選擇和標準化到新數據。
        
        Parameters:
        -----------
        df : pd.DataFrame
            要轉換的數據
            
        Returns:
        --------
        pd.DataFrame
            轉換後的特徵
        """
        return self.fe_instance.transform(df)
    
    def save_features(self, path):
        """
        保存所選特徵到 Excel 文件
        
        Parameters:
        -----------
        path : str
            保存路徑
        """
        features = pd.DataFrame(self.selected_features, columns=['feature'])
        features.to_excel(path, index=False)
        print(f"特徵列表已保存到: {path}")
    
    @classmethod
    def load_features(cls, path, remove_column_name=None, scaler=None):
        """
        從文件加載特徵列表
        
        Parameters:
        -----------
        path : str
            特徵列表文件路徑
        remove_column_name : list, optional
            需要移除的特徵名稱
        scaler : StandardScaler, optional
            已訓練的 scaler
            
        Returns:
        --------
        FeatureEngineeringWrapper
            初始化的特徵工程實例
        """
        features = pd.read_excel(path)
        feature_list = features['feature'].tolist()
        
        return cls(remove_column_name=remove_column_name, 
                  selected_features=feature_list, 
                  scaler=scaler)


# 為了保持與原有代碼兼容，創建別名
FeatureEngineering = FeatureEngineeringWrapper