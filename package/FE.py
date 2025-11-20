import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from ta import add_all_ta_features  # Import `ta` technical indicators library
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold
from sklearn.linear_model import LassoCV
from package.alpha_eric import AlphaFactory

def calculate_realtime_features(
    data, slope_window=3, ema_window=9, avg_vol_window=9, long_ema_window=13, scaler=None
):
    """
    **Real-time Feature Calculation**
    1. Calculate `ta` technical indicators
    2. Calculate Bollinger Bands
    3. Calculate `Lower_Band_Slope` and `Normalized_Slope`
    4. Calculate `Slope_Change`
    5. Calculate `EMA`, `Long_EMA`
    6. Calculate `Rebound_Above_EMA` (breakthrough above short-term moving average)
    7. Calculate `Long_EMA_Downward` (long-term trend decline)
    8. Calculate `Average_Volatility`
    9. **Does not determine `Event`, suitable for real-time data**
    
    :param data: `pd.DataFrame`, containing real-time market data
    :param scaler: `MinMaxScaler`, if provided, use the same standardization
    :return: `pd.DataFrame` with new features, `scaler`
    """

    # Ensure sufficient data for calculation
    if len(data) < max(slope_window, ema_window, avg_vol_window, long_ema_window):
        raise ValueError("Insufficient data, please provide more historical data!")

    # Step 1: Add `ta` technical indicators and custom alpha factors
    data = add_all_ta_features(
        data, open="open", high="high", low="low", close="close", volume="volume", fillna=True
    )

    alpha = AlphaFactory(data)
    data = alpha.add_all_alphas(days=[3, 9, 20, 60, 120, 240])

    # Step 2: Calculate Bollinger Bands
    data['Middle_Band'] = data['close'].rolling(window=20).mean()
    data['Std_Dev'] = data['close'].rolling(window=20).std()
    data['Upper_Band'] = data['Middle_Band'] + 2 * data['Std_Dev']
    data['Lower_Band'] = data['Middle_Band'] - 2 * data['Std_Dev']

    # Step 3: Calculate `Lower_Band_Slope`
    data['Lower_Band_Slope'] = data['Lower_Band'].diff(slope_window) / data['Lower_Band'].shift(slope_window)
    data['Lower_Band_Slope'] = data['Lower_Band_Slope'].fillna(0)

    # Step 4: Normalize `Lower_Band_Slope`
    if scaler is None:
        scaler = MinMaxScaler(feature_range=(-1, 1))

    data['Normalized_Slope'] = scaler.fit_transform(data[['Lower_Band_Slope']])

    # Step 5: Calculate `Slope_Change`
    data['Slope_Change'] = data['Normalized_Slope'] - data['Normalized_Slope'].shift(slope_window)
    data['Slope_Change'] = data['Slope_Change'].fillna(0)

    # Step 6: Calculate short-term & long-term `EMA`
    data['EMA'] = data['close'].ewm(span=ema_window, adjust=False).mean()
    data['Long_EMA'] = data['close'].ewm(span=long_ema_window, adjust=False).mean()

    # Step 7: Calculate `Rebound_Above_EMA, Break_Below_EMA`
    data['Rebound_Above_EMA'] = data['EMA'] > data['Long_EMA']
    data['Break_Below_EMA'] = data['EMA'] < data['Long_EMA']

    # Step 8: Calculate `Long_EMA_Downward`
    data['Long_EMA_Downward'] = data['Long_EMA'].diff() < 0

    # Step 9: Calculate short-term average volatility (measured using volume)
    data['Average_Volatility_short'] = data['volume'].diff().abs().rolling(window=3).mean()

    # Step 10: Calculate medium-term average volatility (measured using volume)
    data['Average_Volatility_long'] = data['volume'].diff().abs().rolling(window=9).mean()

    # Ensure data is a DataFrame
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
        Initialize Feature Engineering
        :param variance_threshold: Variance threshold (lower value retains more features)
        :param lasso_eps: Lasso parameter (affects number of features selected)
        :param corr_threshold: High correlation feature threshold
        :param remove_column_name: Feature names to remove
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
        Train feature selection, suitable for historical data.
        :param df: DataFrame containing all calculated technical indicator features
        :param target_column: Target variable name (default: 'Label')
        :return: DataFrame with only best features retained
        """
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)

        # Step 1: Remove non-feature columns
        features = df.columns.difference(self.remove_column_name)
        X = df[features]
        y = df[target_column]

        # Step 2: Handle NaN and Inf
        X.replace([np.inf, -np.inf], np.nan, inplace=True)
        X.dropna(axis=1, inplace=True)

        # Step 3: VarianceThreshold feature filtering
        self.var_thresh = VarianceThreshold(threshold=self.variance_threshold)
        print(self.variance_threshold)
        X_filtered = self.var_thresh.fit_transform(X)

        # Step 4: Ensure `features` length matches `X_filtered`
        filtered_features = X.columns[self.var_thresh.get_support()]
        X_filtered = pd.DataFrame(X_filtered, columns=filtered_features)

        print(f"Number of features after variance filtering: {X_filtered.shape[1]}")

        # Step 5: Lasso feature selection
        X_scaled = self.scaler.fit_transform(X_filtered)
        self.lasso_model = LassoCV(cv=10, random_state=42, eps=self.lasso_eps, max_iter=1000)
        self.lasso_model.fit(X_scaled, y)

        selected_features = np.array(filtered_features)[self.lasso_model.coef_ != 0]
        X_selected = X_filtered[selected_features]

        # Step 6: Filter highly correlated features
        corr_matrix = X_selected.corr().abs()
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > self.corr_threshold)]
        X_final = X_selected.drop(columns=to_drop, errors='ignore')

        self.selected_features = X_final.columns.tolist()
        print(f"Final number of features retained: {len(self.selected_features)}")
        print(f"Final feature names retained: {self.selected_features}")

        return X_final, self.scaler, self.selected_features

    def transform(self, df):
        """Apply variance filtering, Lasso selection, and high correlation filtering to real-time data"""
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        df.fillna(0, inplace=True)
        
        # Ensure feature order matches `fit()` time
        X = df[self.selected_features]

        # Standardization
        X_scaled = self.scaler.fit_transform(X)
        return pd.DataFrame(X_scaled, columns=self.selected_features)