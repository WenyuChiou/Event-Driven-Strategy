import pandas as pd
# Import FeatureEngineering class and calculation functions from external package
from package.FE import FeatureEngineering as PackageFE, calculate_realtime_features

def calculate_features(data, slope_window=3, ema_window=9, 
                       avg_vol_window=9, long_ema_window=13, 
                       scaler=None):
    """
    Calculate trading features using the calculate_realtime_features function from external package.
    
    Parameters:
    -----------
    data : pd.DataFrame
        DataFrame containing OHLCV data
    slope_window : int, default=3
        Window size for calculating slope
    ema_window : int, default=9
        Window size for calculating EMA
    avg_vol_window : int, default=9
        Window size for calculating average volatility
    long_ema_window : int, default=13
        Window size for calculating long-term EMA
    scaler : MinMaxScaler, optional
        Trained scaler, if provided, use the same scaling method
        
    Returns:
    --------
    tuple
        (DataFrame with features added, scaler used)
    """
    # Call external package function to calculate features
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
    Feature engineering wrapper class that uses external PackageFE and provides additional functionality.
    """
    def __init__(self, variance_threshold=0.005, lasso_eps=1e-4, 
                 corr_threshold=0.9, remove_column_name=None,
                 selected_features=None, scaler=None):
        """
        Initialize feature engineering wrapper class
        
        Parameters:
        -----------
        variance_threshold : float, default=0.005
            Variance threshold for removing low-variance features
        lasso_eps : float, default=1e-4
            Lasso parameter that affects the number of features selected
        corr_threshold : float, default=0.9
            High correlation threshold for removing highly correlated features
        remove_column_name : list, optional
            Feature names to remove
        selected_features : list, optional
            Already selected feature list
        scaler : StandardScaler, optional
            Trained scaler
        """
        self.remove_column_name = remove_column_name or ['date', 'Profit_Loss_Points', 'Event', 'Label']
        self.selected_features = selected_features or []
        self.scaler = scaler
        
        # Initialize external package FeatureEngineering class
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
        Train feature selection using external package FeatureEngineering, suitable for historical data.
        
        Parameters:
        -----------
        df : pd.DataFrame
            DataFrame containing all calculated technical indicator features
        target_column : str, default='Label'
            Target variable name
            
        Returns:
        --------
        tuple
            (DataFrame with only best features retained, scaler, selected feature list)
        """
        X_final, self.scaler, self.selected_features = self.fe_instance.fit(df, target_column)
        
        print(f"Final number of features retained: {len(self.selected_features)}")
        print(f"Final feature names retained: {self.selected_features}")
        
        return X_final, self.scaler, self.selected_features

    def transform(self, df):
        """
        Apply feature selection and standardization to new data using external package FeatureEngineering.
        
        Parameters:
        -----------
        df : pd.DataFrame
            Data to transform
            
        Returns:
        --------
        pd.DataFrame
            Transformed features
        """
        return self.fe_instance.transform(df)
    
    def save_features(self, path):
        """
        Save selected features to Excel file
        
        Parameters:
        -----------
        path : str
            Save path
        """
        features = pd.DataFrame(self.selected_features, columns=['feature'])
        features.to_excel(path, index=False)
        print(f"Feature list saved to: {path}")
    
    @classmethod
    def load_features(cls, path, remove_column_name=None, scaler=None):
        """
        Load feature list from file
        
        Parameters:
        -----------
        path : str
            Feature list file path
        remove_column_name : list, optional
            Feature names to remove
        scaler : StandardScaler, optional
            Trained scaler
            
        Returns:
        --------
        FeatureEngineeringWrapper
            Initialized feature engineering instance
        """
        features = pd.read_excel(path)
        feature_list = features['feature'].tolist()
        
        return cls(remove_column_name=remove_column_name, 
                  selected_features=feature_list, 
                  scaler=scaler)


# Create alias for backward compatibility with existing code
FeatureEngineering = FeatureEngineeringWrapper