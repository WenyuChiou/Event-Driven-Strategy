import pandas as pd
import numpy as np
import os
import json
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier




class BayesianOptimizerWrapper:
    """
    整合 ModelLoader 中的貝氏最佳化功能的包裝類
    """
    def __init__(self, model_name):
        """
        初始化貝氏最佳化包裝類
        
        Parameters:
        -----------
        model_name : str
            模型名稱，支援 'randomforest', 'gradientboosting', 'xgboost', 'lightgbm'
        """
        # 模型名稱映射 (將 TradingModel 的模型名稱映射到 ModelLoader 的模型名稱)
        self.model_name_map = {
            'randomforest': 'random_forest',
            'gradientboosting': None,  # ModelLoader 不直接支援 gradientboosting
            'xgboost': 'xgboost',
            'lightgbm': 'lightgbm'
        }
        
        self.model_name = model_name.lower()
        
        if self.model_name not in self.model_name_map:
            raise ValueError(f"不支援的模型: {model_name}. 支援的模型有: {list(self.model_name_map.keys())}")
        
        if self.model_name_map[self.model_name] is None:
            raise ValueError(f"ModelLoader 的 BayesianOptimizer 不支援 {model_name}")
        
        # 初始化 ModelLoader 的 BayesianOptimizer
        self.ml_optimizer = BayesianOptimizer(model_name=self.model_name_map[self.model_name])
        
        self.best_params = None
        self.weights = None
        self.feature_importances = None
    
    def fit(self, X, y, n_splits=5, n_trials=100):
        """
        使用 ModelLoader 的貝氏最佳化找到最佳超參數
        
        Parameters:
        -----------
        X : array-like
            特徵數據
        y : array-like
            目標數據
        n_splits : int, default=5
            交叉驗證分割數
        n_trials : int, default=100
            優化嘗試次數
        """
        # 確保 X 和 y 是 numpy 數組
        if isinstance(X, pd.DataFrame):
            X_array = X.to_numpy()
        else:
            X_array = X
        
        if isinstance(y, pd.Series):
            y_array = y.to_numpy()
        else:
            y_array = y
        
        # 執行貝氏最佳化
        self.ml_optimizer.fit(X=X_array, y=y_array, n_splits=n_splits, n_trials=n_trials)
        
        # 獲取最佳參數和權重
        best_params, weights = self.ml_optimizer.get_best_params_and_weights()
        
        # 獲取特徵重要性
        try:
            self.feature_importances = self.ml_optimizer.get_feature_importances()
        except Exception as e:
            print(f"無法獲取特徵重要性: {e}")
            self.feature_importances = None
        
        # 將最佳參數儲存起來
        self.best_params = best_params
        self.weights = weights
        
        print(f"貝氏最佳化完成，找到最佳參數: {self.best_params}")
        
        return self
    
    def get_best_params_and_weights(self):
        """
        獲取最佳超參數和權重
        
        Returns:
        --------
        tuple
            (最佳超參數, 權重)
        """
        if self.best_params is None:
            raise ValueError("模型尚未進行貝氏最佳化. 請先呼叫 fit() 方法")
        
        return self.best_params, self.weights
    
    def get_feature_importances(self):
        """
        獲取特徵重要性
        
        Returns:
        --------
        array-like
            特徵重要性
        """
        if self.feature_importances is None:
            raise ValueError("特徵重要性尚未計算或不可用於此模型")
        
        return self.feature_importances

class BayesianOptimizer:
    """
    Uses Bayesian optimization to find the best hyperparameters.
    
    Note: This class needs to be further developed according to your actual implementation,
    as the original code does not provide a complete implementation of BayesianOptimizer.
    """
    def __init__(self, model_name):
        """
        Initialize the Bayesian optimizer.
        
        Parameters:
        -----------
        model_name : str
            The name of the model. Supported models include 'randomforest', 'gradientboosting', 'xgboost', 
            'lightgbm'
        """
        self.model_name = model_name.lower()
        self.search_space = self._get_search_space()
        self.best_params = None
        self.best_score = None
        self.weights = None
        self.feature_importances = None
        self.supported_models = [
            'randomforest', 'gradientboosting', 'xgboost', 'lightgbm'
        ]
        
        if self.model_name not in self.supported_models:
            raise ValueError(f"Unsupported model: {model_name}. Supported models are: {self.supported_models}")
    
    def _get_search_space(self):
        """
        Get the hyperparameter search space for the model.
        
        Returns:
        --------
        dict
            The hyperparameter search space.
        """
        # Define the appropriate search space for each model.
        # The following is an example search space that should be adjusted based on your needs.
        search_spaces = {
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
            }
        }
        return search_spaces.get(self.model_name, {})
    
    def fit(self, X, y, n_splits=5, n_trials=100):
        """
        Use Bayesian optimization to find the best hyperparameters.
        
        Parameters:
        -----------
        X : array-like
            Feature data.
        y : array-like
            Target data.
        n_splits : int, default=5
            Number of cross-validation splits.
        n_trials : int, default=100
            Number of optimization trials.
        """
        # 檢查是否可以使用 ModelLoader 的貝氏最佳化
        try:
            from package.ModelLoader import BayesianOptimizer as MLBayesianOptimizer
            
            # 模型名稱映射
            model_name_map = {
                'randomforest': 'random_forest',
                'xgboost': 'xgboost',
                'lightgbm': 'lightgbm'
            }
            
            if self.model_name in model_name_map:
                print(f"使用 ModelLoader 的 BayesianOptimizer 進行優化...")
                ml_optimizer = MLBayesianOptimizer(model_name=model_name_map[self.model_name])
                
                # 執行優化
                ml_optimizer.fit(X=X, y=y, n_splits=n_splits, n_trials=n_trials)
                
                # 獲取最佳參數和權重
                self.best_params, self.weights = ml_optimizer.get_best_params_and_weights()
                
                # 獲取特徵重要性
                try:
                    self.feature_importances = ml_optimizer.get_feature_importances()
                except:
                    self._calculate_feature_importances(X, y)
                
                print(f"ModelLoader 優化完成，找到最佳參數: {self.best_params}")
                return self
            
        except ImportError:
            print("未找到 ModelLoader 模組，使用原始優化方法...")
        except Exception as e:
            print(f"使用 ModelLoader 優化時發生錯誤: {e}，使用原始優化方法...")
        
        # 如果無法使用 ModelLoader 的優化器，使用原始方法
        default_params = {
            'randomforest': {
                'n_estimators': 100,
                'max_depth': 10,
                'min_samples_split': 5,
                'min_samples_leaf': 2
            },
            'gradientboosting': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
                'subsample': 0.8
            },
            'xgboost': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            },
            'lightgbm': {
                'n_estimators': 100,
                'learning_rate': 0.1,
                'max_depth': 5,
                'num_leaves': 31,
                'subsample': 0.8,
                'colsample_bytree': 0.8
            }
        }
        
        self.best_params = default_params.get(self.model_name, {})
        self.weights = {'accuracy': 0.5, 'precision': 0.25, 'recall': 0.25}
        self._calculate_feature_importances(X, y)
        
        print(f"使用預設參數完成: {self.best_params}")
        
        return self
    
    def _calculate_feature_importances(self, X, y):
        """
        Calculate feature importances.
        
        Parameters:
        -----------
        X : array-like
            Feature data.
        y : array-like
            Target data.
        """
        # Create a model based on the best parameters.
        model = self._create_model()
        
        # Train the model.
        model.fit(X, y)
        
        # Retrieve feature importances.
        if self.model_name in ['randomforest', 'gradientboosting', 'xgboost', 'lightgbm']:
            self.feature_importances = model.feature_importances_
    
    def get_best_params_and_weights(self):
        """
        Retrieve the best hyperparameters and weights.
        
        Returns:
        --------
        tuple
            (best hyperparameters, weights)
        """
        return self.best_params, self.weights
    
    def get_feature_importances(self):
        """
        Retrieve the feature importances.
        
        Returns:
        --------
        array-like
            Feature importances.
        """
        return self.feature_importances
    
    def _create_model(self, num = 3):
        """
        Create a model instance based on the model name.
        
        Returns:
        --------
        model
            A machine learning model.
        """
        if self.model_name == 'randomforest':
            return RandomForestClassifier(**self.best_params, random_state=42)
        elif self.model_name == 'gradientboosting':
            return GradientBoostingClassifier(**self.best_params, random_state=42)
        elif self.model_name == 'xgboost':
            return XGBClassifier(**self.best_params, random_state=42)
        elif self.model_name == 'lightgbm':
            return LGBMClassifier(**self.best_params, random_state=42, num_class=num)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

class TradingModel:
    """
    Trading model class responsible for model training, evaluation, and prediction.
    """
    def __init__(self, model_name='lightgbm', params=None):
        """
        Initialize the trading model.
        
        Parameters:
        -----------
        model_name : str, default='lightgbm'
            The model name. Supported models include 'randomforest', 'gradientboosting', 'xgboost', 
            'lightgbm'
        params : dict, optional
            Model parameters. If None, default parameters will be used.
        """
        self.model_name = model_name.lower()
        self.params = params
        self.model = None
        self.feature_names = None
        self.supported_models = [
            'randomforest', 'gradientboosting', 'xgboost', 'lightgbm'
        ]
        
        if self.model_name not in self.supported_models:
            raise ValueError(f"Unsupported model: {model_name}. Supported models are: {self.supported_models}")
    
    def fit(self, X, y):
        """
        Train the model.
        
        Parameters:
        -----------
        X : pd.DataFrame or array-like
            Feature data.
        y : array-like
            Target data.
        
        Returns:
        --------
        self
            The trained model instance.
        """
        if isinstance(X, pd.DataFrame):
            self.feature_names = X.columns.tolist()
        
        self.model = self._create_model()
        self.model.fit(X, y)
        
        return self
    
    def predict_proba(self, X):
        """
        Predict class probabilities.
        
        Parameters:
        -----------
        X : pd.DataFrame or array-like
            Feature data.
        
        Returns:
        --------
        array-like
            Predicted probabilities.
        """
        if self.model is None:
            raise ValueError("Model is not trained. Please call fit() first.")
        
        return self.model.predict_proba(X)
    
    def predict(self, X):
        """
        Predict class labels.
        
        Parameters:
        -----------
        X : pd.DataFrame or array-like
            Feature data.
        
        Returns:
        --------
        array-like
            Predicted class labels.
        """
        if self.model is None:
            raise ValueError("Model is not trained. Please call fit() first.")
        
        return self.model.predict(X)
    
    def evaluate(self, X, y):
        """
        Evaluate the model's performance.
        
        Parameters:
        -----------
        X : pd.DataFrame or array-like
            Feature data.
        y : array-like
            Target data.
        
        Returns:
        --------
        dict
            Evaluation metrics.
        """
        if self.model is None:
            raise ValueError("Model is not trained. Please call fit() first.")
        
        y_pred = self.model.predict(X)
        y_proba = self.model.predict_proba(X)
        
        # Calculate various evaluation metrics.
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, average='weighted'),
            'recall': recall_score(y, y_pred, average='weighted'),
            'f1': f1_score(y, y_pred, average='weighted')
        }
        
        # For binary classification, calculate AUC.
        if len(np.unique(y)) == 2:
            metrics['auc'] = roc_auc_score(y, y_proba[:, 1])
        
        return metrics
    
    def cross_validate(self, X, y, n_splits=5):
        """
        Evaluate the model using time series cross-validation.
        
        Parameters:
        -----------
        X : pd.DataFrame or array-like
            Feature data.
        y : array-like
            Target data.
        n_splits : int, default=5
            Number of cross-validation splits.
        
        Returns:
        --------
        dict
            Evaluation metrics.
        """
        tscv = TimeSeriesSplit(n_splits=n_splits)
        metrics = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'f1': []
        }
        
        # Add AUC metric for binary classification.
        if len(np.unique(y)) == 2:
            metrics['auc'] = []
        
        for train_index, test_index in tscv.split(X):
            X_train, X_test = (X.iloc[train_index] if isinstance(X, pd.DataFrame) else X[train_index],
                               X.iloc[test_index] if isinstance(X, pd.DataFrame) else X[test_index])
            y_train, y_test = (y.iloc[train_index] if isinstance(y, pd.Series) else y[train_index],
                               y.iloc[test_index] if isinstance(y, pd.Series) else y[test_index])
            
            # Train the model.
            model = self._create_model()
            model.fit(X_train, y_train)
            
            # Make predictions.
            y_pred = model.predict(X_test)
            
            # Calculate metrics.
            metrics['accuracy'].append(accuracy_score(y_test, y_pred))
            metrics['precision'].append(precision_score(y_test, y_pred, average='weighted'))
            metrics['recall'].append(recall_score(y_test, y_pred, average='weighted'))
            metrics['f1'].append(f1_score(y_test, y_pred, average='weighted'))
            
            # For binary classification, calculate AUC.
            if len(np.unique(y)) == 2:
                y_proba = model.predict_proba(X_test)[:, 1]
                metrics['auc'].append(roc_auc_score(y_test, y_proba))
        
        # Calculate mean metrics.
        mean_metrics = {k: np.mean(v) for k, v in metrics.items()}
        std_metrics = {k + '_std': np.std(v) for k, v in metrics.items()}
        
        # Merge mean and standard deviation.
        result = {**mean_metrics, **std_metrics}
        
        return result

    def _create_model(self):
        """
        Create a model instance based on the model name and parameters.
        
        Returns:
        --------
        model
            A machine learning model instance.
        """
        if self.model_name == 'randomforest':
            return RandomForestClassifier(**(self.params or {}), random_state=42)
        elif self.model_name == 'gradientboosting':
            return GradientBoostingClassifier(**(self.params or {}), random_state=42)
        elif self.model_name == 'xgboost':
            return XGBClassifier(**(self.params or {}), random_state=42)
        elif self.model_name == 'lightgbm':
            return LGBMClassifier(**(self.params or {}), random_state=42)
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")

    def save_model(self, model_path, scaler_path=None, feature_path=None):
        """
        Save the model and related components.
        
        Parameters:
        -----------
        model_path : str
            The file path to save the model.
        scaler_path : str, optional
            The file path to save the scaler.
        feature_path : str, optional
            The file path to save the feature names.
        """
        if self.model is None:
            raise ValueError("Model is not trained. Please call fit() first.")
        
        # Save the model.
        joblib.dump(self.model, model_path)
        print(f"Model saved to: {model_path}")
        
        # Save feature names if a path is provided.
        if feature_path is not None and self.feature_names is not None:
            features_df = pd.DataFrame({'feature': self.feature_names})
            features_df.to_excel(feature_path, index=False)
            print(f"Feature names saved to: {feature_path}")

    @classmethod
    def load_model(cls, model_path):
        """
        Load a trained model.
        
        Parameters:
        -----------
        model_path : str
            The file path from which to load the model.
            
        Returns:
        --------
        TradingModel
            The loaded model instance.
        """
        model = joblib.load(model_path)
        
        # Infer the model name based on the loaded model's type.
        model_type = type(model).__name__
        model_name_map = {
            'RandomForestClassifier': 'randomforest',
            'GradientBoostingClassifier': 'gradientboosting',
            'XGBClassifier': 'xgboost',
            'LGBMClassifier': 'lightgbm'
        }
        
        model_name = model_name_map.get(model_type, 'unknown')
        
        # Create a TradingModel instance.
        trading_model = cls(model_name=model_name)
        trading_model.model = model
        
        # If the model has the attribute feature_names_in_, retrieve the feature names.
        if hasattr(model, 'feature_names_in_'):
            trading_model.feature_names = model.feature_names_in_.tolist()
        
        return trading_model

    def plot_feature_importance(self, top_n=20, figsize=(12, 8)):
        """
        Plot the feature importances.
        
        Parameters:
        -----------
        top_n : int, default=20
            Display the top N most important features.
        figsize : tuple, default=(12, 8)
            The figure size.
        """
        if self.model is None:
            raise ValueError("Model is not trained. Please call fit() first.")
        
        if not hasattr(self.model, 'feature_importances_'):
            raise ValueError(f"{self.model_name} model does not have a feature_importances_ attribute")
        
        importances = self.model.feature_importances_
        
        if self.feature_names is None:
            # If feature names are not provided, use indices.
            feature_names = [f'F{i}' for i in range(len(importances))]
        else:
            feature_names = self.feature_names
        
        # Create a DataFrame for the feature importances.
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        })
        
        # Sort and select the top N features.
        feature_importance = feature_importance.sort_values('importance', ascending=False).head(top_n)
        
        # Plot the feature importances.
        plt.figure(figsize=figsize)
        plt.barh(feature_importance['feature'], feature_importance['importance'])
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.title(f'Top {top_n} Feature Importances')
        plt.tight_layout()
        plt.show()
        
        return feature_importance

# 新增一個輔助類，用於整合 TradingModel 和 ModelLoader 的貝氏最佳化功能
class TradingModelOptimizer:
    """
    輔助類，用於優化 TradingModel 的超參數
    """
    @staticmethod
    def optimize(X, y, model_name='lightgbm', n_splits=5, n_trials=100, random_state=42):
        """
        使用 ModelLoader 的貝氏最佳化找到最佳超參數，並建立優化後的 TradingModel
        
        Parameters:
        -----------
        X : pd.DataFrame or array-like
            特徵數據
        y : array-like
            目標數據
        model_name : str, default='lightgbm'
            模型名稱，支援 'randomforest', 'xgboost', 'lightgbm'
        n_splits : int, default=5
            交叉驗證分割數
        n_trials : int, default=100
            優化嘗試次數
        random_state : int, default=42
            隨機種子
            
        Returns:
        --------
        TradingModel
            優化後的交易模型
        """
        # 檢查模型名稱
        model_name = model_name.lower()
        supported_models = ['randomforest', 'xgboost', 'lightgbm']
        
        if model_name not in supported_models:
            raise ValueError(f"不支援的模型: {model_name}. 支援的模型有: {supported_models}")
        
        # 建立 BayesianOptimizerWrapper 類的實例
        optimizer = BayesianOptimizerWrapper(model_name=model_name)
        
        # 執行優化
        optimizer.fit(X=X, y=y, n_splits=n_splits, n_trials=n_trials)
        
        # 獲取最佳參數
        best_params, _ = optimizer.get_best_params_and_weights()
        
        # 建立 TradingModel
        model = TradingModel(model_name=model_name, params=best_params)
        
        # 訓練模型
        model.fit(X, y)
        
        return model