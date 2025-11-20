#%%

from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, r2_score
import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
from optuna.distributions import FloatDistribution, IntDistribution
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit

class HyperparameterConfig:
    """
    Store hyperparameter configurations for different machine learning and deep learning models.
    """
    @staticmethod
    def get_params(model_name):
        if model_name == "random_forest":
            return {
                'n_estimators': IntDistribution(50, 600),
                'max_depth': IntDistribution(5, 50),
                'min_samples_split': IntDistribution(2, 20),
                'min_samples_leaf': IntDistribution(1, 10),
                'max_features': FloatDistribution(0.1, 1.0),
                'bootstrap': [True, False],
            }
        elif model_name == "xgboost":
            return {
                'learning_rate': FloatDistribution(1e-4, 1e-1),
                'n_estimators': IntDistribution(50, 600),
                'max_depth': IntDistribution(3, 50),
                'subsample': FloatDistribution(0.5, 1.0),
                'colsample_bytree': FloatDistribution(0.5, 1.0),
                'reg_alpha': FloatDistribution(1e-4, 1e-1),
                'reg_lambda': FloatDistribution(1e-4, 1e-1),
            }
        elif model_name == "lightgbm":
            return {
                'learning_rate': FloatDistribution(1e-4, 1e-1),
                'n_estimators': IntDistribution(50, 600),
                'max_depth': IntDistribution(3, 50),
                'num_leaves': IntDistribution(15, 300),
                'min_child_samples': IntDistribution(10, 100),
                'subsample': FloatDistribution(0.5, 1.0),
                'colsample_bytree': FloatDistribution(0.5, 1.0),
                'reg_alpha': FloatDistribution(1e-4, 1e-1),
                'reg_lambda': FloatDistribution(1e-4, 1e-1),
            }

        elif model_name == "residual_lstm":
            return {
                'hidden_size': IntDistribution(32, 512),  # Hidden layer size
                'num_layers': IntDistribution(1, 5),     # Number of LSTM layers
                'dropout': FloatDistribution(0.1, 0.5),  # Dropout ratio
                'learning_rate': FloatDistribution(1e-4, 1e-2),  # Learning rate
                'batch_size': IntDistribution(16, 128),  # Batch size
            }

        elif model_name == "transformer":
            return {
                'num_heads': IntDistribution(1, 16),
                'num_layers': IntDistribution(1, 6),
                'd_model': IntDistribution(64, 512),
                'dropout': FloatDistribution(0.0, 0.5),
                'learning_rate': FloatDistribution(1e-5, 1e-3),
                'batch_size': IntDistribution(16, 128),
                'epochs': IntDistribution(10, 200),
            }
        elif model_name == "cnn":
            return {
                'num_filters': IntDistribution(16, 256),
                'kernel_size': IntDistribution(2, 7),
                'pool_size': IntDistribution(2, 4),
                'dropout': FloatDistribution(0.0, 0.5),
                'learning_rate': FloatDistribution(1e-4, 1e-2),
                'batch_size': IntDistribution(16, 128),
                'epochs': IntDistribution(10, 200),
            }
        elif model_name == "mamba":
            return {
                'hidden_units': IntDistribution(10, 512),
                'num_layers': IntDistribution(1, 5),
                'attention_heads': IntDistribution(1, 8),
                'dropout': FloatDistribution(0.0, 0.5),
                'learning_rate': FloatDistribution(1e-5, 1e-3),
                'batch_size': IntDistribution(16, 128),
                'epochs': IntDistribution(10, 200),
            }
        else:
            raise ValueError(f"Unknown model name: {model_name}")




class BayesianOptimizer:
    def __init__(self, model_name, metric=None):
        """
        Initialize Bayesian optimizer.
        
        :param model_name: str, Model name (must correspond to method names in HyperparameterConfig).
        :param metric: callable, Custom loss function, e.g., mean_squared_error. If None, defaults to negative mean squared error.
        """
        self.model_name = model_name
        self.metric = metric or mean_squared_error
        self.param_space = HyperparameterConfig.get_params(model_name)
        self.feature_importances_ = None  # Used to store feature importances

    def fit(self, X, y, n_splits=5, n_trials=50, random_state=42):
        """
        Perform hyperparameter tuning using TimeSeriesCV and Bayesian optimization, and calculate feature weights.
        """

        def objective(trial):
            # Generate parameters
            params = {
                key: trial.suggest_categorical(key, value) if isinstance(value, list)
                else trial.suggest_float(key, value.low, value.high) if isinstance(value, FloatDistribution)
                else trial.suggest_int(key, value.low, value.high)
                for key, value in self.param_space.items()
            }

            # Call model
            model = self._get_model(params)
            tscv = TimeSeriesSplit(n_splits=n_splits)
            errors = []

            for train_idx, val_idx in tscv.split(X):
                X_train, X_val = X[train_idx], X[val_idx]
                y_train, y_val = y[train_idx], y[val_idx]

                if isinstance(model, nn.Module):  # LSTM or PyTorch model
                    errors.append(self._train_lstm(model, X_train, y_train, X_val, y_val, params))
                else:  # Traditional machine learning model
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_val)
                    errors.append(self.metric(y_val, y_pred))

            return np.mean(errors)

        import optuna
        self.study = optuna.create_study(direction="minimize")
        self.study.optimize(objective, n_trials=n_trials)

        self.best_params = self.study.best_params
        self.best_score = self.study.best_value

        print("Best parameters:", self.best_params)
        print("Best score:", self.best_score)

        # Train best model and calculate feature importance
        best_model = self._get_model(self.best_params)
        if isinstance(best_model, nn.Module):
            self._train_lstm(best_model, X, y, None, None, self.best_params, full_train=True)
        else:
            best_model.fit(X, y)

        self.best_model = best_model

        if hasattr(best_model, "feature_importances_"):
            self.feature_importances_ = best_model.feature_importances_
        else:
            print("This model does not support feature importance extraction.")

    def _train_lstm(self, model, X_train, y_train, X_val, y_val, params, full_train=False):
        """
        Training function specifically designed for LSTM, including validation and full training.
        """
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)

        # Build DataLoader
        def get_dataloader(X, y, batch_size):
            X_tensor = torch.tensor(X, dtype=torch.float32)
            y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(1)
            dataset = TensorDataset(X_tensor, y_tensor)
            return DataLoader(dataset, batch_size=batch_size, shuffle=True)

        batch_size = params['batch_size']
        train_loader = get_dataloader(X_train, y_train, batch_size)
        val_loader = get_dataloader(X_val, y_val, batch_size) if X_val is not None else None

        optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])
        criterion = nn.MSELoss()

        # Train model
        epochs = params.get("epochs", 10)
        for epoch in range(epochs):
            model.train()
            train_loss = 0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                predictions = model(X_batch)
                loss = criterion(predictions, y_batch)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

            # Validate model
            if val_loader is not None:
                model.eval()
                val_loss = 0
                with torch.no_grad():
                    for X_batch, y_batch in val_loader:
                        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                        predictions = model(X_batch)
                        loss = criterion(predictions, y_batch)
                        val_loss += loss.item()

                print(f"Epoch {epoch + 1}/{epochs}, Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")

        # If in full training mode (full_train=True), do not return validation loss
        if full_train:
            return

        # Return validation loss
        return val_loss / len(val_loader)



    def _get_model(self, params):
        """
        Return corresponding model instance based on model name and parameters.
        
        :param params: dict, Model hyperparameters.

        :return: Model instance.
        """
        if self.model_name == "random_forest":
            return RandomForestRegressor(random_state=42, **params)
        elif self.model_name == "xgboost":
            from xgboost import XGBRegressor
            return XGBRegressor(random_state=42, **params)
        elif self.model_name == "lightgbm":
            from lightgbm import LGBMRegressor
            return LGBMRegressor(random_state=42, **params)
        elif self.model_name == "residual_lstm":
            input_size = params.get("input_size", 10)  # Default number of input features
            output_size = params.get("output_size", 1)  # Default output dimension
            hidden_size = params["hidden_size"]
            num_layers = params["num_layers"]
            dropout = params["dropout"]
            
            return ResidualLSTMModel(input_size, hidden_size, output_size, num_layers).to('cuda' if torch.cuda.is_available() else 'cpu')

        elif self.model_name == "lstm":
            from keras.models import Sequential
            from keras.layers import LSTM, Dense, Dropout

            model = Sequential()
            model.add(LSTM(params['hidden_units'], return_sequences=True, input_shape=(None, 1)))
            for _ in range(params['num_layers'] - 1):
                model.add(LSTM(params['hidden_units'], return_sequences=False))
            model.add(Dropout(params['dropout']))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mean_squared_error')
            return model
        elif self.model_name == "transformer":
            from transformers import AutoModel
            return AutoModel.from_pretrained(params['pretrained_model_name'])
        elif self.model_name == "cnn":
            from keras.models import Sequential
            from keras.layers import Conv1D, MaxPooling1D, Flatten, Dense, Dropout

            model = Sequential()
            model.add(Conv1D(filters=params['num_filters'], kernel_size=params['kernel_size'], activation='relu', input_shape=(None, 1)))
            model.add(MaxPooling1D(pool_size=params['pool_size']))
            model.add(Flatten())
            model.add(Dropout(params['dropout']))
            model.add(Dense(1))
            model.compile(optimizer='adam', loss='mean_squared_error')
            return model
        elif self.model_name == "mamba":
            # Mamba model instantiation placeholder (replace with actual implementation if available)
            raise NotImplementedError("Mamba model not yet implemented, please replace with specific implementation logic.")
        elif self.model_name == "custom":
            # Support custom models imported from external sources
            model_class = params.get('model_class')
            if not model_class:
                raise ValueError("When using 'custom' model, 'model_class' must be provided in params.")
            return model_class(**params)
        else:
            raise ValueError(f"Unknown model name: {self.model_name}")



    def get_feature_importances(self):
        """
        Return feature importances.
        
        :return: np.array, Feature importance data.
        """
        if self.feature_importances_ is not None:
            return self.feature_importances_
        else:
            raise ValueError("Feature importances have not been calculated yet. Please ensure this method is called after training.")

    def get_best_params_and_weights(self):
        """
        Return best parameters and weights (here weights equal the contribution ratio of each hyperparameter).
        
        :return: tuple, Dictionary containing best parameters and weights.
        """
        total = sum(self.best_params.values()) if all(isinstance(v, (int, float)) for v in self.best_params.values()) else None
        if total:
            weights = {key: value / total for key, value in self.best_params.items()}
        else:
            weights = {key: None for key in self.best_params.keys()}  # Return None when weights cannot be calculated
        return self.best_params, weights

class ResidualLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=3):
        super(ResidualLSTMModel, self).__init__()
        self.lstm1 = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        batch_size = x.size(0)
        h_0 = torch.zeros(self.lstm1.num_layers, batch_size, self.lstm1.hidden_size).to(x.device)
        c_0 = torch.zeros(self.lstm1.num_layers, batch_size, self.lstm1.hidden_size).to(x.device)

        out1, _ = self.lstm1(x.unsqueeze(1) if x.ndim == 2 else x, (h_0, c_0))

        h_0_2 = torch.zeros(self.lstm2.num_layers, batch_size, self.lstm2.hidden_size).to(x.device)
        c_0_2 = torch.zeros(self.lstm2.num_layers, batch_size, self.lstm2.hidden_size).to(x.device)
        out2, _ = self.lstm2(out1, (h_0_2, c_0_2))

        residual = x if x.ndim == 2 else x[:, -1, :]
        if residual.size(1) > out2.size(2):
            residual = residual[:, :out2.size(2)]
        elif residual.size(1) < out2.size(2):
            padding = torch.zeros((batch_size, out2.size(2) - residual.size(1))).to(x.device)
            residual = torch.cat((residual, padding), dim=1)
        out = out2[:, -1, :] + residual

        out = self.dropout(out)
        out = self.fc(out)
        return out

# # 創建數據集
# X, y = make_regression(n_samples=1000, n_features=10, noise=0.1, random_state=42)
# X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# # 使用貝葉斯優化器進行隨機森林超參數調整
# optimizer = BayesianOptimizer(model_name="lightgbm")
# optimizer.fit(X_train, y_train, n_splits=5, n_trials=10)

# # 獲取最佳參數和權重
# best_params, weights = optimizer.get_best_params_and_weights()
# print("最佳參數:", best_params)
# print("參數權重:", weights)

# # 獲取特徵重要性
# feature_importances = optimizer.get_feature_importances()
# print("特徵重要性:", feature_importances)
# # %%
# import numpy as np
# import torch
# from sklearn.metrics import mean_squared_error
# from sklearn.model_selection import train_test_split

# # 模擬數據生成
# np.random.seed(42)
# X = np.random.rand(1000, 10)  # 假設有 1000 條樣本，每條樣本有 10 個特徵
# y = np.random.rand(1000)      # 對應的標籤

# # 將數據拆分為訓練集和測試集
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # 定義 BayesianOptimizer
# optimizer = BayesianOptimizer(model_name="residual_lstm")

# # 執行貝葉斯優化
# optimizer.fit(X_train, y_train, n_splits=5, n_trials=20)

# # 輸出最佳參數和得分
# print("最佳參數:", optimizer.best_params)
# print("最佳得分:", optimizer.best_score)

# # 使用最佳參數訓練的模型
# best_model = optimizer.best_model

# # 測試最佳模型
# if isinstance(best_model, torch.nn.Module):  # 如果是 LSTM 模型
#     best_model.eval()
#     with torch.no_grad():
#         X_test_tensor = torch.tensor(X_test, dtype=torch.float32).to('cuda' if torch.cuda.is_available() else 'cpu')
#         y_test_tensor = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1).to('cuda' if torch.cuda.is_available() else 'cpu')
#         predictions = best_model(X_test_tensor).cpu().numpy()
# else:  # 如果是傳統機器學習模型
#     predictions = best_model.predict(X_test)

# # 計算測試集上的性能
# test_mse = mean_squared_error(y_test, predictions)
# print("測試集上的 MSE:", test_mse)

# %%
