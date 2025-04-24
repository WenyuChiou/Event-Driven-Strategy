# imports.py
import os
import sys

# 添加專案根目錄到 Python 路徑
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.append(project_root)

# 標準庫導入
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta

# src 模組導入
from src.backtesting import *
from src.event_detection import *
from src.feature_engineering import *
from src.ma_strategy import *
from src.model import *
from src.model_utils import *
from src.utils import *
from src.visualization import *

# package 模組導入
# 從 package 資料夾導入所需模組
from package.alpha_eric import *  # 假設 package 目錄下有這些模組
from package.FE import *
from package.SignalGenerator import *
from package.TAIndicator import *
from package.ModelLoader import *
from package.FuturesFilter import * 
from package.scraping_and_indicators import *
# 從 examples 資料夾導入所需模組
from examples.backtesting_example import *  
# from examples.basic_example import *
from examples.model_training import *

# 配置導入
# 例如從 config 讀取配置文件
def load_config(config_name):
    """從 config 目錄加載指定的配置文件"""
    config_path = os.path.join(project_root, "config", f"{config_name}.json")
    # 實現配置加載邏輯
    pass

# 初始化路徑
def init_paths():
    """初始化專案所需的各種路徑"""
    paths = {
        "data": os.path.join(project_root, "data"),
        "models": os.path.join(project_root, "models"),
        "results": os.path.join(project_root, "results"),
        "fig": os.path.join(project_root, "Fig"),
        "examples": os.path.join(project_root, "examples")
    }
    return paths

# 初始化專案所需的資源
def init_project():
    """初始化專案資源，確保所需目錄存在等"""
    paths = init_paths()
    for path in paths.values():
        os.makedirs(path, exist_ok=True)
    return paths