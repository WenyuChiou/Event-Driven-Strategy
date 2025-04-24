# app.py
from imports import *

# Initialize project paths
paths = init_project()

# Set page configuration
st.set_page_config(
    page_title="Trading Event Detection System",
    page_icon="📈",
    layout="wide"
)

# Create sidebar for navigation
st.sidebar.title("Navigation")
app_mode = st.sidebar.selectbox(
    "Choose the app mode",
    ["Home", "Data Upload", "Event Detection", "Strategy Testing", "Model Training", "Backtesting", "Real-time Simulation"]
)

# Main title
st.title("Trading Event Detection and Model Prediction System")
st.subheader("Based on the Hydraulic Jump Concept")

# Show different content based on the selected mode
if app_mode == "Home":
    # Homepage content
    st.markdown("""
    ## Welcome to the Trading Event Detection System
    
    This system helps you analyze financial data, predict trading opportunities, and test various strategies including 
    machine learning models and traditional technical approaches.
    
    ### Key Features:
    - **Multiple Strategies**: Support for ML models and traditional strategies like Moving Average (MA)
    - **Alpha Factor Generation**: Over 200 behavioral finance and technical alpha factors
    - **Signal Generation**: Smart detection of trading events based on price patterns
    - **Advanced Backtesting**: Test different strategies with detailed performance analytics
    - **Real-time Simulation**: Simulate trading in real-time environments
    - **Visualization Tools**: Generate intuitive charts and comprehensive performance reports
    
    ### The Hydraulic Jump Concept
    
    This project was inspired by the hydraulic jump phenomenon in fluid dynamics, which has striking parallels to market behavior.
    
    In financial markets, similar behavior can be observed:
    1. **Slow Price Decline**: Analogous to the gradual flow before the critical point
    2. **Rapid Price Decline**: Corresponds to the high-velocity flow
    3. **Hydraulic Jump**: The point where the market suddenly reverses direction after a rapid decline
    """)
    
    # Display hydraulic jump and market example images
    col1, col2 = st.columns(2)
    with col1:
        st.image(os.path.join(paths["fig"], "HJ.png"), caption="Hydraulic Jump Diagram")
    with col2:
        st.image(os.path.join(paths["fig"], "example.png"), caption="Market Hydraulic Jump Example")

elif app_mode == "Data Upload":
    # Data upload section
    st.header("Data Upload and Preview")
    
    uploaded_file = st.file_uploader("Upload your financial data (CSV or Excel)", type=['csv', 'xlsx'])
    
    if uploaded_file is not None:
        # Try to read the file
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)
            
            # Save the dataframe to session state for use in other sections
            st.session_state['data'] = df
            
            # Show data preview
            st.write("Data Preview:")
            st.dataframe(df.head())
            
            # Display basic statistics
            st.write("Basic Statistics:")
            st.dataframe(df.describe())
            
            # Check for required columns
            required_columns = ['Open', 'High', 'Low', 'Close', 'Volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                st.warning(f"Warning: The following required columns are missing: {', '.join(missing_columns)}")
            else:
                st.success("All required columns are present in the dataset!")
                
                # Display a simple price chart
                st.subheader("Price Chart Preview")
                fig = go.Figure()
                fig.add_trace(go.Candlestick(
                    x=df.index if 'Date' not in df.columns else df['Date'],
                    open=df['Open'],
                    high=df['High'],
                    low=df['Low'],
                    close=df['Close'],
                    name="Price"
                ))
                fig.update_layout(title="Price Chart", xaxis_title="Date", yaxis_title="Price")
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"Error loading file: {e}")

elif app_mode == "Event Detection":
    # Event detection implementation
    st.header("Trading Event Detection")
    
    if 'data' not in st.session_state:
        st.warning("Please upload data first in the 'Data Upload' section.")
    else:
        df = st.session_state['data']
        
        
        st.subheader("Configure Event Detection Parameters")
        
        col1, col2 = st.columns(2)
        with col1:
            profit_loss_window = st.slider("Profit Loss Window", 1, 5, 10)
            vp = st.slider("Volume Multiplier", 1, 2, 10, ) 
        
        with col2:
            long_threshold = st.slider("Long profit threshold", 10, 20, 30)
            short_threshold = st.slider("Short profit threshold", 10, 20, 30)
        
        if st.button("Detect Trading Events"):
            # Placeholder for event detection function call
            
            st.info("Generating features ...")
            df, scaler = calculate_features(df)
            # In a real implementation, you would call your event detection functions here
            st.info("Running event detection algorithm...")
            
            # Simulate event detection results
            # Replace this with actual event detection from your modules
            detected_events =  detect_trading_events(df, 
                                           profit_loss_window=profit_loss_window,
                                           atr_window=14, 
                                           long_profit_threshold=long_threshold,
                                           short_loss_threshold=-short_threshold,
                                           volume_multiplier=vp)
            
            if len(detected_events) > 0:
                st.success(f"Detected  potential trading events!")
                st.info("Analyzing detected events...")
                analysis = analyze_trading_events(detected_events)

                summary_data = {
                    "Metric": [
                        "Total Events",
                        "Long Events",
                        "Short Events",
                        "Win Rate",
                        "Profit Factor",
                        "Average Profit (points)"
                    ],
                    "Value": [
                        analysis['total_events'],
                        f"{analysis['long_events']} ({analysis['long_percentage']:.2f}%)",
                        f"{analysis['short_events']} ({analysis['short_percentage']:.2f}%)",
                        f"{analysis['win_rate']:.2%}",
                        f"{analysis['profit_factor']:.2f}",
                        f"{analysis['expectancy']:.2f}"
                    ]
                }

                summary_df = pd.DataFrame(summary_data)

                st.subheader("Trading Event Analysis Summary")
                st.table(summary_df)
                
                # Display events on chart
                st.subheader("Detected Events Visualization")
                fig = visualize_event_summary(detected_events, analysis, show_plots=True)
                st.pyplot(fig)

                st.session_state['events'] = detected_events  # Save events for later use
                st.success("Trading events have been detected and stored.")


            else:
                st.warning("No trading events detected with the current parameters.")

elif app_mode == "Strategy Testing":
    # Strategy testing section
    st.header("Strategy Testing")
    
    if 'data' not in st.session_state:
        st.warning("Please upload data first in the 'Data Upload' section.")
    else:
        df = st.session_state['data']
        
        st.subheader("Select Strategy")
        strategy_type = st.selectbox("Strategy Type", ["Moving Average", "Machine Learning", "Signal Generator"])
        
        if strategy_type == "Moving Average":
            # MA Strategy configuration
            st.subheader("Moving Average Strategy Configuration")
            
            col1, col2 = st.columns(2)
            with col1:
                ma_period = st.slider("MA Period", 3, 100, 20)
                commission = st.number_input("Commission (%)", 0.0, 1.0, 0.1, 0.01)
            
            with col2:
                price_col = st.selectbox("Price Column", ['Close', 'Open', 'High', 'Low'])
                min_hold_periods = st.slider("Minimum Holding Periods", 1, 50, 5)
            
            optimize_params = st.checkbox("Optimize Parameters")
            
            if st.button("Run MA Strategy"):
                st.info("Running Moving Average strategy...")
                
                # Run MA strategy 
                # Replace with actual implementation using your ma_strategy module
                result = backtest_ma_strategy(df, 
                                        ma_period=ma_period, 
                                        commission=commission, 
                                        min_hold_periods=min_hold_periods)
                
                # Display results
                st.subheader("Strategy Performance")
                for key, value in result.items():
                    if isinstance(value, float):
                        st.write(f"**{key}**: {value:.4f}")
                    else:
                        st.write(f"**{key}**: {value}")
                
        elif strategy_type == "Machine Learning":
            st.subheader("Machine Learning Strategy Configuration")
            
            # Model selection
            model_type = st.selectbox("Model Type", 
                                     ["randomforest", "gradientboosting", "xgboost", "lightgbm", "catboost"])
            
            # Let the user select an existing model or train a new one
            model_option = st.radio("Model Selection", ["Use existing model", "Train new model"])
            
            if model_option == "Use existing model":
                # List available models from the models directory
                model_files = [f for f in os.listdir(paths["models"]) if f.endswith('.joblib')]
                
                if not model_files:
                    st.warning("No saved models found. Please train a new model.")
                else:
                    selected_model = st.selectbox("Select model", model_files)
                    model_path = os.path.join(paths["models"], selected_model)
                    
                    if st.button("Run ML Strategy"):
                        st.info(f"Running strategy with model: {selected_model}")
                        # Implement the ML strategy run with the selected model
            
            else:  # Train new model
                st.subheader("Model Training Parameters")
                
                col1, col2 = st.columns(2)
                with col1:
                    trials = st.slider("Number of Trials", 10, 500, 100)
                    test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
                
                with col2:
                    target_col = st.selectbox("Target Column", ['return_1d', 'return_5d', 'return_10d'])
                    feature_count = st.slider("Max Features", 10, 100, 50)
                
                if st.button("Train Model"):
                    st.info(f"Training {model_type} model...")
                    # Implement the model training logic

elif app_mode == "Model Training":
    # Model training section
    st.header("Machine Learning Model Training")
    
    
    if 'events' not in st.session_state:
        st.warning("Please detect trading events first in the 'Data Upload' section.")
    else:
        if st.button("Start Feature Engineering"):
            st.info("Starting feature engineering...")
            df = st.session_state['events']
            
            st.subheader("Feature Engineering")
            
            # Feature generation options
            st.write("Select feature types to generate:")
            use_ta = st.checkbox("Technical Indicators", value=True)
            use_alpha = st.checkbox("Alpha Factors", value=True)
            use_custom = st.checkbox("Custom Features", value=False)

            st.write("Feature Cleaning...")
            df.drop(columns=['Session', 'Day_Of_Week','Valid_Trading_Time','Event_Type','hour_minute'], inplace=True)
            remove_column_name = [ 'date','Profit_Loss_Points', 'Event', 'Label']
            
            ## Display the columns to be removed
            for col in remove_column_name:
                if col not in df.columns:
                    st.warning(f"Warning: The column '{col}' does not exist in the dataset and will not be removed.")
            
            
            feature_engineering = FeatureEngineeringWrapper(remove_column_name=remove_column_name)
            X_final, scaler, selected_features = feature_engineering.fit(df)      

            st.write(f"Feature engineering completed, total {len(X_final.columns)} features")

            # Model configuration
            st.subheader("Model Configuration")
            
            col1, col2 = st.columns(2)
            with col1:
                model_type = st.selectbox("Model Type", 
                                        ['randomforest', 'gradientboosting', 'xgboost', 'lightgbm', 'catboost'])
                # target_col = st.selectbox("Target Variable", ['return_1d', 'return_5d', 'return_10d'])
            
            with col2:
                trials = st.slider("Hyperparameter Tuning Trials", 10, 500, 100)
                splits = st.slider("Split size", 1, 10, 5, 1)
            
            if st.button("Start Model Training"):
                st.info(f"Hyperparameter tuning for {model_type} model...")

                X = X_final
                y = df['Label']

                optimizer = BayesianOptimizerWrapper(model_name=model_type)
                optimizer.fit(X=X.to_numpy(), y=y.to_numpy(), n_splits=splits, n_trials=trials)

                st.success("Hyperparameter tuning completed!")

                st.info(f"Training {model_type} model...")
                            
                # Model training
                st.write("Training model...")

                # Best parameters and weights
                best_params, weights = optimizer.get_best_params_and_weights()
                st.write("Best parameters:", best_params)
                st.write("Parameter weights:", weights)
                
                # Feature importance
                feature_importances = optimizer.get_feature_importances()
                st.write("Feature importances:", feature_importances)
                
                # Train the final model with the best parameters
                st.info("Training final model with best parameters...")
                model = TradingModel(model_name=model_type, params=best_params)
                model.fit(X, y)
                
                st.success("Model training completed!")

                metrics = model.evaluate(X, y)
                st.write("Model evaluation metrics:")
                metrics_df = pd.DataFrame(metrics.items(), columns=["Metric", "Value"])
                metrics_df["Value"] = metrics_df["Value"].apply(lambda x: f"{x:.4f}" if isinstance(x, float) else x)

                st.subheader("訓練集評估指標")
                st.table(metrics_df)       

            
elif app_mode == "Backtesting":
    # Backtesting section
    st.header("Strategy Backtesting")
    
    if 'data' not in st.session_state:
        st.warning("Please upload data first in the 'Data Upload' section.")
    else:
        df = st.session_state['data']
        
        st.subheader("Backtesting Configuration")
        
        # Strategy selection
        strategy_type = st.selectbox("Strategy Type", ["Moving Average", "Machine Learning", "Signal Generator"])
        
        # Common parameters
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input("Start Date", value=datetime.now() - timedelta(days=365))
            initial_capital = st.number_input("Initial Capital", 10000.0, 1000000.0, 100000.0, 10000.0)
        
        with col2:
            end_date = st.date_input("End Date", value=datetime.now())
            commission = st.number_input("Commission (%)", 0.0, 1.0, 0.1, 0.01)
        
        # Strategy specific parameters
        if strategy_type == "Moving Average":
            ma_period = st.slider("MA Period", 3, 100, 20)
        
        elif strategy_type == "Machine Learning":
            # List available models
            model_files = [f for f in os.listdir(paths["models"]) if f.endswith('.joblib')]
            
            if not model_files:
                st.warning("No saved models found. Please train a model first.")
            else:
                selected_model = st.selectbox("Select model", model_files)
                threshold = st.slider("Prediction Threshold", 0.0, 0.1, 0.005, 0.001)
        
        elif strategy_type == "Signal Generator":
            signal_threshold = st.slider("Signal Threshold", 0.0, 0.1, 0.005, 0.001)
            lookback = st.slider("Lookback Period", 5, 50, 20)
        
        if st.button("Run Backtest"):
            st.info("Running backtest...")
            
            # Simulate backtest execution
            # Replace with actual backtest logic from your modules
            
            # Mock results
            st.success("Backtest completed!")
            
            # Display performance metrics
            st.subheader("Performance Metrics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Return", "42.5%")
                st.metric("Annualized Return", "25.7%")
            with col2:
                st.metric("Sharpe Ratio", "1.85")
                st.metric("Max Drawdown", "-12.3%")
            with col3:
                st.metric("Win Rate", "58.6%")
                st.metric("Profit Factor", "2.1")
            
            # Equity curve chart
            st.subheader("Equity Curve")
            # Replace with actual equity curve visualization
            
            # Trade analysis
            st.subheader("Trade Analysis")
            # Replace with actual trade analysis logic

elif app_mode == "Real-time Simulation":
    # Real-time simulation section
    st.header("Real-time Trading Simulation")
    
    st.subheader("Simulation Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        strategy_type = st.selectbox("Strategy Type", ["Moving Average", "Machine Learning", "Signal Generator"])
        simulation_speed = st.slider("Simulation Speed (x)", 1, 100, 10)
    
    with col2:
        start_capital = st.number_input("Starting Capital", 10000.0, 1000000.0, 100000.0, 10000.0)
        position_size = st.slider("Position Size (%)", 1, 100, 20)
    
    if st.button("Start Simulation"):
        st.info("Starting real-time simulation...")
        
        # Replace with actual simulation logic
        
        # Mock simulation display
        st.subheader("Simulation Progress")
        sim_progress = st.empty()
        chart_placeholder = st.empty()
        metrics_placeholder = st.empty()
        
        # Mock simulation execution
        for i in range(100):
            # Update progress
            sim_progress.progress(i + 1)
            
            # Update chart (this would be replaced with actual chart updates)
            with chart_placeholder.container():
                st.write(f"Simulation step {i+1}/100")
                # Replace with actual chart updates
            
            # Update metrics
            with metrics_placeholder.container():
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Current P&L", f"{np.random.uniform(-5, 5):.2f}%")
                with col2:
                    st.metric("Open Positions", f"{np.random.randint(0, 3)}")
                with col3:
                    st.metric("Last Signal", "BUY" if np.random.random() > 0.5 else "SELL")
            
            time.sleep(0.1)  # Simulate real-time updates
        
        st.success("Simulation completed!")

# Function placeholders (replace with actual implementations)
def detect_events(df, decline_threshold, lookback_period, momentum_threshold, volatility_factor):
    """Placeholder for event detection function"""
    # In actual implementation, this would call your event_detection module
    # For now, just return some mock data
    events = []
    for i in range(5):
        idx = np.random.randint(lookback_period, len(df) - 10)
        events.append({
            'index': idx,
            'date': df.index[idx] if isinstance(df.index, pd.DatetimeIndex) else f"Bar {idx}",
            'price': df['Close'].iloc[idx],
            'decline': decline_threshold * np.random.uniform(1.0, 1.5),
            'momentum': momentum_threshold * np.random.uniform(0.8, 1.2),
            'confidence': np.random.uniform(0.6, 0.95)
        })
    return events

def visualize_events(df, events):
    """Placeholder for event visualization function"""
    # Create a simple plot with events marked
    fig = go.Figure()
    
    # Add price data
    fig.add_trace(go.Candlestick(
        x=df.index if isinstance(df.index, pd.DatetimeIndex) else range(len(df)),
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name="Price"
    ))
    
    # Add event markers
    for event in events:
        idx = event['index']
        fig.add_trace(go.Scatter(
            x=[df.index[idx] if isinstance(df.index, pd.DatetimeIndex) else idx],
            y=[df['Low'].iloc[idx] * 0.99],
            mode='markers',
            marker=dict(size=12, color='green', symbol='triangle-up'),
            name=f"Event at {event['date']}"
        ))
    
    fig.update_layout(title="Price Chart with Detected Events", xaxis_title="Date", yaxis_title="Price")
    return fig

def run_ma_strategy(df, ma_period, commission, price_col, min_hold_periods, optimize):
    """Placeholder for MA strategy execution"""
    # In actual implementation, this would call your ma_strategy module
    # For now, just return some mock results
    return {
        'returns': np.cumsum(np.random.normal(0.001, 0.02, len(df))),
        'trades': np.random.randint(0, 2, len(df)),
        'metrics': {
            'total_return': np.random.uniform(0.1, 0.5),
            'sharpe': np.random.uniform(1.0, 2.5),
            'max_dd': np.random.uniform(-0.2, -0.1),
            'win_rate': np.random.uniform(0.4, 0.6),
            'profit_factor': np.random.uniform(1.2, 2.5)
        }
    }

def display_strategy_results(result):
    """Display strategy backtesting results"""
    # Display metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Return", f"{result['metrics']['total_return']*100:.2f}%")
        st.metric("Win Rate", f"{result['metrics']['win_rate']*100:.2f}%")
    with col2:
        st.metric("Sharpe Ratio", f"{result['metrics']['sharpe']:.2f}")
        st.metric("Profit Factor", f"{result['metrics']['profit_factor']:.2f}")
    with col3:
        st.metric("Max Drawdown", f"{result['metrics']['max_dd']*100:.2f}%")
    
    # Display equity curve
    st.subheader("Equity Curve")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        y=result['returns'],
        mode='lines',
        name="Equity Curve"
    ))
    fig.update_layout(title="Strategy Performance", xaxis_title="Time", yaxis_title="Cumulative Return")
    st.plotly_chart(fig, use_container_width=True)