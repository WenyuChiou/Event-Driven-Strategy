#%% 
import pandas as pd
import matplotlib.pyplot as plt
import os
from src.event_detection import detect_trading_events, analyze_trading_events
from src.visualization import visualize_event_summary, plot_price_with_event_markers
from src.utils import export_event_summary
from package.FE import calculate_realtime_features

#%% 
# Set paths
file_path = r"C:\Users\wenyu\Desktop\trade\investment\python\API\TX00\training data\TX00_1_20240924_20241124.xlsx"
output_dir = r"C:\Users\wenyu\Desktop\trade\investment\python\scrapping\hydraulic jump\project\results\visualization"
os.makedirs(output_dir, exist_ok=True)

# Load data
print(f"Reading data: {file_path}")
df = pd.read_excel(file_path)
df['date'] = pd.to_datetime(df['date'])

# Ensure numeric columns are float64 type
for col in ['open', 'high', 'low', 'close', 'volume']:
    if col in df.columns:
        df[col] = df[col].astype('float64')
df = df.sort_values(by='date').reset_index(drop=True)

# Calculate features
print("Calculating technical indicators...")
df, scaler = calculate_realtime_features(df)

# Detect events
print("Detecting trading events...")
df_events = detect_trading_events(df)

# Analyze events
print("Analyzing trading events...")
analysis = analyze_trading_events(df_events)

# Basic statistics output
print("\n===== Trading Event Analysis =====")
print(f"Total events: {analysis['total_events']}")
print(f"  Long events: {analysis['long_events']} ({analysis['long_percentage']:.2f}%)")
print(f"  Short events: {analysis['short_events']} ({analysis['short_percentage']:.2f}%)")
print(f"Win rate: {analysis['win_rate']:.2%}")
print(f"Profit factor: {analysis['profit_factor']:.2f}")
print(f"Average profit: {analysis['expectancy']:.2f} points")

# Visualize results
print("\nGenerating visualization charts...")
visualize_event_summary(df_events, analysis, os.path.join(output_dir, "event_summary.png"))
plot_price_with_event_markers(df_events, os.path.join(output_dir, "price_with_events.png"))

# Export results
print("\nExporting analysis results...")
export_event_summary(analysis, os.path.join(output_dir, "event_analysis.xlsx"))

print(f"\nAnalysis complete! Results saved to: {output_dir}")