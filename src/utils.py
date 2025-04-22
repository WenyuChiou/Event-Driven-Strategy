import os
import pandas as pd

def export_event_summary(analysis_results, output_path):
    """
    導出事件分析摘要到Excel文件。
    
    Parameters:
    -----------
    analysis_results : dict
        analyze_trading_events的分析結果
    output_path : str
        輸出文件路徑
    """
    # 確保輸出目錄存在
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # 基本統計
    basic_stats = {
        "統計項目": ["總事件數", "做多事件數", "做空事件數", "做多事件比例", "做空事件比例", 
                   "勝率", "盈虧比", "平均獲利"],
        "數值": [
            analysis_results["total_events"],
            analysis_results["long_events"],
            analysis_results["short_events"],
            f"{analysis_results['long_percentage']:.2f}%",
            f"{analysis_results['short_percentage']:.2f}%",
            f"{analysis_results['win_rate']:.2%}",
            f"{analysis_results['profit_factor']:.2f}",
            f"{analysis_results['expectancy']:.2f}"
        ]
    }
    
    # 按星期分布
    day_stats = {"星期": [], "事件數": []}
    days_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
    day_translation = {'Monday': '星期一', 'Tuesday': '星期二', 'Wednesday': '星期三', 
                      'Thursday': '星期四', 'Friday': '星期五'}
    
    for day in days_order:
        if day in analysis_results.get('by_day', {}):
            day_stats["星期"].append(day_translation.get(day, day))
            day_stats["事件數"].append(analysis_results['by_day'][day])
    
    # 按小時分布
    hour_stats = {"小時": [], "事件數": []}
    for hour in range(24):
        if hour in analysis_results.get('by_hour', {}):
            hour_stats["小時"].append(hour)
            hour_stats["事件數"].append(analysis_results['by_hour'][hour])
    
    # 創建Excel文件
    with pd.ExcelWriter(output_path) as writer:
        pd.DataFrame(basic_stats).to_excel(writer, sheet_name="基本統計", index=False)
        pd.DataFrame(day_stats).to_excel(writer, sheet_name="按星期分布", index=False)
        pd.DataFrame(hour_stats).to_excel(writer, sheet_name="按小時分布", index=False)
        
        # 盈虧統計
        stats_df = pd.DataFrame({
            "項目": analysis_results["all_stats"].index,
            "所有事件": analysis_results["all_stats"].values,
            "做多事件": analysis_results["long_stats"].values,
            "做空事件": analysis_results["short_stats"].values
        })
        stats_df.to_excel(writer, sheet_name="盈虧統計", index=False)
    
    print(f"事件摘要已導出到: {output_path}")