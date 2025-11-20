import pandas as pd
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta

class FuturesFilter:
    def __init__(self, data):
        """
        Initialize FuturesFilter class.

        Parameters:
        - data (pd.DataFrame): DataFrame containing time index and futures prices.
        """
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Data must have a DatetimeIndex.")
        self.data = data.copy()

    def adjust_to_next_weekday(self, date_str):
        """
        Check if the input date is a weekend, and if so, adjust to the next Monday.

        Parameters:
        - date_str (str): Original date string in YYYYMMDD format.

        Returns:
        - str: Adjusted date string in YYYYMMDD format.
        """
        date_dt = datetime.strptime(date_str, "%Y%m%d")
        if date_dt.weekday() == 5:  # Saturday
            date_dt += timedelta(days=2)  # Adjust to Monday
        elif date_dt.weekday() == 6:  # Sunday
            date_dt += timedelta(days=1)  # Adjust to Monday
        return date_dt.strftime("%Y%m%d")

    def filter_by_dates_and_time(self, dates, session=None, night_session=False):
        """
        Filter data based on multiple dates and trading sessions.

        Parameters:
        - dates (list of str): List of dates to filter (format: YYYY-MM-DD).
        - session (str or None): Specified trading session ("morning", "afternoon", "evening"). If None, filter only by date.
        - night_session (bool): If True, filter night session data from the previous day evening to the given date early morning.

        Returns:
        - pd.DataFrame: Data with specified dates and sessions removed.
        """
        date_set = set(pd.to_datetime(dates).date)

        if night_session:
            # Filter night session data (previous day 22:30 to specified date 05:00)
            mask = ~self.data.index.to_series().apply(
                lambda dt: any(
                    (dt >= pd.Timestamp(d) - pd.Timedelta(days=1, hours=1, minutes=30) + pd.Timedelta(hours=22) and
                     dt <= pd.Timestamp(d) + pd.Timedelta(hours=5))
                    for d in dates
                )
            )
            filtered_data = self.data[mask]
        else:
            # Filter by date only
            mask = ~self.data.index.map(lambda x: x.date()).isin(date_set)
            filtered_data = self.data[mask]

        if session is not None:
            # Filter specified trading session
            if session == "morning":
                time_mask = (filtered_data.index.time < pd.to_datetime('08:45').time()) | \
                            (filtered_data.index.time > pd.to_datetime('13:30').time())
            elif session == "afternoon":
                time_mask = (filtered_data.index.time < pd.to_datetime('15:00').time()) | \
                            (filtered_data.index.time > pd.to_datetime('22:30').time())
            elif session == "evening":
                time_mask = (filtered_data.index.time < pd.to_datetime('22:30').time()) & \
                            (filtered_data.index.time > pd.to_datetime('05:00').time())
            else:
                raise ValueError("Session must be one of 'morning', 'afternoon', 'evening', or None.")

            filtered_data = filtered_data[time_mask]

        return filtered_data

    def get_three_months_ago_avoiding_weekends(self, date_str):
        """
        給定一個日期字串 (格式: YYYYMMDD)，回推三個月並避免周末，返回新的日期字串 (格式: YYYYMMDD)。

        參數：
        - date_str (str): 原始日期字串，格式為 YYYYMMDD。

        回傳：
        - str: 回推三個月並避免周末的日期字串，格式為 YYYYMMDD。
        """
        date_dt = datetime.strptime(date_str, "%Y%m%d")
        three_months_ago = date_dt - relativedelta(months=3)

        # 避免周末
        def adjust_to_weekday(date):
            if date.weekday() == 5:  # 星期六
                return date - timedelta(days=1)  # 調整到星期五
            elif date.weekday() == 6:  # 星期天
                return date - timedelta(days=2)  # 調整到星期五
            return date

        three_months_ago_adjusted = adjust_to_weekday(three_months_ago)
        return three_months_ago_adjusted.strftime("%Y%m%d")

# 使用範例
# 假設 `data` 是包含時間索引和期貨價格的資料
# data = pd.DataFrame(..., index=pd.to_datetime(data.index))
# futures_filter = FuturesFilter(data)
# 過濾掉指定日期的夜盤資料
# result = futures_filter.filter_by_dates_and_time(["2025-01-10"], night_session=True)
# 調整到下週一的日期
# adjusted_date = futures_filter.adjust_to_next_weekday("20250110")
# 回推三個月並避免周末的日期
# three_months_ago_date = futures_filter.get_three_months_ago_avoiding_weekends("20250110")


# 使用範例
# 假設 `data` 是包含時間索引和期貨價格的資料
# data = pd.DataFrame(..., index=pd.to_datetime(data.index))
# futures_filter = FuturesFilter(data)
# 過濾掉指定日期的夜盤資料
# result = futures_filter.filter_by_dates_and_time(["2025-01-10"], night_session=True)



# 使用範例
# 假設 `data` 是包含時間索引和期貨價格的資料
# data = pd.DataFrame(..., index=pd.to_datetime(data.index))
# futures_filter = FuturesFilter(data)
# 過濾掉指定日期的夜盤資料
# result = futures_filter.filter_by_dates_and_time(["2025-01-10"], night_session=True)


