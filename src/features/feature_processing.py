import pandas as pd 
from sklearn.preprocessing import MinMaxScaler

# Read walmart sales of 45 stores data
#df = load_weekly_sales( path_file = "../../data/walmart-sales-dataset-of-45stores", index_col = "Date", freq = "W" ,target_var = "Weekly_Sales" )
def select_exog(df: pd.DataFrame, exog_cols):
    return df[exog_cols] if exog_cols else None


def inverse_minmax(y_scaled, y_min: float, y_max: float) :
    """
    Invert min–max scaling for y.
    y_scaled: values in scaled space (typically [0,1])
    y_min, y_max: the SAME values used during scaling (from train!)
    Returns the original-scale y with type preserved for pandas Series.
    """
    y_range = float(y_max - y_min)
    if y_range <= 0:
        raise ValueError("y_max must be greater than y_min.")
    if isinstance(y_scaled, pd.Series):
        return pd.Series(y_min + y_range * y_scaled.values, index=y_scaled.index, name=y_scaled.name)
    arr = np.asarray(y_scaled, dtype=float)
    return y_min + y_range * arr

def scale_features(train: pd.DataFrame, test: pd.DataFrame, featrs_to_scale: list ):
 
    preprocessor =  MinMaxScaler(feature_range=(0, 1))  
    preprocessor.fit(train.loc[:, featrs_to_scale])
    train_scld = preprocessor.transform(train.loc[:, featrs_to_scale])
    test_scld = preprocessor.transform(test.loc[:, featrs_to_scale])

   
               
    train_scld = pd.DataFrame(
                          train_scld,
                          columns = featrs_to_scale,
                          index = train.index
               )
    test_scld = pd.DataFrame(
                         test_scld,
                         columns = featrs_to_scale,
                         index = test.index
               )
    train_scld = train_scld.dropna()
    test_scld = test_scld.dropna()

    return  train_scld, test_scld, preprocessor


def month_start_flag(dates):
    """
    Given a list of dates (strings or datetime), return a list where:
    - 1 = first occurrence of each month in the sequence
    - 0 = all other dates
    """
    # Convert to pandas datetime
    dates = pd.to_datetime(dates)
    
    # Create a DataFrame
    df = pd.DataFrame({'date': dates})
    
    # Sort by date (if not already sorted)
    df = df.sort_values('date').reset_index(drop=True)
    
    # Group by year and month, mark first row in each group as 1
    df['month_flag'] = df.groupby([df['date'].dt.year, df['date'].dt.month]).cumcount().eq(0).astype(int)
    
    return df['month_flag'].tolist()



def add_thanksgiving_christmas_flags(
    df,
    date_col=None,
    week_anchor="MON",          # One of {"MON","TUE","WED","THU","FRI","SAT","SUN"}
    windows={"thanksgiving":[0], "christmas":[0]},  # 0 = holiday week; add negatives/positives for lead/lag weeks
    col_prefix=""
):
    """
    Adds binary flags for Thanksgiving and Christmas to `df`.
    
    Parameters
    ----------
    df : pd.DataFrame
        Your dataframe with a DateTimeIndex or a date column.
    date_col : str or None
        Name of the date column if the index is not datetime-like. If None, uses the index.
    week_anchor : str
        The weekday used to define the "week bucket".
        Example: "MON" means the week is labeled by its Monday (common for pandas weekly resampling).
                 If your data is week-ending Sunday, pass week_anchor="SUN".
    windows : dict
        Which week offsets to flag. 
        Example: {"thanksgiving":[-1,0,1], "christmas":[-3,-2,-1,0,1]}
    col_prefix : str
        Optional prefix for the created column names (e.g., "cal_").
    
    Returns
    -------
    df_out : pd.DataFrame
        Original df with new columns like:
        - {prefix}thanksgiving_w0, {prefix}thanksgiving_w-1, ...
        - {prefix}christmas_w0, {prefix}christmas_w-1, ...
    """
    df_out = df.copy()

    # --- Resolve dates ---
    if date_col is None:
        dates = pd.to_datetime(df_out.index)
    else:
        dates = pd.to_datetime(df_out[date_col].values)

    # Map weekday names to pandas weekday numbers (Mon=0,...,Sun=6)
    _wk_map = dict(MON=0, TUE=1, WED=2, THU=3, FRI=4, SAT=5, SUN=6)
    if week_anchor.upper() not in _wk_map:
        raise ValueError("week_anchor must be one of {'MON','TUE','WED','THU','FRI','SAT','SUN'}")
    anchor_num = _wk_map[week_anchor.upper()]

    # Helper: convert any date to its "week label" (the date of the anchor weekday of that week)
    def to_week_label(ts):
        ts = pd.Timestamp(ts)
        # shift back to the anchor weekday
        delta_days = (ts.weekday() - anchor_num) % 7
        return ts - pd.Timedelta(days=delta_days)

    week_labels = pd.to_datetime([to_week_label(d) for d in dates])
    years = pd.Index(dates).year.unique()

    # --- Build holiday base dates ---
    # Thanksgiving: 4th Thursday in November
    thanksgiving_dates = []
    for y in years:
        nov1 = pd.Timestamp(year=int(y), month=11, day=1)
        # weekday numbers: Mon=0,...,Thu=3
        days_to_thu = (3 - nov1.weekday()) % 7
        first_thu = nov1 + pd.Timedelta(days=days_to_thu)
        fourth_thu = first_thu + pd.Timedelta(weeks=3)
        thanksgiving_dates.append(fourth_thu)

    # Christmas: fixed
    christmas_dates = [pd.Timestamp(year=int(y), month=12, day=25) for y in years]

    # Convert to week labels
    tg_week0 = pd.to_datetime([to_week_label(d) for d in thanksgiving_dates])
    xm_week0 = pd.to_datetime([to_week_label(d) for d in christmas_dates])

    # For each requested window offset, compute week labels
    def shifted_weeks(base_weeks, offset):
        return pd.to_datetime([w + pd.Timedelta(weeks=offset) for w in base_weeks])

    # Initialize all requested columns to 0
    for offset in sorted(set(windows.get("thanksgiving", [0]))):
        col = f"{col_prefix}thanksgiving_w{offset:+d}".replace("+", "")
        df_out[col] = 0

    for offset in sorted(set(windows.get("christmas", [0]))):
        col = f"{col_prefix}christmas_w{offset:+d}".replace("+", "")
        df_out[col] = 0

    # Fill flags
    # Thanksgiving
    for offset in windows.get("thanksgiving", [0]):
        weeks = set(shifted_weeks(tg_week0, offset))
        col = f"{col_prefix}thanksgiving_w{offset:+d}".replace("+", "")
        df_out.loc[pd.Index(week_labels).isin(weeks), col] = 1

    # Christmas
    for offset in windows.get("christmas", [0]):
        weeks = set(shifted_weeks(xm_week0, offset))
        col = f"{col_prefix}christmas_w{offset:+d}".replace("+", "")
        df_out.loc[pd.Index(week_labels).isin(weeks), col] = 1

    return df_out

