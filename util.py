import pandas as pd
import requests
import datetime
import spacepy.omni as omni
import spacepy.time as spt
import spacepy.toolbox as tb
from typing import Tuple, List


def lag_features(
    df: pd.DataFrame,
    feature_cols: List[str],
    lags: List[int],
    time_col: str = 'window_start',
    time_end_col: str = 'window_end',
    check_consecutive: bool = True,
    forward: bool = False
) -> pd.DataFrame:
    """
    Creates lagged features for the specified columns in the DataFrame.
    Validates that rows are consecutive in time before assigning lags.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with time series data (must be sorted by time).
    feature_cols : List[str]
        List of column names to create lagged features for.
    lags : List[int]
        List of lag periods (number of time steps) to create.
    time_col : str
        Name of the datetime column for window start (default: 'window_start')
    time_end_col : str
        Name of the datetime column for window end (default: 'window_end')
    check_consecutive : bool
        If True, only assign lagged values when rows are consecutive in time.
        Non-consecutive lags will be set to NaN. (default: True)
    forward : bool
        If False (default), creates backward lags (past values): col_lag_1, col_lag_2, ...
        If True, creates forward lags (future values): col_t_plus_1, col_t_plus_2, ...
    
    Returns
    -------
    pd.DataFrame
        DataFrame with original and lagged features.
        
    Examples
    --------
    # Backward lags (for input features - using past data)
    df = lag_features(df, feature_cols=["speed", "density"], lags=[1, 2], forward=False)
    # Creates: speed_lag_1, speed_lag_2, density_lag_1, density_lag_2
    
    # Forward lags (for targets - predicting future values)
    df = lag_features(df, feature_cols=["kp_index"], lags=[1, 2], forward=True)
    # Creates: kp_index_t_plus_1, kp_index_t_plus_2
    """
    df = df.copy()
    
    # Ensure datetime format
    if time_col in df.columns and time_end_col in df.columns:
        df[time_col] = pd.to_datetime(df[time_col])
        df[time_end_col] = pd.to_datetime(df[time_end_col])
    
    # Pre-compute consecutive masks for all needed lags
    consecutive_masks = {}
    
    if check_consecutive and time_col in df.columns and time_end_col in df.columns:
        for lag in lags:
            mask = pd.Series(True, index=df.index)
            
            for step in range(1, lag + 1):
                if forward:
                    # Forward lag: check if future rows are consecutive
                    # Row at i+step should start where i+(step-1) ends
                    next_window_start = df[time_col].shift(-step)
                    current_window_end = df[time_end_col].shift(-(step - 1))
                    step_mask = (next_window_start == current_window_end)
                else:
                    # Backward lag: check if past rows are consecutive
                    # Row at i should start where i-1 ends
                    current_window_start = df[time_col].shift(step - 1)
                    prev_window_end = df[time_end_col].shift(step)
                    step_mask = (current_window_start == prev_window_end)
                
                mask = mask & step_mask
            
            consecutive_masks[lag] = mask
    else:
        # No time validation - all True
        for lag in lags:
            consecutive_masks[lag] = pd.Series(True, index=df.index)
    
    # Apply lags with naming based on direction
    for col in feature_cols:
        for lag in lags:
            shift_amount = -lag if forward else lag
            suffix = f"_plus_{lag}" if forward else f"_lag_{lag}"
            
            if check_consecutive:
                df[f"{col}{suffix}"] = df[col].shift(shift_amount).where(consecutive_masks[lag])
            else:
                df[f"{col}{suffix}"] = df[col].shift(shift_amount)
    
    return df


def aggregate_solar_wind_3h(
    df: pd.DataFrame,
    time_col: str = 'date_and_time',
    feature_cols: List[str] = ['by_gsm', 'bz_gsm', 'density', 'speed', 'dynamic_pressure'],
    target_col: str = 'kp_index',
    min_samples: int = 3
) -> pd.DataFrame:
    """
    Aggregates solar wind data into 3-hour windows for Kp index prediction.
    
    The Kp index is officially calculated every 3 hours, so we need to aggregate
    the higher-resolution solar wind measurements to match this cadence.
    
    For each 3H window, we compute statistics (mean, min, max, std) for the
    feature columns, and take the first Kp value (they should all be identical
    within a window since Kp is a 3H index).
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with solar wind data at hourly (or finer) resolution.
        Must contain the time column, feature columns, and target column.
    time_col : str
        Name of the datetime column (default: 'date_and_time')
    feature_cols : List[str]
        List of feature column names to aggregate (default: ['by_gsm', 'bz_gsm', 'density', 'speed'])
    target_col : str
        Name of the Kp index column (default: 'kp_index')
    min_samples : int
        Minimum number of samples required in a 3H window to consider it valid
    
    Returns
    -------
    pd.DataFrame
        Aggregated DataFrame with columns:
        - window_start: Start of the 3H window
        - window_end: End of the 3H window
        - {feature}_{stat}: Aggregated statistics for each feature (mean, min, max, std)
        - kp_index: The Kp index value for the window
    """
    # Make a copy to avoid modifying the original
    df = df.copy()
    
    # Set time column as index for resampling
    df = df.set_index(time_col)
    
    # Count samples per 3H window to validate completeness
    # For hourly data, we expect 3 samples per 3H window
    sample_counts = df[feature_cols[0]].resample('3H').count()
    
    # # Determine minimum required samples based on data resolution
    # if len(df) > 0:
    #     time_diff = df.index.to_series().diff().median()
    #     if time_diff <= pd.Timedelta(minutes=5):
    #         # 1-minute resolution: expect ~180 samples, require at least 90 (50%)
    #         min_samples = 90
    #     elif time_diff <= pd.Timedelta(hours=1):
    #         # Hourly resolution: expect 3 samples, require at least 3
    #         min_samples = 3
    #     else:
    #         # Other resolutions: require at least 2 samples
    #         min_samples = 2
    # else:
    #     min_samples = 3
    
    # Get valid windows that have sufficient data
    valid_windows = sample_counts[sample_counts >= min_samples].index
    
    # Aggregate features over 3H windows with statistics
    df_features_3h = df[feature_cols].resample('3H').agg([
        'mean',
        'min',
        'max',
        'std'
    ])
    
    # Flatten column names (e.g., by_gsm_mean, by_gsm_min, etc.)
    df_features_3h.columns = [f"{col}_{stat}" for col, stat in df_features_3h.columns]
    
    # For Kp index, take the first value of each 3H window (they should all be the same)
    kp_3h = df[[target_col]].resample('3H').first()
    
    # Join features with Kp
    result = df_features_3h.join(kp_3h, how='inner')
    
    # Filter to only valid (complete) windows
    result = result[result.index.isin(valid_windows)]
    
    # Drop incomplete windows (NaN values)
    result = result.dropna()
    
    # Reset index to get the window start time
    result = result.reset_index()
    
    # Create explicit window start and end columns
    result['window_start'] = result[time_col]
    result['window_end'] = result[time_col] + pd.Timedelta(hours=3)
    
    # Drop the original time column
    result = result.drop(columns=[time_col])
    
    # Reorder columns to have time columns first
    time_cols = ['window_start', 'window_end']
    other_cols = [c for c in result.columns if c not in time_cols]
    result = result[time_cols + other_cols]
    
    return result


def get_noaa_realtime_hourly_data(mag_url: str, plasma_url: str, kp_url: str) -> pd.DataFrame:
    """
    Fetches NOAA data and aligns everything to 3-hour windows
    for physically correct Kp training.
    """

    # -----------------------------------------
    # Fetch Magnetometer data (1-min) and aggregate to 1H
    # -----------------------------------------
    m_res = requests.get(mag_url).json()
    df_mag = pd.DataFrame(m_res[1:], columns=m_res[0])
    df_mag['time_tag'] = pd.to_datetime(df_mag['time_tag'], utc=True)
    # df_mag = df_mag.set_index('time_tag')

    # Convert to numeric (not the time tag)
    for col in df_mag.columns:
        if col != 'time_tag':
            df_mag[col] = pd.to_numeric(df_mag[col], errors='coerce')

    # Rename the columns for coherence
    df_mag.rename(columns={'time_tag': 'date_and_time'}, inplace=True)

    # Round to nearest hour
    df_mag['rounded_time'] = df_mag['date_and_time'].dt.round('H')
    # Calculate absolute difference in seconds
    df_mag['abs_diff'] = (df_mag['date_and_time'] - df_mag['rounded_time']).abs()
    # Keep only the row closest to the o'clock for each rounded_time
    df_mag = df_mag.loc[df_mag.groupby('rounded_time')['abs_diff'].idxmin()]
    df_mag = df_mag.drop(columns=['date_and_time'])
    # Rename rounded_time to date_and_time for downstream compatibility
    df_mag = df_mag.rename(columns={'rounded_time': 'date_and_time'})
    df_mag = df_mag.drop(columns=['abs_diff'])
    # Reset index
    df_mag = df_mag.reset_index(drop=True)
    
    print("Raw Magnetometer data:\n", df_mag)

    # -----------------------------------------
    # Fetch Plasma data (1-min) and aggregate to 1H
    # -----------------------------------------
    p_res = requests.get(plasma_url).json()
    df_plasma = pd.DataFrame(p_res[1:], columns=p_res[0])
    df_plasma['time_tag'] = pd.to_datetime(df_plasma['time_tag'], utc=True)
    #df_plasma = df_plasma.set_index('time_tag')

    # Convert to numeric
    for col in df_plasma.columns:
        if col != 'time_tag':
            df_plasma[col] = pd.to_numeric(df_plasma[col], errors='coerce')

    # Rename the columns for coherence
    df_plasma.rename(columns={'time_tag': 'date_and_time'}, inplace=True)

    # Round to nearest hour
    df_plasma['rounded_time'] = df_plasma['date_and_time'].dt.round('H')
    # Calculate absolute difference in seconds
    df_plasma['abs_diff'] = (df_plasma['date_and_time'] - df_plasma['rounded_time']).abs()
    # Keep only the row closest to the o'clock for each rounded_time
    df_plasma = df_plasma.loc[df_plasma.groupby('rounded_time')['abs_diff'].idxmin()]
    df_plasma = df_plasma.drop(columns=['date_and_time'])
    # Rename rounded_time to date_and_time for downstream compatibility
    df_plasma = df_plasma.rename(columns={'rounded_time': 'date_and_time'})
    df_plasma = df_plasma.drop(columns=['abs_diff'])
    # Reset index
    df_plasma = df_plasma.reset_index(drop=True)

    print("Raw Plasma data:\n", df_plasma)

    # -----------------------------------------
    # Fetch Kp index (3-hour official)
    # -----------------------------------------
    kp_res = requests.get(kp_url).json()
    df_kp = pd.DataFrame(kp_res[1:], columns=kp_res[0])
    df_kp['time_tag'] = pd.to_datetime(df_kp['time_tag'], utc=True)
    #df_kp = df_kp.set_index('time_tag')

    df_kp['Kp'] = pd.to_numeric(df_kp['Kp'], errors='coerce')

    # Rename the columns for coherence
    df_kp.rename(columns={'time_tag': 'date_and_time'}, inplace=True)
    df_kp.rename(columns={'Kp': 'kp_index'}, inplace=True)

    print("Raw Kp index data:\n", df_kp)

    # -----------------------------------------
    # Data merging
    # -----------------------------------------
    # Merge on timestamp magnetometer and plasma
    df_temp = pd.merge(df_mag, df_plasma, on='date_and_time')
    
    # Round timestamps to nearest hour to match with df_kp (which is on the hour)
    # df_temp['date_and_time'] = df_temp['date_and_time'].dt.round('H')
    
    df = pd.merge(df_temp, df_kp, on='date_and_time', how='left')

    return df

def get_noaa_realtime_data_old(mag_url: str, plasma_url: str, kp_url: str) -> pd.DataFrame:
    """
    Fetches 1-minute solar wind data from NOAA SWPC and merges Magnetometer and Plasma data.
    """
    # -----------------------------------------
    # Data fetching
    # -----------------------------------------
    # Fetch Magnetometer data
    m_res = requests.get(mag_url).json()
    df_mag = pd.DataFrame(m_res[1:], columns=m_res[0])
    df_mag['time_tag'] = pd.to_datetime(df_mag['time_tag'])
    print("Magnetometer data:\n", df_mag)

    # Fetch Plasma data
    p_res = requests.get(plasma_url).json()
    df_plasma = pd.DataFrame(p_res[1:], columns=p_res[0])
    df_plasma['time_tag'] = pd.to_datetime(df_plasma['time_tag'])
    print("Plasma data:\n", df_plasma)

    # Fetch KP index data
    kp_res = requests.get(kp_url).json()
    df_kp = pd.DataFrame(kp_res[1:], columns=kp_res[0])
    df_kp['time_tag'] = pd.to_datetime(df_kp['time_tag'])
    print("KP index data:\n", df_kp)
    print("----------------------------------------- Data manipulation -----------------------------------------")

    # -----------------------------------------
    # Data manipulation
    # -----------------------------------------
    # Forward fill: each 3-hour value fills the previous 2 hours
    df_kp = df_kp.set_index('time_tag')
    df_kp = df_kp.resample('1H').ffill() 
    df_kp = df_kp.reset_index()
    print("KP index data (resampled to 1H):\n", df_kp)

    # Aggregate magnetometer data to hourly resolution (mean)
    df_mag = df_mag.set_index('time_tag')
    # Convert all columns except 'time_tag' to numeric before resampling
    for col in df_mag.columns:
        if col != 'time_tag':
            df_mag[col] = pd.to_numeric(df_mag[col], errors='coerce')
    df_mag_hourly = df_mag.resample('1H').mean(numeric_only=True).reset_index()
    print("Magnetometer data (resampled to 1H):\n", df_mag_hourly)

    # Aggregate plasma data to hourly resolution (mean)
    df_plasma = df_plasma.set_index('time_tag')
    # Convert all columns except 'time_tag' to numeric before resampling
    for col in df_plasma.columns:
        if col != 'time_tag':
            df_plasma[col] = pd.to_numeric(df_plasma[col], errors='coerce')
    df_plasma_hourly = df_plasma.resample('1H').mean(numeric_only=True).reset_index()
    print("Plasma data (resampled to 1H):\n", df_plasma_hourly)

    # -----------------------------------------
    # Data merging
    # -----------------------------------------
    # Merge on timestamp magnetometer and plasma
    df = pd.merge(df_mag_hourly, df_plasma_hourly, on='time_tag')
    df = pd.merge(df, df_kp, on='time_tag')

    return df

def get_noaa_training_data_3h_old(mag_url: str, plasma_url: str, kp_url: str) -> pd.DataFrame:
    """
    Fetches NOAA data and aligns everything to 3-hour windows
    for physically correct Kp training.
    """

    # -----------------------------------------
    # Fetch Magnetometer data (1-min)
    # -----------------------------------------
    m_res = requests.get(mag_url).json()
    df_mag = pd.DataFrame(m_res[1:], columns=m_res[0])
    df_mag['time_tag'] = pd.to_datetime(df_mag['time_tag'], utc=True)
    df_mag = df_mag.set_index('time_tag')

    # Convert to numeric
    for col in df_mag.columns:
        df_mag[col] = pd.to_numeric(df_mag[col], errors='coerce')

    # -----------------------------------------
    # Fetch Plasma data (1-min)
    # -----------------------------------------
    p_res = requests.get(plasma_url).json()
    df_plasma = pd.DataFrame(p_res[1:], columns=p_res[0])
    df_plasma['time_tag'] = pd.to_datetime(df_plasma['time_tag'], utc=True)
    df_plasma = df_plasma.set_index('time_tag')

    for col in df_plasma.columns:
        df_plasma[col] = pd.to_numeric(df_plasma[col], errors='coerce')

    # -----------------------------------------
    # Fetch Kp index (3-hour official)
    # -----------------------------------------
    kp_res = requests.get(kp_url).json()
    df_kp = pd.DataFrame(kp_res[1:], columns=kp_res[0])
    df_kp['time_tag'] = pd.to_datetime(df_kp['time_tag'], utc=True)
    df_kp = df_kp.set_index('time_tag')

    df_kp['Kp'] = pd.to_numeric(df_kp['Kp'], errors='coerce')

    # -----------------------------------------
    # 3H AGGREGATION (CORE STEP)
    # Matches the logic in aggregate_solar_wind_3h for consistency
    # -----------------------------------------

    # Magnetometer: statistics over 3h
    df_mag_3h = df_mag.resample('3H').agg([
        'mean',
        'min',
        'max',
        'std'
    ])

    # Flatten column names
    df_mag_3h.columns = [
        f"{col}_{stat}" for col, stat in df_mag_3h.columns
    ]

    # Plasma: statistics over 3h
    df_plasma_3h = df_plasma.resample('3H').agg([
        'mean',
        'min',
        'max',
        'std'
    ])
    df_plasma_3h.columns = [
        f"{col}_{stat}" for col, stat in df_plasma_3h.columns
    ]

    # -----------------------------------------
    # Validate complete 3H windows
    # Only keep windows that have enough data points
    # -----------------------------------------
    
    # Count samples per 3H window for magnetometer and plasma
    mag_counts = df_mag.resample('3H').count().iloc[:, 0]  # Count from first column
    plasma_counts = df_plasma.resample('3H').count().iloc[:, 0]
    
    # For 1-minute data, a complete 3H window should have ~180 samples
    # We use a threshold of at least 90 samples (50%) to be considered valid
    # For hourly data, we expect 3 samples per window, threshold of 2
    min_samples_1min = 90  # 50% of 180 samples for 1-min data
    min_samples_hourly = 2  # At least 2 out of 3 for hourly data
    
    # Determine data resolution and set appropriate threshold
    if len(df_mag) > 0:
        time_diff = df_mag.index.to_series().diff().median()
        if time_diff <= pd.Timedelta(minutes=5):
            min_samples = min_samples_1min
        else:
            min_samples = min_samples_hourly
    else:
        min_samples = min_samples_hourly
    
    valid_mag_windows = mag_counts[mag_counts >= min_samples].index
    valid_plasma_windows = plasma_counts[plasma_counts >= min_samples].index
    
    # Keep only windows that are valid for both mag and plasma
    valid_windows = valid_mag_windows.intersection(valid_plasma_windows)

    # -----------------------------------------
    # Merge everything on 3H windows
    # -----------------------------------------
    df = df_mag_3h.join(df_plasma_3h, how='inner')
    
    # Rename Kp to kp_index for consistency with historical data
    df_kp = df_kp.rename(columns={'Kp': 'kp_index'})
    df = df.join(df_kp[['kp_index']], how='inner')

    # Filter to only valid (complete) windows
    df = df[df.index.isin(valid_windows)]

    # Drop any remaining incomplete windows (NaN values)
    df = df.dropna()

    # -----------------------------------------
    # Create window_start and window_end columns
    # (matching aggregate_solar_wind_3h output format)
    # -----------------------------------------
    df = df.reset_index()
    df['window_start'] = df['time_tag']
    df['window_end'] = df['time_tag'] + pd.Timedelta(hours=3)
    df = df.drop(columns=['time_tag'])

    # Reorder columns to have time columns first
    time_cols = ['window_start', 'window_end']
    other_cols = [c for c in df.columns if c not in time_cols]
    df = df[time_cols + other_cols]

    return df



def get_city_weather_history(city: str, start_date: str, end_date: str, latitude: float, longitude: float) -> pd.DataFrame:
    """
    Fetches historical cloud cover data for a specific city and coordinate range.
    Uses Open-Meteo Archive API for historical weather data.
    Returns a DataFrame with columns: date, cloud_cover_mean
    """
    params = {
        "latitude": latitude,
        "longitude": longitude,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "cloud_cover",
        "timezone": "auto"
    }
    
    # Use the archive API for historical data
    response = requests.get("https://archive-api.open-meteo.com/v1/archive", params=params)
    
    if response.status_code != 200:
        raise Exception(f"Failed to fetch weather data for {city}: {response.status_code}")
    
    data = response.json()
    
    # Extract hourly data
    hourly_data = data['hourly']
    df = pd.DataFrame({
        'date': hourly_data['time'],
        'cloud_cover': hourly_data['cloud_cover']
    })
    
    return df

def get_city_weather_today(lat: float, lon: float) -> int:
    """
    Fetches the current cloud cover percentage for a specific coordinate.

    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "cloud_cover",
        "timezone": "auto"
    }
    response = requests.get("https://api.open-meteo.com/v1/forecast", params=params).json()
    current_data = response['current']
    return {
        'time': current_data['time'],
        'cloud_cover': current_data['cloud_cover']
    }
    #return response['current']['cloud_cover']

def get_city_weather_forecast(lat: float, lon: float, hours_ahead: int) -> pd.DataFrame:
    """
    Get the forecasted cloud cover percentage for a specific coordinate and number of hours ahead.
    
    Parameters
    ----------
    lat : float
        Latitude of the city.
    lon : float
        Longitude of the city.
    hours_ahead : int
        Number of hours ahead to forecast.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with 'time' (UTC) and 'cloud_cover' columns.
    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "cloud_cover",
        "forecast_hours": hours_ahead,
        "timezone": "UTC"  # Use UTC for consistent time matching
    }
    response = requests.get("https://api.open-meteo.com/v1/forecast", params=params).json()

    # Parse the hourly forecast data
    hourly_data = response['hourly']
    df = pd.DataFrame({
        'time': pd.to_datetime(hourly_data['time'], utc=True),
        'cloud_cover': hourly_data['cloud_cover']
    })

    return df


def get_cloud_cover_for_window(cloud_cover_df: pd.DataFrame, window_start: pd.Timestamp, window_end: pd.Timestamp) -> float:
    """
    Calculate the mean cloud cover for a specific time window.
    
    Parameters
    ----------
    cloud_cover_df : pd.DataFrame
        DataFrame with 'time' and 'cloud_cover' columns (from get_city_weather_forecast).
    window_start : pd.Timestamp
        Start of the prediction window (UTC).
    window_end : pd.Timestamp
        End of the prediction window (UTC).
    
    Returns
    -------
    float
        Mean cloud cover percentage for the window, or NaN if no data available.
    """
    # Ensure timestamps are timezone-aware (UTC)
    if window_start.tzinfo is None:
        window_start = window_start.tz_localize('UTC')
    if window_end.tzinfo is None:
        window_end = window_end.tz_localize('UTC')
    
    # Filter cloud cover data for the window
    mask = (cloud_cover_df['time'] >= window_start) & (cloud_cover_df['time'] < window_end)
    window_data = cloud_cover_df[mask]
    
    if len(window_data) == 0:
        return float('nan')
    
    return window_data['cloud_cover'].mean()


def aurora_visibility_logic(pred_kp: float, kp_threshold: float, cloud_cover: float) -> str:
    """
    Returns the Actionable status for a city.
    """
    if pred_kp >= kp_threshold:
        if cloud_cover <= 30:
            return "GO"
        else:
            return "HIGH ACTIVITY / CLOUDY"
    return "NO ACTIVITY"


# Placeholder for spacepy historical fetching
def get_historical_omni_data(start_date: str, end_date: str):
    """
    Logic to pull historical OMNI data via spacepy or NASA CDAWeb.
    Returns features: [bz_gsm, density, speed] and label: [kp_index]
    """
    # Note: Requires spacepy installation and local config
    # In practice, you would use: from spacepy import omindata
    print(f"Fetching historical data from {start_date} to {end_date}...")

    return 0
    print("carita linda")
    return data


def calculate_dynamic_pressure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate dynamic pressure and add it as a new column to the DataFrame.
    Dynamic Pressure P = N * V^2
    where N is density and V is speed.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame with 'density' and 'speed' columns.
    
    Returns
    -------
    pd.DataFrame
        DataFrame with an additional 'dynamic_pressure' column.
    """
    df = df.copy()
    df['dynamic_pressure'] = df['density'] * (df['speed'] ** 2)
    df['dynamic_pressure'] = df['dynamic_pressure'].astype('float32')
    return df