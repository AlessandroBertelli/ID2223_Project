import pandas as pd
import requests
import datetime
import spacepy.omni as omni
import spacepy.time as spt
import spacepy.toolbox as tb
from typing import Tuple, List


def aggregate_solar_wind_3h(
    df: pd.DataFrame,
    time_col: str = 'date_and_time',
    feature_cols: List[str] = ['by_gsm', 'bz_gsm', 'density', 'speed'],
    target_col: str = 'kp_index'
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

def get_noaa_training_data_3h(mag_url: str, plasma_url: str, kp_url: str) -> pd.DataFrame:
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
    # Merge everything on 3H windows
    # -----------------------------------------
    df = df_mag_3h.join(df_plasma_3h, how='inner')
    df = df.join(df_kp[['Kp']], how='inner')

    # Drop incomplete windows
    df = df.dropna()

    # -----------------------------------------
    # Optional: shift timestamp to window center
    # -----------------------------------------
    df.index = df.index + pd.Timedelta(hours=1.5)

    return df.reset_index().rename(columns={'index': 'time'})



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

def get_city_weather_forecast(lat: float, lon: float, hours_ahead: int) -> int:
    """
    Get the forecasted cloud cover percentage for a specific coordinate and number of hours ahead.

    """
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "cloud_cover",
        "forecast_hours": hours_ahead,
        "timezone": "auto"
    }
    response = requests.get("https://api.open-meteo.com/v1/forecast", params=params).json()

    # Parse the hourly forecast data
    hourly_data = response['hourly']
    df = pd.DataFrame({
        'time': pd.to_datetime(hourly_data['time']),
        'cloud_cover': hourly_data['cloud_cover']
    })

    # # Get the forecast for the specified hour ahead
    # now = pd.Timestamp.now(tz='UTC').tz_localize(None)
    # target_time = now + pd.Timedelta(hours=hours_ahead)
    # closest_forecast = df.iloc[(df['time'] - target_time).abs().argsort()[0]]

    #return int(closest_forecast['cloud_cover'])
    return df


def aurora_visibility_logic(pred_kp: float, kp_threshold: float, cloud_cover: int) -> str:
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