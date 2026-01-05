import pandas as pd
import requests
import datetime
import spacepy.omni as omni
import spacepy.time as spt
import spacepy.toolbox as tb
from typing import Tuple


def get_noaa_realtime_data(mag_url: str, plasma_url: str, kp_url: str) -> pd.DataFrame:
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

def get_city_weather_forecast(lat: float, lon: float) -> int:
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
    return response['current']['cloud_cover']


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