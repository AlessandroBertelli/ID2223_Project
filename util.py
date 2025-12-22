import pandas as pd
import requests
import datetime
from typing import Tuple


def get_noaa_realtime_data(mag_url: str, plasma_url: str) -> pd.DataFrame:
    """
    Fetches 1-minute solar wind data from NOAA SWPC and merges Magnetometer and Plasma data.
    """
    # Fetch Magnetometer data
    m_res = requests.get(mag_url).json()
    df_mag = pd.DataFrame(m_res[1:], columns=m_res[0])
    df_mag['time_tag'] = pd.to_datetime(df_mag['time_tag'])

    # Fetch Plasma data
    p_res = requests.get(plasma_url).json()
    df_plasma = pd.DataFrame(p_res[1:], columns=p_res[0])
    df_plasma['time_tag'] = pd.to_datetime(df_plasma['time_tag'])

    # Merge on timestamp
    df = pd.merge(df_mag, df_plasma, on='time_tag')

    # Clean and cast to float
    cols_to_fix = ['bx_gsm', 'by_gsm', 'bz_gsm', 'density', 'speed']
    for col in cols_to_fix:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    return df.dropna()


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
    pass