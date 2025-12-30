import os
from pathlib import Path
from pydantic import SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class HopsworksSettings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file='.env',
        env_file_encoding='utf-8',
        extra='ignore'
    )

    MLFS_DIR: Path = Path(__file__).parent

    # Hopsworks Credentials
    HOPSWORKS_API_KEY: SecretStr | None = None
    HOPSWORKS_PROJECT: str | None = None
    HOPSWORKS_HOST: str | None = None

    # Aurora Specific Settings
    # Thresholds: Kiruna is high latitude (low Kp needed), Stockholm is lower (high Kp needed)
    CITIES: dict = {
        "Kiruna": {"lat": 67.8557, "lon": 20.2251, "kp_threshold": 1.5},
        "Luleå": {"lat": 65.5848, "lon": 22.1567, "kp_threshold": 3.0},
        "Stockholm": {"lat": 59.3293, "lon": 18.0686, "kp_threshold": 5.0}
    }

    # API Endpoints
    NOAA_MAG_URL: str = "https://services.swpc.noaa.gov/products/solar-wind/mag-1-day.json"
    NOAA_PLASMA_URL: str = "https://services.swpc.noaa.gov/products/solar-wind/plasma-1-day.json"
    KP_INDEX_URL: str = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
    OPEN_METEO_URL: str = "https://api.open-meteo.com/v1/forecast"

    # Threshold for 'Clear Skies'
    MAX_CLOUD_COVER: int = 30  # %

    # Model configuration
    MODEL_NAME: str = "aurora_kp_rf_model"
    MODEL_VERSION: int = 1

    def model_post_init(self, __context):
        print("Aurora Project Settings initialized!")
        if os.getenv("HOPSWORKS_API_KEY") is None and self.HOPSWORKS_API_KEY:
            os.environ['HOPSWORKS_API_KEY'] = self.HOPSWORKS_API_KEY.get_secret_value()