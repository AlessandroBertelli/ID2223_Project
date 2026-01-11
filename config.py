import os
from pathlib import Path
from typing import Literal
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
    CITIES: dict = {
        "Kiruna": {"lat": 67.8557, "lon": 20.2251, "kp_threshold": 1.5},
        "Luleå": {"lat": 65.5848, "lon": 22.1567, "kp_threshold": 3.0},
        "Stockholm": {"lat": 59.3293, "lon": 18.0686, "kp_threshold": 5.0}
    }

    # API Endpoints
    # These can also be moved to Hopsworks Secrets later if preferred
    NOAA_MAG_URL: str = "https://services.swpc.noaa.gov/products/solar-wind/mag-1-day.json"
    NOAA_PLASMA_URL: str = "https://services.swpc.noaa.gov/products/solar-wind/plasma-1-day.json"
    KP_INDEX_URL: str = "https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json"
    
    MODEL_3H_NAME: str = "aurora_kp_3h_model"
    MODEL_3H_VERSION: int = 1
    MODEL_6H_NAME: str = "aurora_kp_6h_model"
    MODEL_6H_VERSION: int = 1

    def model_post_init(self, __context):
        """Runs after the model is initialized to sync environment variables."""
        print("HopsworksSettings initialized!")

        if os.getenv("HOPSWORKS_API_KEY") is None and self.HOPSWORKS_API_KEY:
            os.environ['HOPSWORKS_API_KEY'] = self.HOPSWORKS_API_KEY.get_secret_value()
        
        if os.getenv("HOPSWORKS_PROJECT") is None and self.HOPSWORKS_PROJECT:
            os.environ['HOPSWORKS_PROJECT'] = self.HOPSWORKS_PROJECT

        # Check required credentials
        missing = []
        if not (self.HOPSWORKS_API_KEY or os.getenv("HOPSWORKS_API_KEY")):
            missing.append("HOPSWORKS_API_KEY")
        if not (self.HOPSWORKS_PROJECT or os.getenv("HOPSWORKS_PROJECT")):
            missing.append("HOPSWORKS_PROJECT")

        if missing:
            raise ValueError(
                f"The following required settings are missing: {', '.join(missing)}"
            )