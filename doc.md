
Actionable Aurora Forecasting System

This project is a real-time, scalable machine learning pipeline designed to predict the ground visibility of the Aurora Borealis in three Swedish cities:
	•	Stockholm
	•	Luleå
	•	Kiruna

By combining interplanetary solar wind data with local meteorological constraints, the system produces an actionable visibility signal, rather than relying solely on a planetary geomagnetic activity index.

⸻

🌌 Project Overview

Traditional aurora alerts typically rely on the global Kp index, which measures geomagnetic disturbance but does not account for local weather conditions or geographic visibility.

This system addresses that limitation by integrating:
	•	Space Weather Physics
Real-time solar wind parameters, including magnetic field components and plasma density.
	•	Meteorological Constraints
Local cloud cover data used as a definitive visibility filter.
	•	Latitude-Specific Logic
Custom Kp thresholds per city to ensure actionable, location-aware alerts.

⸻

🛠️ System Architecture

The project is built on the Hopsworks Feature Store framework and follows a modular four-stage pipeline.

⸻

1. Feature Backfill

Notebook: 1_aurora_feature_backfill.ipynb

A historical dataset is constructed using three main data sources:
	•	Solar Wind Data
Historical measurements from NASA and NOAA (ACE and DSCOVR satellites) via the spacepy OMNI dataset.
Features include:
	•	bx_gsm
	•	by_gsm
	•	bz_gsm
	•	Plasma density
	•	Solar wind speed
	•	Target Labels
Historical Kp index values used as ground truth for model training.
	•	Weather Data
Historical cloud cover for each city retrieved from the Open-Meteo API.

⸻

2. Feature Pipeline

Notebook: 2_aurora_feature_pipeline.ipynb

A daily pipeline keeps the system synchronized with real-time conditions:
	•	Satellite Sync
Fetches 1-minute resolution solar wind data from the NOAA SWPC API.
	•	Weather Sync
Retrieves current cloud cover percentages for Stockholm, Luleå, and Kiruna.
	•	Feature Store Ingestion
Updates the following Hopsworks feature groups:
	•	solar_wind_fg
	•	city_weather_fg

⸻

3. Training Pipeline

Notebook: 3_aurora_training_pipeline.ipynb

A machine learning model is trained to map space weather conditions to geomagnetic activity:
	•	Model
Random Forest Regressor.
	•	Training Logic
Uses planetary-scale solar wind features to predict the Kp index.
	•	Model Registry
The trained model is versioned and stored in the Hopsworks Model Registry for deployment.

⸻

4. Batch Inference

Notebook: 4_aurora_batch_inference.ipynb

This is the final actionable stage that generates city-specific visibility signals:
	•	Inference
The model predicts the current Kp index using live satellite data.
	•	Visibility Logic
A Go / No-Go decision is generated per city using local thresholds:
	•	Kiruna: Kp ≥ 1.5
	•	Luleå: Kp ≥ 3.0
	•	Stockholm: Kp ≥ 5.0
	•	Cloud Override
If cloud cover exceeds 30%, visibility is reported as obstructed, even under strong geomagnetic activity.

⸻

📁 Repository Structure

.
├── config.py
│   Project configuration, city coordinates, and Kp thresholds
├── util.py
│   Helper functions for NOAA API ingestion, weather fetching, and visibility logic
├── 1_aurora_feature_backfill.ipynb
│   Historical data ingestion and feature group creation
├── 2_aurora_feature_pipeline.ipynb
│   Daily pipeline for real-time data synchronization
├── 3_aurora_training_pipeline.ipynb
│   Feature view creation and Random Forest model training
└── 4_aurora_batch_inference.ipynb
    Real-time prediction and city-level visibility reporting


⸻

📈 Key Outcome

The system automatically issues a “Go” visibility signal when both:
	•	Geomagnetic activity exceeds location-specific thresholds
	•	Local sky conditions are sufficiently clear

This enables a true ground-visible aurora estimation for both northern and southern Sweden, moving beyond generic planetary activity alerts.
