# Actionable Aurora Forecasting System 🌌

> **Real-time visibility prediction for the Northern Lights in Stockholm, Luleå, and Kiruna.**

![Project Status](https://img.shields.io/badge/Status-Active-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![Machine Learning](https://img.shields.io/badge/ML-Random%20Forest-orange)
![API](https://img.shields.io/badge/Data-NOAA%20%7C%20OpenMeteo-lightgrey)

## 📖 Project Overview

The **Actionable Aurora Forecasting System** is a machine learning project developed for the **Scalable Machine Learning** course. Its primary goal is to predict not only the *theoretical* presence of geomagnetic activity but the actual *visibility* of the Aurora Borealis from the ground.

Traditional aurora applications typically rely solely on the **Kp index**. While this captures global geomagnetic disturbances, it does not indicate whether the aurora will truly be visible to an observer. The most significant bottleneck for aurora hunting is **cloud cover**.

Our system bridges this gap by creating a unified real-time inference pipeline that fuses:
1.  **Solar-wind driven space-weather data** (to predict geomagnetic storms).
2.  **Local meteorological conditions** (to predict sky clarity).

The result is a genuine **"Visible Aurora Event"** estimator for three Swedish cities: **Stockholm**, **Luleå**, and **Kiruna**.

---

## 🎯 The Problem & Solution

### The Gap in Current Solutions
Most existing forecasting tools operate on a single dimension:
* **The Scenario:** The Solar Wind is fast, and the Kp index hits 7 (Storm level).
* **The Reality:** It is 100% overcast in Luleå.
* **The User Experience:** The user receives a notification, goes outside, and sees nothing but grey clouds.

### Our Approach
We treat Aurora Forecasting as a **Dual-Constraint Problem**:
1.  **Geomagnetic Constraint:** Is the solar wind energetic enough to trigger an aurora? (Predicted via ML).
2.  **Visibility Constraint:** Is the sky clear enough to see it? (Retrieved via Weather API).

Our system issues a **"Go" signal** only when *both* conditions are satisfied simultaneously.

---

## ⚙️ Technical Architecture

The project consists of two main workflows: **Historical Training** and **Real-Time Inference**.

### 1. Data Sources & Aggregation
We construct a historical dataset by aggregating archives from NASA and NOAA.
* **Satellites:** ACE (Advanced Composition Explorer) and DSCOVR (Deep Space Climate Observatory).
* **Features:**
    * **Magnetic Field Components:** $B_x$, $B_y$, $B_z$ (GSE/GSM coordinates).
    * **Plasma Parameters:** Proton Density, Solar Wind Speed, Temperature.
* **Target Variable:** Kp Index (Planetary K-index) or Disturbance Storm Time ($Dst$).

### 2. Machine Learning Model
We utilize a **Random Forest Regressor** to map the raw solar wind parameters to the geomagnetic activity level.
* **Preprocessing:** Cleaning missing satellite logs, time-series alignment, and normalization.
* **Training:** The model learns the lag between solar wind measurement (at the L1 Lagrange point) and the impact on Earth's magnetosphere.

### 3. The Inference Pipeline
During live operation, the system follows this logic flow:

1.  **Fetch Live Space Data:** Query the NOAA SWPC API for real-time solar wind stats.
2.  **Predict Activity:** The ML model estimates the current/upcoming Kp index.
3.  **Fetch Local Weather:** Query the **Open-Meteo API** for cloud cover percentages in Stockholm, Luleå, and Kiruna.
4.  **Decision Logic:**
    ```python
    if (Predicted_Kp >= Kp_Threshold) and (Cloud_Cover <= Cloud_Threshold):
        return "VISIBLE_EVENT"
    else:
        return "NO_EVENT"
    ```

---

## 📍 Target Locations

We evaluate performance across different latitudes to test the model's sensitivity:

| City | Latitude | Significance |
| :--- | :--- | :--- |
| **Kiruna** | ~67.8° N | **High Arctic:** Auroras are visible even at low Kp levels. |
| **Luleå** | ~65.5° N | **Sub-Arctic:** Requires moderate activity. |
| **Stockholm** | ~59.3° N | **Mid-Latitude:** Requires strong geomagnetic storms (High Kp). |

---

## 📂 Repository Structure

```bash
├── data/
│   ├── raw/                  # Raw logs from NASA/NOAA archives
│   └── processed/            # Cleaned time-series data ready for training
├── models/
│   └── rf_regressor.pkl      # Serialized trained model
├── src/
│   ├── data_ingestion.py     # Scripts to fetch ACE/DSCOVR data
│   ├── preprocessing.py      # Feature engineering and time alignment
│   ├── train.py              # Model training logic
│   ├── weather_api.py        # Integration with Open-Meteo
│   └── inference.py          # Main pipeline for real-time predictions
├── notebooks/                # EDA and prototyping
├── requirements.txt          # Dependencies
└── README.md                 # Project documentation