# ⚡ WattWise — Smart Electricity Usage Predictor

> AI-powered electricity bill forecasting built with Python, Prophet, Streamlit & SQLite

![Python](https://img.shields.io/badge/Python-3.11-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.x-red?style=flat-square&logo=streamlit)
![Prophet](https://img.shields.io/badge/Prophet-1.1-orange?style=flat-square)
![SQLite](https://img.shields.io/badge/SQLite-3-lightblue?style=flat-square&logo=sqlite)
![License](https://img.shields.io/badge/License-MIT-green?style=flat-square)

---

## Problem Statement

Households receive electricity bills at the end of the month with no way to anticipate the amount. There is no simple, free, personal tool that predicts your specific usage based on your own historical data — without requiring smart meters, utility account access, or expensive hardware.

---

## Solution

WattWise is a machine learning web application that:
- Accepts your monthly electricity readings (kWh from your bill)
- Stores them locally in a SQLite database
- Trains a Facebook Prophet time series model on your personal data
- Predicts your electricity usage and estimated bill for the next 1–12 months
- Shows interactive dashboards with usage trends, breakdowns, and saving tips

---

## Live Demo

> **[Launch WattWise](https://your-app-url.streamlit.app)** ← replace with your deployed URL after deployment

---

## Features

- **Personalised AI Predictions** — model retrains on your own data for accurate forecasts
- **Interactive Dashboard** — usage trend, month-by-month breakdown, usage split chart
- **Bill Forecast Table** — min/expected/max bill range for each predicted month
- **Inline Edit & Delete** — manage entries directly in the data table with icon buttons
- **SQLite Storage** — data persists across sessions without any login required
- **CSV Export** — download your data anytime
- **Energy Saving Tips** — automatic suggestions based on your usage pattern
- **Fully Custom UI** — built with Outfit font, custom CSS on Streamlit

---

## Model Details

| Metric | Value |
|--------|-------|
| Model | Facebook Prophet (Meta) |
| Training Data | UCI Household Power Consumption (Kaggle) |
| Dataset Size | 2+ million readings (2006–2010) |
| MAPE | **10.48%** |
| MAE | 71.77 kWh |
| RMSE | 91.29 kWh |
| Seasonality | Yearly (auto-detected when 13+ months available) |
| Prediction Mode | Additive, conservative changepoint (0.05) |

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Frontend | Streamlit + Custom CSS |
| ML Model | Facebook Prophet |
| Data Cleaning | Pandas, NumPy |
| Visualisation | Plotly |
| Database | SQLite |
| Language | Python 3.11 |
| Deployment | Streamlit Cloud (free) |

---

## Project Structure

```
smart_electricity_usage_predictor/
│
├── app/
│   └── app.py                           ← Main Streamlit application
│
├── data/
│   ├── household_power_consumption.txt  ← Raw Kaggle dataset
│   ├── cleaned_daily.csv                ← Cleaned daily data
│   ├── cleaned_monthly.csv              ← Cleaned monthly data
│   ├── forecast.csv                     ← Saved forecast output
│   ├── monthly_usage_plot.png           ← Usage trend chart
│   └── forecast_plot.png                ← Forecast chart
│
├── models/
│   └── prophet_model.pkl                ← Trained Prophet model
│
├── notebooks/
│   ├── 01_data_cleaning.ipynb           ← Data cleaning & preparation
│   └── 02_model_building.ipynb          ← Model training & evaluation
│
├── requirements.txt                     ← Python dependencies
└── README.md                            ← This file
```

---

## Installation & Setup

### 1. Clone the repository
```bash
git clone https://github.com/Snehar273/wattwise.git
cd smart_electricity_usage_predictor
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the application
```bash
cd app
streamlit run app.py
```

### 4. Open in browser
```
http://localhost:8501
```

---

## How to Use

1. **Add Monthly Entry** — Enter your month, year, and kWh usage from the sidebar. Find the kWh reading on your electricity bill.
2. **Set Your Rate** — Enter the electricity rate in Rs/kWh (default: Rs 6).
3. **View Dashboard** — See your usage trend, seasonal patterns, and month-by-month breakdown.
4. **Get Predictions** — Go to the Predictions tab, select how many months ahead, and view your AI-powered bill forecast.
5. **Edit or Delete** — Use the checkmark and X icon buttons in the My Data tab to manage entries.
6. **Download** — Export your data as CSV anytime.

---

## Sample Data

| Month | Year | kWh | Context |
|-------|------|-----|---------|
| January | 2024 | 420 | Winter — heater usage high |
| February | 2024 | 390 | Shorter month |
| March | 2024 | 310 | Spring — mild weather |
| April | 2024 | 280 | Lowest usage month |
| May | 2024 | 340 | Summer begins |
| June | 2024 | 510 | Peak summer — heavy AC |
| July | 2024 | 530 | Hottest month |
| August | 2024 | 490 | Still hot |
| September | 2024 | 380 | Monsoon |
| October | 2024 | 290 | Festival season |
| November | 2024 | 320 | Cooler nights |
| December | 2024 | 400 | Winter heating |

---

## Data Pipeline

```
Raw Dataset (2M+ readings, per-minute)
        ↓
Data Cleaning  →  Parse datetime, handle missing values, convert to kWh
        ↓
Resampling     →  Per-minute → Daily → Monthly totals
        ↓
Model Training →  Facebook Prophet, additive mode, conservative changepoints
        ↓
Prediction     →  Clamp to [60% min, 140% max] of historical data
        ↓
Dashboard      →  Streamlit interactive UI with Plotly charts
```

---

## Requirements

```
streamlit
pandas
numpy
plotly
prophet
scikit-learn
```

---

## Future Improvements

- Multi-user support with login/authentication
- Integration with utility provider APIs for automatic data import
- Anomaly detection to flag unusually high usage months
- Email alerts when predicted bill crosses a user-defined threshold
- Support for multi-slab electricity tariffs (common in India)

---

## Dataset

**UCI Household Power Consumption Dataset**
- Source: [Kaggle](https://www.kaggle.com/datasets/uciml/electric-power-consumption-data-set)
- Interval: 1-minute readings
- Period: December 2006 – November 2010
- Records: 2,075,259 rows

---

## Author

**Your Name**  
GitHub: [@Sneha](https://github.com/Snehar273)  
LinkedIn: [Sneha](https://www.linkedin.com/in/sneha-r-b90866290)

---

## License

This project is licensed under the MIT License.

---

*Built as a portfolio project demonstrating end-to-end ML application development — from raw data cleaning to deployed web app.*
