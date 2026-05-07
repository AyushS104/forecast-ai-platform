# 📘 AI Sales Forecasting System Documentation

---

# 1. Project Overview

The AI Sales Forecasting System is an end-to-end time series forecasting platform developed to predict future sales for different states using historical sales data.

The system is designed like a real-world production backend service and includes:

- Data preprocessing pipeline
- Feature engineering
- Multiple forecasting models
- Automatic best model selection
- REST API using FastAPI
- Interactive dashboard using Streamlit

The primary objective is to forecast the next 8 weeks (56 days) of sales for each state.

---

# 2. Objectives

The main objectives of the project are:

- Forecast future sales using historical time-series data
- Handle missing dates and missing values
- Capture trend and seasonality
- Compare multiple forecasting models
- Automatically select the best-performing model
- Serve predictions through REST APIs
- Visualize predictions through dashboard

---

# 3. Dataset Description

The dataset contains historical sales records with the following columns:

| Column | Description |
|---|---|
| State | Name of the state |
| Date | Sales date |
| Sales | Sales amount |
| Category | Product category |

The dataset was provided in Excel format.

---

# 4. Project Architecture

```bash
forecasting-system/
│
├── app/
│   ├── api/
│   ├── core/
│   ├── models/
│   ├── preprocessing/
│   ├── services/
│   ├── utils/
│   └── main.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── saved_models/
├── dashboard.py
├── Dockerfile
├── README.md
├── requirements.txt
└── DOCUMENTATION.md
```

---

# 5. Data Preprocessing

The preprocessing pipeline performs the following tasks:

## 5.1 Date Conversion

The Date column is converted into datetime format for time-series processing.

## 5.2 Sorting

The dataset is sorted by:

- State
- Date

to maintain chronological order.

## 5.3 Missing Date Handling

Continuous daily date ranges are created using pandas `date_range()` and `reindex()`.

This ensures there are no missing timestamps in the time series.

## 5.4 Missing Value Handling

Missing sales values are handled using:

- Interpolation
- Backward filling

This ensures continuity in forecasting data.

---

# 6. Feature Engineering

Feature engineering is one of the most critical components of forecasting.

The following features were created:

## 6.1 Lag Features

| Feature | Description |
|---|---|
| lag_1 | Previous day sales |
| lag_7 | Previous week sales |
| lag_30 | Previous month sales |

## 6.2 Rolling Statistics

| Feature | Description |
|---|---|
| rolling_mean_7 | 7-day moving average |
| rolling_std_7 | 7-day moving standard deviation |

## 6.3 Date-Based Features

| Feature | Description |
|---|---|
| day_of_week | Day index |
| month | Month number |
| week_of_year | Week number |
| holiday_flag | Holiday indicator |

These features help capture trend, seasonality, and temporal dependencies.

---

# 7. Forecasting Models

The following forecasting models were implemented and compared.

## 7.1 SARIMA

SARIMA is a statistical forecasting model capable of handling trend and seasonality.

## 7.2 Facebook Prophet

Prophet is designed for business forecasting and supports trend shifts and seasonality.

## 7.3 XGBoost

XGBoost is a machine learning model trained using engineered lag and rolling features.

## 7.4 LSTM

LSTM is a deep learning recurrent neural network capable of learning sequential temporal dependencies.

---

# 8. Train-Test Split

The dataset was split using time-series logic.

- 80% Training Data
- 20% Testing Data

Random shuffling was avoided to prevent future data leakage.

---

# 9. Model Evaluation

The models were evaluated using:

- MAE (Mean Absolute Error)
- RMSE (Root Mean Squared Error)

The best model was selected automatically based on the lowest RMSE value.

---

# 10. Model Performance

| Model | RMSE |
|---|---|
| LSTM | ~3.38M |
| XGBoost | ~9.54M |
| Prophet | ~23.29M |
| SARIMA | ~27.05M |

Best Model Selected: **LSTM**

---

# 11. REST API

The forecasting system exposes predictions using FastAPI.

## API Endpoints

### Health Check

```bash
/health
```

### Best Model

```bash
/best-model
```

### Forecast API

```bash
/forecast/{state}
```

Example:

```bash
/forecast/Alabama
```

The API returns:

- Selected state
- Forecast horizon
- Model used
- Predicted future sales values

---

# 12. Dashboard

A Streamlit dashboard was developed for visualization.

Features include:

- Interactive forecast charts
- KPI metrics
- AI insights
- Downloadable forecast reports
- State-wise analysis

---

# 13. Technologies Used

| Technology | Purpose |
|---|---|
| Python | Core programming |
| Pandas | Data preprocessing |
| NumPy | Numerical operations |
| Scikit-learn | ML utilities |
| TensorFlow/Keras | LSTM model |
| XGBoost | Gradient boosting |
| Prophet | Forecasting |
| Statsmodels | SARIMA |
| FastAPI | REST API |
| Streamlit | Dashboard |
| Plotly | Visualization |

---

# 14. Future Improvements

Possible future improvements include:

- Docker deployment
- Cloud deployment using AWS
- CI/CD pipelines
- PostgreSQL integration
- Real-time streaming forecasts
- Multi-model ensemble forecasting

---

# 15. Conclusion

The AI Sales Forecasting System successfully demonstrates a complete end-to-end forecasting pipeline.

The system includes:

- Data preprocessing
- Feature engineering
- Multiple forecasting models
- Automatic model selection
- REST API deployment
- Dashboard visualization

This project reflects a production-style AI forecasting architecture suitable for real-world business forecasting applications.

---
