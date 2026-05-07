# 🚀 AI Sales Forecasting System

An end-to-end time series forecasting platform built using Machine Learning, Deep Learning, FastAPI, and Streamlit.

This system predicts future state-wise sales for the next 8 weeks using historical sales data.

---

# 📌 Features

✅ Data preprocessing pipeline  
✅ Missing date handling  
✅ Missing value interpolation  
✅ Feature engineering  
✅ Multiple forecasting models  
✅ Automatic best model selection  
✅ REST API using FastAPI  
✅ Interactive Streamlit dashboard  
✅ Forecast visualization & AI insights  

---

# 🤖 Models Implemented

- SARIMA
- Facebook Prophet
- XGBoost
- LSTM Deep Learning

---

# 🧠 Feature Engineering

Implemented features include:

## Lag Features
- lag_1
- lag_7
- lag_30

## Rolling Statistics
- rolling_mean_7
- rolling_std_7

## Date Features
- day_of_week
- month
- week_of_year
- holiday_flag

---

# 🏗️ Project Architecture

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
├── requirements.txt
├── README.md
└── DOCUMENTATION.md
```

---

# 📈 Model Performance

| Model | RMSE |
|---|---|
| LSTM | ~3.38M |
| XGBoost | ~9.54M |
| Prophet | ~23.29M |
| SARIMA | ~27.05M |

🏆 Best Model Selected: **LSTM**

---

# ⚡ FastAPI Endpoints

## Health Check

```bash
/health
```

## Best Model

```bash
/best-model
```

## Forecast API

```bash
/forecast/{state}
```

Example:

```bash
/forecast/Alabama
```

---

# 📊 Streamlit Dashboard

The project includes an interactive dashboard with:

- Forecast charts
- KPI metrics
- AI insights
- Downloadable CSV reports
- State-wise prediction analysis

Run dashboard:

```bash
streamlit run dashboard.py
```

---

# ▶️ Run FastAPI Server

```bash
uvicorn app.main:app --reload
```

Swagger API Docs:

```bash
http://127.0.0.1:8000/docs
```

---

# 🛠️ Tech Stack

- Python
- Pandas
- NumPy
- Scikit-learn
- TensorFlow / Keras
- XGBoost
- Prophet
- Statsmodels
- FastAPI
- Streamlit
- Plotly

---

# 📌 Future Improvements

- Docker deployment
- AWS cloud deployment
- PostgreSQL integration
- CI/CD pipelines
- Real-time forecasting
- Ensemble forecasting models

---

# 👨‍💻 Author

Tathagat Gupta

---

# 📷 Project Highlights

✅ End-to-end forecasting pipeline  
✅ Production-style backend architecture  
✅ AI-powered prediction system  
✅ Interactive visualization dashboard  
✅ Multi-model comparison framework  
✅ REST API deployment  

---
