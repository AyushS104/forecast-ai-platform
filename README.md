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

# ⚙️ Project Setup

## 1. Clone Repository

```bash
git clone https://github.com/yourusername/ai-sales-forecasting-system.git
```

---

## 2. Move Into Project Folder

```bash
cd ai-sales-forecasting-system
```

---

## 3. Create Virtual Environment

### Mac/Linux

```bash
python3 -m venv venv
```

### Windows

```bash
python -m venv venv
```

---

## 4. Activate Virtual Environment

### Mac/Linux

```bash
source venv/bin/activate
```

### Windows

```bash
venv\\Scripts\\activate
```

---

## 5. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 📂 Dataset Setup

Place the Excel dataset inside:

```bash
data/raw/
```

Example:

```bash
data/raw/sales.xlsx
```

---

# 🧹 Run Data Preprocessing

```bash
python app/preprocessing/clean_data.py
```

This will:
- handle missing dates
- handle missing values
- create cleaned dataset

---

# 🧠 Run Feature Engineering

```bash
python app/preprocessing/feature_engineering.py
```

This will generate:
- lag features
- rolling statistics
- date-based features

---

# 🤖 Train Forecasting Models

## XGBoost

```bash
python app/models/xgboost_model.py
```

---

## SARIMA

```bash
python app/models/sarima_model.py
```

---

## Prophet

```bash
python app/models/prophet_model.py
```

---

## LSTM

```bash
python app/models/lstm_model.py
```

---

# 🏆 Select Best Model

```bash
python app/services/model_selector.py
```

The system automatically compares RMSE values and selects the best model.

---

# ▶️ Run FastAPI Backend

```bash
uvicorn app.main:app --reload
```

Server will start at:

```bash
http://127.0.0.1:8000
```

Swagger API Docs:

```bash
http://127.0.0.1:8000/docs
```

---

# ⚡ API Endpoints

## Health Check

```bash
/health
```

---

## Best Model

```bash
/best-model
```

---

## Forecast API

```bash
/forecast/{state}
```

Example:

```bash
/forecast/Alabama
```

---

# 📊 Run Streamlit Dashboard

Open another terminal and run:

```bash
streamlit run dashboard.py
```

Dashboard URL:

```bash
http://localhost:8501
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
